---
name: qec-decoder-hackathon
description: Workflow for the 2026 QEC decoder hackathon in this repo ([[18,4,4]] qLDPC code, 70 detectors -> 4 observables, Kaggle quantum-error-correction-decoder). Use when building, benchmarking, tuning, comparing or submitting a decoder here, or when asked about results so far.
---

# QEC decoder hackathon workflow

Read `team/FINDINGS.md` first. It has every result so far, what worked, what didn't, and the ranked next ideas.
Don't re-run experiments it already answers.

## Hard rules (from the README and the Kaggle rules)

- **Write the decoder yourself.** Never import or copy pymatching, ldpc, stimbposd, beliefmatching,
  tesseract-decoder, fusion-blossom, or pretrained decoder models made by others. numpy, scipy, numba, torch, jax and
  sklearn are fine. Tutorial notebooks use banned decoders, so never carry their code over.
- **Hand-in signature:** `submission_template/decoder.py` defines `decode(circuit: stim.Circuit, detector_events:
  (shots, 70) bool) -> (shots, 4) 0/1`. Everything inside it is timed, including building matrices and loading
  weights.
- **Be deterministic.** The predictions must reproduce the selected Kaggle submission, so fix seeds.
- **Scoring:**
  - Leaderboard: per-bit LER.
  - Top 10 final score: LER × max(1, t / t_cap)^0.2, with **t_cap = 15.4 s** (organisers' BP+OSD, 32 threads) for
    2×100k shots. We're timed with 48 threads on 48 EPYC cores + B200. Hard limit 5 minutes. Each doubling beyond
    t_cap costs about 15%.
  - **Only the two `decode` calls are timed;** import-time work (building matrices, JIT warm-up) is free.
- **Kaggle:** about 5–10 submissions per day, shared by the team. Submitting is outward-facing, so confirm with the
  user first.

## Environment

- **Python env:** `uv sync --all-extras --group team` (numba, scipy, torch are in the `team` group). Run everything
  from the repo root with `uv run python …`.
- **Kaggle CLI:** `uvx --python 3.13 --with six kaggle …`. The token is already in `~/.kaggle/access_token`.
- **Data:**
  - `data/train` (labelled) and `data/test` (unlabelled) are identical to Kaggle's `train.csv` / `test.csv`, which
    are mirrored in `data/kaggle/` (git-ignored).
  - The circuits rebuilt by `team/common.circuit(basis)` match the data exactly.
- **Generated data:** `team/common.generate(basis, shots, seed)` caches bit-packed shots in `data/generated/`
  (git-ignored).
  - Seeds 2000/2001 = the shared validation set.
  - Seeds ≥ 10000 = NN training.
  - Seeds 30000+ = the lookup table.

## How to evaluate a change (always paired)

1. Put experimental decoders in `team/decoders/<name>.py` with the same `decode()` signature.
2. Quick check: `uv run python team/bench.py --decoder team/decoders/<name>.py [--fresh 1000000]`.
3. Compare variants with `team.compare.Table`: every variant decodes the same validation shots, and it prints a
   paired Δ ± SE against the first row. With 2×50k shots, a Δ smaller than about 2 SE is noise.
4. **Time it with nothing else running** (NN training on the same machine inflates CPU timings), using
   `submission_template/run_local.py --runs 5`.
5. Report LER, time for 200k shots, and the final score under t_cap = 3 s.

## Current best building blocks (see FINDINGS.md for numbers)

- **Decoder:** `team/decoders/bposd2.py`. `decode_full()` returns the most-likely-error bits, the 16-class posterior
  and the per-bit marginals.
- **Best decoder:** `submission_ml/` (exact maximum-likelihood transfer matrix, `team/decoders/mldfast.py`):
  LER 0.03386 on `data/train`, Kaggle public 0.03367 (ref 58318168). Profiles `max` (default, ~110 s on a 10-core
  M4) and `fast` (`MLD_PROFILE=fast`, ~7 s). Run with `run_local.py --decoder submission_ml`.
- **Previous BP-OSD hand-in:** `submission_template/decoder.py`, a three-level cascade (cheap → 0.49 heavy → 0.4 SUPER):
  LER 0.03459, 13.4 s on an 18-core Mac, with import-time warm-up.
- **Cheap configuration (LER 0.0371, ~2.4 s on an 18-core Mac):** `schedule=1` (layered), `max_iter=2`,
  `alpha=0.45`, `snaps=(2,)`, `n_single=200`, `n_pair=30`, `degenerate=True`, `shortcut=True`.
- **Cascade:** `team/decoders/cascade.py`. Re-decode shots with min |P − 0.5| < τ using HEAVY (3 orderings, 800/60).
  τ = 0.3 gives LER 0.0358 at ~1.8× the time.
- **Neural network:** `team/nn/train_nn.py` (ResMLP). It's under-trained at 0.047; it needs much longer GPU training.

## Gotchas

- **Numba caches by module name.** Import a decoder file under its real module name (`bench.py` does this). After
  renaming or copying decoder files, delete `submission_template/__pycache__/decoder.*.nb[ic]`.
- **macOS Accelerate** warns on matmuls with subnormal probabilities; use plain sums (as `bposd2.decode_full` does).
- **Few BP iterations beat many.** Even and odd iteration counts differ; retune α whenever the schedule changes.
- **Re-estimating noise is pointless:** the priors from `problem_info` are exact for this data.
- **Before handing in:**
  - Copy `team/decoders/bposd2.py` into `submission_template/`.
  - Update `submission_template/requirements.txt`.
  - Run `run_local.py` on `data/train` (LER) and on `data/test` (the submission file).
