# QEC decoder hackathon 2026: findings

Team quantum_crew_ku, as of 2026-11-13 (afternoon). All numbers were measured in this repo with the scripts listed
at the end. Every variant in a comparison table decodes the same fresh validation shots, so the Δ values are paired
differences (seeds 2000/2001, 50k shots per basis unless noted).

## Summary

- **Task:** read the 70 detector bits of each shot of the `[[18,4,4]]` qLDPC memory experiment and predict the 4
  logical observable flips. Predicting all zeros gives LER 0.288.
- **Current best submitted decoder:** degenerate BP-OSD, written from scratch in numpy + numba.
  - Layered min-sum BP: 2 iterations, α = 0.45.
  - OSD-CS: 200 single flips and 30 paired flips.
  - Decision: each observable is decided by summing the probability of every candidate error per logical class.
  - **LER 0.0371** on `data/train`, about 3.0 s for 200k shots on an 18-core Mac.
  - Submitted to Kaggle as ref 58309242.
- **Biggest single gain:** summing probability per logical class instead of trusting the single most likely error
  (−6% LER).
- **Most accurate so far:**
  - Heavy OSD: 0.0349, but 8× slower.
  - Confidence cascade: 0.0355–0.0358 at 1.6–2× the cost.
- **Not useful (yet):** lookup tables, prior scaling, long BP runs, the neural network (0.047 after 12 minutes of
  training per basis), and NN hybrids.

## The problem in brief

| Item | Value |
|---|---|
| Code | `[[18,4,4]]` qLDPC (Wang et al. 2026): 18 data qubits (blocks L0–L8, R0–R8), 7 X and 7 Z weight-6 checks, 32 qubits in total. The circuit-level distance drops to 3 |
| Noise | SI1000, p = 0.002, 5 rounds; separate X and Z memory circuits |
| Per shot | 70 detectors in (7 + 4×14 + 7), 4 observables out |
| Data | `data/train`: 200k labelled shots. `data/test` (= Kaggle `test.csv`): 200k unlabelled, split 30% Public / 70% Private. Stim gives unlimited extra labelled data |
| Metric | Per-bit LER: the mean of wrong predictions over shots × 4 observables |
| Final score (top 10) | LER × max(1, t / t_cap)^0.1, t_cap ≈ 3 s for 2×100k shots. Each doubling of runtime costs about 7%; hard limit 5 minutes |
| Rules | Own decoder only. Banned: pymatching, ldpc, stimbposd, beliefmatching, tesseract-decoder, fusion-blossom, others' pretrained models. Allowed: numpy, scipy, numba, torch, jax, … |

The hand-in signature is `decode(circuit, detector_events) -> (shots, 4)`. Everything inside `decode` is timed,
including building the matrices.

## Error and data analysis

- **DEM size:** about 2470 merged mechanisms per basis. `H` is 70 × ~2470, with ~10.7k nonzeros and full rank 70.
- **Errors per shot:** about 3.1 on average; only 4.4% of shots have no error.
- **Hyperedges:** mechanisms trigger 1–7 detectors, mostly 3–7. Matching-style decoders don't fit.
- **Probability range:** 1.3×10⁻⁴ to 2.3×10⁻².
- **Most common errors:** ancilla measurement and reset faults, p ≈ 0.016–0.023, triggering 1–2 detectors. About 60%
  of all errors are these, and they rarely flip observables.
- **Most dangerous errors:** data-qubit errors while idling during measurement layers, p ≈ 0.0037, triggering 2–6
  detectors. 59% of them flip an observable.
- **Long tail:** about 1150 single-CZ faults (p/15 ≈ 1.3×10⁻⁴) triggering up to 7 detectors. They cause most of the
  hard syndromes.
- **Syndrome diversity:** about 7.4 detectors fire per shot. 200k shots contain about 152k distinct syndromes; 20M
  shots cover only 49% of fresh shots.
- **Cancelling triples:** about 9.6k triples with col_a ⊕ col_b = col_c; 341–356 of them flip an observable.
- **The priors are exact:** the Kaggle data comes from exactly these circuits (verified), so re-estimating noise from
  the data cannot help.
- **Worked example (verified in Stim):** an X error on L0 after round 1 fires Z0, Z1 and Z6 (D28, D29, D32) and flips
  L0, L1 and L3. Layered BP converges on it in 2 iterations with the correct answer.

## What worked

### 1. Very short BP

| BP (OSD-CS 100/20) | LER |
|---|---|
| flooding, 30 iterations, α = 0.8 | 0.0971 |
| flooding, 10 iterations, α = 0.6 | 0.0710 |
| flooding, 3 iterations, α = 0.45 | 0.0413 |

Longer BP oscillates and gets overconfident, which gives OSD a worse column ordering. Even and odd iteration counts
behave differently.

### 2. A large OSD-CS is cheap here

The information set is found after scanning only about 70–150 columns, and each re-solve is a few 64-bit XORs.

| Singles / pairs (flooding, 3 it, α = 0.45) | LER | µs/shot |
|---|---|---|
| 40/12 | 0.0438 | 15.9 |
| 200/30 | 0.0403 | 18.5 |
| 400/40 | 0.0397 | 22.5 |
| 1500/80 | 0.0387 | 47.9 |

### 3. Total probability per logical class (degenerate decoding): the biggest gain

Every distinct valid error found by OSD-CS adds exp(−cost) to its logical class (16 classes). Duplicates are removed
with Zobrist hashes, and each bit is decided by its marginal probability, which is optimal for the per-bit metric.

| Variant (flooding, 3 it) | LER | Δ |
|---|---|---|
| most likely error | 0.04043 | — |
| degenerate 200/30 | 0.03793 | −0.0025 ± 0.0002 (−6%) |
| degenerate 800/60 | 0.03624 | −0.0042 |

The huge candidate set judged by its single most likely error (0.0382) loses to the cheap degenerate decoder
(0.0372).

### 4. Layered (serial) BP schedule

Layered BP, 2 iterations, α = 0.45: 0.03944 against 0.04032 for flooding (−0.0009 ± 0.0002), and slightly faster.
Speed gains are limited because OSD dominates the time. Combined with the degenerate sum it gives **0.0368 at v1
speed**.

### 5. Skipping OSD when BP converges

Same LER (Δ < 0.0001), about 15% faster.

### 6. Confidence cascade

The cheap pass runs everywhere. Shots whose least certain bit has |P − 0.5| < τ are re-decoded with the heavy
configuration (3 OSD orderings, 800/60).

| τ | Shots sent to heavy | LER | Δ vs cheap |
|---|---|---|---|
| 0.1 | 2.5% | 0.03620 | −0.0006 |
| 0.2 | 6.2% | 0.03602 | −0.0008 |
| 0.3 | 8.1% | 0.03580 | −0.0010 |
| 0.4 | 11% | 0.03559 | −0.0012 |

### 7. Multiple OSD orderings

Using the posteriors after BP iterations 1, 2 and 3 as three separate seeds gives −0.0006 to −0.0009 at 2–2.5× the
cost. It's best used only inside the heavy pass.

## What did not work

| Idea | Result | Why |
|---|---|---|
| Lookup table alone | LER 0.253 (200k-shot table) → 0.202 (20M-shot table) | Only 28–49% of syndromes are covered; unseen ones have LER ~0.38 |
| Lookup table (≥ 80 counts) in front of BP-OSD | Δ = 0.0000, slower | BP-OSD already gets the common syndromes right; covered shots hold only 10% of the flips |
| Long BP (10–30 iterations) | +0.003 to +0.007 worse | Oscillation and overconfidence |
| Scaling the BP priors (×0.7 … ×1.2) | Δ = 0 exactly | Min-sum is invariant to scaling all LLRs by the same factor |
| Prior ordering only (0 BP iterations) | LER 0.185 | OSD needs BP's posterior to order the columns |
| Triple-swap local search (top 20 candidates) | −0.0004 at 2–4× cost | Worse value than simply enlarging OSD-CS |
| NN alone (ResMLP 8.5M params, 64M shots, 12 min per basis) | 0.0474 (X 0.0481, Z 0.0456) | Under-trained; still improving when stopped |
| Averaging NN and BP-OSD probabilities | at best Δ = −0.00005 (0.3 NN weight) | The NN adds no information yet |
| NN decides, BP-OSD on uncertain shots | +0.0005 to +0.0012 worse | The NN is overconfident when wrong |
| Mega OSD (4 orderings, 3000/120, local search) | 0.0349, 50× cost | The OSD-family ceiling: near 0.035 |

## Scoreboard (full `data/train`, 2×100k shots, 18-core Mac, no other load)

| Decoder | LER | Time | Score if timing machine = Mac | Score if timing machine is 2× faster |
|---|---|---|---|---|
| All zeros | 0.288 | 0 s | 0.288 | 0.288 |
| v1 BP-OSD (flooding, most likely error) | 0.04045 | 2.98 s | 0.0405 | 0.0405 |
| **Cheap: layered + degenerate (submitted)** | **0.03706** | **3.00 s** | **0.0371** | 0.0371 |
| Cascade τ = 0.2 | 0.03612 | 4.69 s | 0.0378 | 0.0361 |
| Cascade τ = 0.3 | 0.03579 | 5.36 s | 0.0379 | **0.0358** |
| Cascade τ = 0.4 | 0.03552 | 6.16 s | 0.0382 | 0.0356 |
| NN gate + heavy BP-OSD | 0.03599 | 6.88 s | 0.0386 | 0.0362 |
| Heavy on every shot | 0.03494 | 24.3 s | 0.0431 | 0.0402 |

Score = LER × max(1, t / 3 s)^0.1, using the provisional t_cap. The real speed ratio of the timing machine
(48 EPYC 9655 cores + B200) to the Mac is unknown.

## What to try next (ranked)

1. **Train the NN much longer, on the GPU.**
   - Hours, hundreds of millions of shots, bigger width or depth.
   - Validation LER was still dropping (0.053 → 0.048 over the last 28M shots).
   - Inference on a B200 is nearly free, so a NN that beats 0.035 would win on both accuracy and speed.
2. **Give the NN BP-OSD features.** Add the 16-class posterior from the cheap pass as extra inputs, so the network
   learns corrections; or train only on the shots the cascade marks uncertain.
3. **Structure-aware NN.** A transformer over detectors with (x, y, t, check type) embeddings, or a GNN on the Tanner
   graph (learned BP).
4. **Speed up the cheap pass so the cascade fits under t_cap.**
   - Only the OSD part dominates; faster pivot selection would help.
   - Early-exit OSD-CS once one class holds > 0.999 of the probability.
   - Reuse OSD across the X/Z bases (not possible; different circuits).
5. **Better candidate generation for the degenerate sum.**
   - OSD-E (exhaustive over the top λ columns) instead of CS.
   - Add weight-4 null combinations.
   - Importance-sample around the best errors.
6. **Exact-ish maximum-likelihood on small subproblems.** When few detectors fire, enumerate all errors up to
   weight 3–4 restricted to columns touching them, and sum per class exactly.
7. **Tune τ and the heavy configuration** once the timing-machine speed or the official t_cap is published.
8. **Measure on Kaggle:** check the public score of ref 58309242 against local 0.0371. Keep about 5 submissions a day
   for the strongest LER variants (heavy, cascade) to secure a top-10 rank.

## Repo map and commands

| Path | What it is |
|---|---|
| `team/common.py` | Circuits, `H`/`L`/`p`, cached data generation (`data/generated/*.npz`), LER |
| `team/bench.py` | Benchmark any `decode()` file: LER ± SE and µs/shot per basis |
| `team/decoders/bposd.py` | v1 BP-OSD (flooding, most likely error) |
| `team/decoders/bposd2.py` | v2: layered BP, degenerate class sum, multiple OSD orderings, local search, class posteriors (`decode_full`) |
| `team/decoders/cascade.py` | Cheap + heavy confidence cascade |
| `team/hybrid.py` | Lookup table, cascade and NN helpers |
| `team/compare.py`, `team/compare_hybrids.py` | Paired comparison suites |
| `team/tune_bposd.py` | Grid searches for v1 |
| `team/nn/train_nn.py` | ResMLP NN training (MPS/CUDA), weights in `team/nn/*.pt` |
| `submission_template/decoder.py` + `bposd2.py` | The hand-in (cheap degenerate; `TAU = 0.3` enables the cascade) |

```bash
uv run python team/bench.py --decoder team/decoders/cascade.py
uv run python team/compare.py --suite degenerate
uv run python team/compare_hybrids.py --suite lut_cascade
uv run python team/nn/train_nn.py --basis X --minutes 60
uv run python submission_template/run_local.py --threads 18
uv run python submission_template/run_local.py --problem-set data/test --out submission_test.csv
uvx --python 3.13 --with six kaggle competitions submit -c quantum-error-correction-decoder -f submission_test.csv -m "message"
```

**Watch out:**
- Numba caches compiled code by module name. Benchmark a decoder under its real module name; `bench.py` does this.
- macOS Accelerate prints spurious matmul warnings on subnormal probabilities; `bposd2` avoids the matmul for that
  reason.
