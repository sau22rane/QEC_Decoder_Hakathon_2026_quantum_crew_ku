# QEC decoder hackathon 2026: findings

Team quantum_crew_ku, as of 2026-11-13 (afternoon). All numbers were measured in this repo with the scripts listed
at the end. Every variant in a comparison table decodes the same fresh validation shots, so the Δ values are paired
differences (seeds 2000/2001, 50k shots per basis unless noted).

## Summary

- **Task:** read the 70 detector bits of each shot of the `[[18,4,4]]` qLDPC memory experiment and predict the 4
  logical observable flips. Predicting all zeros gives LER 0.288.
- **Current best (evening):** exact maximum-likelihood decoding by a transfer matrix over time layers
  (`submission_ml/`, `team/decoders/mldfast.py`). **LER 0.03386** on `data/train`, **0.03367 on the Kaggle public
  board** (ref 58318168). It has converged to the accuracy floor; see "Exact maximum-likelihood decoder" below.
- **Earlier best submitted decoder:** degenerate BP-OSD, written from scratch in numpy + numba.
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

## Faster OSD-CS via GF(2) linearity, and the new hand-in

The solve is linear: solve(s ⊕ col_a ⊕ col_b) = solve(s) ⊕ solve(col_a) ⊕ solve(col_b). `bposd2._osd` now reduces
the syndrome and each candidate column once, and builds every OSD-CS candidate from two 128-bit XORs plus a walk
over the set bits (de Bruijn trailing-zero count). The predictions are **bit-identical** to before. It's 1.2× faster
for cheap and 1.4–1.5× for heavy. The BP shortcut barely helps heavy (80 → 77 µs/shot), because shots that reach
OSD rarely have a converged BP.

Clean timing on the full `data/train` (2×100k shots, 18-core Mac, median of 3, nothing else running).
Score = LER × max(1, t / 3 s)^0.1.

| Decoder | LER | Time | Score if timing machine = Mac | Score if 2× faster |
|---|---|---|---|---|
| cheap | 0.03706 | 2.36 s | 0.03706 | 0.03706 |
| 1 ordering, 800/60 | 0.03599 | 4.08 s | 0.03711 | 0.03599 |
| 2 orderings, 400/40 | 0.03568 | 5.56 s | 0.03796 | 0.03568 |
| 3 orderings, 400/40 | 0.03546 | 8.02 s | 0.03912 | 0.03650 |
| heavy (3 orderings, 800/60) | 0.03494 | 14.20 s | 0.04082 | 0.03808 |
| cascade τ = 0.3 → heavy | 0.03579 | 4.05 s | 0.03689 | 0.03579 |
| **cascade τ = 0.4 → heavy (new hand-in)** | **0.03552** | **4.36 s** | **0.03687** | 0.03552 |
| cascade τ = 0.45 → heavy | 0.03535 | 4.86 s | 0.03710 | 0.03535 |
| cascade τ = 0.4 → 1500/80 | 0.03539 | 5.32 s | 0.03748 | 0.03539 |
| cascade τ = 0.45 → 1500/80 | 0.03518 | 5.98 s | 0.03770 | **0.03518** |

The cascade τ = 0.4 → heavy is best or near-best under both speed assumptions, so it's now the hand-in
(`submission_template/decoder.py`, `TAU = 0.4`). Official `run_local.py`: LER 0.03552, median 3.98 s over 5 runs,
deterministic.

The transformer (sparse fired-detector tokens, d = 128, 4 layers, bf16 on MPS) trains 2–5× slower than the MLP and
learned worse per shot. With length-grouped batches it reached 0.105 at 28M shots; with random batches 0.115 at 14M
(X basis), against the MLP's 0.065 at 12M. We stopped it.

## AutDEC / multiple BPs instead of OSD: not feasible here

- **Automorphisms:**
  - The code's stabiliser group is invariant under all 9 translations of the L/R 3×3 torus.
  - The measured check graph (7 X + 7 Z, with 4 redundant checks dropped) has only 2 automorphisms.
  - An exhaustive search (networkx VF2, detectors coloured by round, mechanisms by prior) finds **only the identity**
    for the circuit-level DEM in both bases. There are no symmetric copies of the decoding graph for AutDEC to run
    BP on.
- **Multiple BPs without OSD** (2×50k shots; unconverged shots predict 0):

| BP alone | Converged | LER on converged shots | Overall LER |
|---|---|---|---|
| layered 3 it, α = 0.45 (our setting) | 24% | 0.0005 | 0.247 |
| layered 30 it, α = 1.0 | 82% | 0.019 | 0.090 |
| layered 100 it, α = 1.0 (best single) | 86% | 0.024 | 0.079 |
| flooding 100 it, α = 1.0 | 60% | 0.013 | 0.166 |
| ensemble of all 24 BP runs (lowest-cost converged solution) | 89% | 0.025 | **0.070** |

- **Conclusion:** BP converges on at most ~89% of shots even with a 24-run ensemble. The remaining 11% are the hard
  shots and still need OSD. Convergence also doesn't mean correct: converged BP solutions ignore degeneracy, and
  their LER (0.025) is far worse than BP-OSD on the same shots. Our cascade already does the useful part (a cheap
  pass with BP shortcut, OSD only where needed).

## Rule update (upstream README, seen 16:40): t_cap = 15.4 s, α = 0.2, import time is free

- **t_cap is the organisers' BP+OSD on all test shots with 32 threads: 15.4 s** (was provisionally 3 s). We are
  timed with 48 threads.
- **α = 0.2:** each doubling beyond t_cap costs about 15% (was 7%). So stay under the cap rather than trading LER
  for time.
- **Only the two `decode` calls are timed.** Importing the decoder, and anything it builds or loads at import, is
  free. The hand-in now builds the matrices and compiles the numba kernels at import, using bundled copies of the
  competition circuits.
- **The earlier final-score tables** (t_cap = 3 s, α = 0.1) are superseded. All our cascades are far under
  15.4 s.

Thread scaling on the Mac (Apple M5 Pro, 6 performance + 12 efficiency cores), three-level cascade, 2×50k shots:

| Threads | Time | Speed-up |
|---|---|---|
| 1 | 84.0 s | 1× |
| 6 | 16.2 s | 5.2× |
| 12 | 8.8 s | 9.5× |
| 18 | 7.2 s | 11.7× |

Per-shot parallelism is near-linear, so 48 EPYC 9655 threads are probably about 2–3× the Mac. That's an estimate,
not measured.

Accuracy-focused cascades, full `data/train`, clean timing (median of 3, 18 threads):

| Decoder | LER | Mac time | Score if timing machine = Mac (t_cap 15.4 s, α 0.2) |
|---|---|---|---|
| cascade 0.4 → heavy (previous hand-in) | 0.03552 | 4.18 s | 0.03552 |
| heavy on every shot | 0.03494 | 13.05 s | 0.03494 |
| cheap → 0.49 heavy + 32 perturbed | 0.03469 | 11.48 s | 0.03469 |
| **cheap → 0.49 heavy → 0.4 SUPER (new hand-in)** | **0.03459** | **13.73 s** | **0.03459** |
| heavy every shot → 0.4 SUPER | 0.03436 | 22.29 s | 0.03700 (fits only if the machine is ≥ 1.45× the Mac) |

SUPER = heavy with OSD-CS 1500/80 plus 32 perturbed OSD-CS 200/30 runs.

**New hand-in** (`submission_template/decoder.py`, three levels):
- `run_local.py`: LER **0.03459** (X 0.03517, Z 0.03401), median **13.3–13.4 s** over 5 runs;
- deterministic;
- the first run is as fast as the rest, thanks to the import-time warm-up.

It stays under t_cap even if the timing machine is no faster than the Mac. If the organisers confirm the machine
is ≥ 1.5× faster, "heavy every shot → 0.4 SUPER" (0.03436) is the next step up.

## Accuracy round 2 (17:00–17:45): routing is the lever; extra BP orderings and weight-3 flips aren't

New `bposd2` options:
- `extra_bp=((schedule, iterations, α), ...)`: independent BP runs with other settings, each seeding an OSD-CS run;
- `n_triple`: OSD-CS also tries every triple among the first n candidates.

**Inside SUPER (full cascade, 2×50k paired validation shots):**

| Variant | LER | Δ vs hand-in |
|---|---|---|
| hand-in (cheap → 0.49 heavy → 0.4 SUPER) | 0.03471 | — |
| SUPER + one extra BP (flood 3 it α 0.8 / layered 2 it α 0.8 / layered 1 it / flood 10 it α 1.0) | 0.03467–0.03471 | −0.00004 to 0 |
| SUPER + all four extra BPs | 0.03463 | −0.00007 ± 0.00003 |
| SUPER + triples among 10 / 15 / 20 | 0.03470–0.03471 | 0 |

**Where the hand-in's remaining errors are** (2×50k validation shots):

| Final level | Share of shots | Share of wrong bits |
|---|---|---|
| cheap only | 79.6% | 12.6% |
| heavy | 8.5% | 10.4% |
| SUPER | 11.9% | 77.0% |

- Shots where BP converged (one-hot, never escalated) are 19.9% of shots but only 0.2% of the wrong bits.
- Cheap-only shots where BP didn't converge but the decoder was confident (margin ≥ 0.49) hold **12.4%** of the
  wrong bits. That's the one routing lever left.

**Routing variants** (same paired shots):

| Variant | LER | Δ | µs/shot |
|---|---|---|---|
| hand-in | 0.03471 | — | 62 |
| cheap → **0.5** heavy (all unconverged) → 0.4 SUPER | 0.03442 | **−0.00029 ± 0.00006** | 106–110 |
| cheap → 0.5 heavy → 0.45 SUPER | 0.03437 | −0.00034 ± 0.00007 | 122 |
| cheap → 0.5 MED (it 1, 3, 200/30) → 0.49 heavy → 0.4 SUPER | 0.03456 | −0.00015 | 94 |
| cheap → 0.5 MED1 (it 1 only) → … | 0.03516 | +0.00046 (a worse middle estimate overrides the cheap one) | 82 |
| **cheap with 2 orderings (it 1, 2)** → 0.49 heavy → 0.4 SUPER | 0.03454 | −0.00016 ± 0.00005 | 79 |

**Full `data/train`, clean timing (median of 3).** Score columns use t_cap 15.4 s and α 0.2, assuming the timing
machine is equal to / 1.5× / 2× the Mac.

| Decoder | LER | Mac time | Score at equal / 1.5× / 2× |
|---|---|---|---|
| previous hand-in | 0.03459 | 12.6 s | 0.03459 / 0.03459 / 0.03459 |
| **robust: cheap2 → 0.49 heavy → 0.4 SUPER (new hand-in)** | **0.03445** | **15.6 s** (14.5 s in `run_local`) | 0.03452 / 0.03445 / 0.03445 |
| cheap2 → 0.49 heavy → 0.45 SUPER + 4 extra BP | 0.03440 | 22.9 s | 0.03724 / 0.03440 / 0.03440 |
| **wide: cheap → 0.5 heavy → 0.4 SUPER** | **0.03436** | **23.2 s** | 0.03730 / 0.03439 / 0.03436 |
| wide + 0.45 SUPER + 4 extra BP | 0.03433 | 28.6 s | 0.03884 / 0.03582 / 0.03433 |

`submission_template/decoder.py` now has `PROFILE = "robust"` (default) or `"wide"`:
- robust: `run_local` LER 0.03445, median 14.5 s, file `submission_test_robust.csv`;
- wide: LER 0.03436, median 23.2 s, file `submission_test_wide.csv`.

Both are deterministic. The hand-in must match the selected Kaggle submission, so set `PROFILE` accordingly before
handing in.

## Stacked NN on BP-OSD (PLAN_NEXT Part 2): no-go

Setup:
- `team/nn/make_stack_features.py` generated cheap-pass features (seeds ≥ 50000, 15 chunks of 1M for X).
- `team/nn/train_stack.py` trained a residual MLP 512×3 on 89 features. Its output is added to the decoder's
  clipped per-bit logit, with a zero-initialised last layer.

Result, X basis, 392 s of training (119M samples, re-using up to 15 chunks): validation LER **0.0378–0.0384**
against the decoder's own **0.0376** on the same shots. It never beat the decoder, so I stopped it at the go/no-go.
The likely reasons are that the decoder's class sums are already close to calibrated (see Calibration), and that
chunk re-use invites fitting label noise.

## Search diversity (PLAN_NEXT Part 1): perturbed orderings help LER, not the final score

New `bposd2` options:
- `n_perturb`, `perturb_scale`, `perturb_mode`, `perturb_single`, `perturb_pair`: extra OSD-CS runs seeded by
  fixed-seed noisy copies of the last BP posterior. Only the first 512 columns of the base order are re-sorted,
  1.7× cheaper with identical LER.
- `dec_*`: BP guided decimation.

Heavy pass, 2×20k paired validation shots:

| Variant | LER | Δ vs heavy | µs/shot |
|---|---|---|---|
| heavy (3 BP orderings, 800/60) | 0.03544 | — | 61 |
| + 8 perturbed 100/20, additive sd 0.5 / 1 / 2 / 4 | 0.03513 / 0.03509 / 0.03521 / 0.03546 | −0.0003 / −0.00035 / −0.0002 / 0 | 135–148 |
| + 8 perturbed, multiplicative sd 0.3 / 0.6 | 0.03515 / 0.03537 | −0.0003 / −0.0001 | 156–164 |
| + 16 perturbed 100/20 (head sort) | 0.03508 | −0.00036 ± 0.00016 | 121 |
| + 16 perturbed 200/30 (head sort) | 0.03491 | −0.00052 ± 0.00018 | 153 |
| + 32 perturbed 100/20 (head sort) | 0.03479 | −0.00064 ± 0.00018 | 180 |
| 1 BP ordering 400/40 + 16–32 perturbed | 0.03536–0.03539 | ≈0 | 170–326 |
| + BPGD solutions only (15 steps, 3 iterations, 1 or 3 branches) | 0.03544 | 0.0000 | 91–135 |
| + BPGD, 4 decimation orderings | 0.03519 | −0.00025 | 156 |
| + BPGD 5 branches + 32 perturbed | 0.03478 | −0.00066 (= perturbed alone) | 339 |

Cascade, full `data/train`, clean timing (median of 3, 18-core Mac). Score uses t_cap = 3 s; "2×" assumes the
timing machine is twice as fast.

| Cascade | LER | Time | Score if machine = Mac | Score if 2× faster |
|---|---|---|---|---|
| **τ 0.4 → heavy (hand-in)** | **0.03552** | **3.90 s** | **0.03646** | **0.03552** |
| τ 0.4 → heavy + 16 perturbed 200/30 | 0.03523 | 6.57 s | 0.03811 | 0.03555 |
| τ 0.4 → heavy + 32 perturbed 100/20 | 0.03522 | 7.44 s | 0.03856 | 0.03598 |
| τ 0.45 → heavy + 16 perturbed 200/30 | 0.03501 | 7.98 s | 0.03861 | 0.03602 |
| τ 0.45 → heavy + 32 perturbed 100/20 | 0.03499 | 8.88 s | 0.03900 | 0.03639 |

- **Perturbed orderings are the only search-diversity idea that helped** (−0.0003 to −0.0006).
- **BPGD's own solutions add nothing,** and its orderings are a weaker version of the same effect.
- **Inside the cascade the gain is about −0.0003 for +70–130% runtime,** a net loss in final score under both
  speed assumptions. The hand-in stays as it is.
- **Still useful for leaderboard rank (step 1 is LER only):** τ 0.45 → heavy + 16 perturbed 200/30 gives 0.0350 on
  `data/train`.

## Relay-BP (memory-BP legs, Müller et al. 2025): converges, but no useful gain

Implemented in `bposd2` (`relay_*` options, `_relay` kernel):
- min-sum legs with per-mechanism memory strengths γ (leg 0: γ = 0.125; later legs uniform in [−0.24, 0.66],
  fixed seed);
- each leg starts from the previous leg's marginals;
- every converged leg adds a candidate, and the first `relay_osd` legs' marginals seed extra OSD-CS runs.

| Relay-BP alone (no OSD, 2×10k shots) | Any leg converged | LER | µs/shot |
|---|---|---|---|
| 1 leg | 78% | 0.108 | 57 |
| 10 legs, stop at 1 solution | 99.6% | 0.059 | 97 |
| 30 legs, stop at 5 solutions | 100% | 0.053 | 159 |
| 100 legs, S = 5, α = 0.8 | 100% | **0.050** | 275 |

| Inside the heavy pass (2×20k paired shots) | LER | Δ vs heavy | µs/shot |
|---|---|---|---|
| heavy (3 BP orderings, 800/60) | 0.03544 | — | 62 |
| + relay 10 legs, candidates only | 0.03541 | −0.00003 ± 0.00006 | 188 |
| + relay, 1 relay ordering | 0.03529 | −0.00015 ± 0.00012 | 231 |
| + relay, 3 relay orderings | 0.03521 | −0.00023 ± 0.00014 | 286 |
| 1 BP ordering + 3 relay orderings | 0.03589 | +0.00045 | 248 |

- **Memory works for convergence:** ~100% against 89% for a 24-run plain-BP ensemble.
- **The few solutions it finds are no substitute for degenerate OSD** (0.050 against 0.0355).
- **Relay candidates add nothing,** and relay orderings are no better than BP snapshot orderings. The best gain,
  −0.0002 (1.6 SE), costs 4–5× the heavy pass, which is a net loss in the final score. Kept as options, off by
  default.

## Failure diagnostic: search miss vs ambiguity (`team/diagnose.py`)

Stim's DEM sampler returns each shot's true error. We mapped it to the merged columns (verified H e = D and
L e = O) and compared its cost with the cost of the most likely error the decoder found. 2×50k fresh shots
(seeds 4000/4001); this sample gives LER about 0.001 lower than the validation set, consistently across decoders.

| Decoder | LER | Search misses (truth cheaper than best found) | Ambiguity (found something at least as likely) | True class absent | Oracle LER (add the missed truth) |
|---|---|---|---|---|---|
| cheap (submitted) | 0.03578 | 12% of bit errors | 88% | 173 of 7427 wrong shots | 0.03206 (−0.0037) |
| heavy (3 orderings, 800/60) | 0.03389 | 4.3% | 95.7% | 0 | 0.03271 (−0.0012) |

- **The cheap decoder's search misses are worth up to −0.0037.** The heavy pass recovers about 70% of that; only
  −0.0012 is left even for a perfect search on top of heavy.
- **Missed true errors are heavy:** about 5.5 merged mechanisms on average. Wrong shots come mostly from true errors
  of weight 4–7, so exact enumeration up to weight 3–4 (meet in the middle) would mostly re-find errors OSD already
  finds. It's de-prioritised.
- **88–96% of the remaining failures are ambiguity:** the decoder found an error at least as likely as the truth,
  and gives the true class only P ≈ 0.18 on average. A sizeable part of this is irreducible; better summation or
  calibration can only move the shots near 0.5.
- **Confidence on wrong bits (heavy decoder):** the probability it gave the true value of each wrongly predicted
  bit was:
  - 0–0.05: 19%
  - 0.05–0.1: 8%
  - 0.1–0.2: 14%
  - 0.2–0.3: 13%
  - 0.3–0.4: **29%**
  - 0.4–0.5: 17%

  So **46% of wrong bits were near coin-flips (0.3–0.5)**. That's where calibration (a temperature on the costs) or
  more complete class sums could still flip answers. The ~19% confident-wrong bits are mostly irreducible.
- **The heavy pass (0.0339 on this sample) is already about at the leader's public score (0.0337).** So the main
  gain is making heavy-quality search cheap enough to fit the time budget.

## Calibration: no gain (`bposd2` option `temp`)

| Test | Result |
|---|---|
| Temperature on the class sum, exp(−cost / T), cheap decoder, 2×50k shots | T = 1.0 is best. T = 0.6–1.25 are within +0.0002; T = 2.0 is +0.0013 |
| Temperature, heavy decoder, 2×20k shots | T = 0.9: 0.03542, T = 1.0: 0.03544, T = 1.1: 0.03549 (all noise) |
| Decision threshold (per-bit marginal), cheap / heavy | 0.5 is optimal; 0.45 and 0.55 are +0.0000 to +0.0002; 0.4 and 0.6 are worse |

Reliability curves are close to the diagonal. For example, predicted 0.40–0.50 → observed 0.47–0.48, and
predicted 0.60–0.70 → observed 0.66–0.67. The decoder is only slightly overconfident near the extremes.

Candidate costs are exact log-probabilities, so the class sums are already calibrated. Any monotone recalibration
that keeps the 0.5 crossing cannot change a decision anyway. Only a model that uses extra features (the full
16-class posterior, number of fired detectors, BP convergence) could still help, which is the NN + BP-OSD features
idea below.

## Exact maximum-likelihood decoder (19:00–20:00): LER 0.0339 on `data/train`, the accuracy floor

Files: `team/decoders/mldfast.py` (self-contained: stim + numpy + numba), hand-in folder `submission_ml/`,
experiments in `team/decoders/mld.py` and `team/mld_eval.py`. No BP, no OSD, no search, no training.

**Idea.** Every DEM mechanism touches detectors in at most two adjacent time layers (7, 14, 14, 14, 14, 7
detectors; checked for all ~2470 mechanisms in both bases), and the three middle layer pairs are identical. So the
probability of a syndrome D together with a logical class is an exact chain of five tables on 14-bit indices:

    P_chi(D) = sum_{b0..b3} T0[D0, b0] N[b0^D1, b1] N[b1^D2, b2] N[b2^D3, b3] T4[b3^D4, D5]

- chi runs over characters of the 4 observable bits. chi = e_i gives P(D, L_i = 0) − P(D, L_i = 1), so bit i is
  predicted 1 exactly when P_{e_i}(D) < 0. That is the per-bit optimal decision; only 5 characters are needed.
- Tables come from Walsh–Hadamard transforms (the XOR of independent mechanisms is a product in the transformed
  domain): two 2^28 transforms per character. Mechanisms inside a middle layer are split into two independent halves
  (p = 2q(1 − q)) so the three middle tables are the same table N.
- Decoding is one left-to-right pass per shot. Before each N, only running-vector entries within a factor delta of
  the best are expanded (ranked with a one-step lookahead), rows of N are cut to the M largest entries, and row
  entries below eta × the layer's largest contribution are skipped.

**Results.** Hand-in = the "robust" BP-OSD cascade.

| Set | ML decoder | BP-OSD hand-in | Δ |
|---|---|---|---|
| validation 2×50k (seeds 2000/2001) | 0.03392 | 0.03454 | −0.00062 ± 0.00012 |
| `data/train` 2×100k, `run_local.py` | **0.03389** (X 0.03430, Z 0.03348) | 0.03445 | −0.00056 |

**Convergence** (2×200k validation shots, M = 4096, paired against R; times on a 10-core Apple M4):

| delta / kmax / eta | LER | Δ vs R | Time per 200k |
|---|---|---|---|
| R: 1e-5 / 256 / 1e-8 | 0.03364 | — | 29.5 s |
| 1e-6 / 512 / 1e-9 (profile `max`) | 0.03363 | −0.000005 ± 0.000010 | 110 s |
| 1e-5 / 256 / 1e-7 | 0.03364 | +0.000003 ± 0.000008 | 23.2 s |
| 1e-4 / 128 / 1e-7 | 0.03365 | +0.000016 ± 0.000016 | 19.0 s |
| 1e-4 / 128 / 1e-6 (profile `fast`) | 0.03367 | +0.000029 ± 0.000020 | 9.9 s (6.6 s in `run_local`) |
| 1e-3 / 64 / 1e-6 | 0.03380 | +0.00016 ± 0.00003 | 10.2 s |
| 1e-3 / 128 / 1e-5 | 0.03384 | +0.00020 ± 0.00004 | 6.6 s |

- The decoder has converged: loosening every limit by 10× changes nothing measurable. This is the maximum-likelihood
  floor; a lower score on a given set can only come from sampling noise (about ±0.0003 on 200k shots).
- `fast` costs about +0.00003 LER and runs in 6.6 s on a 10-core M4, against 14.5 s on an 18-core M5 Pro for the
  BP-OSD hand-in.
- Profile `max` on `data/train`: **0.03386** (X 0.03429, Z 0.03343), 104 s on the M4.
- Row truncation (M = 4096) costs nothing: re-decoding the 16k ambiguous shots of 2×100k with the full, untruncated
  table changed 162 of 800,000 bits and the LER by +0.00003 (noise).
- Test-set files (not submitted): `submission_test_ml_max.csv`, `submission_test_ml_fast.csv`; they differ in 483
  of 800,000 bits. The hand-in must use the same `MLD_PROFILE` as the selected submission.
- **Kaggle:** `submission_test_ml_max.csv` submitted at 19:34 CET as ref 58318168, **public score 0.03367**
  (previous team best 0.03408, the coset-walk cascade). It still has to be selected as a final submission on the
  Kaggle page, and the hand-in is then `submission_ml/` with `MLD_PROFILE` unset (= `max`).
- **Open decision:** `max` takes 105–120 s on the 10-core M4, far over t_cap = 15.4 s unless the timing machine is
  about 7× faster; `fast` (6.6–12 s on the M4, +0.00003 LER) has a test file but is not submitted.
- **NN plan dropped:** a power-law fit to `team/nn/train_log.txt` puts a plain MLP at 10–1,000 billion shots per basis
  to reach 0.0336, against 5–40 billion trainable on one H100/B200 in 3.5 hours. The ML decoder makes it moot.
- Table build: about 30–160 s per basis at import (untimed), then cached in `submission_ml/mld_cache/`
  (1.7 GB per basis, git-ignored). Needs about 4 GB of RAM while decoding.
- No shot lost every path under pruning in 600k shots; if one does, it's re-decoded without beam limits.

```bash
.venv/bin/python submission_template/run_local.py --decoder submission_ml --problem-set data/train
MLD_PROFILE=fast .venv/bin/python submission_template/run_local.py --decoder submission_ml --problem-set data/train
.venv/bin/python team/mld_eval.py --shots 50000 --M 1024 --delta 1e-5 --kmax 256
```

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
