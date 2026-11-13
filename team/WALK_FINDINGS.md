# Coset walk and capture analytics: what was done and observed

Written 2026-11-13, evening. Builds on `team/FINDINGS.md` (read that first) and on commit `2cd213b`.
All numbers were measured on a 10-core Apple M4, so absolute times are about 2–3× the team's 18-core M5 Pro; use
the time columns for relative cost only.

## Summary

- **New analytic: capture rate.** For shots sampled with their true error, check whether that error is in the
  decoder's candidate list. Averaged over shots, this equals the share of posterior probability mass the list holds.
  The heavy pass captures only **80%** on the cascade's uncertain shots (63% on shots it gets wrong).
- **New decoder option: a coset walk** (`team/decoders/bposd_walk.py`, options `walk_*`). A deterministic best-first
  search over errors with the same syndrome, moving through the weight-3 null combinations. It raises capture to
  92–95% with few extra candidates.
- **LER effect on the uncertain shots:** SUPER + walk gives **−0.00069 ± 0.00007** against heavy, compared with
  −0.00056 for SUPER alone. So the walk adds about **−0.00013** on top of the current hand-in's last level.
- **It saturates.** Raising capture from 92% to 95% does not move LER (0.03486 → 0.03485). What is left on the
  uncertain shots is ambiguity, not missing candidates.
- **The largest untouched piece is routing:** 30% of the remaining LER (0.0105 of 0.0349) comes from shots the cheap
  pass was confident about and never re-decoded.
- **BP guided decimation: no gain** (independently confirms the result in `FINDINGS.md`).
- **Full-cascade numbers on `data/train`:** the team hand-in is reproduced (0.03445); the walk profiles are pending
  (see the last section).

## Files

| Path | What it is |
|---|---|
| `team/decoders/bposd_walk.py` | `bposd2.py` at `2cd213b` plus the walk (`walk_exp`, `walk_seeds`, `walk_max`, `walk_gap`, `walk_w`) and the `truth=` probe. With the walk off, it is the same decoder as `bposd2` |
| `team/decoders/cascade_walk.py` | The three-level cascade with the walk in the last level; run it to compare profiles on `data/train` |
| `team/gen_eval.py` | Harness: capture rate and paired LER on the uncertain shots only (sets cached in `data/generated/`) |

```bash
.venv/bin/python team/gen_eval.py walk           # walk settings against heavy and SUPER
.venv/bin/python team/gen_eval.py walk2          # deeper walks, saturation check
.venv/bin/python team/decoders/cascade_walk.py   # all cascade profiles on data/train
```

## The capture analytic

`diagnose.true_errors` gives each shot's true error on the merged columns. `decode_full(..., truth=e)` hashes it
with the decoder's Zobrist keys and reports in `stats[:, 2]` whether that hash is among the distinct candidates.
Because the true error is a sample from the posterior P(e | D), the mean of that flag is the average posterior mass
the candidate list holds. No oracle decision is involved, so it is an unbiased completeness measure for any
candidate generator.

First measurement, heavy pass (3 orderings, 800/60), 2×20k shots:

| Shots | True error in the list |
|---|---|
| all | 94.5% |
| confident (cheap margin ≥ 0.4) | 96.8% |
| uncertain (margin < 0.4) | 77.4% |
| wrongly decoded | 63.1% |

**Why OSD-CS misses it.** Any error is the OSD solution plus its set T of non-basis columns. OSD-CS finds it only if
T is empty, one column within `n_single`, or two columns within `n_pair`. On uncertain shots (ordering after BP
iteration 3):

| \|T\| | Share of uncertain shots | Rank of its worst column, 50% / 75% / 90% |
|---|---|---|
| 0 | 16% | — |
| 1 | 39% | 4 / 18 / 58 |
| 2 | 27% | 30 / 95 / 228 |
| 3 | 12% | 94 / 199 / 393 |
| 4 | 4% | 159 / 375 / 655 |
| ≥ 5 | 2% | — |

So 18% of uncertain shots need three or more simultaneous flips, at ranks far beyond what triples can enumerate
(triples among the first 80 columns are already 82k candidates). That is why enlarging OSD-CS stalls.

## Candidate generators compared

2×150k validation shots (seeds 2000/2001); cheap pass everywhere, then the 32,309 uncertain shots (10.8%, τ = 0.4)
re-decoded. Capture is measured on a separate set of 12,969 uncertain shots with known true errors (seeds 6000+).
Δ is paired against heavy.

| Second stage | Capture | Distinct candidates | LER | Δ vs heavy | µs per uncertain shot |
|---|---|---|---|---|---|
| cheap (no second stage) | 69.8% | 636 | 0.03685 | +0.00129 ± 0.00009 | 40 |
| heavy (3 orderings, 800/60) | 79.8% | 5,919 | 0.03555 | — | 177–250 |
| heavy, pairs 120 | 83.0% | 18,766 | 0.03528 | −0.00028 ± 0.00005 | 399 |
| heavy, pairs 250 | 85.4% | 77,650 | 0.03505 | −0.00051 ± 0.00006 | 1,761 |
| heavy, 2400/120 | 83.0% | 22,803 | 0.03528 | −0.00027 ± 0.00005 | 738 |
| heavy + triples among 40 | 82.6% | 29,176 | 0.03540 | −0.00015 ± 0.00004 | 1,000 |
| heavy + triples among 80 | 85.1% | 208,508 | 0.03518 | −0.00037 ± 0.00005 | 7,095 |
| heavy + BPGD solutions (8 steps, 1 or 4 branches) | 79.8% | 5,919 | 0.03555 | 0.00000 | 438–756 |
| heavy + 3 BPGD orderings | 81.5% | 10,323 | 0.03545 | −0.00011 ± 0.00004 | 808 |
| heavy + 6 BPGD orderings | 81.7% | 13,939 | 0.03544 | −0.00011 ± 0.00004 | 901 |
| SUPER (1500/80 + 32 perturbed 200/30) | 87.1% | 23,480 | 0.03500 | −0.00056 ± 0.00006 | 858–968 |
| **heavy + walk, 50 expansions** | 87.2% | 7,743 | 0.03522 | −0.00034 ± 0.00006 | 440 |
| **heavy + walk, 200 expansions** | 90.2% | 9,537 | 0.03502 | −0.00053 ± 0.00006 | 665 |
| heavy + walk 200, 8 seeds | 89.0% | 9,051 | 0.03512 | −0.00043 ± 0.00006 | 716 |
| heavy + walk 200, 400 seeds | 90.2% | 9,564 | 0.03501 | −0.00054 ± 0.00006 | 657 |
| **heavy + walk, 1,000 expansions** | 93.2% | 12,311 | 0.03495 | −0.00061 ± 0.00007 | 1,129 |
| heavy + walk 1,000, gap 20 | 92.3% | 48,540 | 0.03498 | −0.00057 ± 0.00007 | 3,208 |
| heavy 200/30 ×3 + walk 1,000 | 92.0% | 8,414 | 0.03500 | −0.00056 ± 0.00007 | 2,127 |
| **SUPER + walk, 200 expansions** | 92.2% | 26,707 | 0.03486 | **−0.00069 ± 0.00007** | 1,947 |
| SUPER + walk, 1,000 expansions | 94.3% | 29,007 | 0.03486 | −0.00069 ± 0.00007 | 2,666 |
| SUPER + walk, 3,000 expansions, gap 18 | 95.1% | 66,078 | 0.03485 | −0.00071 ± 0.00007 | 5,429 |

The BPGD and higher-order rows were measured with an earlier prototype on the older `bposd2`; the team's own
`dec_*` and `n_triple` options now cover them, so that prototype is not committed.

Observations:

- **The walk is the most efficient generator per candidate.** Heavy + walk 200 matches SUPER's LER with 40% of
  SUPER's candidates and less time.
- **Walk and perturbed orderings find different errors.** They stack: SUPER + walk is the best row.
- **Roughly −0.0001 LER per +1% capture up to about 90%, then nothing.** Past 92% capture the LER is flat.
- **A wider cost window does not help.** Gap 20 produces 4× the candidates for no gain; the default gap 14
  (candidates down to e⁻¹⁴ of the best) is enough.
- **Seeds matter a little:** 8 seeds is worse than 64; 400 is no better.
- **BPGD:** its own solutions are always already in the list; its orderings are a weak form of perturbation.

## How the walk works

- **Moves.** The 9.6k weight-3 null combinations (col_a ⊕ col_b = col_c) that `bposd2` already builds for
  `n_expand`. Applying one to an error keeps the syndrome: one mechanism becomes two, two become one, or three vanish.
- **Search.** Seed a min-heap with the `walk_seeds` most likely distinct candidates OSD found (main orderings only,
  weight ≤ `walk_w`). Pop the most likely unexpanded error, apply every triple touching it, and keep each new distinct
  error whose cost is within `walk_gap` of the best. Each kept error joins the class sum and the heap.
- **Stops** after `walk_exp` expansions or `walk_max` new errors.
- **Deterministic:** no randomness; visited errors are tracked by Zobrist hash in an open-addressing table.
- **Difference from `n_expand`:** that was a single step around the top candidates (−0.0004 at 2–4× cost in
  `FINDINGS.md`). This is the multi-step version, ordered by likelihood, so it reaches errors that need several
  swaps.
- **Cost note:** the walk needs each seed's column list, so the main OSD runs record their columns. `n_triple` is
  ignored while the walk is on (as with `n_expand`).

## Where the remaining error is

On the 300k validation shots, with SUPER + walk as the second stage (LER 0.03486):

| Part | LER contribution |
|---|---|
| Confident shots (cheap margin ≥ 0.4, never re-decoded), 89.2% of shots | 0.01048 |
| Uncertain shots after SUPER + walk, 10.8% of shots | 0.02438 |

- The uncertain part no longer responds to better search (95% capture, flat LER).
- The confident part is 30% of the total and is only reachable by routing more shots to the later levels, as the
  "wide" profile does.

## Full cascade on `data/train` (2×100k shots)

Measured by `team/decoders/cascade_walk.py`. Times are from the 10-core M4 with other jobs running, so they are not
comparable with `FINDINGS.md`.

| Profile | LER | X | Z | Δ vs team hand-in |
|---|---|---|---|---|
| robust (team hand-in): cheap2 → 0.49 heavy → 0.4 SUPER | 0.03445 | 0.03501 | 0.03389 | — |

This reproduces the hand-in's 0.03445 exactly. **The walk profiles were still running when this was committed;**
their rows are added in a follow-up commit.

## Not done

- No Kaggle submission was made and `submission_template/` is unchanged.
- Timing on the team's Mac and thread scaling for the walk profiles.
- The walk uses only weight-3 null combinations. Weight-4 combinations (col_a ⊕ col_b = col_c ⊕ col_d) were not
  tried; given the saturation above, they are unlikely to help the uncertain shots.
- A Tesseract reference (`tesseract-decoder`, banned for submissions) scored LER 0.03764 on `data/train` in
  213.7 s on the 10-core M4. It is kept out of the repo on purpose.
