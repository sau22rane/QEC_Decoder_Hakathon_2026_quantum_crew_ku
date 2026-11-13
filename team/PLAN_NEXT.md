# Plan: close the gap to the leaders (search diversity, stacked NN, time budget)

Written 2026-11-13 16:15 CET, revised 16:20. The competition closes at **23:59 CET**. This is a brief for a fresh
session: read `team/FINDINGS.md` and `.claude/skills/qec-decoder-hackathon/SKILL.md` first, then follow the steps
below in order.

## Situation

- **Leaderboard (public, 16:07 CET):** four teams at 0.0336–0.0339, 5th–7th at 0.0342–0.0344, 10th at 0.0362.
- **Us:** the submitted cheap decoder scores 0.0368 (ref 58309242), which is now **outside the top 10**. Only the
  private top 10 go on to the speed round.
- **Our best decoders, on `data/train` (2×100k shots, 18-core Mac):**

| Decoder | LER | Time |
|---|---|---|
| cheap (submitted) | 0.03706 | 2.4 s |
| cascade τ = 0.4 → heavy (current hand-in, not submitted) | 0.03552 | 4.4 s |
| cascade τ = 0.45 → 1500/80 | 0.03518 | 6.0 s |
| heavy on every shot (3 orderings, 800/60) | 0.03494 | 14.2 s |
| 3 orderings, 1500/80 on every shot | ≈0.0348 (validation set, not `data/train`) | ≈30 s (estimated) |

- **Public vs local scores:** our one submission scored 0.0368 public against 0.0371 on `data/train`. That's a
  single data point within public-board noise (about ±0.0004), so don't rely on a fixed offset.
- **Why we think part of the gap is closable:** `team/diagnose.py` shows that adding the missed true errors to the
  heavy decoder's class sum gives 0.0327 against 0.0339 on that sample (−0.0012). That's an **oracle bound**, the
  gain if every missed error were found. The missed errors are heavy (about 5.5 merged mechanisms), and more OSD-CS
  candidates around the same BP ordering have stopped helping (3000/120 with 4 orderings only reaches 0.0349).
  **A realistic gain from Part 1 is −0.0003 to −0.0008**, not the full −0.0012.
- **Already ruled out (don't repeat):** lookup tables, prior scaling, long BP, temperature and threshold
  calibration, Relay-BP, AutDEC-style BP ensembles, the triple-swap local search, the plain MLP and the transformer
  as standalone decoders, RL.

## Rules that constrain everything

- Own decoder only: no pymatching, ldpc, stimbposd, beliefmatching, tesseract-decoder, fusion-blossom.
- The hand-in `submission_template/decoder.py` must be deterministic and **reproduce the selected Kaggle
  submission**. Any change to the decoder after a submission needs a new submission before it can be the hand-in.
- Final score = LER × max(1, t / t_cap)^0.1, t_cap ≈ 3 s provisional, on 48 EPYC cores + one B200. Hard limit
  5 minutes. Each doubling of runtime costs about 7%; a 3% LER gain is worth up to about 1.34× the runtime
  (1.03^10).
- Submitting to Kaggle uses a shared daily quota (4 left at 13:00). **Ask the user before every submission.**
- The 2 final submissions must be **selected on the Kaggle submissions page before 23:59**. Remind the user.
- Evaluate every change as a paired comparison with `team.compare.Table` on the validation shots (seeds 2000/2001).
  A difference under about 2 standard errors is noise.
- **CPU discipline:** LER-only sweeps can share the machine; **timing runs need nothing else running** (no feature
  generation, no data sampling, no other decoder runs). NN training on the GPU slows CPU timings slightly, so time
  critical candidates again with training paused.

## Timeline (time boxes)

| Until | Step |
|---|---|
| 16:40 | Step 0: commit, then a stronger submission |
| 18:30 | Part 1: search diversity (LER sweeps first, timing last) |
| 18:30–19:00 | Part 2 feature generation (CPU-heavy, nothing else on the CPU) |
| 19:00–20:00 | Part 2 NN training on the GPU; meanwhile Part 3 analysis that doesn't need clean timings |
| 20:00 | Part 2 go/no-go |
| 20:00–22:00 | Integrate the winners, clean timings, pick the hand-in |
| **22:30** | **Last code change.** After this only `run_local.py` checks, submission files and commits |
| 23:30 | Final submissions done and the 2 finals selected on Kaggle |

If a part overruns its box, stop it, record what was learned in `FINDINGS.md`, and move on.

## Step 0: secure the current state (before any new work)

1. **Commit the uncommitted work on `dev/sarane`:**
   - cascade hand-in (`submission_template/decoder.py` with `TAU = 0.4`, fast `bposd2.py`);
   - linear-OSD rewrite, Relay-BP, `temp` and `return_stats` options in `team/decoders/bposd2.py`;
   - `team/diagnose.py`, transformer changes in `team/nn/train_nn.py` and `team/hybrid.py`, `compare_hybrids.py`;
   - `FINDINGS.md` sections and this plan.

   Check `git status` first; the `.pt` weights are git-ignored.
2. **Get back into the top 10.** Offer the user two options (ask before submitting):
   - **cascade τ = 0.4** (`submission_test_cascade.csv` already exists, LER 0.0355 locally): fast, likely about 9th;
   - **max-accuracy** (3 orderings, 1500/80 on every shot, about 0.0348): slower, likely 7th–8th.

   Submitting both uses 2 of the 4 left today and keeps both hand-in options open.

## Part 1: make the search more diverse (until 18:30)

Goal: find the low-cost errors that OSD-CS around a single BP ordering misses, and add them to the
class-probability sum. Target: heavy-pass LER from 0.0351 (validation) down by 0.0003–0.0008.

All work goes in `team/decoders/bposd2.py`. The candidate pool (`cc`, `co`, `ch` arrays, Zobrist de-duplication,
class sum) already accepts candidates from any source; `_relay` is an example of plugging one in.

1. **Randomly perturbed orderings.**
   - Add options `n_perturb` (number of extra orderings) and `perturb_scale`.
   - For each extra ordering, add fixed-seed noise to the chosen BP posterior before `np.argsort`, then call
     `_osd` with a small OSD-CS (try 100/20 and 200/30).
   - Noise to try: Gaussian with scale 0.5, 1, 2, 4 in LLR units; also multiplicative noise on the priors. Keep it
     modest: ordering by priors alone (no BP) gave LER 0.185.
   - Pre-generate the noise once per problem (shape `(n_perturb, n)`), as `_gammas` does, so decoding stays
     deterministic and cheap.
   - Sweep `n_perturb` in 4, 8, 16, 32.
2. **BP guided decimation (BPGD).**
   - After the normal BP pass: fix the mechanism with the most negative posterior to 1 (XOR its column into the
     syndrome, set its prior LLR to a large positive value so it can't fire again), re-run 2–3 BP iterations,
     repeat until the residual syndrome is zero or a step limit (about 15) is hit.
   - On success, add the fixed set as one candidate. Also use the posterior after each decimation step as an OSD
     ordering (optionally only every few steps).
   - Variant: at the first one or two steps, branch on the top 2–3 mechanisms to get several distinct solutions.
   - Cost estimate: 15 steps × 2–3 iterations at about 5 µs per iteration per shot is about 150–225 µs per shot.
     That's affordable only on the cascade's uncertain shots.
3. **Measure with the diagnostic.** Extend `team/diagnose.py` `CONFIGS` with the new settings. The number to push
   down is "search miss" (4.3% of bit errors for heavy) and the oracle gap (−0.0012).
4. **Accept if,** inside the cascade, the paired LER gain is at least −0.0003 and the **total** runtime of the
   cascade grows by at most about 40% (4.4 s → 6.2 s on the Mac). The heavy pass runs on only ~11% of shots, so a
   heavier pass costs much less in total than per shot. Then re-tune τ.

## Part 2: stacked NN on top of BP-OSD (features 18:30–19:00, training until 20:00)

Goal: a small network that corrects the decoder's ambiguous shots. The plain MLP needs about 3 billion shots to
get near 0.036, so the NN must start from the decoder's output.

1. **Features per shot:**
   - the 70 detector bits;
   - the 16-class posterior from the cheap pass, as log-probabilities **clipped to [log 1e-6, 0]**;
   - the two stats from `decode_full(..., return_stats=True)` (clip the 1e300 sentinel);
   - the number of fired detectors.
2. **Data:** generate with `team/common.generate` (new seeds ≥ 50000), decode with the cheap configuration
   (about 85k shots/s on the Mac), and cache features + labels as `.npz` in `data/generated/`.
   - About 20 minutes per basis for 50M shots, using the whole CPU. Run it in one block, with nothing else on the
     CPU.
   - Start with 20–30M shots per basis; generate more only if the 20:00 check looks promising.
3. **Model and loss:** extend `team/nn/train_nn.py` with an `--features` mode. Use the residual MLP (width
   512–1024, depth 3–4) and per-bit BCE.
   - Add the decoder's per-bit logit to the network output, so the model starts at the decoder's accuracy and
     learns only a correction.
   - **Clip that logit:** for shots where BP converged, the cheap pass returns one-hot probabilities, so clip the
     per-bit probabilities to [1e-6, 1 − 1e-6] before taking logits.
4. **Go/no-go at 20:00:** validation LER must start at about 0.0368 (cheap) and go down. Compare it with the
   cascade on the same shots. If it doesn't beat cheap by −0.0005, stop.
5. **If it works:** try heavy-pass features on the uncertain shots only, and wire inference into the hand-in
   (torch on the GPU; loading weights counts toward the runtime). Hand in the training code and data-generation
   script too.

## Part 3: spend the time budget on the timing machine

1. **Estimate the speed ratio** (needs clean timing, so run it while nothing else uses the CPU). The timing machine
   has 48 EPYC 9655 vCPUs; the Mac has 6 performance + 12 efficiency cores. Measure how our decoder scales with
   threads on the Mac (`NUMBA_NUM_THREADS` = 6, 12, 18) and extrapolate. Treat the result as a range, not a fact.
2. **Check for rule updates:** the Kaggle Rules and Overview pages may give a final t_cap and the software
   environment. Re-read them before choosing (doable during GPU training).
3. **Pick the hand-in configuration** by final score under the estimated range, using the table format in
   `FINDINGS.md` (score if machine = Mac, score if 2× faster).
4. **Cheap speed-ups worth doing:**
   - early exit from OSD-CS when one class holds more than 0.999 of the mass;
   - partial sort instead of a full `argsort` over 2470 columns;
   - reuse the BP state between the cheap and heavy passes of the cascade.

   Each must give bit-identical or paired-tested predictions.
5. **Two-submission strategy:** keep one max-accuracy submission to secure a top-10 place and one fast submission
   for the best final score. Confirm with the user which one becomes the hand-in, and remind them to select both
   on Kaggle before 23:59.

## Definition of done

- `FINDINGS.md` has a section per part with paired tables, including what failed.
- `submission_template/` holds the chosen decoder; `run_local.py` on `data/train` reports its LER and a
  deterministic median time; a matching `submission_test_*.csv` exists and has been submitted (with the user's
  approval).
- Everything is committed on `dev/sarane` (the user decides when to push).
- The user has been reminded to select the 2 final submissions on Kaggle before 23:59.
