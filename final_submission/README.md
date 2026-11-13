# Hand-in: exact maximum-likelihood decoder (team quantum_crew_ku)

Selected Kaggle submission: ref 58321275 (`submission_test_ml_exact.csv`), public score 0.03368.
Local check on `data/train` (2 x 100k shots): LER 0.03385 (X 0.03430, Z 0.03340).

## Files

| File | What it is |
|---|---|
| `decoder.py` | Entry point: `decode(circuit, detector_events) -> (shots, 4)` |
| `mldfast.py` | Builds the lookup tables from the circuit (Walsh-Hadamard transforms) and holds the CPU kernel |
| `mlddense.py` | The exact, unpruned pass on the GPU (PyTorch) |
| `X_circuit.stim`, `Z_circuit.stim` | The competition circuits, so the tables are built at import, before the clock starts |
| `requirements.txt` | Exact package versions |
| `run_local.py` | The organisers' timing script, unchanged |

No QEC decoder library, no pretrained model, no training data and no randomness are used.

## Method in short

Every mechanism of the detector error model touches detectors in at most two adjacent time layers, and the three
middle layer pairs are identical. The probability of each logical class given the syndrome is therefore an exact
chain of five tables on 14-bit indices. Observable i is predicted 1 exactly when
P(D, L_i = 0) - P(D, L_i = 1) < 0, which is the optimal decision for the per-bit metric. The derivation is in the
docstring of `mldfast.py`.

On a GPU (the default, `MODE = "exact"`) every shot is decoded with the full tables: no beam, no truncation, no
thresholds. Bits whose float32 probability is within 5e-4 of 0.5 are recomputed in float64, so the output does not
depend on the GPU model.

## Lookup tables

The tables `decode` uses are created by the code in this folder (`mldfast.build` and `mldfast.build_dense`) from
the circuit alone. They are built on the first import and cached in `mld_cache/` (about 7 GB per basis, 14 GB in
total); later imports load them. Building needs no internet access. It took 30-160 s per basis and table kind on a
10-core Apple M4, and needs about 6 GB of RAM. Deleting `mld_cache/` just triggers a rebuild.

## Running

```bash
pip install -r final_submission/requirements.txt
python final_submission/run_local.py --decoder final_submission --problem-set data/kaggle_private --threads 48
```

- The selected submission needs a CUDA GPU with at least 16 GB (both bases stay loaded above 40 GB). GPU memory used:
  about 5.4 GB per basis for the table plus up to a few GB of working arrays.
- Without a GPU, or with `MLD_GPU=0`, the decoder falls back to a lightly pruned CPU pass (profile `cap`). Its
  predictions differ from the selected submission in about 200 of 800,000 bits, so it is a fallback only.
- `decoder.STATS` holds per-call timings and counts after a run.

## What has and hasn't been tested

- Exact mode: run on an Apple GPU (MPS) only: 570-646 s for 2 x 100k shots there, deterministic across runs.
- An earlier CPU + GPU version of this decoder ran on a Kaggle T4 (CUDA, Linux) with the same LER.
- Exact mode has not been run on a CUDA GPU, and not on a B200. Our estimate for the timing machine is 6-11 s
  if the B200 sustains 60-120 TFLOPS in float32, and 15-16 s at 40 TFLOPS.
