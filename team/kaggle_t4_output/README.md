# Kaggle T4 run of `team/kaggle_t4_check.ipynb` (2026-11-13, 19:55–20:22 CET)

Kernel `rane22sau/qec-mld-t4-check` (private), GPU T4 accelerator, 4 vCPUs, 31 GB RAM. Code: snapshot of
`dev/coset-walk` @ 8b0e0cc, attached as a private dataset, because the repo is private and `git clone` fails on
Kaggle. `kaggle_t4_check_as_run.ipynb` is the notebook as run: only the clone line is replaced.

Decoder settings: the defaults at 8b0e0cc, i.e. CPU profile `fast` + GPU dense pass (TAU 0.3).
Environment: Python 3.13, torch 2.11.0+cu130, numba 0.61.2, numpy 2.1.3. Kaggle's preinstalled versions are
used, not the pins in `submission_ml/requirements.txt`.

| Check | Result |
|---|---|
| `data/train` LER | 0.03385 (X 0.03430, Z 0.03340) |
| `data/train` runtime, 3 runs | 51.5 / 56.6 / 56.1 s (median 56.1 s) |
| `data/test` runtime | 57.0 s |
| Test bits differing from the Mac (MPS) reference, compared by position | 1 of 799,988 |
| Metrics cell (`mld_metrics.json`) | failed: `ModuleNotFoundError: hackathon_2026` |

**The two CSVs aren't valid submissions.** With numpy 2.1.3, `load_kaggle_csv` (`np.loadtxt`) dropped 3 of
200,000 rows, both in `test` and `train`: `gen_X_50000`, `gen_Z_00001` and `gen_Z_50002` for test. With the pinned
numpy 2.5.4 all rows load. Install the pinned numpy when running the hand-in. The 1-bit check above compares by
position, so it is only a rough cross-machine check.

Files: `qec-mld-t4-check.log` (full kernel log, JSON), `submission_train_ml_gpu.csv`, `submission_test_ml_gpu.csv`.
