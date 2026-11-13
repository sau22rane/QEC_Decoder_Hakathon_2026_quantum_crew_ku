"""Write a Kaggle submission file for data/test with one of the cascade_walk profiles.

    python team/make_submission.py "<profile name>" submission_test_walk.csv
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "decoders")]
import numpy as np  # noqa: E402
import stim  # noqa: E402
from hackathon_2026 import load_kaggle_csv, save_submission  # noqa: E402

import cascade_walk  # noqa: E402

profile, out = sys.argv[1], sys.argv[2]
levels = cascade_walk.PROFILES[profile]
root = os.path.join(os.path.dirname(HERE), "data", "test")
data = load_kaggle_csv(os.path.join(root, "test.csv"))
pred = np.zeros((len(data["id"]), 4), np.uint8)
t0 = time.perf_counter()
for b in "XZ":
    c = stim.Circuit.from_file(os.path.join(root, f"{b}_circuit.stim"))
    rows = data["basis"] == b
    pred[rows] = cascade_walk.decode_levels(c, data["detector_events"][rows], levels)
print(f"{profile}: decoded {len(pred)} shots in {time.perf_counter() - t0:.1f} s; mean flips per bit {pred.mean():.4f}")
save_submission(out, data["id"], pred)
print("wrote", out)
