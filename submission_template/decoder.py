"""Hand-in decoder: degenerate BP-OSD with an optional confidence cascade (numpy + numba only, no QEC libraries).

TAU = 0.4 (default): heavy pass on ~11% of shots: LER 0.0355 on data/train, ~4.4 s for 200k shots on an 18-core Mac.
TAU = 0 runs only the cheap pass: LER 0.0371, ~2.4 s on the same Mac.

1. Cheap pass on every shot: layered min-sum BP (2 iterations, alpha 0.45); if BP converges use its error, otherwise
   OSD-CS 200/30 seeded by BP's posterior. Every distinct error found adds its probability to its logical class, and
   each observable is decided by its marginal probability (approximate maximum-likelihood / degenerate decoding).
2. Shots whose least certain observable has |P(L_j = 1) - 0.5| < TAU are re-decoded by a heavy pass: three OSD-CS 800/60
   runs seeded by the posteriors after BP iterations 1, 2 and 3, summed together (duplicates removed).

Needs bposd2.py next to this file.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import bposd2  # noqa: E402

TAU = 0.4
CHEAP = dict(schedule=1, max_iter=2, alpha=0.45, snaps=(2,), n_single=200, n_pair=30, degenerate=True, shortcut=True)
HEAVY = dict(schedule=1, max_iter=3, alpha=0.45, snaps=(1, 2, 3), n_single=800, n_pair=60, degenerate=True,
             shortcut=False)


def decode(circuit, detector_events, tau=None):
    tau = TAU if tau is None else tau
    dets = np.asarray(detector_events)
    marg = bposd2.decode_full(circuit, dets, **CHEAP)[2]
    unsure = np.abs(marg - 0.5).min(1) < tau
    if tau > 0 and unsure.any():
        marg[unsure] = bposd2.decode_full(circuit, dets[unsure], **HEAVY)[2]
    return (marg > 0.5).astype(np.uint8)
