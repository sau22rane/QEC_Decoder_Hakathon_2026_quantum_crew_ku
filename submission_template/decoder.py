"""Hand-in decoder: degenerate BP-OSD in a three-level confidence cascade (numpy + numba only, no QEC libraries).

1. Cheap pass on every shot: layered min-sum BP (2 iterations, alpha 0.45); if BP converges use its error,
   otherwise OSD-CS 200/30 seeded by BP's posterior. Every distinct error found adds its probability to its
   logical class, and each observable is decided by its marginal probability (approximate maximum-likelihood
   / degenerate decoding).
2. Shots whose least certain observable has |P(L_j = 1) - 0.5| < 0.49 are re-decoded by the heavy pass: three
   OSD-CS 800/60 runs seeded by the posteriors after BP iterations 1, 2 and 3.
3. Shots still uncertain (|P - 0.5| < 0.4) get the SUPER pass: heavy with OSD-CS 1500/80 plus 32 extra OSD-CS
   200/30 runs seeded by fixed-seed perturbations of the BP posterior.

Measured on data/train (2x100k shots): LER 0.03459, 13.7 s on an 18-core Apple M5 Pro (t_cap = 15.4 s).
Deterministic: all randomness uses fixed seeds.

Import time is not timed (README, Final ranking), so the matrices for the competition circuits are built and the
numba kernels compiled here, at import. decode() still works for any circuit.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import stim  # noqa: E402

import bposd2  # noqa: E402

CHEAP = dict(schedule=1, max_iter=2, alpha=0.45, snaps=(2,), n_single=200, n_pair=30, degenerate=True, shortcut=True)
HEAVY = dict(schedule=1, max_iter=3, alpha=0.45, snaps=(1, 2, 3), n_single=800, n_pair=60, degenerate=True,
             shortcut=False)
SUPER = {**HEAVY, "n_single": 1500, "n_pair": 80, "n_perturb": 32, "perturb_single": 200, "perturb_pair": 30}
LEVELS = [(None, CHEAP), (0.49, HEAVY), (0.4, SUPER)]


def decode(circuit: stim.Circuit, detector_events: np.ndarray) -> np.ndarray:
    """
    circuit: the X or Z circuit, loaded with stim.Circuit.from_file.
    detector_events: bool array of shape (shots, 70), columns D0 to D69.
    Returns a 0/1 array of shape (shots, 4), columns L0 to L3.
    """
    dets = np.asarray(detector_events)
    marg = None
    for tau, cfg in LEVELS:
        if marg is None:
            marg = bposd2.decode_full(circuit, dets, **cfg)[2]
            continue
        unsure = np.flatnonzero(np.abs(marg - 0.5).min(1) < tau)
        if len(unsure):
            marg[unsure] = bposd2.decode_full(circuit, dets[unsure], **cfg)[2]
    return (marg > 0.5).astype(np.uint8)


def _warm_up():
    """Build the problem matrices for the bundled competition circuits and compile every kernel (untimed)."""
    for b in "XZ":
        path = os.path.join(HERE, f"{b}_circuit.stim")
        if os.path.exists(path):
            c = stim.Circuit.from_file(path)
            sample, _ = c.compile_detector_sampler(seed=0).sample(64, separate_observables=True)
            decode(c, sample)


_warm_up()
