"""Hand-in decoder: maximum-likelihood decoding by a transfer matrix over time layers (stim + numpy + numba only).

Every mechanism of the detector error model touches at most two adjacent time layers of detectors, so the exact
probability of each logical class given the syndrome is a chain of five tables on 14-bit indices (see mldfast.py for
the derivation). The tables are built from the circuit with Walsh-Hadamard transforms; decoding is one left-to-right
pass per shot that keeps only the entries that matter. No BP, no OSD, no search, no training data, no randomness.

Measured on data/train (2 x 100k shots): see team/FINDINGS.md, section "Exact maximum-likelihood decoder".

Import time is not timed (README, Final ranking), so the tables for the two competition circuits are built here, at
import (a few minutes the first time, then loaded from mld_cache/), and the numba kernel is compiled. decode() also
works for any other circuit with the same layer structure, building its tables on first use.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import stim  # noqa: E402

import mldfast  # noqa: E402

M = 4096        # entries kept per row of the middle table
# (delta, kmax, eta): expand running-vector entries within a factor delta of the best one, at most kmax per layer;
# skip row entries contributing less than eta x the layer's largest contribution.
PROFILES = {"max": (1e-6, 512, 1e-9),    # converged: looser limits change nothing measurable; ~110 s on a 10-core M4
            "fast": (1e-4, 128, 1e-6)}   # +0.00003 +- 0.00002 LER; 6.6 s on a 10-core M4
PROFILE = os.environ.get("MLD_PROFILE", "max")
DELTA, KMAX, ETA = PROFILES[PROFILE]
_TABLES = {}


def _tables(circuit):
    key = str(circuit)
    if key not in _TABLES:
        _TABLES[key] = mldfast.build(circuit, M=M, cache_dir=os.path.join(HERE, "mld_cache"))
    return _TABLES[key]


def decode(circuit: stim.Circuit, detector_events: np.ndarray) -> np.ndarray:
    """
    circuit: the X or Z circuit, loaded with stim.Circuit.from_file.
    detector_events: bool array of shape (shots, 70), columns D0 to D69.
    Returns a 0/1 array of shape (shots, 4), columns L0 to L3.
    """
    dets = np.asarray(detector_events)
    tb = _tables(circuit)
    pchi = mldfast.run(tb, dets, DELTA, KMAX, ETA)
    pred = (pchi[:, 1:] < 0).astype(np.uint8)
    lost = np.flatnonzero(~(pchi[:, 0] > 0))
    if len(lost):  # pruning removed every path (not seen in 600k shots): redo those shots without beam limits
        pred[lost] = (mldfast.run(tb, dets[lost], 0.0, mldfast.SZ, 0.0)[:, 1:] < 0).astype(np.uint8)
    return pred


def _warm_up():
    """Build or load the tables for the bundled competition circuits and compile the kernel (untimed)."""
    for b in "XZ":
        path = os.path.join(HERE, f"{b}_circuit.stim")
        if os.path.exists(path):
            c = stim.Circuit.from_file(path)
            sample, _ = c.compile_detector_sampler(seed=0).sample(64, separate_observables=True)
            decode(c, sample)


_warm_up()
