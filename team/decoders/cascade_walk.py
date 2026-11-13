"""Three-level confidence cascade with the coset walk (bposd_walk) in the last level(s).

Same structure as submission_template/decoder.py; WALK adds the best-first walk over weight-3 null combinations to
the SUPER pass. PROFILES are compared on data/train by `python team/decoders/cascade_walk.py`.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import numpy as np  # noqa: E402

import bposd_walk  # noqa: E402

CHEAP1 = dict(schedule=1, max_iter=2, alpha=0.45, snaps=(2,), n_single=200, n_pair=30, degenerate=True, shortcut=True)
CHEAP2 = {**CHEAP1, "snaps": (1, 2)}
HEAVY = dict(schedule=1, max_iter=3, alpha=0.45, snaps=(1, 2, 3), n_single=800, n_pair=60, degenerate=True,
             shortcut=False)
SUPER = {**HEAVY, "n_single": 1500, "n_pair": 80, "n_perturb": 32, "perturb_single": 200, "perturb_pair": 30}
WALK = dict(walk_exp=200, walk_seeds=64, walk_max=20000, walk_gap=14.0)
HEAVY_W = {**HEAVY, "walk_exp": 50, "walk_max": 5000}
SUPER_W = {**SUPER, **WALK}

PROFILES = {
    "robust (team hand-in)": [(None, CHEAP2), (0.49, HEAVY), (0.4, SUPER)],
    "robust + walk": [(None, CHEAP2), (0.49, HEAVY), (0.4, SUPER_W)],
    "robust + walk, SUPER at 0.45": [(None, CHEAP2), (0.49, HEAVY), (0.45, SUPER_W)],
    "wide (team)": [(None, CHEAP1), (0.5, HEAVY), (0.4, SUPER)],
    "wide + walk": [(None, CHEAP1), (0.5, HEAVY), (0.4, SUPER_W)],
    "wide + walk, SUPER at 0.45": [(None, CHEAP1), (0.5, HEAVY), (0.45, SUPER_W)],
    "wide, heavy+walk50 -> 0.45 SUPER+walk": [(None, CHEAP1), (0.5, HEAVY_W), (0.45, SUPER_W)],
}
PROFILE = "robust + walk"


def decode_levels(circuit, detector_events, levels):
    dets = np.asarray(detector_events)
    marg = None
    for tau, cfg in levels:
        if marg is None:
            marg = bposd_walk.decode_full(circuit, dets, **cfg)[2]
            continue
        unsure = np.flatnonzero(np.abs(marg - 0.5).min(1) < tau)
        if len(unsure):
            marg[unsure] = bposd_walk.decode_full(circuit, dets[unsure], **cfg)[2]
    return (marg > 0.5).astype(np.uint8)


def decode(circuit, detector_events):
    return decode_levels(circuit, detector_events, PROFILES[PROFILE])


if __name__ == "__main__":
    from common import circuit, labelled_set
    data = labelled_set("train")
    ref = None
    for name, levels in PROFILES.items():
        for b in "XZ":
            decode_levels(circuit(b), data[b][0][:300], levels)
        t0 = time.perf_counter()
        pred = {b: decode_levels(circuit(b), data[b][0], levels) for b in "XZ"}
        dt = time.perf_counter() - t0
        w = np.concatenate([(pred[b].astype(bool) != data[b][1]).mean(1) for b in "XZ"])
        lers = [float((pred[b].astype(bool) != data[b][1]).mean()) for b in "XZ"]
        line = f"{name:<40} LER {w.mean():.5f} (X {lers[0]:.5f}, Z {lers[1]:.5f})  {dt:6.1f} s"
        if ref is None:
            ref = w
        else:
            d = w - ref
            line += f"  d vs team hand-in {d.mean():+.5f} ± {d.std() / np.sqrt(len(d)):.5f}"
        print(line, flush=True)
