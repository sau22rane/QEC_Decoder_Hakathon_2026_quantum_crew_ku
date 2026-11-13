"""Hybrid decoders built from the pieces in team/decoders/ (and team/nn/ once trained).

  lut_table(basis)                 small lookup table (syndromes seen >= MIN_COUNT times in 20M shots, per-bit majority)
  bposd_cascade(...)               cheap degenerate BP-OSD on every shot, expensive BP-OSD only on uncertain shots
  nn_probs(basis, dets)            neural-network P(L_j = 1 | D) per bit
  combinations of the above in team/compare_hybrids.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "decoders"), os.path.join(HERE, "nn")]
import numpy as np  # noqa: E402

import bposd2  # noqa: E402
from common import circuit  # noqa: E402

CHEAP = dict(schedule=1, max_iter=2, alpha=0.45, snaps=(2,), n_single=200, n_pair=30, degenerate=True, shortcut=True)
HEAVY = dict(schedule=1, max_iter=3, alpha=0.45, snaps=(1, 2, 3), n_single=800, n_pair=60, degenerate=True,
             shortcut=False)
LUT_SHOTS, MIN_COUNT = 20_000_000, 80


def _keys(bits):
    packed = np.packbits(np.asarray(bits, dtype=bool), axis=1)
    return np.ascontiguousarray(packed).view(np.dtype((np.void, packed.shape[1]))).ravel()


_LUT = {}


def lut_table(basis):
    """(sorted keys, per-bit P(L_j = 1)) for syndromes seen at least MIN_COUNT times; cached in data/generated/."""
    if basis in _LUT:
        return _LUT[basis]
    path = os.path.join(os.path.dirname(HERE), "data", "generated", f"lut_{basis}_{LUT_SHOTS}_{MIN_COUNT}.npz")
    if os.path.exists(path):
        f = np.load(path)
        keys = f["keys"].view(np.dtype((np.void, 9))).ravel()
        _LUT[basis] = (keys, f["p1"])
        return _LUT[basis]
    k = "XZ".index(basis)
    d, o = circuit(basis).compile_detector_sampler(seed=30_000 + k).sample(LUT_SHOTS, separate_observables=True)
    uk, inv, cnt = np.unique(_keys(d), return_inverse=True, return_counts=True)
    ones = np.zeros((len(uk), 4))
    np.add.at(ones, inv, o)
    keep = cnt >= MIN_COUNT
    keys, p1 = uk[keep], ones[keep] / cnt[keep, None]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, keys=np.frombuffer(keys.tobytes(), np.uint8).reshape(-1, 9), p1=p1)
    _LUT[basis] = (keys, p1)
    return _LUT[basis]


def lut_lookup(basis, dets):
    """(hit mask, per-bit P(L_j = 1) for hits)."""
    keys, p1 = lut_table(basis)
    q = _keys(dets)
    pos = np.searchsorted(keys, q)
    pos[pos == len(keys)] = 0
    hit = keys[pos] == q
    return hit, p1[pos[hit]]


def bposd_marginals(c, dets, **kw):
    return bposd2.decode_full(c, dets, **kw)[2]


def cascade_marginals(c, dets, tau=0.3, cheap=CHEAP, heavy=HEAVY):
    """Cheap decoder everywhere; re-decode shots whose least certain bit has |P - 0.5| < tau with the heavy one."""
    marg = bposd_marginals(c, dets, **cheap)
    unsure = np.abs(marg - 0.5).min(1) < tau
    if unsure.any():
        marg[unsure] = bposd_marginals(c, dets[unsure], **heavy)
    return marg, unsure.mean()


_NN = {}


def nn_probs(basis, dets, tag="w1024_d4"):
    """P(L_j = 1 | D) from the trained network team/nn/<basis>_<tag>.pt (torch, CPU or MPS)."""
    import torch
    from train_nn import ResMLP
    key = (basis, tag)
    if key not in _NN:
        ck = torch.load(os.path.join(HERE, "nn", f"{basis}_{tag}.pt"), map_location="cpu")
        m = ResMLP(width=ck["width"], depth=ck["depth"])
        m.load_state_dict(ck["state_dict"])
        _NN[key] = m.eval()
    with torch.no_grad():
        out = []
        x = torch.tensor(np.asarray(dets), dtype=torch.float32)
        for i in range(0, len(x), 65536):
            out.append(torch.sigmoid(_NN[key](x[i:i + 65536])).numpy())
    return np.concatenate(out).astype(np.float64)
