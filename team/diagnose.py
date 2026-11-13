"""Failure diagnostic for BP-OSD: is a wrong prediction a SEARCH failure or genuine AMBIGUITY?

    uv run python team/diagnose.py [--shots 50000]

Stim's DEM sampler returns the true error of every shot. We map it onto the merged columns of problem_info (same
key and order), check H e = D and L e = O, and compare its cost sum_j e_j log((1-p_j)/p_j) with the cost of the most
likely error the decoder found:

  search miss  the true error is MORE likely (cheaper) than anything the decoder found: better search can fix it
  ambiguity    the decoder found an error at least as likely as the truth: only better summation / calibration helps,
               and part of it is irreducible
  class absent the true logical class got zero probability: no candidate in the true class was found at all

The 'oracle' line adds the true error to the decoder's class sum when it was missed. Its LER bounds what a perfect
search on top of this decoder could reach.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "decoders")]
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

import bposd2  # noqa: E402
from common import circuit, matrices  # noqa: E402

CONFIGS = {
    "cheap (submitted)": dict(schedule=1, max_iter=2, alpha=0.45, snaps=(2,), n_single=200, n_pair=30,
                              degenerate=True, shortcut=True),
    "cheap, no BP shortcut": dict(schedule=1, max_iter=2, alpha=0.45, snaps=(2,), n_single=200, n_pair=30,
                                  degenerate=True, shortcut=False),
    "heavy (3 orderings, 800/60)": dict(schedule=1, max_iter=3, alpha=0.45, snaps=(1, 2, 3), n_single=800, n_pair=60,
                                        degenerate=True, shortcut=False),
}


def true_errors(basis, shots, seed):
    """(detectors, observables, true error over merged columns) sampled from the DEM."""
    c = circuit(basis)
    dem = c.detector_error_model()
    cols, inst_col = {}, []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        dets, obs = set(), set()
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                dets ^= {t.val}
            elif t.is_logical_observable_id():
                obs ^= {t.val}
        key = (frozenset(dets), frozenset(obs))
        if key not in cols:
            cols[key] = len(cols)
        inst_col.append(cols[key])
    H, L, p = matrices(basis)
    assert len(cols) == H.shape[1], "column mapping differs from problem_info"
    M = sp.csr_matrix((np.ones(len(inst_col), np.uint8), (np.arange(len(inst_col)), inst_col)),
                      shape=(len(inst_col), len(cols)))
    d, o, err = dem.compile_sampler(seed=seed).sample(shots, return_errors=True)
    e = (sp.csr_matrix(err.astype(np.uint8)) @ M).toarray() % 2
    assert np.array_equal((e @ H.T.astype(np.int64)) % 2, d.astype(np.int64)), "H e != D"
    assert np.array_equal((e @ L.T.astype(np.int64)) % 2, o.astype(np.int64)), "L e != O"
    return d, o, e.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=50_000)
    args = ap.parse_args()
    bits = 1 << np.arange(4)
    data = {}
    for k, b in enumerate("XZ"):
        d, o, e = true_errors(b, args.shots, seed=4000 + k)
        H, L, p = matrices(b)
        w = np.log((1 - p) / p)
        data[b] = (d, o, e, e @ w, (o.astype(np.int64) * bits).sum(1), e.sum(1))
        print(f"{b}: {args.shots} shots, true error weight mean {e.sum(1).mean():.2f} (max {e.sum(1).max()}), "
              f"mapping verified (H e = D, L e = O)")

    for name, cfg in CONFIGS.items():
        for b in "XZ":
            bposd2.decode_full(circuit(b), data[b][0][:20], **cfg)
        rows = {"shots": 0, "wrong_shots": 0, "bit_err": 0, "miss": 0, "miss_bits": 0, "amb": 0, "amb_bits": 0,
                "absent": 0, "oracle_bit_err": 0, "p_true_amb": [], "w_true_miss": [], "w_true_all": []}
        for b in "XZ":
            d, o, e, cost_true, cls_true, wt = data[b]
            _, probs, marg, stats = bposd2.decode_full(circuit(b), d, return_stats=True, **cfg)
            pred = marg > 0.5
            wrong_bits = (pred != o).sum(1)
            wrong = wrong_bits > 0
            best_cost = stats[:, 0]
            miss = cost_true < best_cost - 1e-6
            absent = probs[np.arange(len(d)), cls_true] == 0
            # oracle: add the true error to the class sum when the search missed it
            unnorm = probs * stats[:, 1:2]
            add = np.where(miss, np.exp(np.clip(best_cost - cost_true, -700, 700)), 0.0)
            unnorm[np.arange(len(d)), cls_true] += add
            unnorm /= unnorm.sum(1, keepdims=True)
            omarg = np.stack([unnorm[:, (np.arange(16) >> j) & 1 == 1].sum(1) for j in range(4)], 1)
            rows["oracle_bit_err"] += ((omarg > 0.5) != o).sum()
            rows["shots"] += len(d)
            rows["wrong_shots"] += wrong.sum()
            rows["bit_err"] += wrong_bits.sum()
            rows["miss"] += (wrong & miss).sum()
            rows["miss_bits"] += wrong_bits[wrong & miss].sum()
            rows["amb"] += (wrong & ~miss).sum()
            rows["amb_bits"] += wrong_bits[wrong & ~miss].sum()
            rows["absent"] += (wrong & absent).sum()
            rows["p_true_amb"].append(probs[np.arange(len(d)), cls_true][wrong & ~miss])
            rows["w_true_miss"].append(wt[wrong & miss])
            rows["w_true_all"].append(wt[wrong])
        n = rows["shots"] * 4
        ler, oler = rows["bit_err"] / n, rows["oracle_bit_err"] / n
        print(f"\n=== {name}: LER {ler:.5f}  ({rows['wrong_shots']} wrong shots of {rows['shots']})")
        print(f"  search miss (truth cheaper than best found): {rows['miss']:5d} shots = "
              f"{rows['miss_bits'] / rows['bit_err']:.1%} of bit errors; true error weight mean "
              f"{np.concatenate(rows['w_true_miss']).mean():.2f}")
        print(f"  ambiguity (found something at least as likely): {rows['amb']:5d} shots = "
              f"{rows['amb_bits'] / rows['bit_err']:.1%} of bit errors; mean P(true class) given to them "
              f"{np.concatenate(rows['p_true_amb']).mean():.3f}")
        print(f"  true class absent from candidates: {rows['absent']} of {rows['wrong_shots']} wrong shots")
        print(f"  oracle (add the missed true error): LER {oler:.5f}  -> perfect search would gain {oler - ler:+.5f}")
        wts = np.concatenate(rows["w_true_all"])
        print(f"  true-error weight of wrong shots: " +
              ", ".join(f"w={k}: {(wts == k).mean():.1%}" for k in range(1, 9)) + f", w>8: {(wts > 8).mean():.1%}")


if __name__ == "__main__":
    main()
