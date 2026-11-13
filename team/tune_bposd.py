"""Grid-search BP-OSD parameters on fresh validation shots (both bases). Run from the repository root:

    uv run python team/tune_bposd.py [--shots 50000]
"""
import argparse
import itertools
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "decoders"))
import numpy as np  # noqa: E402

import bposd  # noqa: E402
from common import circuit, generate, ler  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--shots", type=int, default=50_000)
parser.add_argument("--grid", choices=["coarse", "bp", "osd"], default="coarse")
parser.add_argument("--iters", type=int, default=3)
parser.add_argument("--alpha", type=float, default=0.6)
args = parser.parse_args()

data = {b: tuple(x[: args.shots] for x in generate(b, 1_000_000, seed=2000 + k)) for k, b in enumerate("XZ")}
bposd.decode(circuit("X"), data["X"][0][:10])  # compile


def run(**kw):
    t0, preds, trues = time.perf_counter(), [], []
    for b in "XZ":
        preds.append(bposd.decode(circuit(b), data[b][0], **kw))
        trues.append(data[b][1])
    dt = time.perf_counter() - t0
    e = ler(np.concatenate(preds), np.concatenate(trues))
    se = np.sqrt(e * (1 - e) / (4 * 2 * args.shots))
    print(f"LER {e:.5f} ± {se:.5f}  {dt:6.2f} s ({1e6 * dt / (2 * args.shots):5.1f} µs/shot)  {kw}", flush=True)
    return e, dt


if args.grid == "coarse":
    print("--- BP iterations / normalisation (OSD-CS 40/12)")
    for it, a in itertools.product([0, 3, 10, 30], [0.6, 0.8, 1.0]):
        if it == 0 and a != 0.8:
            continue
        run(max_iter=it, alpha=a)
    print("--- OSD-CS size (BP 10 iterations, alpha 0.8)")
    for s, p in [(0, 0), (40, 0), (40, 12), (100, 20), (200, 30), (300, 40)]:
        run(max_iter=10, alpha=0.8, osd_singles=s, osd_pairs=p)
elif args.grid == "bp":
    print("--- few BP iterations x alpha (OSD-CS 100/20)")
    for it, a in itertools.product([1, 2, 3, 4, 6], [0.3, 0.45, 0.6, 0.75]):
        run(max_iter=it, alpha=a, osd_singles=100, osd_pairs=20)
elif args.grid == "osd":
    print(f"--- OSD-CS size (BP {args.iters} iterations, alpha {args.alpha})")
    for s, p in [(40, 12), (100, 20), (200, 30), (400, 40), (800, 60), (1500, 80)]:
        run(max_iter=args.iters, alpha=args.alpha, osd_singles=s, osd_pairs=p)
