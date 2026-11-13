"""Paired LER of the transfer-matrix ML decoder against the current hand-in, on the shared validation shots.

    .venv/bin/python team/mld_eval.py [--shots 50000] [--K 128]

Caches predictions in data/generated/mld_eval_*.npz so later comparisons (other K, cascades) reuse them.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "decoders")]
import numpy as np  # noqa: E402

import mld  # noqa: E402
from common import GEN_DIR, circuit, generate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--shots", type=int, default=50_000)
ap.add_argument("--K", type=int, default=0, help="dense beam size; 0 = fast sparse kernel")
ap.add_argument("--M", type=int, default=1024)
ap.add_argument("--delta", type=float, default=1e-4)
ap.add_argument("--kmax", type=int, default=128)
ap.add_argument("--no-baseline", action="store_true")
args = ap.parse_args()

wrong_m, wrong_b = [], []
for k, b in enumerate("XZ"):
    d, o = generate(b, 1_000_000, seed=2000 + k)
    d, o = d[:args.shots], o[:args.shots]
    tag = f"K{args.K}" if args.K else f"M{args.M}_d{args.delta:g}_k{args.kmax}"
    run = (lambda x, tb: mld.decode_beam(b, x, K=args.K, tables=tb)) if args.K else \
        (lambda x, tb: mld.decode_fast(b, x, args.delta, args.kmax, args.M, tables=tb))
    path = os.path.join(GEN_DIR, f"mld_eval_{b}_{args.shots}_{tag}.npz")
    if os.path.exists(path):
        pchi = np.load(path)["pchi"]
    else:
        tb = mld.build(b)
        run(d[:64], tb)
        t0 = time.time()
        pchi = run(d, tb)
        print(f"[{b}] ML {tag}: {time.time() - t0:.1f} s for {len(d)} shots", flush=True)
        np.savez(path, pchi=pchi)
    pred, p1 = mld.predict(pchi)
    wm = (pred.astype(bool) != o).mean(1)
    wrong_m.append(wm)
    line = f"[{b}] ML {tag}: LER {wm.mean():.5f} ± {wm.std() / np.sqrt(len(wm)):.5f}"
    if not args.no_baseline:
        bpath = os.path.join(GEN_DIR, f"mld_eval_{b}_{args.shots}_handin.npz")
        if os.path.exists(bpath):
            base = np.load(bpath)["pred"]
        else:
            sys.path.insert(0, os.path.join(os.path.dirname(HERE), "submission_template"))
            import decoder  # noqa: E402
            base = decoder.decode(circuit(b), d)
            np.savez(bpath, pred=base)
        wb = (base.astype(bool) != o).mean(1)
        wrong_b.append(wb)
        dd = wm - wb
        line += f"   hand-in {wb.mean():.5f}   Δ {dd.mean():+.5f} ± {dd.std() / np.sqrt(len(dd)):.5f}"
    print(line, flush=True)

wm = np.concatenate(wrong_m)
line = f"[both] ML {tag}: LER {wm.mean():.5f} ± {wm.std() / np.sqrt(len(wm)):.5f}"
if wrong_b:
    wb = np.concatenate(wrong_b)
    dd = wm - wb
    line += f"   hand-in {wb.mean():.5f}   Δ {dd.mean():+.5f} ± {dd.std() / np.sqrt(len(dd)):.5f}"
print(line, flush=True)
