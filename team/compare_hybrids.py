"""Paired comparison of hybrid decoders on the shared validation shots.

    uv run python team/compare_hybrids.py --suite lut_cascade|nn [--shots 50000]
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "decoders")]
import numpy as np  # noqa: E402

import hybrid as hy  # noqa: E402
from compare import Table, val  # noqa: E402
from common import circuit  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--suite", default="lut_cascade")
ap.add_argument("--shots", type=int, default=50_000)
args = ap.parse_args()
v = val(args.shots)
for b in "XZ":
    hy.bposd_marginals(circuit(b), v[b][0][:50], **hy.CHEAP)
    hy.bposd_marginals(circuit(b), v[b][0][:50], **hy.HEAVY)
frac = {}


def cheap(b, c, d):
    return hy.bposd_marginals(c, d, **hy.CHEAP) > 0.5


def lut_then(inner):
    def f(b, c, d):
        out = np.zeros((len(d), 4), np.uint8)
        hit, p1 = hy.lut_lookup(b, d)
        out[hit] = p1 > 0.5
        if (~hit).any():
            out[~hit] = inner(b, c, d[~hit])
        frac[b] = hit.mean()
        return out
    return f


def cascade(tau):
    def f(b, c, d):
        m, u = hy.cascade_marginals(c, d, tau=tau)
        frac[b] = u
        return m > 0.5
    return f


if args.suite == "lut_cascade":
    for b in "XZ":
        hy.lut_table(b)
    print("=== LUT front-end and confidence cascade (cheap = layered 2 it, degenerate 200/30; "
          "heavy = layered 3 it, snaps 1-3, 800/60)")
    T = Table(args.shots)
    T.run("cheap degenerate BP-OSD", cheap)
    T.run("heavy degenerate BP-OSD on every shot", lambda b, c, d: hy.bposd_marginals(c, d, **hy.HEAVY) > 0.5)
    T.run(f"LUT (>= {hy.MIN_COUNT} counts) + cheap", lut_then(lambda b, c, d: cheap(b, c, d)))
    print(f"   LUT covers {np.mean(list(frac.values())):.3f} of shots")
    for tau in [0.1, 0.2, 0.3, 0.4]:
        T.run(f"cascade tau={tau}", cascade(tau))
        print(f"   heavy decoder used on {np.mean(list(frac.values())):.3f} of shots")
    T.run("LUT + cascade tau=0.3", lut_then(lambda b, c, d: cascade(0.3)(b, c, d)))

if args.suite == "nn":
    print("=== Neural network and NN + BP-OSD hybrids")
    T = Table(args.shots)
    T.run("cheap degenerate BP-OSD", cheap)
    T.run("NN alone", lambda b, c, d: hy.nn_probs(b, d) > 0.5)

    def avg(wt):
        return lambda b, c, d: (wt * hy.nn_probs(b, d) + (1 - wt) * hy.bposd_marginals(c, d, **hy.CHEAP)) > 0.5
    for wt in [0.3, 0.5, 0.7]:
        T.run(f"average P: {wt} NN + {1 - wt:.1f} BP-OSD", avg(wt))

    def gate(tau):
        def f(b, c, d):
            p = hy.nn_probs(b, d)
            unsure = np.abs(p - 0.5).min(1) < tau
            out = p > 0.5
            if unsure.any():
                q = hy.bposd_marginals(c, d[unsure], **hy.CHEAP)
                out[unsure] = (p[unsure] + q) / 2 > 0.5
            frac[b] = unsure.mean()
            return out
        return f
    for tau in [0.2, 0.35, 0.45]:
        T.run(f"NN, uncertain (tau={tau}) -> avg with BP-OSD", gate(tau))
        print(f"   BP-OSD used on {np.mean(list(frac.values())):.3f} of shots")

    def gate_heavy(tau):
        def f(b, c, d):
            p = hy.nn_probs(b, d)
            unsure = np.abs(p - 0.5).min(1) < tau
            out = p > 0.5
            if unsure.any():
                q = hy.bposd_marginals(c, d[unsure], **hy.HEAVY)
                out[unsure] = (p[unsure] + q) / 2 > 0.5
            frac[b] = unsure.mean()
            return out
        return f
    T.run("NN, uncertain (tau=0.35) -> avg with HEAVY BP-OSD", gate_heavy(0.35))
    print(f"   BP-OSD used on {np.mean(list(frac.values())):.3f} of shots")
