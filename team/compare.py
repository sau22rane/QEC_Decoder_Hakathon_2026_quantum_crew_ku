"""Paired comparison of decoder variants on the same fresh validation shots.

    uv run python team/compare.py --suite schedule|degenerate|ideas|all [--shots 100000]

Every variant decodes the same shots (validation seeds 2000/2001, never used for training), so differences between
variants are measured on identical errors. 'Δ vs ref' is the LER difference against the first line of the suite, with
its paired standard error.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "decoders")]
import numpy as np  # noqa: E402

from common import circuit, generate  # noqa: E402

_VAL = {}


def val(shots):
    if shots not in _VAL:
        _VAL[shots] = {b: tuple(x[:shots] for x in generate(b, 1_000_000, seed=2000 + k)) for k, b in enumerate("XZ")}
    return _VAL[shots]


class Table:
    def __init__(self, shots):
        self.shots, self.ref = shots, None

    def run(self, name, fn):
        """fn(basis, circuit, detectors) -> (shots, 4) predictions."""
        data = val(self.shots)
        t0 = time.perf_counter()
        preds = {b: np.asarray(fn(b, circuit(b), data[b][0])) for b in "XZ"}
        dt = time.perf_counter() - t0
        wrong = np.concatenate([(preds[b].astype(bool) != data[b][1]).mean(1) for b in "XZ"])
        e = wrong.mean()
        line = f"{name:<55} LER {e:.5f} ± {wrong.std() / np.sqrt(len(wrong)):.5f}   {1e6 * dt / len(wrong):6.1f} µs/shot"
        if self.ref is None:
            self.ref = wrong
        else:
            d = wrong - self.ref
            line += f"   Δ vs ref {d.mean():+.5f} ± {d.std() / np.sqrt(len(d)):.5f}"
        print(line, flush=True)
        return e, dt


if __name__ == "__main__":
    import bposd
    import bposd2

    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="all")
    parser.add_argument("--shots", type=int, default=100_000)
    args = parser.parse_args()
    v = val(args.shots)
    bposd.decode(circuit("X"), v["X"][0][:50])
    bposd2.decode(circuit("X"), v["X"][0][:50])
    bposd2.decode(circuit("X"), v["X"][0][:50], schedule=0)

    def v2(**kw):
        return lambda b, c, d: bposd2.decode(c, d, **kw)

    if args.suite in ("schedule", "all"):
        print("\n=== BP schedule (most-likely-error OSD-CS 200/30, BP shortcut on)")
        T = Table(args.shots)
        T.run("v1 flooding, 3 it, a=0.45", lambda b, c, d: bposd.decode(c, d))
        for it in [1, 2, 3, 4, 6, 10]:
            T.run(f"flooding {it} it, a=0.45", v2(schedule=0, max_iter=it, snaps=(it,), degenerate=False, shortcut=True))
        for it, a in [(1, 0.45), (1, 0.7), (2, 0.45), (2, 0.6), (2, 0.8), (3, 0.6), (3, 0.8), (5, 0.8), (10, 0.8), (10, 1.0)]:
            T.run(f"layered {it} it, a={a}", v2(schedule=1, max_iter=it, alpha=a, snaps=(it,), degenerate=False, shortcut=True))

    if args.suite in ("degenerate", "all"):
        print("\n=== Degenerate (total probability per logical class) vs most likely error")
        T = Table(args.shots)
        base = dict(schedule=0, max_iter=3, alpha=0.45, snaps=(3,))
        T.run("most likely error, shortcut (= v1)", v2(**base, degenerate=False, shortcut=True))
        T.run("most likely error, OSD always", v2(**base, degenerate=False, shortcut=False))
        T.run("degenerate sum, 200/30", v2(**base, degenerate=True))
        T.run("degenerate sum, 400/40", v2(**base, degenerate=True, n_single=400, n_pair=40))
        T.run("degenerate sum, 800/60", v2(**base, degenerate=True, n_single=800, n_pair=60))
        T.run("degenerate sum, shortcut", v2(**base, degenerate=True, shortcut=True))

    if args.suite in ("ideas", "all"):
        print("\n=== Multiple OSD orderings, prior scaling (layered BP, degenerate sum, OSD-CS 200/30, shortcut)")
        T = Table(args.shots)
        base = dict(schedule=1, alpha=0.45, degenerate=True, shortcut=True)
        T.run("layered 2 it, snaps (2)", v2(**base, max_iter=2, snaps=(2,)))
        T.run("layered 2 it, snaps (1,2)", v2(**base, max_iter=2, snaps=(1, 2)))
        T.run("layered 2 it, snaps (0,1,2)", v2(**base, max_iter=2, snaps=(0, 1, 2)))
        T.run("layered 3 it, snaps (1,2,3)", v2(**base, max_iter=3, snaps=(1, 2, 3)))
        T.run("layered 2 it, snaps (1,2), 400/40", v2(**base, max_iter=2, snaps=(1, 2), n_single=400, n_pair=40))
        T.run("layered 2 it, snaps (2), 800/60", v2(**base, max_iter=2, snaps=(2,), n_single=800, n_pair=60))
        for s in [0.7, 0.85, 1.2]:
            T.run(f"layered 2 it, snaps (2), bp_scale {s}", v2(**base, max_iter=2, snaps=(2,), bp_scale=s))
