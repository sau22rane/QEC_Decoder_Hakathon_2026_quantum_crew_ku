"""Generate training data for the stacked NN: fresh shots decoded by the cheap degenerate BP-OSD pass.

    uv run python team/nn/make_stack_features.py --basis X --millions 20

Writes data/generated/stack_<basis>/chunk_<i>.npz (git-ignored), one per 1M shots:
  d     detector events, bit-packed (shots, 9) uint8
  o     true observable flips, bit-packed (shots, 1) uint8 (low 4 bits)
  p     16-class posterior from the cheap pass, float16 (shots, 16)
  st    [cost of the most likely error found, log of the class-sum normaliser], float16 (shots, 2)
Seeds: 50000 + 1000 * basis_index + chunk (never used elsewhere). The validation features use the shared validation
seeds 2000/2001 (first 100k shots) and are written to stack_<basis>/val.npz.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "decoders")]
import numpy as np  # noqa: E402

import bposd2  # noqa: E402
import cascade  # noqa: E402
from common import GEN_DIR, circuit, generate  # noqa: E402


def features(c, d):
    _, probs, _, st = bposd2.decode_full(c, d, return_stats=True, **cascade.CHEAP)
    st = st.copy()
    st[:, 0] = np.minimum(st[:, 0], 200.0)  # 1e300 sentinel (no candidate) -> 200
    st[:, 1] = np.log(np.maximum(st[:, 1], 1e-30))
    return probs.astype(np.float16), st.astype(np.float16)


def save(path, d, o, p, st):
    np.savez(path, d=np.packbits(d, axis=1), o=np.packbits(o, axis=1, bitorder="little"), p=p, st=st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", required=True, choices=["X", "Z"])
    ap.add_argument("--millions", type=int, default=20)
    args = ap.parse_args()
    k = "XZ".index(args.basis)
    out = os.path.join(GEN_DIR, f"stack_{args.basis}")
    os.makedirs(out, exist_ok=True)
    c = circuit(args.basis)
    vd, vo = generate(args.basis, 1_000_000, seed=2000 + k)
    vd, vo = vd[:100_000], vo[:100_000]
    p, st = features(c, vd)
    save(os.path.join(out, "val.npz"), vd, vo, p, st)
    for i in range(args.millions):
        path = os.path.join(out, f"chunk_{i:03d}.npz")
        if os.path.exists(path):
            continue
        t = time.time()
        d, o = c.compile_detector_sampler(seed=50_000 + 1000 * k + i).sample(1_000_000, separate_observables=True)
        p, st = features(c, d)
        save(path, d, o, p, st)
        print(f"[{args.basis}] chunk {i} done in {time.time() - t:.1f} s", flush=True)


if __name__ == "__main__":
    main()
