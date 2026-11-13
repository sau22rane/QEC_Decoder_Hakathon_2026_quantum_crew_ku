"""Benchmark a decoder quickly: LER (with a standard error) and time per shot, per basis.

Run from the repository root:

    uv run python team/bench.py                                   # submission_template/decoder.py on data/train
    uv run python team/bench.py --decoder team/decoders/zeros.py  # any file that defines decode(circuit, detector_events)
    uv run python team/bench.py --fresh 1000000 --seed 7          # fresh generated shots instead of data/train

The decoder file must define decode(circuit, detector_events) -> (shots, 4) 0/1 array, same as the hand-in.
"""
import argparse
import importlib.util
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

from common import ROOT, circuit, generate, labelled_set, ler  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--decoder", default=os.path.join(ROOT, "submission_template", "decoder.py"))
parser.add_argument("--shots", type=int, default=None, help="only the first N shots per basis")
parser.add_argument("--fresh", type=int, default=None, help="use N freshly generated shots per basis instead of data/train")
parser.add_argument("--seed", type=int, default=12345)
args = parser.parse_args()

# Import under the file's own module name (e.g. "decoder"), as run_local.py does: numba's on-disk cache records the
# module name, and a different name here would leave cache entries that break the next run_local.py import.
name = os.path.splitext(os.path.basename(args.decoder))[0]
sys.path.insert(0, os.path.dirname(os.path.abspath(args.decoder)))
spec = importlib.util.spec_from_file_location(name, args.decoder)
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)

total_time, preds, trues = 0.0, [], []
for k, b in enumerate("XZ"):
    if args.fresh:
        d, o = generate(b, args.fresh, seed=args.seed + k)
    else:
        d, o = labelled_set("train")[b]
    d, o = d[: args.shots], o[: args.shots]
    start = time.perf_counter()
    pred = np.asarray(module.decode(circuit(b), d))
    dt = time.perf_counter() - start
    total_time += dt
    preds.append(pred), trues.append(o)
    e = ler(pred, o)
    print(f"{b}: LER {e:.5f} ± {np.sqrt(e * (1 - e) / o.size):.5f}   {dt:.2f} s   {1e6 * dt / len(d):.1f} µs/shot   ({len(d)} shots)")
e = ler(np.concatenate(preds), np.concatenate(trues))
print(f"both: LER {e:.5f}   total {total_time:.2f} s (t_cap ≈ 3 s for 2×100k shots on the timing machine)")
