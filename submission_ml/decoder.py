"""Hand-in decoder: maximum-likelihood decoding by a transfer matrix over time layers (stim + numpy + numba + torch).

Every mechanism of the detector error model touches at most two adjacent time layers of detectors, so the exact
probability of each logical class given the syndrome is a chain of five tables on 14-bit indices (derivation in
mldfast.py). No BP, no OSD, no search, no training data, no randomness.

Default with a GPU (MODE = "exact"): every shot is decoded with the full, untruncated tables on the GPU
(mlddense.run_all): no beam, no row truncation, no thresholds, so the result is the exact maximum-likelihood
decision for every observable of every shot. Work shared between shots (identical syndromes, identical first two
or last two layers) is done once, which changes the cost but not the result. Bits whose float32 probability lands
within TIE of 0.5 are recomputed in float64, so the output does not depend on the GPU model.

MODE = "hybrid" (MLD_MODE=hybrid) is the earlier, cheaper scheme. Two passes, running at the same time:
1. CPU (numba, all threads): a pruned left-to-right pass on every shot (mldfast.run). It gives each observable's
   probability P(L_i = 1 | D).
2. GPU (torch): shots with any observable within TAU of 0.5 are re-decoded with the full, untruncated tables
   (mlddense.run): three dense 16384 x 16384 products per shot, so no path can be missed. The GPU works on one chunk
   of shots while the CPU decodes the next.
A GPU answer replaces the CPU answer unless it is a numerical tie (|P - 0.5| < TIE), so the output does not depend
on the GPU model.

Without a GPU (or with MLD_GPU=0) the CPU pass alone is used, with the "cap" limits: light pruning sized to stay
under t_cap on the timing machine.

Import time is not timed (README, Final ranking): the tables for the two competition circuits are built or loaded
from mld_cache/ here, moved to the GPU, and the kernels are compiled.
"""
import os
import queue
import sys
import threading
import time

os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")  # numba's OpenMP layer can clash with torch's
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import stim  # noqa: E402

import mldfast  # noqa: E402

try:
    import mlddense
    DEVICE = mlddense.pick_device()
except Exception:  # torch missing or broken: CPU only
    mlddense, DEVICE = None, None
USE_GPU = os.environ.get("MLD_GPU", "1") != "0" and DEVICE is not None and DEVICE.type != "cpu"

CACHE = os.path.join(HERE, "mld_cache")
M = 4096        # entries kept per row of the middle table (CPU pass)
# CPU pass (delta, kmax, eta): expand running-vector entries within a factor delta of the best one, at most kmax per
# layer; skip row entries contributing less than eta x the layer's largest contribution.
# "cap" is the CPU-only default, sized to finish 2 x 100k shots under t_cap = 15.4 s on the 48-thread timing machine:
# it takes 35.6 s on a 10-core Apple M4, which scales like 3 of its fast cores on this kernel (measured), and 48 EPYC
# 9655 threads are estimated at 12-29 such cores, i.e. 4-10x faster, giving about 4-9 s (see FINDINGS.md).
PROFILES = {"max": (1e-6, 512, 1e-9), "cap": (1e-5, 256, 1e-8), "fast": (1e-4, 128, 1e-6)}
CPU_PROFILE = os.environ.get("MLD_PROFILE", "fast" if USE_GPU else "cap")
MODE = os.environ.get("MLD_MODE", "exact")   # "exact": dense GPU pass on every shot; "hybrid": CPU pass + GPU on uncertain shots
TAU = 0.3       # hybrid: GPU re-decodes shots with an observable at |P - 0.5| < TAU
TIE = 5e-4      # hybrid: ... and its answer is used unless |P - 0.5| < TIE; exact: such bits are redone in float64
CHUNK = 25_000  # shots per CPU chunk; the GPU works on chunk i while the CPU decodes chunk i + 1
GPU_BATCH = 1024
_TABLES, _DENSE = {}, {}
STATS = []      # one dict per decode() call: shot counts and seconds per stage (for timing estimates)


def _tables(circuit):
    key = str(circuit)
    if key not in _TABLES:
        _TABLES[key] = mldfast.build(circuit, M=M, cache_dir=CACHE)
    return key, _TABLES[key]


def _dense(circuit, key, tb):
    if key not in _DENSE:
        big = DEVICE.type == "cuda" and mlddense.torch.cuda.get_device_properties(0).total_memory > 40 * 2 ** 30
        if not big:  # GPUs under 40 GB (T4, Apple MPS): keep one basis at a time
            _DENSE.clear()
            if DEVICE.type == "cuda":
                mlddense.torch.cuda.empty_cache()
        N = mldfast.build_dense(circuit, CACHE)
        _DENSE[key] = mlddense.load(tb, N, DEVICE)
        _DENSE[key]["Nnp"] = N
    return _DENSE[key]


def _cpu(tb, dets):
    """Predictions (shots, 4) and P(L_i = 1 | D) from the pruned CPU pass."""
    pchi = mldfast.run(tb, dets, *PROFILES[CPU_PROFILE])
    lost = np.flatnonzero(~(pchi[:, 0] > 0))
    if len(lost):  # pruning removed every path (not seen in 600k shots): redo those shots without beam limits
        pchi[lost] = mldfast.run(tb, dets[lost], 0.0, mldfast.SZ, 0.0)
    p0 = np.where(pchi[:, :1] > 0, pchi[:, :1], 1.0)
    return (pchi[:, 1:] < 0).astype(np.uint8), 0.5 * (1.0 - pchi[:, 1:] / p0)


def decode(circuit: stim.Circuit, detector_events: np.ndarray) -> np.ndarray:
    """
    circuit: the X or Z circuit, loaded with stim.Circuit.from_file.
    detector_events: bool array of shape (shots, 70), columns D0 to D69.
    Returns a 0/1 array of shape (shots, 4), columns L0 to L3.
    """
    t_start = time.perf_counter()
    dets = np.asarray(detector_events)
    key, tb = _tables(circuit)
    pred = np.zeros((len(dets), 4), np.uint8)
    st = dict(shots=len(dets), gpu=USE_GPU, cpu_profile=CPU_PROFILE, cpu_seconds=0.0, gpu_seconds=0.0, gpu_shots=0,
              gpu_ties=0, gpu_changed_bits=0, gpu_errors=0, load_seconds=0.0)
    STATS.append(st)
    if not USE_GPU:
        pred[:] = _cpu(tb, dets)[0]
        st["cpu_seconds"] = st["total_seconds"] = time.perf_counter() - t_start
        return pred
    dt = _dense(circuit, key, tb)
    st["load_seconds"] = time.perf_counter() - t_start
    if MODE == "exact":
        st["mode"] = "exact"
        t0 = time.perf_counter()
        pchi, shared = mlddense.run_all(dt, dets, GPU_BATCH)
        st.update(shared, gpu_seconds=time.perf_counter() - t0, gpu_shots=len(dets))
        with np.errstate(divide="ignore", invalid="ignore"):
            p1 = 0.5 * (1.0 - pchi[:, 1:] / pchi[:, :1])
        ties = np.flatnonzero(~(np.abs(p1 - 0.5).min(1) >= TIE))  # also catches non-finite values
        if len(ties):
            t0 = time.perf_counter()
            pchi[ties] = mlddense.run64(dt, dt["Nnp"], dets[ties])
            st["tie_seconds"] = time.perf_counter() - t0
        st["gpu_ties"] = len(ties)
        pred[:] = pchi[:, 1:] < 0
        st["total_seconds"] = time.perf_counter() - t_start
        return pred
    jobs = queue.Queue()
    errors = []

    def gpu_worker():
        while True:
            rows = jobs.get()
            if rows is None:
                return
            try:
                t0 = time.perf_counter()
                pchi = mlddense.run(dt, dets[rows], GPU_BATCH)
                st["gpu_seconds"] += time.perf_counter() - t0
                st["gpu_shots"] += len(rows)
                ok = pchi[:, :1] > 0
                p1 = 0.5 * (1.0 - pchi[:, 1:] / np.where(ok, pchi[:, :1], 1.0))
                use = ok & np.isfinite(p1) & (np.abs(p1 - 0.5) >= TIE)
                sub = pred[rows]
                st["gpu_ties"] += int((~use).sum())
                st["gpu_changed_bits"] += int((sub[use] != (p1 > 0.5)[use]).sum())
                sub[use] = (p1 > 0.5)[use]
                pred[rows] = sub
            except Exception as exc:  # keep the CPU answers for this chunk
                errors.append(exc)
                st["gpu_errors"] += 1

    worker = threading.Thread(target=gpu_worker)
    worker.start()
    for s in range(0, len(dets), CHUNK):
        sl = slice(s, min(s + CHUNK, len(dets)))
        t0 = time.perf_counter()
        p, p1 = _cpu(tb, dets[sl])
        st["cpu_seconds"] += time.perf_counter() - t0
        pred[sl] = p
        rows = s + np.flatnonzero(np.abs(p1 - 0.5).min(1) < TAU)
        if len(rows):
            jobs.put(rows)
    jobs.put(None)
    worker.join()
    st["total_seconds"] = time.perf_counter() - t_start
    return pred


def _warm_up():
    """Build or load the tables for the bundled competition circuits and compile the kernels (untimed)."""
    for b in "XZ":
        path = os.path.join(HERE, f"{b}_circuit.stim")
        if os.path.exists(path):
            c = stim.Circuit.from_file(path)
            sample, _ = c.compile_detector_sampler(seed=0).sample(256, separate_observables=True)
            decode(c, sample)
    STATS.clear()


_warm_up()
