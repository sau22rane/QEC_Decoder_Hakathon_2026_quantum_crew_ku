"""Dense, unpruned evaluation of the transfer-matrix chain on a GPU (PyTorch). See mldfast.py for the method.

    P_chi(D) = sum_{b0..b3} T0[D0, b0] N[b0^D1, b1] N[b1^D2, b2] N[b2^D3, b3] T4[b3^D4, D5]

Per batch of shots and per chi this is three products of a (batch, 16384) matrix with the full 16384 x 16384 table:
nothing is truncated, so no path can be missed. Runs on CUDA, Apple MPS or (slowly) the CPU.
"""
import numpy as np
import torch

SZ = 1 << 14


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load(tb, N, device):
    """tb: tables from mldfast.build (for T0i, T4i); N: (5, SZ, SZ) array from mldfast.build_dense."""
    torch.backends.cuda.matmul.allow_tf32 = False  # full float32 products: the chi != 0 sums cancel
    return dict(device=device,
                N=[torch.from_numpy(np.ascontiguousarray(N[k])).to(device) for k in range(5)],
                T0=torch.from_numpy(np.ascontiguousarray(tb["T0i"].transpose(2, 0, 1))).to(device),
                T4=torch.from_numpy(np.ascontiguousarray(tb["T4i"].transpose(2, 0, 1))).to(device),
                ar=torch.arange(SZ, device=device))


def indices(detector_events):
    d = np.asarray(detector_events).astype(np.int64)
    w7, w14 = 1 << np.arange(7), 1 << np.arange(14)
    return [d[:, :7] @ w7] + [d[:, 7 + 14 * t: 21 + 14 * t] @ w14 for t in range(4)] + [d[:, 63:] @ w7]


@torch.no_grad()
def run(dt, detector_events, batch=1024):
    """(shots, 5) float64 array of P_chi(D) for chi = 0, e0..e3, up to a common positive factor per shot."""
    dev = dt["device"]
    idx = [torch.from_numpy(i).to(dev) for i in indices(detector_events)]
    n = len(idx[0])
    out = torch.empty((n, 5), dtype=torch.float32, device=dev)
    for s in range(0, n, batch):
        sl = slice(s, min(s + batch, n))
        scales = []
        for k in range(5):
            v = dt["T0"][k][idx[0][sl]]
            for t in (1, 2, 3):
                v = torch.gather(v, 1, dt["ar"][None, :] ^ idx[t][sl][:, None]) @ dt["N"][k]
                if k == 0:
                    scales.append(v.amax(1, keepdim=True).clamp_min(1e-30))
                v = v / scales[t - 1]
            v = torch.gather(v, 1, dt["ar"][None, :] ^ idx[4][sl][:, None])
            out[sl, k] = (v * dt["T4"][k][idx[5][sl]]).sum(1)
    return out.cpu().double().numpy()


@torch.no_grad()
def run_all(dt, detector_events, batch=1024):
    """Exact P_chi(D) for every shot, with no pruning, as (shots, 5) float64 (common positive factor per shot).

    Same sums as run(), reorganised so that work shared between shots is done once (the result is unchanged):
    - identical syndromes are decoded once;
    - the left part  V[b1] = sum_b0 T0[D0, b0] N[b0^D1, b1]   depends only on (D0, D1): once per distinct pair;
    - the right part U[a]  = sum_b3 N[a, b3] T4[b3^D4, D5]    depends only on (D4, D5): once per distinct pair;
    - per shot one product remains:  P = sum_b2 (V[. ^ D2] N)[b2] U[b2 ^ D3].
    """
    dev, ar = dt["device"], dt["ar"]
    full = np.stack(indices(detector_events), 1)
    uniq, inv = np.unique(full, axis=0, return_inverse=True)
    pre, pinv = np.unique(uniq[:, 0] * SZ + uniq[:, 1], return_inverse=True)
    suf, sinv = np.unique(uniq[:, 5] * SZ + uniq[:, 4], return_inverse=True)
    T = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(dev)  # noqa: E731
    p0, p1, s5, s4 = T(pre // SZ), T(pre % SZ), T(suf // SZ), T(suf % SZ)
    pinv_t, sinv_t, i2, i3 = T(pinv.reshape(-1)), T(sinv.reshape(-1)), T(uniq[:, 2]), T(uniq[:, 3])
    out = torch.empty((len(uniq), 5), dtype=torch.float32, device=dev)
    V = torch.empty((len(pre), SZ), dtype=torch.float32, device=dev)
    U = torch.empty((len(suf), SZ), dtype=torch.float32, device=dev)
    sv = torch.empty((len(pre), 1), dtype=torch.float32, device=dev)
    su = torch.empty((len(suf), 1), dtype=torch.float32, device=dev)
    for k in range(5):
        Nk = dt["N"][k]
        for s in range(0, len(pre), batch):
            sl = slice(s, s + batch)
            v = torch.gather(dt["T0"][k][p0[sl]], 1, ar[None, :] ^ p1[sl][:, None]) @ Nk
            if k == 0:
                sv[sl] = v.amax(1, keepdim=True).clamp_min(1e-30)
            V[sl] = v / sv[sl]
        for s in range(0, len(suf), batch):
            sl = slice(s, s + batch)
            u = torch.gather(dt["T4"][k][s5[sl]], 1, ar[None, :] ^ s4[sl][:, None]) @ Nk.T
            if k == 0:
                su[sl] = u.amax(1, keepdim=True).clamp_min(1e-30)
            U[sl] = u / su[sl]
        for s in range(0, len(uniq), batch):
            sl = slice(s, s + batch)
            v = torch.gather(V[pinv_t[sl]], 1, ar[None, :] ^ i2[sl][:, None]) @ Nk
            out[sl, k] = (v * torch.gather(U[sinv_t[sl]], 1, ar[None, :] ^ i3[sl][:, None])).sum(1)
    stats = dict(shots=len(full), distinct=len(uniq), prefixes=len(pre), suffixes=len(suf))
    return out.cpu().double().numpy()[inv.reshape(-1)], stats


@torch.no_grad()
def run64(dt, N, detector_events, block=1024):
    """P_chi(D) in float64 for a few shots (numerical ties of the float32 pass). N: the (5, SZ, SZ) float32 array."""
    idx = indices(detector_events)
    n = len(idx[0])
    out = np.empty((n, 5))
    if n == 0:
        return out
    cuda = dt["device"].type == "cuda"
    dev = dt["device"] if cuda else torch.device("cpu")
    ar = torch.arange(SZ, device=dev)
    ix = [torch.from_numpy(i).to(dev) for i in idx]
    for k in range(5):
        v = dt["T0"][k].to(dev).double()[ix[0]]
        for t in (1, 2, 3):
            v = torch.gather(v, 1, ar[None, :] ^ ix[t][:, None])
            acc = torch.zeros((n, SZ), dtype=torch.float64, device=dev)
            for a in range(0, SZ, block):
                blk = dt["N"][k][a:a + block].double() if cuda else torch.from_numpy(np.ascontiguousarray(N[k, a:a + block])).double()
                acc += v[:, a:a + block] @ blk
            v = acc / acc.abs().amax(1, keepdim=True).clamp_min(1e-300) if k == 0 else acc
            if k == 0:
                dt.setdefault("_s64", {})[t] = acc.abs().amax(1, keepdim=True).clamp_min(1e-300)
            else:
                v = acc / dt["_s64"][t]
        v = torch.gather(v, 1, ar[None, :] ^ ix[4][:, None])
        out[:, k] = (v * dt["T4"][k].to(dev).double()[ix[5]]).sum(1).cpu().numpy()
    dt.pop("_s64", None)
    return out
