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
