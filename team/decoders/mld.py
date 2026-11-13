"""Exact maximum-likelihood decoder by a transfer matrix over time layers (numpy + numba only).

Every DEM mechanism touches detectors in at most two adjacent time layers (7, 14, 14, 14, 14, 7 detectors), and the
three middle layer pairs are identical. So the exact probability of a syndrome together with a logical class is a
chain of five tables joined on 14-bit indices:

    P_chi(D) = sum_{b0..b3} T0[D0, b0] N[b0^D1, b1] N[b1^D2, b2] N[b2^D3, b3] T4[b3^D4, D5]

where chi is a character of the 4 observable bits: P_chi(D) = sum_c (-1)^(chi.c) P(D, c). chi = 0 gives P(D), and
chi = e_i gives P(D, L_i = 0) - P(D, L_i = 1), so observable i is predicted 1 exactly when P_{e_i}(D) < 0. That is
the per-bit optimal decision, with no search and no approximation.

Tables are built with Walsh-Hadamard transforms: the XOR-convolution of independent mechanisms is a product in the
transformed domain. Mechanisms inside a middle layer are split into two independent halves (p = 2q(1-q)), one for the
table on each side, which makes all three middle tables equal (N); the two end tables absorb the difference.

    build(basis)                 -> tables (cached in data/generated/mld_<basis>.npz + .npy)
    decode_exact(basis, dets)    -> (shots, 5) array of P_chi for chi = 0, e0, e1, e2, e3 (dense, slow, exact)
    decode_beam(basis, dets, K)  -> same, keeping only the K largest entries of the running vector at each layer
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numba as nb  # noqa: E402
import numpy as np  # noqa: E402

from common import GEN_DIR, matrices  # noqa: E402

NA = 14
SZ = 1 << NA
CHIS = (0, 1, 2, 4, 8)  # chi as a bit mask over the 4 observables


@nb.njit(parallel=True, cache=True)
def fwht(x):
    """In-place unnormalised Walsh-Hadamard transform of a float64 array whose length is a power of two."""
    n = x.size
    blk = min(n, 1 << 15)
    for c in nb.prange(n // blk):
        base = c * blk
        h = 1
        while h < blk:
            for i in range(base, base + blk, 2 * h):
                for j in range(i, i + h):
                    a = x[j]
                    b = x[j + h]
                    x[j] = a + b
                    x[j + h] = a - b
            h *= 2
    h = blk
    while h < n:
        nchunk = h // blk
        for i in range(0, n, 2 * h):
            for c in nb.prange(nchunk):
                for j in range(i + c * blk, i + (c + 1) * blk):
                    a = x[j]
                    b = x[j + h]
                    x[j] = a + b
                    x[j + h] = a - b
        h *= 2


@nb.njit(parallel=True, cache=True)
def _exp_half(x, total, scale):
    """x <- exp((total - x) / 2) * scale, in place."""
    for i in nb.prange(x.size):
        x[i] = np.exp(0.5 * (total - x[i])) * scale


def _table(nbits, terms, chi):
    """Signed measure of the XOR of independent mechanisms. terms: list of (index, log(1-2p) weight, obs mask)."""
    g = np.zeros(1 << nbits)
    total = 0.0
    for x, l, c in terms:
        s = -1.0 if bin(c & chi).count("1") & 1 else 1.0
        g[x] += s * l
        total += l
    fwht(g)
    _exp_half(g, total, 1.0 / (1 << nbits))
    fwht(g)
    return g


def _layer(d):
    return 0 if d < 7 else (5 if d >= 63 else 1 + (d - 7) // 14)


def _bit(d):
    return d if d < 7 else (d - 63 if d >= 63 else (d - 7) % 14)


def _groups(basis):
    """{(lo, hi): [(bits in layer lo, bits in layer hi, log(1-2p), obs mask)]} over all mechanisms."""
    H, L, p = matrices(basis)
    H = np.asarray(H.todense() if hasattr(H, "todense") else H).astype(np.uint8)
    L = np.asarray(L.todense() if hasattr(L, "todense") else L).astype(np.uint8)
    out = {}
    for j in range(H.shape[1]):
        dets = np.flatnonzero(H[:, j])
        lay = [_layer(d) for d in dets]
        lo, hi = min(lay), max(lay)
        assert hi - lo <= 1, "mechanism spans more than two layers"
        a = sum(1 << _bit(d) for d, t in zip(dets, lay) if t == lo)
        b = sum(1 << _bit(d) for d, t in zip(dets, lay) if t == hi) if hi > lo else 0
        c = int(sum(int(L[i, j]) << i for i in range(4)))
        out.setdefault((lo, hi), []).append((a, b, float(np.log1p(-2.0 * p[j])), c))
    return out


def build(basis, verbose=True):
    """Returns dict(T0 (5, 128, 16384), N (5, 16384, 16384), T4t (5, 128, 16384)) as float32; T4t[chi, D5, a]."""
    small = os.path.join(GEN_DIR, f"mld_{basis}_ends.npz")
    big = os.path.join(GEN_DIR, f"mld_{basis}_N.npy")
    if os.path.exists(small) and os.path.exists(big):
        f = np.load(small)
        return {"T0": f["T0"], "T4t": f["T4t"], "N": np.load(big, mmap_mode="r")}
    g = _groups(basis)
    key = lambda v: sorted(v)  # noqa: E731
    assert key(g[(1, 2)]) == key(g[(2, 3)]) == key(g[(3, 4)]) and key(g[(2, 2)]) == key(g[(3, 3)])
    half = g[(2, 2)]
    t0 = [(a | (b << 7), l, c) for a, b, l, c in g[(0, 1)]]
    t0 += [(a, l, c) for a, _, l, c in g[(0, 0)]]
    t0 += [(a << 7, l, c) for a, _, l, c in g[(1, 1)]]
    t0 += [(a << 7, -0.5 * l, c) for a, _, l, c in half]
    nn = [(a | (b << NA), l, c) for a, b, l, c in g[(2, 3)]]
    nn += [(a, 0.5 * l, c) for a, _, l, c in half] + [(a << NA, 0.5 * l, c) for a, _, l, c in half]
    t4 = [(a | (b << NA), l, c) for a, b, l, c in g[(4, 5)]]
    t4 += [(a, l, c) for a, _, l, c in g[(4, 4)]]
    t4 += [(a << NA, l, c) for a, _, l, c in g[(5, 5)]]
    t4 += [(a, -0.5 * l, c) for a, _, l, c in half]
    T0 = np.empty((5, 128, SZ), np.float32)
    T4t = np.empty((5, 128, SZ), np.float32)
    os.makedirs(GEN_DIR, exist_ok=True)
    N = np.lib.format.open_memmap(big + ".tmp.npy", mode="w+", dtype=np.float32, shape=(5, SZ, SZ))
    for k, chi in enumerate(CHIS):
        T0[k] = _table(21, t0, chi).reshape(SZ, 128).T          # index = a | b << 7  ->  [a, b]
        T4t[k] = _table(21, t4, chi).reshape(128, SZ)            # index = a | b << 14 ->  [b, a]
        x = _table(28, nn, chi)
        if verbose:
            print(f"[{basis}] chi {chi}: N sum {x.sum():+.6f} min {x.min():.3e} max {x.max():.4f}", flush=True)
        N[k] = x.reshape(SZ, SZ).T                               # [a, b]
        del x
    N.flush()
    del N
    os.replace(big + ".tmp.npy", big)
    np.savez(small, T0=T0, T4t=T4t)
    return build(basis)


def _indices(dets):
    d = np.asarray(dets).astype(np.int64)
    w7, w14 = 1 << np.arange(7), 1 << np.arange(14)
    return [d[:, :7] @ w7] + [d[:, 7 + 14 * t: 21 + 14 * t] @ w14 for t in range(4)] + [d[:, 63:] @ w7]


def decode_exact(basis, dets, batch=512, tables=None):
    """Exact P_chi(D), shape (shots, 5). Dense: three 16384 x 16384 matrix products per chi and batch."""
    tb = tables or build(basis)
    idx = _indices(dets)
    ar = np.arange(SZ)
    out = np.empty((len(dets), 5))
    for s in range(0, len(dets), batch):
        sl = slice(s, min(s + batch, len(dets)))
        scale = np.zeros(sl.stop - sl.start)
        vs = []
        for k in range(5):
            v = tb["T0"][k][idx[0][sl]]
            for t in (1, 2, 3):
                v = np.take_along_axis(v, ar[None, :] ^ idx[t][sl][:, None], axis=1)
                v = v @ np.asarray(tb["N"][k])
                if k == 0:
                    m = v.max(1)
                    scale += np.log(m)
                    vs.append(m)
                v /= vs[t - 1][:, None]
            v = np.take_along_axis(v, ar[None, :] ^ idx[4][sl][:, None], axis=1)
            out[sl, k] = (v * tb["T4t"][k][idx[5][sl]]).sum(1, dtype=np.float64)
        out[sl] *= 1.0  # relative values only: the common scale exp(scale) is dropped
    return out


@nb.njit(parallel=True, cache=True)
def _beam(i0, i1, i2, i3, i4, i5, T0, N, T4t, r, K, out):
    n = len(i0)
    for s in nb.prange(n):
        v = np.empty((5, SZ), np.float32)
        w = np.empty((5, SZ), np.float32)
        mag = np.empty(SZ, np.float32)
        for k in range(5):
            v[k, :] = T0[k, i0[s], :]
        for t in range(3):
            sh = i1[s] if t == 0 else (i2[s] if t == 1 else i3[s])
            for a in range(SZ):  # one-step lookahead: weight so far x chance the next table absorbs the mismatch
                mag[a] = abs(v[0, a]) * r[a ^ sh]
            if K < SZ:
                thr = np.partition(mag, SZ - K)[SZ - K]
            else:
                thr = np.float32(-1.0)
            w[:, :] = 0.0
            for b in range(SZ):
                if mag[b] >= thr and mag[b] > 0.0:
                    a = b ^ sh
                    for k in range(5):
                        c = v[k, b]
                        row = N[k, a]
                        wk = w[k]
                        for j in range(SZ):
                            wk[j] += c * row[j]
            m = np.float32(0.0)
            for j in range(SZ):
                if w[0, j] > m:
                    m = w[0, j]
            if m <= 0.0:
                m = np.float32(1.0)
            for k in range(5):
                for j in range(SZ):
                    v[k, j] = w[k, j] / m
        for k in range(5):
            acc = 0.0
            row = T4t[k, i5[s]]
            for b in range(SZ):
                acc += v[k, b] * row[b ^ i4[s]]
            out[s, k] = acc


_R = {}


def rowsum(basis, tb):
    """r[a] = sum_b N[a, b] for chi = 0: the probability that a middle table shows pattern a on its left side."""
    if basis not in _R:
        path = os.path.join(GEN_DIR, f"mld_{basis}_r.npy")
        if not os.path.exists(path):
            np.save(path, np.asarray(tb["N"][0]).sum(1, dtype=np.float64).astype(np.float32))
        _R[basis] = np.load(path)
    return _R[basis]


def decode_beam(basis, dets, K=256, tables=None):
    """P_chi(D) up to a common factor, keeping the K largest entries of the running vector before each big table."""
    tb = tables or build(basis)
    i = _indices(dets)
    out = np.empty((len(dets), 5))
    _beam(i[0], i[1], i[2], i[3], i[4], i[5], tb["T0"], np.asarray(tb["N"]), tb["T4t"], rowsum(basis, tb), K, out)
    return out


def predict(pchi):
    """(shots, 4) 0/1 predictions and per-bit P(L_i = 1 | D) from P_chi."""
    p1 = 0.5 * (1.0 - pchi[:, 1:] / pchi[:, :1])
    return (pchi[:, 1:] < 0).astype(np.uint8), p1


# ---------------------------------------------------------------------------------------------------------------------
# Fast version: sparse rows of N (largest entries only) and an adaptive beam (entries within a factor delta of the best)
# ---------------------------------------------------------------------------------------------------------------------

@nb.njit(parallel=True, cache=True)
def _row_select(N0, M, eps, counts, idx):
    """Per row of N (chi = 0): the indices of the up-to-M largest entries that are >= eps * row maximum."""
    for a in nb.prange(SZ):
        row = N0[a]
        order = np.argsort(-row)[:M]
        lim = eps * row[order[0]]
        c = 0
        for j in order:
            if row[j] >= lim and row[j] > 0.0:
                idx[a, c] = j
                c += 1
        counts[a] = c


@nb.njit(parallel=True, cache=True)
def _row_gather(Nk, k, ptr, idx, counts, vals):
    for a in nb.prange(SZ):
        for c in range(counts[a]):
            vals[ptr[a] + c, k] = Nk[a, idx[a, c]]


_S = {}


def sparse(basis, M=256, eps=0.0, tables=None):
    """(ptr, cols, vals (nnz, 5), T0i (128, SZ, 5), T4i (128, SZ, 5), r) for the fast kernel."""
    key = (basis, M, eps)
    if key not in _S:
        tb = tables or build(basis)
        counts = np.zeros(SZ, np.int64)
        idx = np.zeros((SZ, M), np.int32)
        _row_select(np.asarray(tb["N"][0]), M, eps, counts, idx)
        ptr = np.zeros(SZ + 1, np.int64)
        ptr[1:] = np.cumsum(counts)
        vals = np.empty((ptr[-1], 5), np.float32)
        for k in range(5):
            _row_gather(np.asarray(tb["N"][k]), k, ptr, idx, counts, vals)
        cols = np.concatenate([idx[a, :counts[a]] for a in range(SZ)]).astype(np.int32)
        _S[key] = (ptr, cols, vals, np.ascontiguousarray(tb["T0"].transpose(1, 2, 0)),
                   np.ascontiguousarray(tb["T4t"].transpose(1, 2, 0)), rowsum(basis, tb))
    return _S[key]


@nb.njit(parallel=True, cache=True)
def _fast(i0, i1, i2, i3, i4, i5, T0i, T4i, ptr, cols, vals, r, delta, kmax, out, used):
    n = len(i0)
    for s in nb.prange(n):
        bufa = np.empty((SZ, 5), np.float32)
        bufb = np.empty((SZ, 5), np.float32)
        score = np.empty(SZ, np.float32)
        sel = np.empty(kmax, np.int64)
        cur = T0i[i0[s]]
        nused = 0
        for t in range(3):
            sh = i1[s] if t == 0 else (i2[s] if t == 1 else i3[s])
            mx = np.float32(0.0)
            mv = np.float32(0.0)
            for b in range(SZ):
                c = cur[b, 0]
                sc = c * r[b ^ sh]
                score[b] = sc
                if sc > mx:
                    mx = sc
                if c > mv:
                    mv = c
            thr = mx * delta
            cnt = 0
            for b in range(SZ):
                if score[b] >= thr and score[b] > 0.0:
                    cnt += 1
            if cnt > kmax:
                thr = np.partition(score, SZ - kmax)[SZ - kmax]
            cnt = 0
            for b in range(SZ):
                if score[b] >= thr and score[b] > 0.0 and cnt < kmax:
                    sel[cnt] = b
                    cnt += 1
            nused += cnt
            nxt = bufa if t % 2 == 0 else bufb
            nxt[:, :] = 0.0
            inv = np.float32(1.0) / mv if mv > 0.0 else np.float32(1.0)
            for q in range(cnt):
                b = sel[q]
                a = b ^ sh
                c0 = cur[b, 0] * inv
                c1 = cur[b, 1] * inv
                c2 = cur[b, 2] * inv
                c3 = cur[b, 3] * inv
                c4 = cur[b, 4] * inv
                for e in range(ptr[a], ptr[a + 1]):
                    j = cols[e]
                    nxt[j, 0] += c0 * vals[e, 0]
                    nxt[j, 1] += c1 * vals[e, 1]
                    nxt[j, 2] += c2 * vals[e, 2]
                    nxt[j, 3] += c3 * vals[e, 3]
                    nxt[j, 4] += c4 * vals[e, 4]
            cur = nxt
        mv = np.float32(0.0)
        for b in range(SZ):
            if cur[b, 0] > mv:
                mv = cur[b, 0]
        inv = 1.0 / mv if mv > 0.0 else 1.0
        row = T4i[i5[s]]
        sh = i4[s]
        a0 = a1 = a2 = a3 = a4 = 0.0
        for b in range(SZ):
            c = cur[b, 0]
            if c != 0.0:
                a = b ^ sh
                a0 += c * inv * row[a, 0]
                a1 += cur[b, 1] * inv * row[a, 1]
                a2 += cur[b, 2] * inv * row[a, 2]
                a3 += cur[b, 3] * inv * row[a, 3]
                a4 += cur[b, 4] * inv * row[a, 4]
        out[s, 0] = a0
        out[s, 1] = a1
        out[s, 2] = a2
        out[s, 3] = a3
        out[s, 4] = a4
        used[s] = nused


def decode_fast(basis, dets, delta=1e-4, kmax=128, M=256, eps=0.0, tables=None, return_used=False):
    """P_chi(D) up to a common factor from the sparse-row adaptive-beam kernel."""
    ptr, cols, vals, T0i, T4i, r = sparse(basis, M, eps, tables)
    i = _indices(dets)
    out = np.empty((len(dets), 5))
    used = np.zeros(len(dets), np.int64)
    _fast(i[0], i[1], i[2], i[3], i[4], i[5], T0i, T4i, ptr, cols, vals, r, np.float32(delta), kmax, out, used)
    return (out, used) if return_used else out
