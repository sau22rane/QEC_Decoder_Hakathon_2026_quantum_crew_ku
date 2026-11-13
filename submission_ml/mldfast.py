"""Maximum-likelihood decoder by a transfer matrix over time layers. Self-contained: stim + numpy + numba only.

Idea. Every error mechanism of the detector error model touches detectors in at most two adjacent time layers
(7, 14, 14, 14, 14, 7 detectors), and the three middle layer pairs are identical. The probability of a syndrome D
together with the parity chi.c of the logical class c is therefore an exact chain of five tables on 14-bit indices:

    P_chi(D) = sum_{b0..b3} T0[D0, b0] N[b0^D1, b1] N[b1^D2, b2] N[b2^D3, b3] T4[b3^D4, D5]

chi = 0 gives P(D); chi = e_i gives P(D, L_i = 0) - P(D, L_i = 1), so observable i is predicted 1 exactly when
P_{e_i}(D) < 0. This is the per-bit optimal (maximum-likelihood) decision: no search, no candidate list.

Tables. The XOR of independent mechanisms is a product after a Walsh-Hadamard transform, so each table is two
transforms: log-factors -> transform -> exp -> transform. Mechanisms inside a middle layer are split into two
independent halves (p = 2q(1 - q)), one per neighbouring table, which makes the three middle tables equal (N); the
end tables T0 and T4 absorb the difference. N has 2^28 entries per chi; we keep the M largest entries of every row.

Decoding. One left-to-right pass per shot. Before each N, only entries of the running vector within a factor delta
of the best are expanded (at most kmax), ranked by weight x r[b ^ D_t], where r[a] = sum_b N[a, b] is the chance that
the next table absorbs the mismatch (one-step lookahead). Row entries whose contribution is below eta x the largest
contribution of the layer are skipped. With M, kmax large and delta, eta -> 0 the result is exact.

    tables = build(circuit, M=1024)          # minutes; cached on disk if cache_dir is given
    pchi = run(tables, detector_events)      # (shots, 5): chi = 0, e0, e1, e2, e3, up to a common positive factor
"""
import hashlib
import os

import numba as nb
import numpy as np

NA = 14
SZ = 1 << NA
CHIS = (0, 1, 2, 4, 8)
LAYERS = (7, 14, 14, 14, 14, 7)


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
    for i in nb.prange(x.size):
        x[i] = np.exp(0.5 * (total - x[i])) * scale


def _table(nbits, terms, chi):
    """Signed measure of the XOR of independent mechanisms; terms = [(index, log(1 - 2p), observable mask)]."""
    g = np.zeros(1 << nbits)
    total = 0.0
    for x, l, c in terms:
        g[x] += -l if bin(c & chi).count("1") & 1 else l
        total += l
    fwht(g)
    _exp_half(g, total, 1.0 / (1 << nbits))
    fwht(g)
    return g


def _layer(d):
    return 0 if d < 7 else (5 if d >= 63 else 1 + (d - 7) // 14)


def _bit(d):
    return d if d < 7 else (d - 63 if d >= 63 else (d - 7) % 14)


def _groups(circuit):
    """{(lo, hi): [(bits in layer lo, bits in layer hi, log(1 - 2p), observable mask)]} over all DEM mechanisms."""
    dem = circuit.detector_error_model()
    if dem.num_detectors != 70 or dem.num_observables != 4:
        raise ValueError("expected 70 detectors and 4 observables")
    coords = circuit.get_detector_coordinates()
    if any(int(coords[d][-1]) != _layer(d) for d in range(70)):
        raise ValueError("unexpected detector time layers")
    out = {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        dets, obs = set(), 0
        for tg in inst.targets_copy():
            if tg.is_relative_detector_id():
                dets ^= {tg.val}
            elif tg.is_logical_observable_id():
                obs ^= 1 << tg.val
        if not dets:
            raise ValueError("mechanism without detectors")
        lay = {d: _layer(d) for d in dets}
        lo, hi = min(lay.values()), max(lay.values())
        if hi - lo > 1:
            raise ValueError("mechanism spans more than two layers")
        a = sum(1 << _bit(d) for d in dets if lay[d] == lo)
        b = sum(1 << _bit(d) for d in dets if lay[d] == hi) if hi > lo else 0
        out.setdefault((lo, hi), []).append((a, b, float(np.log1p(-2.0 * inst.args_copy()[0])), obs))
    return out


def _merged(terms):
    """Sum the log-factors of mechanisms with the same signature, for comparing layers."""
    acc = {}
    for a, b, l, c in terms:
        acc[(a, b, c)] = acc.get((a, b, c), 0.0) + l
    return sorted((k, round(v, 12)) for k, v in acc.items())


@nb.njit(parallel=True, cache=True)
def _row_select(x, M, counts, idx, r):
    """Per row of N (chi = 0, float64 (SZ, SZ)): indices of the M largest entries, largest first, and the row sum."""
    for a in nb.prange(SZ):
        row = x[a]
        order = np.argsort(-row, kind="mergesort")
        c = 0
        for q in range(M):
            j = order[q]
            if row[j] > 0.0:
                idx[a, c] = j
                c += 1
        counts[a] = c
        r[a] = row.sum()


@nb.njit(parallel=True, cache=True)
def _row_gather(x, k, M, idx, counts, vals):
    for a in nb.prange(SZ):
        for c in range(counts[a]):
            vals[a * M + c, k] = x[a, idx[a, c]]


def build(circuit, M=1024, cache_dir=None, verbose=False):
    """Tables for one circuit: dict(M, counts, cols (SZ*M,), vals (SZ*M, 5), T0i (128, SZ, 5), T4i (128, SZ, 5), r)."""
    path = None
    if cache_dir is not None:
        key = hashlib.sha1((str(circuit) + f"|M={M}|v2").encode()).hexdigest()[:16]
        path = os.path.join(cache_dir, f"mld_{key}.npz")
        if os.path.exists(path):
            f = np.load(path)
            return {k: (int(f[k]) if k == "M" else f[k]) for k in f.files}
    g = _groups(circuit)
    if not (_merged(g[(1, 2)]) == _merged(g[(2, 3)]) == _merged(g[(3, 4)]) and _merged(g[(2, 2)]) == _merged(g[(3, 3)])):
        raise ValueError("middle layers are not identical")
    half = g[(2, 2)]
    t0 = [(b | (a << NA), l, c) for a, b, l, c in g[(0, 1)]]        # index = b | a << 14  ->  [a (7 bits), b]
    t0 += [(a << NA, l, c) for a, _, l, c in g.get((0, 0), [])]
    t0 += [(a, l, c) for a, _, l, c in g.get((1, 1), [])]
    t0 += [(a, -0.5 * l, c) for a, _, l, c in half]
    nn = [(b | (a << NA), l, c) for a, b, l, c in g[(2, 3)]]        # index = b | a << 14  ->  [a, b]
    nn += [(a << NA, 0.5 * l, c) for a, _, l, c in half] + [(a, 0.5 * l, c) for a, _, l, c in half]
    t4 = [(a | (b << NA), l, c) for a, b, l, c in g[(4, 5)]]        # index = a | b << 14  ->  [b (7 bits), a]
    t4 += [(a, l, c) for a, _, l, c in g.get((4, 4), [])]
    t4 += [(a << NA, l, c) for a, _, l, c in g.get((5, 5), [])]
    t4 += [(a, -0.5 * l, c) for a, _, l, c in half]
    T0i = np.empty((128, SZ, 5), np.float32)
    T4i = np.empty((128, SZ, 5), np.float32)
    counts = np.zeros(SZ, np.int64)
    idx = np.zeros((SZ, M), np.int32)
    r = np.zeros(SZ, np.float64)
    vals = np.zeros((SZ * M, 5), np.float32)
    for k, chi in enumerate(CHIS):
        T0i[:, :, k] = _table(21, t0, chi).reshape(128, SZ)
        T4i[:, :, k] = _table(21, t4, chi).reshape(128, SZ)
        x = _table(28, nn, chi).reshape(SZ, SZ)
        if k == 0:
            _row_select(x, M, counts, idx, r)
        _row_gather(x, k, M, idx, counts, vals)
        if verbose:
            print(f"  chi {chi}: table sum {x.sum():+.6f}", flush=True)
        del x
    tb = dict(M=M, counts=counts, cols=idx.reshape(-1), vals=vals, T0i=T0i, T4i=T4i, r=r.astype(np.float32))
    if path is not None:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(path + ".tmp.npz", **tb)
        os.replace(path + ".tmp.npz", path)
    return tb


@nb.njit(parallel=True, cache=True)
def _fast(i0, i1, i2, i3, i4, i5, T0i, T4i, M, counts, cols, vals, r, delta, kmax, eta, out):
    n = len(i0)
    CH = 64
    for ch in nb.prange((n + CH - 1) // CH):
        bufa = np.empty((SZ, 5), np.float32)
        bufb = np.empty((SZ, 5), np.float32)
        score = np.empty(SZ, np.float32)
        sel = np.empty(kmax, np.int64)
        for s in range(ch * CH, min(n, (ch + 1) * CH)):
            cur = T0i[i0[s]]
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
                over = False
                for b in range(SZ):
                    if score[b] >= thr and score[b] > 0.0:
                        if cnt == kmax:
                            over = True
                            break
                        sel[cnt] = b
                        cnt += 1
                if over:
                    thr = np.partition(score, SZ - kmax)[SZ - kmax]
                    cnt = 0
                    for b in range(SZ):
                        if score[b] >= thr and score[b] > 0.0 and cnt < kmax:
                            sel[cnt] = b
                            cnt += 1
                inv = np.float32(1.0) / mv if mv > 0.0 else np.float32(1.0)
                best = np.float32(0.0)
                for q in range(cnt):
                    b = sel[q]
                    a = b ^ sh
                    if counts[a] > 0:
                        w = cur[b, 0] * inv * vals[a * M, 0]
                        if w > best:
                            best = w
                cut = best * eta
                nxt = bufa if t % 2 == 0 else bufb
                nxt[:, :] = 0.0
                for q in range(cnt):
                    b = sel[q]
                    a = b ^ sh
                    c0 = cur[b, 0] * inv
                    c1 = cur[b, 1] * inv
                    c2 = cur[b, 2] * inv
                    c3 = cur[b, 3] * inv
                    c4 = cur[b, 4] * inv
                    e0 = a * M
                    for e in range(e0, e0 + counts[a]):
                        v0 = vals[e, 0]
                        if c0 * v0 < cut:
                            break
                        j = cols[e]
                        nxt[j, 0] += c0 * v0
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
            a0 = 0.0
            a1 = 0.0
            a2 = 0.0
            a3 = 0.0
            a4 = 0.0
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


def run(tb, detector_events, delta=1e-4, kmax=128, eta=0.0):
    """(shots, 5) array of P_chi(D) for chi = 0, e0, e1, e2, e3, up to a common positive factor per shot."""
    d = np.asarray(detector_events).astype(np.int64)
    w7, w14 = 1 << np.arange(7), 1 << np.arange(14)
    i = [d[:, :7] @ w7] + [d[:, 7 + 14 * t: 21 + 14 * t] @ w14 for t in range(4)] + [d[:, 63:] @ w7]
    out = np.zeros((len(d), 5))
    _fast(i[0], i[1], i[2], i[3], i[4], i[5], tb["T0i"], tb["T4i"], tb["M"], tb["counts"], tb["cols"], tb["vals"],
          tb["r"], np.float32(delta), int(kmax), np.float32(eta), out)
    return out
