"""BP-OSD v2: experimental extensions of team/decoders/bposd.py (numpy + numba only, no QEC decoder libraries).

New options compared with v1:
  schedule      0 = flooding (v1), 1 = layered / serial over checks (uses the newest messages, converges faster)
  degenerate    False = pick the single most likely error (v1);
                True  = approximate maximum-likelihood (degenerate) decoding: every distinct error found by OSD-CS adds
                        its probability exp(-cost) to the logical class it implies (one of 16 for 4 observables), and
                        each observable is decided by its marginal probability, which is optimal for the per-bit metric
  snaps         BP iterations whose posteriors each seed an OSD-CS run (0 = the priors). Several orderings give a
                bigger, more diverse candidate set; duplicates are removed with Zobrist hashes before summing
  shortcut      skip OSD when BP converges (the class posterior is then one-hot on BP's solution)
  bp_scale      multiply the BP priors' LLRs by this (OSD costs always use the true priors)

decode_full() also returns the 16-class posterior per shot, used for confidence-gated hybrids.
"""
import numpy as np
import numba as nb
import stim

from hackathon_2026 import problem_info

DEFAULTS = dict(schedule=1, max_iter=3, alpha=0.45, snaps=(3,), n_single=200, n_pair=30, degenerate=True,
                shortcut=False, bp_scale=1.0, n_expand=0)


class _Problem:
    def __init__(self, circuit: stim.Circuit):
        info = problem_info(circuit)
        H, L, p = info["H"].astype(np.uint8), info["L"].astype(np.uint8), info["p"].astype(np.float64)
        m, n = H.shape
        self.m, self.n, self.num_obs = m, n, L.shape[0]
        self.prior_llr = np.log((1 - p) / p)
        chk, var = np.nonzero(H)
        self.edge_var = var.astype(np.int32)
        self.chk_ptr = np.concatenate([[0], np.cumsum(np.bincount(chk, minlength=m))]).astype(np.int32)
        self.var_edges = np.argsort(var, kind="stable").astype(np.int32)
        self.var_ptr = np.concatenate([[0], np.cumsum(np.bincount(var, minlength=n))]).astype(np.int32)
        self.max_deg = int(np.diff(self.chk_ptr).max())
        self.colbits = np.zeros((n, 2), np.uint64)
        for j in range(n):
            for i in np.flatnonzero(H[:, j]):
                self.colbits[j, i >> 6] |= np.uint64(1) << np.uint64(i & 63)
        self.obsmask = np.zeros(n, np.uint8)
        for k in range(L.shape[0]):
            self.obsmask |= L[k].astype(np.uint8) << k
        self.zobrist = np.random.default_rng(12345).integers(1, 2 ** 63, n, dtype=np.int64).astype(np.uint64)
        self.rank = _rank(self.colbits)
        self._tri = None

    def triples(self):
        """Lazily built CSR (tri_ptr, tri_other) of weight-3 null combinations; only needed when n_expand > 0."""
        if self._tri is None:
            n = self.n
            # weight-3 null combinations: columns a, b, c with col_a ^ col_b == col_c (same syndrome), used by the local
            # search; per column, the list of the other two columns of each triple containing it (CSR)
            order = np.lexsort((self.colbits[:, 1], self.colbits[:, 0]))
            tri = _triples(self.colbits, order, self.colbits[order, 0].copy(), self.colbits[order, 1].copy())
            self.n_triples = len(tri)
            owner = np.concatenate([tri[:, 0], tri[:, 1], tri[:, 2]])
            partners = np.concatenate([tri[:, [1, 2]], tri[:, [0, 2]], tri[:, [0, 1]]])
            srt = np.argsort(owner, kind="stable")
            tri_ptr = np.concatenate([[0], np.cumsum(np.bincount(owner, minlength=n))]).astype(np.int32)
            tri_other = partners[srt].astype(np.int32)
            self._tri = (tri_ptr, tri_other)
        return self._tri


_CACHE = {}


def problem(circuit):
    key = str(circuit)
    if key not in _CACHE:
        _CACHE[key] = _Problem(circuit)
    return _CACHE[key]


@nb.njit(cache=True, inline="always")
def _getbit(v0, v1, i):
    if i < 64:
        return (v0 >> np.uint64(i)) & np.uint64(1)
    return (v1 >> np.uint64(i - 64)) & np.uint64(1)


@nb.njit(cache=True)
def _lowbit(v0, v1):
    for i in range(64):
        if (v0 >> np.uint64(i)) & np.uint64(1):
            return i
    for i in range(64):
        if (v1 >> np.uint64(i)) & np.uint64(1):
            return 64 + i
    return -1


@nb.njit(cache=True)
def _rank(colbits):
    basis = np.zeros((128, 2), np.uint64)
    piv = np.zeros(128, np.int64)
    r = 0
    for j in range(colbits.shape[0]):
        v0, v1 = colbits[j, 0], colbits[j, 1]
        for k in range(r):
            if _getbit(v0, v1, piv[k]):
                v0 ^= basis[k, 0]
                v1 ^= basis[k, 1]
        if v0 != 0 or v1 != 0:
            basis[r, 0], basis[r, 1] = v0, v1
            piv[r] = _lowbit(v0, v1)
            r += 1
    return r


@nb.njit(cache=True)
def _triples(colbits, order, k0, k1):
    """All (a, b, c), a < b < c, with col_a ^ col_b == col_c; k0/k1 are the sorted column keys, order the column ids."""
    n = colbits.shape[0]
    out = np.empty((200_000, 3), np.int64)
    cnt = 0
    for a in range(n):
        for b in range(a + 1, n):
            x0 = colbits[a, 0] ^ colbits[b, 0]
            x1 = colbits[a, 1] ^ colbits[b, 1]
            lo, hi = 0, n
            while lo < hi:  # binary search for (x0, x1) in the sorted keys
                mid = (lo + hi) // 2
                if k0[mid] < x0 or (k0[mid] == x0 and k1[mid] < x1):
                    lo = mid + 1
                else:
                    hi = mid
            if lo < n and k0[lo] == x0 and k1[lo] == x1:
                c = order[lo]
                if c > b and cnt < out.shape[0]:
                    out[cnt, 0], out[cnt, 1], out[cnt, 2] = a, b, c
                    cnt += 1
    return out[:cnt]


@nb.njit(cache=True)
def _osd(order, s0, s1, colbits, rank_full, weight, obsmask, zob, n_single, n_pair, cc, co, ch, nc, cols, ncol):
    """OSD-CS seeded by `order`; appends (cost, obs, hash) of every valid candidate to cc/co/ch from index nc."""
    n = order.shape[0]
    bvec = np.zeros((rank_full, 2), np.uint64)
    bcomb = np.zeros((rank_full, 2), np.uint64)
    piv = np.zeros(rank_full, np.int64)
    sel = np.zeros(rank_full, np.int64)
    ncmax = max(n_single, n_pair)
    cand = np.empty(max(ncmax, 1), np.int64)
    n_cand = 0
    rank = 0
    for idx in range(n):
        j = order[idx]
        if rank == rank_full:
            if n_cand < ncmax:
                cand[n_cand] = j
                n_cand += 1
                continue
            break
        v0, v1 = colbits[j, 0], colbits[j, 1]
        k0 = np.uint64(0)
        k1 = np.uint64(0)
        for k in range(rank):
            if _getbit(v0, v1, piv[k]):
                v0 ^= bvec[k, 0]
                v1 ^= bvec[k, 1]
                k0 ^= bcomb[k, 0]
                k1 ^= bcomb[k, 1]
        if v0 != 0 or v1 != 0:
            if rank < 64:
                k0 ^= np.uint64(1) << np.uint64(rank)
            else:
                k1 ^= np.uint64(1) << np.uint64(rank - 64)
            bvec[rank, 0], bvec[rank, 1] = v0, v1
            bcomb[rank, 0], bcomb[rank, 1] = k0, k1
            piv[rank] = _lowbit(v0, v1)
            sel[rank] = j
            rank += 1
        elif n_cand < ncmax:
            cand[n_cand] = j
            n_cand += 1
    npair = min(n_pair, n_cand)
    nsing = min(n_single, n_cand)
    total = 1 + nsing + npair * (npair - 1) // 2
    for t in range(total):
        # t = 0: no flips; 1..nsing: one flip; then pairs (a, b) with a < b < npair
        x0, x1 = s0, s1
        extra_cost = 0.0
        extra_obs = np.uint8(0)
        extra_hash = np.uint64(0)
        ja = 0
        jb = 0
        if t >= 1 and t <= nsing:
            ja = cand[t - 1]
            x0 ^= colbits[ja, 0]
            x1 ^= colbits[ja, 1]
            extra_cost = weight[ja]
            extra_obs = obsmask[ja]
            extra_hash = zob[ja]
        elif t > nsing:
            r = t - nsing - 1
            a = 0
            while r >= npair - 1 - a:
                r -= npair - 1 - a
                a += 1
            b = a + 1 + r
            ja, jb = cand[a], cand[b]
            x0 ^= colbits[ja, 0] ^ colbits[jb, 0]
            x1 ^= colbits[ja, 1] ^ colbits[jb, 1]
            extra_cost = weight[ja] + weight[jb]
            extra_obs = obsmask[ja] ^ obsmask[jb]
            extra_hash = zob[ja] ^ zob[jb]
        c0 = np.uint64(0)
        c1 = np.uint64(0)
        for k in range(rank):
            if _getbit(x0, x1, piv[k]):
                x0 ^= bvec[k, 0]
                x1 ^= bvec[k, 1]
                c0 ^= bcomb[k, 0]
                c1 ^= bcomb[k, 1]
        if x0 != 0 or x1 != 0:
            continue
        cost = extra_cost
        obs = extra_obs
        h = extra_hash
        nk = 0
        if t >= 1 and t <= nsing:
            cols[nc, 0] = cand[t - 1]
            nk = 1
        elif t > nsing:
            cols[nc, 0] = ja
            cols[nc, 1] = jb
            nk = 2
        for k in range(rank):
            if _getbit(c0, c1, k):
                j = sel[k]
                cost += weight[j]
                obs ^= obsmask[j]
                h ^= zob[j]
                cols[nc, nk] = j
                nk += 1
        ncol[nc] = nk
        cc[nc] = cost
        co[nc] = obs
        ch[nc] = h
        nc += 1
    return nc


@nb.njit(cache=True)
def _shot(syn, s0, s1, prior_llr, bp_llr, edge_var, chk_ptr, var_edges, var_ptr, max_deg, colbits, obsmask, zob,
          rank_full, schedule, max_iter, alpha, snaps, n_single, n_pair, shortcut, tri_ptr, tri_other, n_expand,
          probs_out):
    m = chk_ptr.shape[0] - 1
    n = var_ptr.shape[0] - 1
    E = edge_var.shape[0]
    c2v = np.zeros(E)
    v2c = np.empty(E)
    tmp = np.empty(max_deg)
    post = bp_llr.copy()
    hard = np.zeros(n, np.uint8)
    nsnap = snaps.shape[0]
    snap_post = np.empty((nsnap, n))
    si = 0
    while si < nsnap and snaps[si] == 0:
        snap_post[si] = bp_llr
        si += 1
    if schedule == 0:
        for e in range(E):
            v2c[e] = bp_llr[edge_var[e]]
    converged = False
    for it in range(max_iter):
        if schedule == 1:
            for c in range(m):
                a, b = chk_ptr[c], chk_ptr[c + 1]
                sgn = -1.0 if syn[c] else 1.0
                min1 = 1e300
                min2 = 1e300
                amin = -1
                for e in range(a, b):
                    x = post[edge_var[e]] - c2v[e]
                    tmp[e - a] = x
                    if x < 0:
                        sgn = -sgn
                        x = -x
                    if x < min1:
                        min2 = min1
                        min1 = x
                        amin = e
                    elif x < min2:
                        min2 = x
                for e in range(a, b):
                    x = tmp[e - a]
                    mag = min2 if e == amin else min1
                    new = alpha * (sgn if x >= 0 else -sgn) * mag
                    post[edge_var[e]] = x + new
                    c2v[e] = new
            for v in range(n):
                hard[v] = 1 if post[v] < 0 else 0
        else:
            for c in range(m):
                a, b = chk_ptr[c], chk_ptr[c + 1]
                sgn = -1.0 if syn[c] else 1.0
                min1 = 1e300
                min2 = 1e300
                amin = -1
                for e in range(a, b):
                    x = v2c[e]
                    if x < 0:
                        sgn = -sgn
                        x = -x
                    if x < min1:
                        min2 = min1
                        min1 = x
                        amin = e
                    elif x < min2:
                        min2 = x
                for e in range(a, b):
                    mag = min2 if e == amin else min1
                    c2v[e] = alpha * (sgn if v2c[e] >= 0 else -sgn) * mag
            for v in range(n):
                tot = bp_llr[v]
                for k in range(var_ptr[v], var_ptr[v + 1]):
                    tot += c2v[var_edges[k]]
                post[v] = tot
                hard[v] = 1 if tot < 0 else 0
                for k in range(var_ptr[v], var_ptr[v + 1]):
                    e = var_edges[k]
                    v2c[e] = tot - c2v[e]
        while si < nsnap and snaps[si] == it + 1:
            snap_post[si] = post
            si += 1
        ok = True
        for c in range(m):
            par = 0
            for e in range(chk_ptr[c], chk_ptr[c + 1]):
                par ^= hard[edge_var[e]]
            if par != syn[c]:
                ok = False
                break
        if ok:
            converged = True
            break
    while si < nsnap:
        snap_post[si] = post
        si += 1

    maxlen = rank_full + 4
    base_c = 1 + nsnap * (1 + n_single + n_pair * (n_pair - 1) // 2)
    max_c = base_c + n_expand * 600
    cc = np.empty(max_c)
    co = np.empty(max_c, np.uint8)
    ch = np.empty(max_c, np.uint64)
    cols = np.empty((base_c, maxlen), np.int32)
    ncol = np.zeros(base_c, np.int32)
    nc = 0
    if converged:
        cost = 0.0
        obs = np.uint8(0)
        h = np.uint64(0)
        nk = 0
        for v in range(n):
            if hard[v]:
                cost += prior_llr[v]
                obs ^= obsmask[v]
                h ^= zob[v]
                if nk < maxlen:
                    cols[0, nk] = v
                nk += 1
        ncol[0] = nk if nk <= maxlen else 0
        cc[0], co[0], ch[0] = cost, obs, h
        nc = 1
        if shortcut:
            probs_out[obs] = 1.0
            return obs
    for s in range(nsnap):
        order = np.argsort(snap_post[s])
        nc = _osd(order, s0, s1, colbits, rank_full, prior_llr, obsmask, zob, n_single, n_pair, cc, co, ch, nc, cols,
                  ncol)
    if nc == 0:
        probs_out[0] = 1.0
        return np.uint8(0)
    # local search: around the n_expand cheapest candidates, swap in every weight-3 null combination (a, b, c) that
    # touches the error (col_a ^ col_b == col_c, so the syndrome is unchanged; some also flip observables)
    if n_expand > 0:
        n_osd = nc
        top = np.argsort(cc[:n_osd])[:n_expand]
        mark = np.zeros(n, np.uint8)
        for r in range(top.shape[0]):
            i = top[r]
            if i >= base_c:
                continue
            k_n = ncol[i]
            for q in range(k_n):
                mark[cols[i, q]] = 1
            for q in range(k_n):
                j = cols[i, q]
                for t in range(tri_ptr[j], tri_ptr[j + 1]):
                    if nc >= max_c:
                        break
                    u = tri_other[t, 0]
                    w = tri_other[t, 1]
                    dc = -prior_llr[j]
                    dc += -prior_llr[u] if mark[u] else prior_llr[u]
                    dc += -prior_llr[w] if mark[w] else prior_llr[w]
                    cc[nc] = cc[i] + dc
                    co[nc] = co[i] ^ obsmask[j] ^ obsmask[u] ^ obsmask[w]
                    ch[nc] = ch[i] ^ zob[j] ^ zob[u] ^ zob[w]
                    nc += 1
            for q in range(k_n):
                mark[cols[i, q]] = 0
    # remove duplicate errors (same Zobrist hash), then sum probabilities per logical class
    idx = np.argsort(ch[:nc])
    cmin = cc[:nc].min()
    best = 0
    total = 0.0
    prev = np.uint64(0)
    for r in range(nc):
        i = idx[r]
        if r > 0 and ch[i] == prev:
            continue
        prev = ch[i]
        w = np.exp(cmin - cc[i])
        probs_out[co[i]] += w
        total += w
        if cc[i] < cc[best]:
            best = i
    for k in range(16):
        probs_out[k] /= total
    return co[best]


@nb.njit(parallel=True, cache=True)
def _run(dets, prior_llr, bp_llr, edge_var, chk_ptr, var_edges, var_ptr, max_deg, colbits, obsmask, zob, rank_full,
         schedule, max_iter, alpha, snaps, n_single, n_pair, shortcut, tri_ptr, tri_other, n_expand):
    shots, m = dets.shape
    best = np.zeros(shots, np.uint8)
    probs = np.zeros((shots, 16))
    for i in nb.prange(shots):
        syn = dets[i]
        s0 = np.uint64(0)
        s1 = np.uint64(0)
        nz = False
        for c in range(m):
            if syn[c]:
                nz = True
                if c < 64:
                    s0 |= np.uint64(1) << np.uint64(c)
                else:
                    s1 |= np.uint64(1) << np.uint64(c - 64)
        if not nz:
            probs[i, 0] = 1.0
            continue
        best[i] = _shot(syn, s0, s1, prior_llr, bp_llr, edge_var, chk_ptr, var_edges, var_ptr, max_deg, colbits,
                        obsmask, zob, rank_full, schedule, max_iter, alpha, snaps, n_single, n_pair, shortcut,
                        tri_ptr, tri_other, n_expand, probs[i])
    return best, probs


CLASS_BITS = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(np.float64)  # (16, 4)


def decode_full(circuit, detector_events, **kw):
    """Returns (best-error prediction (shots, 4), class posterior (shots, 16), per-bit marginal P(L_j = 1) (shots, 4))."""
    o = {**DEFAULTS, **kw}
    P = problem(circuit)
    dets = np.ascontiguousarray(detector_events, dtype=np.uint8)
    best, probs = _run(dets, P.prior_llr, P.prior_llr * o["bp_scale"], P.edge_var, P.chk_ptr, P.var_edges, P.var_ptr,
                       P.max_deg, P.colbits, P.obsmask, P.zobrist, P.rank, o["schedule"], o["max_iter"], o["alpha"],
                       np.asarray(sorted(o["snaps"]), np.int64), o["n_single"], o["n_pair"], o["shortcut"],
                       *(P.triples() if o["n_expand"] > 0 else (np.zeros(P.n + 1, np.int32), np.zeros((0, 2), np.int32))),
                       o["n_expand"])
    best_bits = ((best[:, None] >> np.arange(P.num_obs, dtype=np.uint8)) & 1).astype(np.uint8)
    # plain sums instead of a matmul: macOS Accelerate raises spurious warnings on subnormal probabilities
    marg = np.stack([probs[:, CLASS_BITS[:, j] == 1].sum(1) for j in range(P.num_obs)], axis=1)
    return best_bits, probs, marg


def decode(circuit, detector_events, **kw):
    o = {**DEFAULTS, **kw}
    best_bits, _, marg = decode_full(circuit, detector_events, **o)
    return (marg > 0.5).astype(np.uint8) if o["degenerate"] else best_bits
