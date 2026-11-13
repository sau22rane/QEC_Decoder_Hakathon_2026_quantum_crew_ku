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
  temp          temperature of the class sum: each candidate adds exp(-cost / temp) (1.0 = true probabilities)
  n_perturb     extra OSD-CS runs (perturb_single/perturb_pair) seeded by noisy copies of the last BP posterior
  dec_*         BP guided decimation: fix the likeliest mechanism, re-run dec_iters BP iterations, repeat up to
                dec_steps times; dec_branch first-step branches; dec_osd decimation posteriors seed OSD-CS runs
  relay_*       Relay-BP (memory-BP legs) after the normal pass: every converged leg adds a candidate, and the first
                relay_osd legs' marginals seed extra OSD-CS runs (relay_legs = 0 disables it)

decode_full() also returns the 16-class posterior per shot, used for confidence-gated hybrids.
"""
import numpy as np
import numba as nb
import stim

from hackathon_2026 import problem_info

DEFAULTS = dict(schedule=1, max_iter=3, alpha=0.45, snaps=(3,), n_single=200, n_pair=30, degenerate=True,
                shortcut=False, bp_scale=1.0, n_expand=0, temp=1.0,
                relay_legs=0, relay_T0=80, relay_T=60, relay_S=5, relay_alpha=1.0, relay_osd=0,
                relay_gmin=-0.24, relay_gmax=0.66, relay_g0=0.125,
                n_perturb=0, perturb_scale=1.0, perturb_mode=0, perturb_single=100, perturb_pair=20,
                dec_steps=0, dec_iters=3, dec_branch=1, dec_osd=0)


_DEBRUIJN = np.zeros(64, np.int64)
for _i in range(64):
    _DEBRUIJN[((1 << _i) * 0x03F79D71B4CB0A89 % (1 << 64)) >> 58] = _i


class _Problem:
    def __init__(self, circuit: stim.Circuit):
        info = problem_info(circuit)
        H, L, p = info["H"].astype(np.uint8), info["L"].astype(np.uint8), info["p"].astype(np.float64)
        m, n = H.shape
        self.m, self.n, self.num_obs = m, n, L.shape[0]
        self.prior_llr = np.log((1 - p) / p)
        chk, var = np.nonzero(H)
        self.edge_var = var.astype(np.int32)
        self.edge_chk = chk.astype(np.int32)  # check of each edge (edges are sorted by check)
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
def _reduce(v0, v1, rank, bvec, piv, bcomb):
    """Reduce a 128-bit vector by the basis: (residual0, residual1, comb0, comb1)."""
    k0 = np.uint64(0)
    k1 = np.uint64(0)
    for k in range(rank):
        if _getbit(v0, v1, piv[k]):
            v0 ^= bvec[k, 0]
            v1 ^= bvec[k, 1]
            k0 ^= bcomb[k, 0]
            k1 ^= bcomb[k, 1]
    return v0, v1, k0, k1


@nb.njit(cache=True)
def _osd(order, s0, s1, colbits, rank_full, weight, obsmask, zob, n_single, n_pair, cc, co, ch, nc, cols, ncol,
         want_cols):
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
    nsol = max(nsing, npair)
    # The solve is linear over GF(2): solve(s ^ col_a ^ col_b) = solve(s) ^ solve(col_a) ^ solve(col_b). So reduce the
    # syndrome and each candidate column ONCE; every OSD-CS candidate is then two 128-bit XORs of these solutions
    # (comb = which basis columns are used; res = residual, nonzero only if H is rank deficient).
    sr0, sr1, sc0, sc1 = _reduce(s0, s1, rank, bvec, piv, bcomb)
    cres = np.empty((max(nsol, 1), 2), np.uint64)
    ccmb = np.empty((max(nsol, 1), 2), np.uint64)
    for a in range(nsol):
        ja = cand[a]
        r0, r1, k0, k1 = _reduce(colbits[ja, 0], colbits[ja, 1], rank, bvec, piv, bcomb)
        cres[a, 0], cres[a, 1], ccmb[a, 0], ccmb[a, 1] = r0, r1, k0, k1
    total = 1 + nsing + npair * (npair - 1) // 2
    a = 0
    b = 0
    for t in range(total):
        # t = 0: no flips; 1..nsing: one flip; then pairs (a, b), a < b < npair, enumerated incrementally
        r0, r1, c0, c1 = sr0, sr1, sc0, sc1
        cost = 0.0
        obs = np.uint8(0)
        h = np.uint64(0)
        nk = 0
        if t >= 1 and t <= nsing:
            i = t - 1
            ja = cand[i]
            r0 ^= cres[i, 0]
            r1 ^= cres[i, 1]
            c0 ^= ccmb[i, 0]
            c1 ^= ccmb[i, 1]
            cost = weight[ja]
            obs = obsmask[ja]
            h = zob[ja]
            if want_cols:
                cols[nc, 0] = ja
            nk = 1
        elif t > nsing:
            if t == nsing + 1:
                a, b = 0, 1
            else:
                b += 1
                if b == npair:
                    a += 1
                    b = a + 1
            ja, jb = cand[a], cand[b]
            r0 ^= cres[a, 0] ^ cres[b, 0]
            r1 ^= cres[a, 1] ^ cres[b, 1]
            c0 ^= ccmb[a, 0] ^ ccmb[b, 0]
            c1 ^= ccmb[a, 1] ^ ccmb[b, 1]
            cost = weight[ja] + weight[jb]
            obs = obsmask[ja] ^ obsmask[jb]
            h = zob[ja] ^ zob[jb]
            if want_cols:
                cols[nc, 0] = ja
                cols[nc, 1] = jb
            nk = 2
        if r0 != 0 or r1 != 0:
            continue
        # walk only the set bits of comb (de Bruijn trailing-zero count)
        for word in range(2):
            x = c0 if word == 0 else c1
            while x != 0:
                low = x & (~x + np.uint64(1))
                k = _DEBRUIJN[(low * np.uint64(0x03F79D71B4CB0A89)) >> np.uint64(58)] + 64 * word
                j = sel[k]
                cost += weight[j]
                obs ^= obsmask[j]
                h ^= zob[j]
                if want_cols:
                    cols[nc, nk] = j
                nk += 1
                x ^= low
        if want_cols:
            ncol[nc] = nk
        cc[nc] = cost
        co[nc] = obs
        ch[nc] = h
        nc += 1
    return nc


@nb.njit(cache=True)
def _relay(syn, prior_llr, edge_var, chk_ptr, var_edges, var_ptr, obsmask, zob, gam, T0, T, S, alpha, n_keep,
           keep_post, cc, co, ch, nc):
    """Relay-BP (Mueller et al. 2025): legs of min-sum BP with per-mechanism memory strengths gam[leg, v].

    Each iteration uses the prior mixed with the previous marginal, lam = (1 - g) * prior + g * M; negative g pushes BP
    away from where it is stuck. Each leg starts from the previous leg's marginals ("relay"). Every converged leg adds
    its error (cost, observables, Zobrist hash) as a candidate; we stop after S solutions. The final marginals of the
    first n_keep legs are kept as extra OSD orderings. Returns (new nc, number of kept marginals).
    """
    m = chk_ptr.shape[0] - 1
    n = var_ptr.shape[0] - 1
    E = edge_var.shape[0]
    M = prior_llr.copy()
    lam = np.empty(n)
    v2c = np.empty(E)
    c2v = np.zeros(E)
    hard = np.zeros(n, np.uint8)
    found = 0
    kept = 0
    for r in range(gam.shape[0]):
        iters = T0 if r == 0 else T
        for e in range(E):
            v2c[e] = M[edge_var[e]]
            c2v[e] = 0.0
        for it in range(iters):
            for v in range(n):
                g = gam[r, v]
                lam[v] = (1.0 - g) * prior_llr[v] + g * M[v]
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
                tot = lam[v]
                for k in range(var_ptr[v], var_ptr[v + 1]):
                    tot += c2v[var_edges[k]]
                M[v] = tot
                hard[v] = 1 if tot < 0 else 0
                for k in range(var_ptr[v], var_ptr[v + 1]):
                    e = var_edges[k]
                    v2c[e] = tot - c2v[e]
            ok = True
            for c in range(m):
                par = 0
                for e in range(chk_ptr[c], chk_ptr[c + 1]):
                    par ^= hard[edge_var[e]]
                if par != syn[c]:
                    ok = False
                    break
            if ok:
                cost = 0.0
                obs = np.uint8(0)
                h = np.uint64(0)
                for v in range(n):
                    if hard[v]:
                        cost += prior_llr[v]
                        obs ^= obsmask[v]
                        h ^= zob[v]
                cc[nc], co[nc], ch[nc] = cost, obs, h
                nc += 1
                found += 1
                break
        if kept < n_keep:
            keep_post[kept] = M
            kept += 1
        if found >= S:
            break
    return nc, kept


@nb.njit(cache=True)
def _bp_layered(syn, llr, edge_var, chk_ptr, max_deg, iters, alpha, post, hard):
    """Layered min-sum BP from fresh messages on (syn, llr); fills post/hard; returns True if H hard = syn."""
    m = chk_ptr.shape[0] - 1
    n = llr.shape[0]
    E = edge_var.shape[0]
    c2v = np.zeros(E)
    tmp = np.empty(max_deg)
    for v in range(n):
        post[v] = llr[v]
    for it in range(iters):
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
        ok = True
        for c in range(m):
            par = 0
            for e in range(chk_ptr[c], chk_ptr[c + 1]):
                par ^= hard[edge_var[e]]
            if par != syn[c]:
                ok = False
                break
        if ok:
            return True
    return False


@nb.njit(cache=True)
def _bpgd(syn, prior_llr, edge_var, chk_ptr, var_edges, var_ptr, edge_chk, max_deg, obsmask, zob, steps, iters,
          alpha, branch, n_keep, keep_post, cc, co, ch, nc):
    """BP guided decimation: repeatedly fix the most likely remaining mechanism (XOR its column into the residual
    syndrome, forbid it in BP) and re-run a few BP iterations, until BP converges on the residual. Each converged run
    adds (fixed set + BP solution) as a candidate. With branch > 1, the first decision is taken for each of the top
    `branch` mechanisms, giving several distinct solutions. The posteriors after the first n_keep decimation steps
    (decimated mechanisms forced to the front) are kept as extra OSD orderings."""
    m = chk_ptr.shape[0] - 1
    n = prior_llr.shape[0]
    post = np.empty(n)
    hard = np.zeros(n, np.uint8)
    kept = 0
    # first BP on the full problem decides the branch candidates
    _bp_layered(syn, prior_llr, edge_var, chk_ptr, max_deg, iters, alpha, post, hard)
    first = np.argsort(post)[:max(branch, 1)]
    for br in range(first.shape[0]):
        llr = prior_llr.copy()
        res = syn.copy()
        fixed = np.zeros(n, np.uint8)
        cost = 0.0
        obs = np.uint8(0)
        h = np.uint64(0)
        v = first[br]
        for step in range(steps):
            # fix mechanism v
            fixed[v] = 1
            cost += prior_llr[v]
            obs ^= obsmask[v]
            h ^= zob[v]
            for k in range(var_ptr[v], var_ptr[v + 1]):
                c = edge_chk[var_edges[k]]
                res[c] ^= 1
            llr[v] = 1e6
            ok = _bp_layered(res, llr, edge_var, chk_ptr, max_deg, iters, alpha, post, hard)
            if br == 0 and kept < n_keep:
                for u in range(n):
                    keep_post[kept, u] = -1e6 if fixed[u] else post[u]
                kept += 1
            if ok:
                c2 = cost
                o2 = obs
                h2 = h
                for u in range(n):
                    if hard[u]:
                        c2 += prior_llr[u]
                        o2 ^= obsmask[u]
                        h2 ^= zob[u]
                cc[nc], co[nc], ch[nc] = c2, o2, h2
                nc += 1
                break
            # next: most likely mechanism not yet fixed
            best = 1e300
            v = -1
            for u in range(n):
                if not fixed[u] and post[u] < best:
                    best = post[u]
                    v = u
            if v < 0:
                break
    return nc, kept


@nb.njit(cache=True)
def _shot(syn, s0, s1, prior_llr, bp_llr, edge_var, chk_ptr, var_edges, var_ptr, max_deg, colbits, obsmask, zob,
          rank_full, schedule, max_iter, alpha, snaps, n_single, n_pair, shortcut, tri_ptr, tri_other, n_expand,
          inv_temp, gam, relay_T0, relay_T, relay_S, relay_alpha, relay_osd, pert, pert_mode, pert_single,
          pert_pair, edge_chk, dec_steps, dec_iters, dec_branch, dec_osd, probs_out, stats_out):
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
    per_osd = 1 + n_single + n_pair * (n_pair - 1) // 2
    base_c += gam.shape[0] + relay_osd * per_osd
    base_c += pert.shape[0] * (1 + pert_single + pert_pair * (pert_pair - 1) // 2)
    base_c += max(dec_branch, 1) * (dec_steps > 0) + dec_osd * per_osd
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
            stats_out[0] = cost
            stats_out[1] = 1.0
            return obs
    for s in range(nsnap):
        order = np.argsort(snap_post[s])
        nc = _osd(order, s0, s1, colbits, rank_full, prior_llr, obsmask, zob, n_single, n_pair, cc, co, ch, nc, cols,
                  ncol, n_expand > 0)
    # randomly perturbed copies of the last BP posterior: each seeds a small OSD-CS run (more diverse candidates)
    if pert.shape[0] > 0 and nsnap > 0:
        base = snap_post[nsnap - 1]
        border = np.argsort(base)
        # small noise only reshuffles the front of the order: re-sort just the first K columns of the base order
        K = min(n, 512)
        head = np.empty(K)
        order = border.copy()
        for q in range(pert.shape[0]):
            for r in range(K):
                v = border[r]
                if pert_mode == 0:
                    head[r] = base[v] + pert[q, v]
                else:
                    head[r] = base[v] * np.exp(pert[q, v])
            hs = np.argsort(head)
            for r in range(K):
                order[r] = border[hs[r]]
            nc = _osd(order, s0, s1, colbits, rank_full, prior_llr, obsmask, zob, pert_single, pert_pair, cc, co, ch,
                      nc, cols, ncol, n_expand > 0)
    if dec_steps > 0:
        dpost = np.empty((max(dec_osd, 1), n))
        nc, dkept = _bpgd(syn, prior_llr, edge_var, chk_ptr, var_edges, var_ptr, edge_chk, max_deg, obsmask, zob,
                          dec_steps, dec_iters, alpha, dec_branch, dec_osd, dpost, cc, co, ch, nc)
        for q in range(dkept):
            order = np.argsort(dpost[q])
            nc = _osd(order, s0, s1, colbits, rank_full, prior_llr, obsmask, zob, n_single, n_pair, cc, co, ch, nc,
                      cols, ncol, n_expand > 0)
    if gam.shape[0] > 0:
        keep_post = np.empty((max(relay_osd, 1), n))
        nc, kept = _relay(syn, prior_llr, edge_var, chk_ptr, var_edges, var_ptr, obsmask, zob, gam, relay_T0, relay_T,
                          relay_S, relay_alpha, relay_osd, keep_post, cc, co, ch, nc)
        for s in range(kept):
            order = np.argsort(keep_post[s])
            nc = _osd(order, s0, s1, colbits, rank_full, prior_llr, obsmask, zob, n_single, n_pair, cc, co, ch, nc,
                      cols, ncol, n_expand > 0)
    if nc == 0:
        probs_out[0] = 1.0
        stats_out[0] = 1e300
        stats_out[1] = 0.0
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
        w = np.exp((cmin - cc[i]) * inv_temp)
        probs_out[co[i]] += w
        total += w
        if cc[i] < cc[best]:
            best = i
    for k in range(16):
        probs_out[k] /= total
    stats_out[0] = cmin  # cost of the most likely error found
    stats_out[1] = total  # sum of exp(cmin - cost) over distinct candidates
    return co[best]


@nb.njit(parallel=True, cache=True)
def _run(dets, prior_llr, bp_llr, edge_var, chk_ptr, var_edges, var_ptr, max_deg, colbits, obsmask, zob, rank_full,
         schedule, max_iter, alpha, snaps, n_single, n_pair, shortcut, tri_ptr, tri_other, n_expand, inv_temp, gam,
         relay_T0, relay_T, relay_S, relay_alpha, relay_osd, pert, pert_mode, pert_single, pert_pair, edge_chk,
         dec_steps, dec_iters, dec_branch, dec_osd):
    shots, m = dets.shape
    best = np.zeros(shots, np.uint8)
    probs = np.zeros((shots, 16))
    stats = np.zeros((shots, 2))
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
            stats[i, 1] = 1.0
            continue
        best[i] = _shot(syn, s0, s1, prior_llr, bp_llr, edge_var, chk_ptr, var_edges, var_ptr, max_deg, colbits,
                        obsmask, zob, rank_full, schedule, max_iter, alpha, snaps, n_single, n_pair, shortcut,
                        tri_ptr, tri_other, n_expand, inv_temp, gam, relay_T0, relay_T, relay_S,
                        relay_alpha, relay_osd, pert, pert_mode, pert_single, pert_pair, edge_chk, dec_steps,
                        dec_iters, dec_branch, dec_osd, probs[i], stats[i])
    return best, probs, stats


_GAM = {}


def _gammas(n, o):
    """Memory strengths for the Relay-BP legs, (legs, n): leg 0 uses relay_g0 everywhere, later legs are uniform in
    [relay_gmin, relay_gmax]. Fixed seed, so decoding stays deterministic."""
    key = (n, o["relay_legs"], o["relay_gmin"], o["relay_gmax"], o["relay_g0"])
    if key not in _GAM:
        g = np.random.default_rng(7).uniform(o["relay_gmin"], o["relay_gmax"], (o["relay_legs"], n))
        if o["relay_legs"] > 0:
            g[0] = o["relay_g0"]
        _GAM[key] = g
    return _GAM[key]


_PERT = {}


def _perturbations(n, o):
    """Fixed-seed Gaussian noise (n_perturb, n) for the perturbed OSD orderings (deterministic decoding).
    perturb_mode 0: added to the posterior LLRs; 1: multiplies them by exp(noise)."""
    key = (n, o["n_perturb"], o["perturb_scale"])
    if key not in _PERT:
        _PERT[key] = np.random.default_rng(11).normal(0.0, o["perturb_scale"], (o["n_perturb"], n))
    return _PERT[key]


CLASS_BITS = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(np.float64)  # (16, 4)


def decode_full(circuit, detector_events, return_stats=False, **kw):
    """Returns (best-error prediction (shots, 4), class posterior (shots, 16), per-bit marginal P(L_j = 1) (shots, 4)),
    plus with return_stats=True a (shots, 2) array: [cost of the most likely error found, sum of exp(best - cost)]."""
    o = {**DEFAULTS, **kw}
    P = problem(circuit)
    dets = np.ascontiguousarray(detector_events, dtype=np.uint8)
    best, probs, stats = _run(dets, P.prior_llr, P.prior_llr * o["bp_scale"], P.edge_var, P.chk_ptr, P.var_edges, P.var_ptr,
                       P.max_deg, P.colbits, P.obsmask, P.zobrist, P.rank, o["schedule"], o["max_iter"], o["alpha"],
                       np.asarray(sorted(o["snaps"]), np.int64), o["n_single"], o["n_pair"], o["shortcut"],
                       *(P.triples() if o["n_expand"] > 0 else (np.zeros(P.n + 1, np.int32), np.zeros((0, 2), np.int32))),
                       o["n_expand"], 1.0 / o["temp"], _gammas(P.n, o), o["relay_T0"], o["relay_T"],
                       o["relay_S"], o["relay_alpha"], o["relay_osd"], _perturbations(P.n, o), o["perturb_mode"],
                       o["perturb_single"], o["perturb_pair"], P.edge_chk, o["dec_steps"], o["dec_iters"],
                       o["dec_branch"], o["dec_osd"])
    best_bits = ((best[:, None] >> np.arange(P.num_obs, dtype=np.uint8)) & 1).astype(np.uint8)
    # plain sums instead of a matmul: macOS Accelerate raises spurious warnings on subnormal probabilities
    marg = np.stack([probs[:, CLASS_BITS[:, j] == 1].sum(1) for j in range(P.num_obs)], axis=1)
    if return_stats:
        return best_bits, probs, marg, stats
    return best_bits, probs, marg


def decode(circuit, detector_events, **kw):
    o = {**DEFAULTS, **kw}
    best_bits, _, marg = decode_full(circuit, detector_events, **o)
    return (marg > 0.5).astype(np.uint8) if o["degenerate"] else best_bits
