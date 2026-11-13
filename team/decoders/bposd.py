"""BP-OSD decoder for the [[18,4,4]] hackathon problem, written from scratch with numpy + numba.

Pipeline per shot (all shots in parallel with numba.prange):
  1. Belief propagation (normalised min-sum, flooding schedule) on the Tanner graph of H, starting from the DEM priors.
  2. If BP converges (H e = D) and OSD_ALWAYS is False, use BP's hard decision.
  3. Otherwise ordered statistics decoding: sort the columns by BP's posterior (most likely error first), pick the first
     linearly independent columns (an information set) and solve H e = D on them exactly (OSD-0). With OSD-CS, also try
     flipping single and paired columns among the most likely non-basis columns, and keep the solution with the lowest
     cost sum_j e_j * log((1-p_j)/p_j), i.e. the most likely error under the priors.
  4. Predict the observable flips O = L e mod 2.

Everything is GF(2) bit-twiddling on 70-bit columns (two uint64 words), so OSD here costs microseconds per shot:
H has only 70 rows, so the information set is found after scanning ~70-100 columns.

The only inputs are the circuit and the detector events (same signature as the hand-in decoder.py). No QEC decoder
library is used.
"""
import numpy as np
import numba as nb
import stim

from hackathon_2026 import problem_info

# Tunable parameters, tuned with team/tune_bposd.py on fresh shots (LER 0.0402 ± 0.0001 on 2×300k shots).
# Few BP iterations with strong normalisation work best: longer BP oscillates and gives OSD a worse column order.
MAX_ITER = 3          # BP iterations
ALPHA = 0.45          # min-sum normalisation factor (1.0 = plain min-sum)
OSD_ALWAYS = False    # run OSD even when BP converged (same LER here, 15% slower)
OSD_SINGLES = 200     # OSD-CS: try flipping each of the first OSD_SINGLES non-basis columns
OSD_PAIRS = 30        # OSD-CS: try flipping every pair among the first OSD_PAIRS non-basis columns


# ---------------------------------------------------------------- graph preparation (once per circuit, cached)

class _Problem:
    def __init__(self, circuit: stim.Circuit):
        info = problem_info(circuit)
        H, L, p = info["H"].astype(np.uint8), info["L"].astype(np.uint8), info["p"].astype(np.float64)
        m, n = H.shape
        assert m <= 128, "column bitsets assume at most 128 detectors"
        self.m, self.n = m, n
        self.prior_llr = np.log((1 - p) / p)
        self.weight = self.prior_llr.copy()  # cost of including column j in an error
        # check-major edge list (CSR over checks) and variable-major view of the same edges
        chk, var = np.nonzero(H)  # sorted by check
        self.edge_var = var.astype(np.int32)
        self.chk_ptr = np.zeros(m + 1, np.int32)
        np.add.at(self.chk_ptr, chk + 1, 1)
        self.chk_ptr = np.cumsum(self.chk_ptr).astype(np.int32)
        order = np.argsort(var, kind="stable")
        self.var_edges = order.astype(np.int32)
        self.var_ptr = np.zeros(n + 1, np.int32)
        np.add.at(self.var_ptr, var + 1, 1)
        self.var_ptr = np.cumsum(self.var_ptr).astype(np.int32)
        # columns of H as 128-bit sets, observables of each column as a 4-bit mask
        self.colbits = np.zeros((n, 2), np.uint64)
        for j in range(n):
            for i in np.flatnonzero(H[:, j]):
                self.colbits[j, i >> 6] |= np.uint64(1) << np.uint64(i & 63)
        self.obsmask = np.zeros(n, np.uint8)
        for k in range(L.shape[0]):
            self.obsmask |= (L[k].astype(np.uint8) << k)
        self.num_obs = L.shape[0]
        self.rank = _rank(self.colbits)


_CACHE: dict[str, _Problem] = {}


def _problem(circuit: stim.Circuit) -> _Problem:
    key = str(circuit)
    if key not in _CACHE:
        _CACHE[key] = _Problem(circuit)
    return _CACHE[key]


# ---------------------------------------------------------------- numba kernels

@nb.njit(cache=True)
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
def _solve(s0, s1, rank, bvec, piv, bcomb):
    """Reduce syndrome (s0, s1) by the basis; returns (ok, comb0, comb1) where comb marks the basis columns used."""
    c0 = np.uint64(0)
    c1 = np.uint64(0)
    for k in range(rank):
        if _getbit(s0, s1, piv[k]):
            s0 ^= bvec[k, 0]
            s1 ^= bvec[k, 1]
            c0 ^= bcomb[k, 0]
            c1 ^= bcomb[k, 1]
    return (s0 == 0 and s1 == 0), c0, c1


@nb.njit(cache=True)
def _cost_obs(c0, c1, rank, sel, weight, obsmask):
    cost = 0.0
    obs = np.uint8(0)
    for k in range(rank):
        if _getbit(c0, c1, k):
            cost += weight[sel[k]]
            obs ^= obsmask[sel[k]]
    return cost, obs


@nb.njit(cache=True)
def _decode_shot(syn, s0, s1, prior_llr, weight, edge_var, chk_ptr, var_edges, var_ptr, colbits, obsmask, full_rank,
                 max_iter, alpha, osd_always, n_single, n_pair):
    m = chk_ptr.shape[0] - 1
    n = var_ptr.shape[0] - 1
    E = edge_var.shape[0]
    v2c = np.empty(E)
    c2v = np.zeros(E)
    post = prior_llr.copy()
    hard = np.zeros(n, np.uint8)
    for e in range(E):
        v2c[e] = prior_llr[edge_var[e]]

    converged = False
    for it in range(max_iter):
        # check -> variable (normalised min-sum)
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
                s = sgn if v2c[e] >= 0 else -sgn
                c2v[e] = alpha * s * mag
        # variable -> check, posterior, hard decision
        for v in range(n):
            tot = prior_llr[v]
            for k in range(var_ptr[v], var_ptr[v + 1]):
                tot += c2v[var_edges[k]]
            post[v] = tot
            hard[v] = 1 if tot < 0 else 0
            for k in range(var_ptr[v], var_ptr[v + 1]):
                e = var_edges[k]
                v2c[e] = tot - c2v[e]
        # syndrome check
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

    best_cost = 1e300
    best_obs = np.uint8(0)
    if converged:
        best_cost = 0.0
        for v in range(n):
            if hard[v]:
                best_cost += weight[v]
                best_obs ^= obsmask[v]
        if not osd_always:
            return best_obs

    # ---- OSD: information set from columns sorted by posterior LLR (most likely to be an error first)
    order = np.argsort(post)
    bvec = np.zeros((full_rank, 2), np.uint64)
    bcomb = np.zeros((full_rank, 2), np.uint64)
    piv = np.zeros(full_rank, np.int64)
    sel = np.zeros(full_rank, np.int64)
    cand = np.empty(max(n_single, n_pair), np.int64)
    n_cand = 0
    rank = 0
    for idx in range(n):
        j = order[idx]
        if rank == full_rank:
            if n_cand < cand.shape[0]:
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
        elif n_cand < cand.shape[0]:
            cand[n_cand] = j  # dependent column: candidate for OSD-CS flips
            n_cand += 1

    ok, c0, c1 = _solve(s0, s1, rank, bvec, piv, bcomb)
    if ok:
        cost, obs = _cost_obs(c0, c1, rank, sel, weight, obsmask)
        if cost < best_cost:
            best_cost, best_obs = cost, obs
    # OSD-CS order 1: flip one candidate column
    for a in range(min(n_single, n_cand)):
        ja = cand[a]
        ok, c0, c1 = _solve(s0 ^ colbits[ja, 0], s1 ^ colbits[ja, 1], rank, bvec, piv, bcomb)
        if ok:
            cost, obs = _cost_obs(c0, c1, rank, sel, weight, obsmask)
            cost += weight[ja]
            if cost < best_cost:
                best_cost, best_obs = cost, obs ^ obsmask[ja]
    # OSD-CS order 2: flip two candidate columns
    npair = min(n_pair, n_cand)
    for a in range(npair):
        ja = cand[a]
        for b in range(a + 1, npair):
            jb = cand[b]
            ok, c0, c1 = _solve(s0 ^ colbits[ja, 0] ^ colbits[jb, 0], s1 ^ colbits[ja, 1] ^ colbits[jb, 1],
                                rank, bvec, piv, bcomb)
            if ok:
                cost, obs = _cost_obs(c0, c1, rank, sel, weight, obsmask)
                cost += weight[ja] + weight[jb]
                if cost < best_cost:
                    best_cost, best_obs = cost, obs ^ obsmask[ja] ^ obsmask[jb]
    return best_obs


@nb.njit(parallel=True, cache=True)
def _decode_all(dets, prior_llr, weight, edge_var, chk_ptr, var_edges, var_ptr, colbits, obsmask, full_rank,
                max_iter, alpha, osd_always, n_single, n_pair):
    shots, m = dets.shape
    out = np.zeros(shots, np.uint8)
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
            continue  # trivial syndrome: predict no flips
        out[i] = _decode_shot(syn, s0, s1, prior_llr, weight, edge_var, chk_ptr, var_edges, var_ptr, colbits, obsmask,
                              full_rank, max_iter, alpha, osd_always, n_single, n_pair)
    return out


# ---------------------------------------------------------------- public API

def decode(circuit: stim.Circuit, detector_events: np.ndarray, max_iter=None, alpha=None, osd_always=None,
           osd_singles=None, osd_pairs=None) -> np.ndarray:
    """(shots, 70) bool detector events -> (shots, 4) uint8 predicted observable flips."""
    P = _problem(circuit)
    dets = np.ascontiguousarray(detector_events, dtype=np.uint8)
    masks = _decode_all(dets, P.prior_llr, P.weight, P.edge_var, P.chk_ptr, P.var_edges, P.var_ptr, P.colbits,
                        P.obsmask, P.rank,
                        MAX_ITER if max_iter is None else max_iter,
                        ALPHA if alpha is None else alpha,
                        OSD_ALWAYS if osd_always is None else osd_always,
                        OSD_SINGLES if osd_singles is None else osd_singles,
                        OSD_PAIRS if osd_pairs is None else osd_pairs)
    return ((masks[:, None] >> np.arange(P.num_obs, dtype=np.uint8)) & 1).astype(np.uint8)
