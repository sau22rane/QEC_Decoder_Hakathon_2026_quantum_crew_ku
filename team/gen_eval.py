"""Compare candidate generators on the shots that matter (the cascade's uncertain shots).

Builds two fixed sets once (cached): (A) uncertain shots WITH their true error, from the DEM sampler, for the capture
rate = fraction of shots whose true error is in the candidate list (= average posterior mass captured); (B) uncertain
shots from the shared validation seeds for the paired LER. Then runs every config in CONFIGS on both.

    python team/gen_eval.py <suite>
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "decoders")]
import numpy as np
import bposd_walk as bposd3  # same API as bposd2, plus walk_* options and truth=
from common import circuit, generate
from diagnose import true_errors

CACHE = os.path.join(os.path.dirname(HERE), "data", "generated")
TAU = 0.4
CHEAP = dict(schedule=1, max_iter=2, alpha=0.45, snaps=(2,), n_single=200, n_pair=30, degenerate=True, shortcut=True)
HEAVY = dict(schedule=1, max_iter=3, alpha=0.45, snaps=(1, 2, 3), n_single=800, n_pair=60, degenerate=True, shortcut=False)


def sets(n_truth=60000, n_val=150000):
    path = os.path.join(CACHE, f"gen_eval_sets_{n_truth}_{n_val}.npz")
    if os.path.exists(path):
        f = np.load(path)
        return {k: f[k] for k in f.files}
    out = {}
    for k, b in enumerate("XZ"):
        c = circuit(b)
        ds, os_, es = [], [], []
        for part in range(n_truth // 20000):
            d, o, e = true_errors(b, 20000, seed=6000 + 10 * part + k)
            u = np.abs(bposd3.decode_full(c, d, **CHEAP)[2] - 0.5).min(1) < TAU
            ds.append(d[u]); os_.append(o[u]); es.append(e[u])
        out[f"td_{b}"], out[f"to_{b}"], out[f"te_{b}"] = np.concatenate(ds), np.concatenate(os_), np.concatenate(es)
        d, o = generate(b, 1_000_000, seed=2000 + k)
        d, o = d[:n_val], o[:n_val]
        marg = bposd3.decode_full(c, d, **CHEAP)[2]
        u = np.abs(marg - 0.5).min(1) < TAU
        out[f"vd_{b}"], out[f"vo_{b}"] = d[u], o[u]
        out[f"sure_err_{b}"] = np.array([((marg[~u] > 0.5) != o[~u]).sum(), len(d)])
    np.savez(path, **out)
    return out


def run(configs, ref="heavy"):
    S = sets()
    n_total = sum(S[f"sure_err_{b}"][1] for b in "XZ")
    sure_err = sum(S[f"sure_err_{b}"][0] for b in "XZ")
    nu = sum(len(S[f"vd_{b}"]) for b in "XZ")
    print(f"confident shots contribute LER {sure_err / (4 * n_total):.5f} (not re-decoded here)")
    print(f"validation: {n_total} shots, {nu} uncertain ({nu / n_total:.1%}); truth set: {sum(len(S[f'td_{b}']) for b in 'XZ')} uncertain shots")
    refw = None
    for name, cfg in configs.items():
        for b in "XZ":
            bposd3.decode_full(circuit(b), S[f"vd_{b}"][:64], **cfg)
        cap, nd = [], []
        for b in "XZ":
            st = bposd3.decode_full(circuit(b), S[f"td_{b}"], truth=S[f"te_{b}"], **cfg)[3]
            cap.append(st[:, 2]); nd.append(st[:, 3])
        cap, nd = np.concatenate(cap), np.concatenate(nd)
        t0 = time.perf_counter()
        w = np.concatenate([((bposd3.decode_full(circuit(b), S[f"vd_{b}"], **cfg)[2] > 0.5) != S[f"vo_{b}"]).sum(1) for b in "XZ"])
        dt = time.perf_counter() - t0
        ler = (w.sum() + sure_err) / (4 * n_total)
        line = f"{name:<46} capture {cap.mean():.4f}  cands {nd.mean():8.0f}  LER {ler:.5f}  {1e6 * dt / len(w):7.0f} us/unsure-shot"
        if refw is None:
            refw = w
        else:
            d = (w - refw) / 4.0
            line += f"  dLER {d.sum() / n_total:+.5f} ± {d.std() * np.sqrt(len(d)) / n_total:.5f}"
        print(line, flush=True)


if __name__ == "__main__":
    suite = sys.argv[1] if len(sys.argv) > 1 else "walk"
    H = HEAVY
    SUPER = {**H, "n_single": 1500, "n_pair": 80, "n_perturb": 32, "perturb_single": 200, "perturb_pair": 30}
    C = {"heavy (3 ord, 800/60)": H}
    if suite == "walk":
        C["team SUPER (1500/80 + 32 perturbed)"] = SUPER
        for ex, mx in [(50, 5000), (200, 20000), (1000, 60000)]:
            C[f"heavy + walk exp {ex} max {mx}"] = dict(H, walk_exp=ex, walk_max=mx)
        C["heavy + walk 200, seeds 8"] = dict(H, walk_exp=200, walk_seeds=8)
        C["heavy + walk 200, seeds 400"] = dict(H, walk_exp=200, walk_seeds=400)
        C["SUPER + walk exp 200"] = dict(SUPER, walk_exp=200)
    if suite == "walk2":
        C["team SUPER"] = SUPER
        C["SUPER + walk 200"] = dict(SUPER, walk_exp=200)
        C["SUPER + walk 1000 max 60k"] = dict(SUPER, walk_exp=1000, walk_max=60000)
        C["SUPER + walk 3000 max 100k gap 18"] = dict(SUPER, walk_exp=3000, walk_max=100000, walk_gap=18.0)
        C["heavy + walk 1000 gap 20"] = dict(H, walk_exp=1000, walk_max=60000, walk_gap=20.0)
        C["heavy + walk 3000 max 100k gap 18"] = dict(H, walk_exp=3000, walk_max=100000, walk_gap=18.0)
        C["heavy 200/30 x3 + walk 1000"] = dict(H, n_single=200, n_pair=30, walk_exp=1000, walk_max=60000)
    run(C)
