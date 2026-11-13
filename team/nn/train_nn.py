"""Train a neural-network decoder (residual MLP: 70 detector bits -> 4 observable logits) on freshly sampled shots.

    uv run python team/nn/train_nn.py --basis X --minutes 8 [--width 1024 --depth 4]

Training data is sampled from the competition circuit with Stim, never reused (so no overfitting); training seeds
start at 10_000 and never touch the validation seeds (2000/2001) or the shipped data/train set. The loss is per-bit
binary cross-entropy, which targets the per-bit marginal P(L_j = 1 | D): exactly what the per-bit LER metric rewards.
Weights are saved to team/nn/<basis>_w<width>_d<depth>.pt together with the settings needed to rebuild the model.
"""
import argparse
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from common import circuit, generate  # noqa: E402


class ResMLP(nn.Module):
    def __init__(self, n_in=70, n_out=4, width=1024, depth=4):
        super().__init__()
        self.inp = nn.Linear(n_in, width)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
            for _ in range(depth))
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, n_out))

    def forward(self, x):
        h = self.inp(x)
        for blk in self.blocks:
            h = h + blk(h)
        return self.out(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", required=True, choices=["X", "Z"])
    ap.add_argument("--minutes", type=float, default=8)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--chunk", type=int, default=4_000_000)
    args = ap.parse_args()

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    model = ResMLP(width=args.width, depth=args.depth).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    k = "XZ".index(args.basis)
    vd, vo = generate(args.basis, 1_000_000, seed=2000 + k)
    vx = torch.tensor(vd[:100_000], dtype=torch.float32, device=dev)
    vy = vo[:100_000]
    sampler = circuit(args.basis).compile_detector_sampler(seed=10_000 + k)

    budget = args.minutes * 60
    t0 = time.time()
    step, seen, chunk_id = 0, 0, 0
    est_total_steps = None
    while time.time() - t0 < budget:
        d, o = sampler.sample(args.chunk, separate_observables=True)
        X = torch.tensor(d, dtype=torch.float32, device=dev)
        Y = torch.tensor(o, dtype=torch.float32, device=dev)
        perm = torch.randperm(args.chunk, device=dev)
        model.train()
        for i in range(0, args.chunk - args.batch + 1, args.batch):
            frac = min(1.0, (time.time() - t0) / budget)
            for g in opt.param_groups:  # warmup then cosine decay over the time budget
                g["lr"] = args.lr * min(1.0, step / 500) * 0.5 * (1 + math.cos(math.pi * frac))
            idx = perm[i:i + args.batch]
            loss = lossf(model(X[idx]), Y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            seen += args.batch
            if time.time() - t0 >= budget:
                break
        chunk_id += 1
        model.eval()
        with torch.no_grad():
            pv = (torch.sigmoid(model(vx)) > 0.5).cpu().numpy()
        print(f"[{args.basis}] {time.time() - t0:6.0f}s  chunk {chunk_id:3d}  {seen / 1e6:7.1f}M shots  "
              f"loss {loss.item():.4f}  val LER {np.mean(pv != vy):.5f}", flush=True)

    path = os.path.join(HERE, f"{args.basis}_w{args.width}_d{args.depth}.pt")
    torch.save({"state_dict": model.state_dict(), "width": args.width, "depth": args.depth, "shots_seen": seen}, path)
    print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
