"""Train the stacked NN: a residual MLP that corrects the cheap degenerate BP-OSD pass.

    uv run python team/nn/train_stack.py --basis X --minutes 20 [--width 512 --depth 3]

Inputs per shot (89 features): 70 detector bits, log of the decoder's 16-class posterior (clipped at 1e-6),
the decoder's best-error cost / 30 and log normaliser, and the number of fired detectors / 20.
Output logits = the decoder's own per-bit logit (marginals clipped to [1e-6, 1 - 1e-6]) + the network's output.
The network's last layer starts at zero, so training starts exactly at the decoder's accuracy and only learns a
correction. Data: data/generated/stack_<basis>/chunk_*.npz from make_stack_features.py; chunks that appear while
training are picked up. Saves team/nn/stack_<basis>_w<width>_d<depth>.pt.
"""
import argparse
import glob
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from common import GEN_DIR  # noqa: E402
from train_nn import ResMLP  # noqa: E402

CLASS_BITS = torch.tensor(((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(np.float32))
N_FEAT = 70 + 16 + 2 + 1


def load(path, dev):
    f = np.load(path)
    d = np.unpackbits(f["d"], axis=1, count=70).astype(np.float32)
    o = np.unpackbits(f["o"], axis=1, count=4, bitorder="little").astype(np.float32)
    return (torch.tensor(d, device=dev), torch.tensor(f["p"].astype(np.float32), device=dev),
            torch.tensor(f["st"].astype(np.float32), device=dev), torch.tensor(o, device=dev))


def featurize(d, p, st):
    """Returns (features (B, 89), decoder per-bit logit (B, 4))."""
    logp = torch.log(p.clamp(min=1e-6))
    x = torch.cat([d, logp, st[:, :1] / 30.0, st[:, 1:2], d.sum(1, keepdim=True) / 20.0], dim=1)
    marg = (p @ CLASS_BITS.to(p.device)).clamp(1e-6, 1 - 1e-6)
    return x, torch.log(marg) - torch.log1p(-marg)


class Stacked(nn.Module):
    def __init__(self, width=512, depth=3):
        super().__init__()
        self.net = ResMLP(n_in=N_FEAT, n_out=4, width=width, depth=depth)
        nn.init.zeros_(self.net.out[1].weight)
        nn.init.zeros_(self.net.out[1].bias)

    def forward(self, d, p, st):
        x, dec_logit = featurize(d, p, st)
        return dec_logit + self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", required=True, choices=["X", "Z"])
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=5e-4)
    args = ap.parse_args()
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    folder = os.path.join(GEN_DIR, f"stack_{args.basis}")
    model = Stacked(args.width, args.depth).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    vd, vp, vst, vo = load(os.path.join(folder, "val.npz"), dev)
    vo_np = vo.cpu().numpy().astype(bool)
    dec_val = ((vp @ CLASS_BITS.to(dev)) > 0.5).cpu().numpy()
    print(f"[{args.basis}] stacked {args.width}x{args.depth} on {dev}; decoder val LER {np.mean(dec_val != vo_np):.5f}",
          flush=True)

    def val_ler():
        model.eval()
        with torch.no_grad():
            z = torch.cat([model(vd[i:i + 16384], vp[i:i + 16384], vst[i:i + 16384]) for i in range(0, len(vd), 16384)])
        model.train()
        return float(np.mean((z > 0).cpu().numpy() != vo_np))

    t0, step, seen, used = time.time(), 0, 0, 0
    budget = args.minutes * 60
    while time.time() - t0 < budget:
        chunks = sorted(glob.glob(os.path.join(folder, "chunk_*.npz")))
        if not chunks:
            time.sleep(5)
            continue
        path = chunks[used % len(chunks)]
        used += 1
        d, p, st, o = load(path, dev)
        perm = torch.randperm(len(d), device=dev)
        for i in range(0, len(d) - args.batch + 1, args.batch):
            frac = min(1.0, (time.time() - t0) / budget)
            for g in opt.param_groups:
                g["lr"] = args.lr * min(1.0, step / 300) * 0.5 * (1 + math.cos(math.pi * frac))
            idx = perm[i:i + args.batch]
            loss = lossf(model(d[idx], p[idx], st[idx]), o[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            seen += args.batch
        print(f"[{args.basis}] {time.time() - t0:6.0f}s  {seen / 1e6:6.1f}M shots ({len(chunks)} chunks on disk)  "
              f"loss {loss.item():.4f}  val LER {val_ler():.5f}", flush=True)
    path = os.path.join(HERE, f"stack_{args.basis}_w{args.width}_d{args.depth}.pt")
    torch.save({"state_dict": model.state_dict(), "width": args.width, "depth": args.depth, "shots_seen": seen}, path)
    print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
