"""Train a neural-network decoder (70 detector bits -> 4 observable logits) on freshly sampled shots.

    uv run python team/nn/train_nn.py --basis X --minutes 8 --arch mlp [--width 1024 --depth 4]
    uv run python team/nn/train_nn.py --basis X --minutes 8 --arch transformer [--d 128 --layers 4 --heads 4]

Architectures:
  mlp          residual MLP on the 70-bit vector
  transformer  one token per detector: embedding(fired or not) + embedding(detector id) + embeddings of its (x, y, t)
               coordinates and check type (X if x is odd), plus a CLS token whose output gives the 4 logits;
               pre-norm transformer encoder (GELU, no dropout: every training shot is fresh)

Training data is sampled from the competition circuit with Stim, never reused (so no overfitting); training seeds
start at 10_000 and never touch the validation seeds (2000/2001) or the shipped data/train set. The loss is per-bit
binary cross-entropy, which targets the per-bit marginal P(L_j = 1 | D): exactly what the per-bit LER metric rewards.
Weights are saved to team/nn/<basis>_<tag>.pt together with the settings needed to rebuild the model
(tag = w<width>_d<depth> for the MLP, tf_d<d>_l<layers>_h<heads> for the transformer).
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


class DetTransformer(nn.Module):
    def __init__(self, coords, n_out=4, d=128, layers=4, heads=4, ff=4, sparse=False):
        """coords: (n_det, 3) integer (x, y, t) detector coordinates from circuit.get_detector_coordinates().

        sparse=False: 70 tokens, one per detector (fired or not).
        sparse=True:  one token per FIRED detector only (~7 per shot), padded to the longest shot in the batch;
                      ~5-10x cheaper. Unfired detectors are implicit: the token set says exactly which ones fired.
        """
        super().__init__()
        self.sparse = sparse
        coords = torch.as_tensor(np.asarray(coords), dtype=torch.long)
        self.register_buffer("coords", coords)
        n_det = coords.shape[0]
        self.val = nn.Embedding(2, d)
        self.det = nn.Embedding(n_det, d)
        self.ex = nn.Embedding(int(coords[:, 0].max()) + 1, d)
        self.ey = nn.Embedding(int(coords[:, 1].max()) + 1, d)
        self.et = nn.Embedding(int(coords[:, 2].max()) + 1, d)
        self.etype = nn.Embedding(2, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, heads, ff * d, dropout=0.0, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_out))

    def _static(self):
        c = self.coords
        return self.det.weight + self.ex(c[:, 0]) + self.ey(c[:, 1]) + self.et(c[:, 2]) + self.etype(c[:, 0] % 2)

    def forward(self, x):
        static = self._static()  # (n_det, d)
        B, n_det = x.shape
        if not self.sparse:
            h = self.val(x.long()) + static
            h = torch.cat([self.cls.expand(B, -1, -1), h], dim=1)
            return self.out(self.enc(h)[:, 0])
        fired = x > 0.5
        count = fired.sum(1)
        L = max(int(count.max()), 1)
        # fired detector indices first (stable order), then pad
        key = (~fired).float() * n_det + torch.arange(n_det, device=x.device, dtype=torch.float32)
        idx = torch.argsort(key, dim=1)[:, :L]
        pad = torch.arange(L, device=x.device)[None, :] >= count[:, None]
        h = static[idx] + self.val.weight[1]
        h = h.masked_fill(pad[..., None], 0.0)
        h = torch.cat([self.cls.expand(B, -1, -1), h], dim=1)
        mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), pad], dim=1)
        return self.out(self.enc(h, src_key_padding_mask=mask)[:, 0])


def build_model(ck):
    """Rebuild a saved model from its checkpoint dict."""
    if ck.get("arch", "mlp") == "transformer":
        m = DetTransformer(ck["coords"], d=ck["d"], layers=ck["layers"], heads=ck["heads"], sparse=ck.get("sparse", False))
    else:
        m = ResMLP(width=ck["width"], depth=ck["depth"])
    m.load_state_dict(ck["state_dict"])
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", required=True, choices=["X", "Z"])
    ap.add_argument("--minutes", type=float, default=8)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--chunk", type=int, default=4_000_000)
    ap.add_argument("--arch", choices=["mlp", "transformer"], default="mlp")
    ap.add_argument("--sparse", action="store_true", help="transformer: tokens for fired detectors only")
    ap.add_argument("--bucket", action="store_true",
                    help="sparse transformer: group batches by #fired detectors (faster, but biased gradients; "
                         "trained ~2x worse per shot than random batches in our test)")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--amp", action="store_true", help="bfloat16 autocast for the forward pass")
    ap.add_argument("--bench", action="store_true", help="only measure training throughput (shots/s) and exit")
    args = ap.parse_args()

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    co = circuit(args.basis).get_detector_coordinates()
    coords = np.array([co[i] for i in range(len(co))], dtype=np.int64)
    if args.arch == "transformer":
        model = DetTransformer(coords, d=args.d, layers=args.layers, heads=args.heads, sparse=args.sparse).to(dev)
        tag = f"tf{'s' if args.sparse else ''}_d{args.d}_l{args.layers}_h{args.heads}"
    else:
        model = ResMLP(width=args.width, depth=args.depth).to(dev)
        tag = f"w{args.width}_d{args.depth}"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.basis}] {args.arch} {tag}: {n_params / 1e6:.2f}M parameters on {dev}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()

    def fwd(xs):
        if args.amp:
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
                return model(xs).float()
        return model(xs)
    k = "XZ".index(args.basis)
    vd, vo = generate(args.basis, 1_000_000, seed=2000 + k)
    vx = torch.tensor(vd[:100_000], dtype=torch.float32, device=dev)
    vy = vo[:100_000]

    def predict(xs):
        with torch.no_grad():
            return torch.cat([torch.sigmoid(fwd(xs[i:i + 16384])) for i in range(0, len(xs), 16384)]).cpu().numpy()

    if args.bench:
        order = np.argsort(vd[:100_000].sum(1), kind="stable")
        mid = order[len(order) // 2 - args.batch // 2: len(order) // 2 + args.batch // 2]  # typical-length batch
        xb = torch.tensor(vd[mid], dtype=torch.float32, device=dev)
        yb = torch.tensor(vo[mid], dtype=torch.float32, device=dev)
        for i in range(25):
            if i == 5:
                torch.mps.synchronize() if dev.type == "mps" else None
                tb = time.time()
            loss = lossf(fwd(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        torch.mps.synchronize() if dev.type == "mps" else None
        print(f"training throughput: {20 * args.batch / (time.time() - tb):,.0f} shots/s", flush=True)
        return
    sampler = circuit(args.basis).compile_detector_sampler(seed=10_000 + k)

    budget = args.minutes * 60
    t0 = time.time()
    step, seen, chunk_id = 0, 0, 0
    est_total_steps = None
    while time.time() - t0 < budget:
        d, o = sampler.sample(args.chunk, separate_observables=True)
        X = torch.tensor(d, dtype=torch.float32, device=dev)
        Y = torch.tensor(o, dtype=torch.float32, device=dev)
        if args.arch == "transformer" and args.sparse and args.bucket:
            srt = np.argsort(d.sum(1) + np.random.rand(args.chunk), kind="stable")  # by length, random within ties
            nb_ = args.chunk // args.batch
            blocks = srt[: nb_ * args.batch].reshape(nb_, args.batch)[np.random.permutation(nb_)]
            perm = torch.tensor(blocks.reshape(-1), device=dev)
        else:
            perm = torch.randperm(args.chunk, device=dev)
        model.train()
        for i in range(0, args.chunk - args.batch + 1, args.batch):
            frac = min(1.0, (time.time() - t0) / budget)
            for g in opt.param_groups:  # warmup then cosine decay over the time budget
                g["lr"] = args.lr * min(1.0, step / 500) * 0.5 * (1 + math.cos(math.pi * frac))
            idx = perm[i:i + args.batch]
            loss = lossf(fwd(X[idx]), Y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            seen += args.batch
            if time.time() - t0 >= budget:
                break
        chunk_id += 1
        model.eval()
        pv = predict(vx) > 0.5
        print(f"[{args.basis}] {time.time() - t0:6.0f}s  chunk {chunk_id:3d}  {seen / 1e6:7.1f}M shots  "
              f"loss {loss.item():.4f}  val LER {np.mean(pv != vy):.5f}", flush=True)

    path = os.path.join(HERE, f"{args.basis}_{tag}.pt")
    torch.save({"state_dict": model.state_dict(), "arch": args.arch, "width": args.width, "depth": args.depth,
                "d": args.d, "layers": args.layers, "heads": args.heads, "sparse": args.sparse, "coords": coords, "shots_seen": seen}, path)
    print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
