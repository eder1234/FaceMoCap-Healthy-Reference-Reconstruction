#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_static_neutral_ae_v3.py

Train a static neutral-face denoising autoencoder on healthy-train neutral FaceMoCap geometries.

Goal
----
Learn a healthy neutral morphology manifold. At inference, the model can project a
pathological/asymmetric neutral face toward a healthier neutral configuration.

Input
-----
healthy_twin_dataset_v3/static_neutral/N_train_healthy.npy  (S,105,3)
healthy_twin_dataset_v3/static_neutral/M_train_healthy.npy  (S,105)

Output
------
static_neutral_ae_v3/model.pt
static_neutral_ae_v3/normalization.json
static_neutral_ae_v3/train_history.csv
static_neutral_ae_v3/config.json

Recommended run
---------------
python train_static_neutral_ae_v3.py \
  --dataset_dir /path/to/healthy_twin_dataset_v3 \
  --out_dir /path/to/static_neutral_ae_v3 \
  --epochs 300 --batch_size 16 --patience 40
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset, random_split
except Exception as exc:  # pragma: no cover
    raise RuntimeError("This script requires PyTorch.") from exc


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_json(p: Path, obj: object) -> None:
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def compute_norm(N: np.ndarray, M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    valid = M.astype(bool)
    mean, std = [], []
    for c in range(3):
        vals = N[..., c][valid]
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            mean.append(0.0); std.append(1.0)
        else:
            mean.append(float(vals.mean()))
            s = float(vals.std())
            std.append(s if s > 1e-8 else 1.0)
    return np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)


def normalize(N: np.ndarray, M: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    Z = (N.astype(np.float32) - mean.reshape(1, 1, 3)) / std.reshape(1, 1, 3)
    Z = np.where(M[..., None].astype(bool), Z, 0.0)
    return np.where(np.isfinite(Z), Z, 0.0).astype(np.float32)


class NeutralDataset(Dataset):
    def __init__(self, N: np.ndarray, M: np.ndarray):
        self.N = torch.from_numpy(N.astype(np.float32))
        self.M = torch.from_numpy(M.astype(np.float32))
    def __len__(self) -> int: return int(self.N.shape[0])
    def __getitem__(self, idx: int): return self.N[idx], self.M[idx]


class StaticNeutralAE(nn.Module):
    def __init__(self, n_markers: int = 105, hidden: int = 256, latent: int = 48, dropout: float = 0.20):
        super().__init__()
        self.n_markers = int(n_markers)
        in_dim = n_markers * 4
        out_dim = n_markers * 3
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, latent), nn.GELU(),
            nn.Linear(latent, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, n: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        B, N, _ = n.shape
        x = torch.cat([n, m.unsqueeze(-1)], dim=-1).reshape(B, N * 4)
        y = self.net(x).reshape(B, N, 3)
        return y


def corrupt(n: torch.Tensor, m: torch.Tensor, args: argparse.Namespace) -> Tuple[torch.Tensor, torch.Tensor]:
    B, N, _ = n.shape
    keep = m > 0.5
    rand_marker_drop = torch.rand(B, N, device=n.device) < args.marker_mask_prob
    keep = keep & (~rand_marker_drop)
    # Whole-half random drop encourages inference from partial contralateral/central face.
    if args.half_mask_prob > 0:
        for b in range(B):
            if torch.rand((), device=n.device) < args.half_mask_prob:
                if torch.rand((), device=n.device) < 0.5:
                    keep[b, : N // 2] = False
                else:
                    keep[b, N // 2 :] = False
    noisy = n + torch.randn_like(n) * args.noise_std
    noisy = torch.where(keep.unsqueeze(-1), noisy, torch.zeros_like(noisy))
    return noisy, keep.float()


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.unsqueeze(-1)
    denom = valid.sum() * target.shape[-1] + 1e-8
    return (((pred - target) ** 2) * valid).sum() / denom


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--latent", type=int, default=48)
    ap.add_argument("--dropout", type=float, default=0.20)
    ap.add_argument("--marker_mask_prob", type=float, default=0.30)
    ap.add_argument("--half_mask_prob", type=float, default=0.20)
    ap.add_argument("--noise_std", type=float, default=0.03)
    ap.add_argument("--val_fraction", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return ap.parse_args()


def main() -> None:
    args = parse_args(); set_seed(args.seed)
    out_dir = Path(args.out_dir); ensure_dir(out_dir)
    static_dir = Path(args.dataset_dir) / "static_neutral"
    N = np.load(static_dir / "N_train_healthy.npy").astype(np.float32)
    M = np.load(static_dir / "M_train_healthy.npy").astype(np.uint8)
    if N.shape[0] < 2:
        raise RuntimeError("Need at least two healthy train neutral samples.")
    mean, std = compute_norm(N, M)
    Z = normalize(N, M, mean, std)
    dataset = NeutralDataset(Z, M)
    n_val = max(1, int(round(len(dataset) * args.val_fraction)))
    n_train = max(1, len(dataset) - n_val)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    model = StaticNeutralAE(n_markers=N.shape[1], hidden=args.hidden, latent=args.latent, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = float("inf"); best_state = None; bad = 0; hist = []
    for epoch in range(1, args.epochs + 1):
        model.train(); tr = []
        for n, m in train_loader:
            n = n.to(device); m = m.to(device)
            nin, minp = corrupt(n, m, args)
            pred = model(nin, minp)
            loss = masked_mse(pred, n, m)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tr.append(float(loss.detach().cpu()))
        model.eval(); va = []
        with torch.no_grad():
            for n, m in val_loader:
                n = n.to(device); m = m.to(device)
                pred = model(n, m)  # uncorrupted val pass
                va.append(float(masked_mse(pred, n, m).detach().cpu()))
        row = {"epoch": epoch, "train_loss": float(np.mean(tr)), "val_loss": float(np.mean(va))}
        hist.append(row)
        print(f"epoch {epoch:04d} train={row['train_loss']:.6f} val={row['val_loss']:.6f}")
        if row["val_loss"] < best - 1e-6:
            best = row["val_loss"]; best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}; bad = 0
        else:
            bad += 1
        if bad >= args.patience:
            print(f"Early stopping at epoch {epoch}"); break
    if best_state is not None:
        model.load_state_dict(best_state)
    pd.DataFrame(hist).to_csv(out_dir / "train_history.csv", index=False)
    write_json(out_dir / "normalization.json", {"mean": mean.tolist(), "std": std.tolist()})
    write_json(out_dir / "config.json", vars(args))
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_markers": int(N.shape[1]), "hidden": args.hidden, "latent": args.latent, "dropout": args.dropout,
        "mean": mean, "std": std, "config": vars(args),
    }, out_dir / "model.pt")
    print(f"[OK] Wrote static neutral AE: {out_dir}")


if __name__ == "__main__":
    main()
