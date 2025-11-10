# train.py
import os
import json
import yaml
import argparse
from datetime import datetime

import torch
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import pandas as pd

from data import make_loaders
from models.autoencoder import UNetAE
from losses import L1_SSIM_Loss, topk_error

from datetime import datetime, timezone

import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)


# ---------------- Utils ----------------
def set_seed(s: int):
    import random, numpy as np, torch
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def pick_device(pref: str = "cuda"):
    pref = (pref or "cuda").lower()
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def save_ckpt(path, model, opt, epoch, best_val):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "best_val": best_val},
        path,
    )

def percentile_threshold(scores, pct=95):
    # scores: list/array of floats
    arr = np.asarray(scores, dtype=np.float32)
    pct = float(pct)
    pct = min(max(pct, 0.0), 100.0)
    return float(np.percentile(arr, pct))


def resolve_cfg(global_cfg: dict, category: str) -> dict:
    """Merge global cfg with per-category overrides (if present)."""
    cfg = dict(global_cfg)  # shallow copy
    overrides = (global_cfg.get("per_category") or {}).get(category, {})
    cfg.update(overrides)  # only keys listed in per_category override
    cfg["category"] = category
    return cfg


# ---------------- Train / Eval ----------------
def _get_images_from_batch(batch):
    # Supports either dict batches or tuple batches from your MVTECDataset
    if isinstance(batch, dict):
        return batch["image"]
    # tuple like: (x, y, category, path)
    return batch[0]

def do_epoch(model, loader, loss_fn, opt=None, device=torch.device("cpu")):
    is_train = opt is not None
    model.train(is_train)
    running, n = 0.0, 0
    for batch in loader:
        x = _get_images_from_batch(batch).to(device, non_blocking=True)
        if is_train:
            opt.zero_grad(set_to_none=True)
        y_hat = model(x)
        loss = loss_fn(y_hat, x)
        if is_train:
            loss.backward()
            opt.step()
        running += loss.item() * x.size(0)
        n += x.size(0)
    return running / max(n, 1)

@torch.no_grad()
def collect_val_scores(model, loader, device, topk_frac):
    """Compute per-image reconstruction error using top-k pixel errors.
       Rebuild a 0-worker loader to avoid CUDA+fork issues, and gracefully
       fall back to CPU if CUDA init fails mid-flight.
    """
    from torch.utils.data import DataLoader

    # Rebuild a single-process val loader (no workers, no pinning)
    val_ds = loader.dataset
    val_loader_sp = DataLoader(
        val_ds,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    def _run(the_device):
        model.eval().to(the_device)
        out_scores = []
        for batch in val_loader_sp:
            # batch comes from your MVTECDataset: (x, y, cat, path)
            x = batch[0].to(the_device, non_blocking=False)
            y_hat = model(x)
            out = topk_error(y_hat, x, topk_frac=topk_frac)  # (scores, heatmap?) or tensor
            s = out[0] if isinstance(out, (tuple, list)) else out
            out_scores.extend(s.detach().cpu().tolist())
        return out_scores

    try:
        # try on the requested device (CUDA)
        return _run(device)
    except RuntimeError as e:
        msg = str(e).lower()
        if "cuda" in msg or "accelerator" in msg:
            print("[collect_val_scores] CUDA issue detected; falling back to CPU just for scoring.")
            torch.cuda.empty_cache()
            return _run(torch.device("cpu"))
        raise

def train_one(global_cfg: dict, category: str):
    cfg = resolve_cfg(global_cfg, category)

    # housekeeping / seeds / device
    set_seed(int(cfg.get("seed", 42)))
    device = pick_device(cfg.get("device", "cuda"))

    print("[train.py] using CSV:", os.path.abspath(cfg["index_csv"]), "| category:", category)

    # data
    train_loader, val_loader = make_loaders(
        index_csv=cfg["index_csv"],
        category=category,
        image_size=cfg["image_size"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        val_split=cfg["val_split"],
        augment_prob=cfg["augment_prob"],
        seed=cfg.get("seed", 42),
    )

    # model / loss / opt
    model = UNetAE(base=int(cfg["base_channels"]), latent=int(cfg["latent_dim"])).to(device)
    loss_fn = L1_SSIM_Loss(float(cfg["weight_l1"]), float(cfg["weight_ssim"]))
    opt = Adam(model.parameters(), lr=float(cfg["lr"]))

    # out dirs
    run_dir = os.path.join(cfg["out_dir"], category)
    os.makedirs(run_dir, exist_ok=True)

    # save the resolved config used for this category
    with open(os.path.join(run_dir, "config_updated.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)

    best_val = float("inf")
    epochs = int(cfg["epochs"])

    for ep in range(1, epochs + 1):
        tr = do_epoch(model, train_loader, loss_fn, opt=opt, device=device)
        va = do_epoch(model, val_loader, loss_fn, opt=None, device=device)

        print(f"[{category}] epoch {ep:03d}/{epochs} | train={tr:.5f} | val={va:.5f}")

        if va < best_val:
            best_val = va
            save_ckpt(os.path.join(run_dir, "best.pth"), model, opt, ep, best_val)

        # optional: checkpoint every 10 epochs too
        if ep % 10 == 0 or ep == epochs:
            save_ckpt(os.path.join(run_dir, f"ep{ep:03d}.pth"), model, opt, ep, best_val)

    # ---- derive threshold from validation (good) images
    # reload best and compute scores
    ckpt = torch.load(os.path.join(run_dir, "best.pth"), map_location=device)
    model.load_state_dict(ckpt["model"])

    val_scores = collect_val_scores(model, val_loader, device, topk_frac=float(cfg["topk_frac"]))
    thr = percentile_threshold(val_scores, pct=float(cfg["threshold_percentile"]))

    with open(os.path.join(run_dir, "threshold.json"), "w") as f:
        json.dump(
            {
                "category": category,
                "topk_frac": float(cfg["topk_frac"]),
                "threshold_percentile": float(cfg["threshold_percentile"]),
                "val_good_scores": {
                    "mean": float(np.mean(val_scores)),
                    "std": float(np.std(val_scores)),
                    "n": int(len(val_scores)),
                },
                "threshold": float(thr),
                "generated_at": datetime.now(timezone.utc).isoformat(),

            },
            f,
            indent=2,
        )

    print(f"[{category}] Saved best model + threshold at: {run_dir}")
    return run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--category", type=str, default="global",
                    help="'global' for pooled model, a single category name, or 'all' to loop all categories in index_csv")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        global_cfg = yaml.safe_load(f)

    # Normalize types that YAML sometimes reads as str
    def _to_float(k):
        if k in global_cfg:
            global_cfg[k] = float(global_cfg[k])
    for k in ["lr", "weight_l1", "weight_ssim", "val_split", "augment_prob", "topk_frac"]:
        _to_float(k)
    def _to_int(k):
        if k in global_cfg:
            global_cfg[k] = int(global_cfg[k])
    for k in ["epochs", "batch_size", "num_workers", "threshold_percentile", "base_channels", "latent_dim"]:
        _to_int(k)

    if args.category == "all":
        cats = pd.read_csv(global_cfg["index_csv"])["category"].unique().tolist()
        for c in cats:
            train_one(global_cfg, c)
    else:
        # "global" (pooled) or a single category name
        train_one(global_cfg, args.category)


if __name__ == "__main__":
    main()
