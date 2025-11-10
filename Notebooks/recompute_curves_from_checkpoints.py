# recompute_curves_from_checkpoints.py
import os, argparse, glob, json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data import make_loaders
from models.autoencoder import UNetAE
from losses import L1_SSIM_Loss

def _get_images_from_batch(batch):
    # Works with (x, y, cat, path) or dict-style batches
    if isinstance(batch, (list, tuple)):
        return batch[0]
    if isinstance(batch, dict):
        for k in ("image", "x", "img", "inputs"):
            if k in batch:
                return batch[k]
    raise KeyError("Could not find image tensor in batch (tried keys: image/x/img/inputs).")

@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        x = _get_images_from_batch(batch).to(device, non_blocking=True)
        y = model(x)
        loss = loss_fn(y, x)
        bs = x.size(0)
        total += float(loss.item()) * bs
        n += bs
    return total / max(n, 1)

def plot_curves(hist_csv, out_png):
    df = pd.read_csv(hist_csv)
    plt.figure(figsize=(7,4))
    plt.plot(df["epoch"], df["train_loss"], label="train")
    plt.plot(df["epoch"], df["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(os.path.basename(os.path.dirname(out_png)))
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device=="cpu" or torch.cuda.is_available()) else "cpu")

    runs = sorted([d for d in glob.glob(os.path.join(args.runs_dir, "*")) if os.path.isdir(d)])
    plots_dir = os.path.join(args.runs_dir, "_plots")
    os.makedirs(plots_dir, exist_ok=True)

    any_done = False
    for run_dir in runs:
        name = os.path.basename(run_dir)
        if name.startswith("_"):
            print(f"Skip {run_dir}: helper dir")
            continue

        cfg_path = os.path.join(run_dir, "config_updated.yaml")
        if not os.path.exists(cfg_path):
            print(f"Skip {run_dir}: no config_updated.yaml")
            continue

        with open(cfg_path, "r") as f:
            cfg = json.loads(json.dumps(__import__("yaml").safe_load(f)))  # normalize YAML types

        # Build loaders (we only need train/val, no aug on val)
        tr_loader, va_loader = make_loaders(
            index_csv=cfg["index_csv"],
            category=cfg.get("category", name),  # fallback to folder name
            image_size=cfg["image_size"],
            batch_size=cfg["batch_size"],
            num_workers=0,              # safer for post-hoc eval
            val_split=cfg["val_split"],
            augment_prob=0.0,           # no aug when evaluating losses
            seed=int(cfg.get("seed", 42)),
        )

        # Model & loss
        model = UNetAE(base=int(cfg["base_channels"]), latent=int(cfg["latent_dim"])).to(device)
        loss_fn = L1_SSIM_Loss(float(cfg["weight_l1"]), float(cfg["weight_ssim"]))

        # Find checkpoints (epXXX.pth + best.pth)
        ckpts = sorted(glob.glob(os.path.join(run_dir, "ep*.pth")))
        best_ckpt = os.path.join(run_dir, "best.pth")
        if os.path.exists(best_ckpt):
            ckpts.append(best_ckpt)

        if not ckpts:
            print(f"Skip {run_dir}: no checkpoints found")
            continue

        rows = []
        for ck in ckpts:
            # Load state dict robustly
            state = torch.load(ck, map_location=device)
            if isinstance(state, dict) and "model" in state:
                model.load_state_dict(state["model"])
                epoch = int(state.get("epoch", -1))
            elif isinstance(state, dict):
                model.load_state_dict(state)
                # try infer epoch from filename epXYZ.pth
                base = os.path.basename(ck)
                epoch = int(base[2:5]) if base.startswith("ep") and base[2:5].isdigit() else -1
            else:
                raise RuntimeError(f"Unexpected checkpoint format: {ck}")

            tr_loss = eval_epoch(model, tr_loader, loss_fn, device)
            va_loss = eval_epoch(model, va_loader, loss_fn, device)

            rows.append({"epoch": epoch, "ckpt": os.path.basename(ck),
                         "train_loss": tr_loss, "val_loss": va_loss})
            print(f"[{name}] {os.path.basename(ck)} -> train={tr_loss:.5f} val={va_loss:.5f}")

        # Sort by epoch, keep best at the end if epoch=-1
        rows_sorted = sorted(rows, key=lambda r: (9999 if r["epoch"] < 0 else r["epoch"], r["ckpt"]))
        hist_path = os.path.join(run_dir, "history.csv")
        pd.DataFrame(rows_sorted).to_csv(hist_path, index=False)

        plot_curves(hist_path, os.path.join(plots_dir, f"{name}.png"))
        any_done = True

    if not any_done:
        print("No suitable run folders found. Make sure each category folder contains config_updated.yaml and checkpoints.")

if __name__ == "__main__":
    main()

