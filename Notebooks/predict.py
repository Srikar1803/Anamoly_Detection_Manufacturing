
# predict.py
import os, json, yaml, argparse, glob
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import pandas as pd

from models.autoencoder import UNetAE
from losses import topk_error

# ----------------- helpers -----------------
def load_artifacts(run_dir, device):
    cfg_path = os.path.join(run_dir, "config_updated.yaml")
    thr_path = os.path.join(run_dir, "threshold.json")
    ckpt_path = None
    for name in ["best.pth", "best.pt"]:
        p = os.path.join(run_dir, name)
        if os.path.isfile(p):
            ckpt_path = p
            break
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Missing {cfg_path}")
    if not os.path.isfile(thr_path):
        raise FileNotFoundError(f"Missing {thr_path}")
    if ckpt_path is None:
        raise FileNotFoundError(f"Missing best.pth/pt in {run_dir}")

    cfg = yaml.safe_load(open(cfg_path))
    thr = float(json.load(open(thr_path))["threshold"])

    model = UNetAE(base=int(cfg["base_channels"]), latent=int(cfg["latent_dim"])).to(device)
    obj = torch.load(ckpt_path, map_location=device)
    state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    model.load_state_dict(state); model.eval()

    size = int(cfg["image_size"])
    tfm = T.Compose([T.Resize((size, size)), T.ToTensor()])
    return model, tfm, thr, cfg

def first_test_image_from_index(index_csv, category):
    if not os.path.isfile(index_csv):
        return None
    df = pd.read_csv(index_csv)
    sub = df[(df["category"] == category) & (df["split"] == "test")]
    if len(sub) == 0:
        return None
    # prefer an anomaly if available
    if "label" in sub.columns:
        sub = sub.sort_values("label", ascending=False)
    return sub.iloc[0]["path"]

@torch.no_grad()
def infer_with_heat(model, tfm, img_path, device, topk_frac=0.01):
    img = Image.open(img_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    y = model(x)
    s = topk_error(y, x, topk_frac=topk_frac)
    if isinstance(s, (tuple, list)): s = s[0]
    score = float(s.detach().cpu().item())
    # simple heat: mean abs recon error
    heat = (y - x).abs().mean(dim=1)[0].detach().cpu().numpy()
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-9)
    return np.array(img), heat, score

def save_overlay(rgb, heat, out_path, title=None):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(10,3.5))
    plt.subplot(1,3,1); plt.imshow(rgb); plt.axis("off"); plt.title("Original")
    plt.subplot(1,3,2); plt.imshow(heat, cmap="jet"); plt.axis("off"); plt.title("Error heatmap")
    plt.subplot(1,3,3); plt.imshow(rgb); plt.imshow(heat, cmap="jet", alpha=0.5); plt.axis("off"); plt.title("Overlay")
    if title: plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

def discover_categories(runs_dir):
    cats = []
    for d in sorted(glob.glob(os.path.join(runs_dir, "*"))):
        if os.path.isdir(d) and os.path.isfile(os.path.join(d, "threshold.json")):
            cats.append(os.path.basename(d))
    return cats

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", required=True)
    ap.add_argument("--category", default="each", help="'each' to run all, or a single category name")
    ap.add_argument("--image", default=None, help="Path to a single image; if omitted we auto-pick from index_csv")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--topk_frac", type=float, default=0.01)
    ap.add_argument("--save_all", action="store_true", help="Save overlays even for OK predictions")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device=="cpu" or torch.cuda.is_available()) else "cpu")

    cats = [args.category] if args.category != "each" else discover_categories(args.runs_dir)
    if len(cats) == 0:
        print("No categories discovered. Check --runs_dir.")
        return

    print(f"Discovered {len(cats)} categories under '{args.runs_dir}'.")

    rows = []
    for c in cats:
        run_dir = os.path.join(args.runs_dir, c)
        try:
            model, tfm, thr, cfg = load_artifacts(run_dir, device)
        except Exception as e:
            print(f"{c:>12s} | ERROR loading: {e}")
            continue

        img_path = args.image
        if img_path is None:
            img_path = first_test_image_from_index(cfg.get("index_csv", ""), c)
        if img_path is None or not os.path.isfile(img_path):
            # as a last resort, try a common mvtec pattern (may not exist for all)
            guess = os.path.expanduser(f"~/mvtec_ad/{c}/test")
            found = sorted(glob.glob(os.path.join(guess, "*", "*.png")))
            img_path = found[0] if found else None

        if img_path is None:
            print(f"{c:>12s} | ERROR: could not find a test image (pass --image to override).")
            continue

        try:
            rgb, heat, score = infer_with_heat(model, tfm, img_path, device, args.topk_frac)
            pred = "ANOMALY" if score >= thr else "OK"
            print(f"{c:>12s} | score={score:.6f} | thr={thr:.6f} | pred={pred} | img={img_path}")

            # save overlay if anomaly or user asked to save all
            if pred == "ANOMALY" or args.save_all:
                out_dir = os.path.join(run_dir, "_pred_vis")
                base = os.path.splitext(os.path.basename(img_path))[0]
                out_path = os.path.join(out_dir, f"{c}_{base}_overlay.png")
                title = f"{c} | score={score:.4f}, thr={thr:.4f}, pred={pred}"
                save_overlay(rgb, heat, out_path, title=title)
                vis_path = out_path
            else:
                vis_path = ""

            rows.append({
                "category": c,
                "score": score,
                "threshold": thr,
                "pred": pred,
                "image": img_path,
                "vis": vis_path
            })
        except Exception as e:
            print(f"{c:>12s} | ERROR during inference: {e}")

    if rows:
        df = pd.DataFrame(rows)
        # Pretty print summary table
        with pd.option_context("display.max_colwidth", 60):
            print(df.to_string(index=False))
        # also save CSV summary
        out_csv = os.path.join(args.runs_dir, "_pred_vis", "predict_summary.csv")
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"Saved summary: {out_csv}")

if __name__ == "__main__":
    main()
