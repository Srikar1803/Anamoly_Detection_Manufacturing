
# ===== score.py (summary table + inline ROC plots) =====
import os, json, yaml, argparse, glob
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# plotting / metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

from models.autoencoder import UNetAE
from losses import topk_error

# -----------------------------
# Data helpers
# -----------------------------
class SimpleTestDataset(Dataset):
    def __init__(self, df, image_size=384):
        self.df = df.reset_index(drop=True)
        self.tfm = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["path"]).convert("RGB")
        x = self.tfm(img)

        # ---- robust label mapping ----
        lab = row.get("label", None)

        if lab is None or (isinstance(lab, float) and pd.isna(lab)):
            y = np.nan  # unlabeled
        elif isinstance(lab, (int, np.integer)):
            # trust numeric labels if already 0/1
            y = int(lab)
        elif isinstance(lab, str):
            # mvtec: "good" => 0, anything else (e.g., "broken_large") => 1
            y = 0 if lab.strip().lower() == "good" else 1
        else:
            y = np.nan

        return x, y, row["category"], row["path"]


def make_test_loader(index_csv, category, image_size, batch_size, num_workers=0):
    df = pd.read_csv(index_csv)
    df = df[(df["category"] == category) & (df["split"] == "test")]
    assert len(df) > 0, f"No test rows for category={category} in {index_csv}"
    ds = SimpleTestDataset(df, image_size=image_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=False)

# -----------------------------
# Config helpers
# -----------------------------
def apply_overrides(base_cfg, category):
    cfg = dict(base_cfg)
    pc = (base_cfg.get("per_category") or {})
    if category in pc:
        for k, v in pc[category].items():
            cfg[k] = v
    return cfg

def try_auroc(y_true, scores):
    try:
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return float("nan")

# -----------------------------
# Core evaluation
# -----------------------------
def evaluate(config, category):
    cfg = config if category == "global" else apply_overrides(config, category)
    device = torch.device(cfg["device"] if torch.cuda.is_available() or cfg["device"] == "cpu" else "cpu")

    # locate run dir, ckpt, threshold
    cand_dirs = [
        os.path.join(cfg['out_dir'], category.replace('/', '_')),
        os.path.join(cfg['out_dir'], f"ae_{category.replace('/', '_')}"),
    ]
    out_dir = next((d for d in cand_dirs if os.path.isdir(d)), None)
    if out_dir is None:
        raise FileNotFoundError(f"No run directory found for {category}. Tried: {cand_dirs}")

    cand_ckpts = [os.path.join(out_dir, "best.pth"), os.path.join(out_dir, "best.pt")]
    ckpt = next((p for p in cand_ckpts if os.path.exists(p)), None)
    thp  = os.path.join(out_dir, "threshold.json")
    assert ckpt is not None and os.path.exists(thp), (
        f"Train first for category={category}. Missing ckpt/threshold in {out_dir}"
    )

    # build model
    model = UNetAE(base=int(cfg['base_channels']), latent=int(cfg['latent_dim'])).to(device)
    obj = torch.load(ckpt, map_location=device)
    state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    model.load_state_dict(state)
    model.eval()

    thr = json.load(open(thp))["threshold"]

    # test loader
    dl_te = make_test_loader(
        index_csv=cfg['index_csv'],
        category=category,
        image_size=cfg['image_size'],
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
    )

    # scoring loop
    rows = []
    TP=FP=TN=FN=0
    with torch.no_grad():
        for (x, y, cat, paths) in tqdm(dl_te, desc=f"score {category}"):
            x = x.to(device)
            xhat = model(x)
            s = topk_error(xhat, x, cfg['topk_frac'])
            if isinstance(s, (list, tuple)):
                s = s[0]
            s = s.detach().cpu().numpy()

            # y may be float NaN; keep it as np.nan for metrics mask
            y_np = np.array(y)

            for i in range(len(s)):
                pred = int(s[i] >= thr)
                label_val = y_np[i]
                rows.append({
                    "category": cat[i],
                    "path": paths[i],
                    "score": float(s[i]),
                    "label": (int(label_val) if not np.isnan(label_val) else np.nan),
                    "pred": pred,
                    "thr": thr
                })
                if not np.isnan(label_val):
                    if   label_val==1 and pred==1: TP+=1
                    elif label_val==0 and pred==1: FP+=1
                    elif label_val==0 and pred==0: TN+=1
                    elif label_val==1 and pred==0: FN+=1

    df = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, "test_predictions.csv")
    df.to_csv(out_csv, index=False)

    # compute metrics (mask to labeled rows)
    labeled = df["label"].notna().values
    y_lab = df.loc[labeled, "label"].astype(int).values if labeled.any() else np.array([])
    s_lab = df.loc[labeled, "score"].astype(float).values if labeled.any() else np.array([])

    eps=1e-9
    precision = TP/(TP+FP+eps) if (TP+FP)>0 else 0.0
    recall    = TP/(TP+FN+eps) if (TP+FN)>0 else 0.0
    f1        = 2*precision*recall/(precision+recall+eps) if (precision+recall)>0 else 0.0

    # ROC (only if both classes present)
    roc = None
    auc = float("nan")
    if labeled.any() and len(np.unique(y_lab)) == 2:
        fpr, tpr, _ = roc_curve(y_lab, s_lab)
        auc = roc_auc_score(y_lab, s_lab)
        roc = (fpr, tpr)

    # return everything needed for summary + plotting
    return {
        "category": category,
        "out_dir": out_dir,
        "csv": out_csv,
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "precision": precision, "recall": recall, "f1": f1,
        "auroc": auc,
        "roc": roc,  # None or (fpr, tpr)
        "labeled_n": int(labeled.sum())
    }

# -----------------------------
# Main with summary + plots
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--category", default="each", help="'each' for all per-category models; a specific category; or 'global'")
    ap.add_argument("--show", action="store_true", help="Show ROC plots inline (and also save them).")
    ap.add_argument("--save_dir", default="runs/_plots", help="Directory to save ROC images and summary CSV.")
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))

    cats = (pd.read_csv(config['index_csv'])['category'].unique().tolist()
            if args.category == "each" else [args.category])

    os.makedirs(args.save_dir, exist_ok=True)

    results = []
    for c in cats:
        try:
            res = evaluate(config, c)
            results.append(res)
        except Exception as e:
            print(f"[{c}] ERROR: {e}")

    if not results:
        print("No results to summarize.")
        return

    # Build and print summary table
    summary = pd.DataFrame([{
        "category": r["category"],
        "AUROC": r["auroc"],
        "F1": r["f1"],
        "Precision": r["precision"],
        "Recall": r["recall"],
        "TP": r["TP"], "FP": r["FP"], "TN": r["TN"], "FN": r["FN"],
        "n_labeled": r["labeled_n"],
        "pred_csv": r["csv"]
    } for r in results])

    # order by AUROC (NaN last)
    summary = summary.sort_values(by=["AUROC"], ascending=False, na_position="last").reset_index(drop=True)
    print("\n=== Evaluation Summary ===")
    print(summary.to_string(index=False))

    # Save summary CSV
    summary_path = os.path.join(args.save_dir, "summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved summary: {summary_path}")

    # Plot ROC per category (inline + save)
    for r in results:
        cat = r["category"]
        roc = r["roc"]
        if roc is None:
            print(f"[{cat}] ROC: n/a (needs labeled positives and negatives)")
            continue
        fpr, tpr = roc
        auc = r["auroc"]

        plt.figure()
        plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.4f}")
        plt.plot([0,1],[0,1], linestyle="--", linewidth=1)
        plt.xlim([0,1]); plt.ylim([0,1])
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title(f"ROC — {cat}")
        plt.legend(loc="lower right")

        # save image
        out_png = os.path.join(args.save_dir, f"roc_{cat.replace('/','_')}.png")
        plt.savefig(out_png, bbox_inches="tight", dpi=150)
        if args.show:
            plt.show()
        else:
            plt.close()
        print(f"[{cat}] Saved ROC: {out_png}")

if __name__ == "__main__":
    main()
