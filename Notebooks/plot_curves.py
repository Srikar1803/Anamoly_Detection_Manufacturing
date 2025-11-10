import os, argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_one(run_dir, out_png=None, title=None):
    csv_p = os.path.join(run_dir, "history.csv")
    df = pd.read_csv(csv_p)
    plt.figure(figsize=(7,4.5))
    plt.plot(df["epoch"], df["train"], label="train")
    plt.plot(df["epoch"], df["val"],   label="val")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(title or f"Loss — {os.path.basename(run_dir)}")
    plt.legend(); plt.tight_layout()
    out_png = out_png or os.path.join(run_dir, "loss_curve.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved: {out_png}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="e.g., runs/bottle")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    plot_one(args.run_dir, args.out)
