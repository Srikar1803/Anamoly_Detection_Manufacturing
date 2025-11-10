# plot_all_curves.py
import os, glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main(runs_root="runs"):
    cats = sorted([d for d in glob.glob(os.path.join(runs_root, "*")) if os.path.isdir(d)])
    if not cats:
        print(f"No run dirs under {runs_root}")
        return

    out_dir = os.path.join(runs_root, "_plots")
    os.makedirs(out_dir, exist_ok=True)

    any_found = False
    for d in cats:
        csv_path = os.path.join(d, "history.csv")
        if not os.path.exists(csv_path):
            print(f"Skip {d}: no history.csv")
            continue
        any_found = True
        cat = os.path.basename(d)
        df = pd.read_csv(csv_path)

        plt.figure(figsize=(7,4))
        plt.plot(df["epoch"], df["train_loss"], label="train")
        plt.plot(df["epoch"], df["val_loss"], label="val")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(cat)
        plt.legend(); plt.tight_layout()
        out_png = os.path.join(out_dir, f"{cat}_loss.png")
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"Wrote {out_png}")

    if not any_found:
        print("No history.csv files found. Re-run training after applying logging patch.")

if __name__ == "__main__":
    main()
