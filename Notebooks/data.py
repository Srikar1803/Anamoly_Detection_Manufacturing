# data.py
import os, random
import pandas as pd
from PIL import Image
from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

def _seed_all(seed: int):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class MVTECDataset(Dataset):
    """
    Returns dict: {"image": tensor, "label": 0(good)/1(anomaly), "category": str, "path": str}
    """
    def __init__(self, df: pd.DataFrame, img_size: int, is_train: bool, aug_p: float):
        self.df = df.reset_index(drop=True)
        self.img_size = int(img_size)
        self.is_train = is_train
        self.aug_p = float(aug_p)
        self.base_tf = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor()
        ])
        self.aug = T.Compose([
            T.RandomApply([T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)], p=0.7),
            T.RandomApply([T.RandomRotation(degrees=5)], p=0.5),
        ])

    def __len__(self): return len(self.df)

    def _open_rgb(self, p: str) -> Image.Image:
        img = Image.open(p)
        if img.mode != "RGB": img = img.convert("RGB")
        return img

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = row["path"]
        label_txt = str(row["label"]).strip().lower()
        y = 0 if label_txt == "good" else 1
        img = self._open_rgb(path)
        if self.is_train and self.aug_p > 0:
            img = self.aug(img)
        x = self.base_tf(img)
        return {"image": x, "label": y, "category": row["category"], "path": path}

def _split_train_val(train_good_df: pd.DataFrame, val_split: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(train_good_df)
    if n == 0:
        return train_good_df.copy(), train_good_df.copy()
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    val_n = int(n * float(val_split))
    # keep at least 1 train sample if possible
    val_n = min(max(val_n, 0), max(n - 1, 0))
    val_idx = set(idx[:val_n])
    tr_rows, va_rows = [], []
    for i, r in enumerate(train_good_df.itertuples(index=False)):
        (va_rows if i in val_idx else tr_rows).append(r._asdict())
    return pd.DataFrame(tr_rows), pd.DataFrame(va_rows)

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["split", "label", "category"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
    return df

def make_loaders(index_csv: str,
                 category: str,
                 img_size=None,
                 image_size=None,
                 batch_size=16,
                 num_workers=4,
                 val_split=0.1,
                 augment_prob=0.5,
                 aug_p=None,
                 seed=42):
    _seed_all(seed)

    # aliases
    if img_size is None and image_size is not None:
        img_size = image_size
    if aug_p is None:
        aug_p = augment_prob
    img_size = int(img_size)

    # load + normalize
    df = pd.read_csv(index_csv)
    df = _normalize_columns(df)

    # subset category (compare in lowercase)
    cat_key = str(category).strip().lower()
    df_cat = df.copy() if cat_key == "global" else df[df["category"] == cat_key].copy()

    # train = only train-good
    tr_good = df_cat[(df_cat["split"] == "train") & (df_cat["label"] == "good")].copy()
    tr_df, va_df = _split_train_val(tr_good, float(val_split), seed)

    # final sanity
    if len(tr_df) == 0:
        raise RuntimeError(
            "\n".join([
                "[make_loaders] EMPTY TRAIN after filtering.",
                f"  category requested : {category}",
                f"  CSV                : {os.path.abspath(index_csv)}",
                f"  df_cat rows        : {len(df_cat)}",
                f"  train-good rows    : {len(tr_good)}",
                f"  val_split          : {val_split}",
                "  Hints:",
                "   - Ensure split is 'train' / 'test' and label 'good' (any case/space OK now).",
                "   - Ensure category matches (we compare in lowercase).",
                "   - Try val_split: 0.0 if the category is tiny."
            ])
        )

    ds_tr = MVTECDataset(tr_df, img_size, is_train=True,  aug_p=aug_p)
    ds_va = MVTECDataset(va_df, img_size, is_train=False, aug_p=0.0)

    dl_tr = DataLoader(ds_tr, batch_size=int(batch_size), shuffle=True,
                       num_workers=int(num_workers), pin_memory=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=int(batch_size), shuffle=False,
                       num_workers=int(num_workers), pin_memory=True)

    print(f"[make_loaders] category={category} | train={len(ds_tr)} | val={len(ds_va)} | img_size={img_size} | bs={batch_size}")
    return dl_tr, dl_va
