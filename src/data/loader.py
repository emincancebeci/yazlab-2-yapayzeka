"""


Tüm parametreler configs/experiments.yaml'dan okunur.
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
import torch
from torch.utils.data import Dataset, DataLoader



def load_skab(config: dict) -> pd.DataFrame:
    
    
    
    skab_cfg = config["datasets"]["skab"]
    dfs = []

    for group_name, path_key in [("valve1", "valve1_path"), ("valve2", "valve2_path")]:
        folder = skab_cfg[path_key]
        csv_files = sorted(glob.glob(os.path.join(folder, "*.csv")))

        if not csv_files:
            raise FileNotFoundError(f"{folder} klasöründe CSV bulunamadı.")

        for fpath in csv_files:
            fname = os.path.basename(fpath)
            
            try:
                df = pd.read_csv(fpath, sep=";")
            except Exception:
                df = pd.read_csv(fpath)

            df["source_group"] = group_name
            df["source_file"] = fname
            dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    return combined


def get_skab_features_labels(df: pd.DataFrame, config: dict):
   
    skab_cfg = config["datasets"]["skab"]
    target_col = skab_cfg["target_col"]           # "anomaly"
    exclude_cols = skab_cfg["exclude_cols"]        # [datetime, changepoint, source_group, source_file]
    group_col = skab_cfg["group_col"]             # "source_file" — GroupKFold 

    groups = df[group_col].values
    y = df[target_col].values.astype(int)

    drop_cols = exclude_cols + [target_col]
    X = df.drop(columns=drop_cols, errors="ignore").values.astype(float)

    return X, y, groups


def get_skab_kfold_splits(X, y, groups, config: dict):
   
    n_splits = config["kfold"]["n_splits"]
    kfold = GroupKFold(n_splits=n_splits)

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y, groups)):
        yield (
            fold_idx,
            X[train_idx], X[val_idx],
            y[train_idx], y[val_idx],
            groups[train_idx], groups[val_idx],
        )



def load_batadal(config: dict) -> pd.DataFrame:
   
    batadal_cfg = config["datasets"]["batadal"]
    fpath = batadal_cfg["path"]

    if not os.path.exists(fpath):
        raise FileNotFoundError(f"BATADAL dosyası bulunamadı: {fpath}")

    df = pd.read_csv(fpath)

    df.columns = [c.strip() for c in df.columns]
    return df


def get_batadal_features_labels(df: pd.DataFrame, config: dict):
  
    batadal_cfg = config["datasets"]["batadal"]
    target_col = batadal_cfg["target_col"]        
    exclude_cols = batadal_cfg["exclude_cols"]     

  
    y = (df[target_col].values != -999).astype(int)

    drop_cols = exclude_cols + [target_col]
    X = df.drop(columns=drop_cols, errors="ignore").values.astype(float)

    return X, y


def get_batadal_temporal_split(X, y, config: dict):
    """
    Zaman sırasını koruyarak kronolojik bölme yapar.
    Rastgele satır bölmesi YAPILMAZ (shuffle=False).

    Returns: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    ratios = config["datasets"]["batadal"]["split_ratio"]  # [0.60, 0.20, 0.20]
    n = len(X)
    train_end = int(n * ratios[0])
    val_end = train_end + int(n * ratios[1])

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val     = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test   = X[val_end:], y[val_end:]

    return X_train, X_val, X_test, y_train, y_val, y_test


# ===========================================================================
# PyTorch Dataset
# ===========================================================================

class TimeSeriesDataset(Dataset):
    """
    Kayan pencere tabanlı PyTorch Dataset.
    window_size configs/experiments.yaml'dan okunur.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, window_size: int):
        """
        Args:
            X: (n_samples, n_features) özellik matrisi
            y: (n_samples,) etiket dizisi
            window_size: kayan pencere boyutu (config'den gelir)
        """
        self.window_size = window_size
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X) - self.window_size + 1

    def __getitem__(self, idx):
        x_window = self.X[idx: idx + self.window_size]   # (window_size, n_features)
        label    = self.y[idx + self.window_size - 1]    # son adımın etiketi
        return x_window, label


def make_dataloader(X: np.ndarray, y: np.ndarray, config: dict,
                    shuffle: bool = False) -> DataLoader:
    """
    Config'den batch_size ve window_size'ı okuyarak DataLoader döner.
    Train için shuffle=True, Val/Test için shuffle=False.
    """
    window_size = config["fixed"]["window_size"]
    batch_size  = config["training"]["batch_size"]

    dataset = TimeSeriesDataset(X, y, window_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
