"""
pipeline_cross_dataset.py - Cross-Dataset Genellenebilirlik Deneyi

Tablo 3 icin:
  - Train: SKAB  → Test: BATADAL
  - Train: BATADAL → Test: SKAB

DL modelleri icin her iki veri seti PCA ile ortak boyuta (8 bileşen) indirgenir.
Automata icin her ikisi de PC1 (1D) kullanir.

Kullanim:
    python src/pipeline_cross_dataset.py --config configs/experiments.yaml
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.loader import (
    load_skab, get_skab_features_labels,
    load_batadal, get_batadal_features_labels, get_batadal_temporal_split,
)
from src.data.preprocess import handle_missing
from src.models.automata.automata import ProbabilisticAutomata
from src.models.deep_learning import train_model, predict_model
from src.models.deep_learning.lstm_model import build_lstm
from src.models.deep_learning.gru_model import build_gru
from src.models.deep_learning.cnn1d_model import build_cnn1d

SEEDS = [42, 123, 2026, 7, 999]
COMMON_DIM = 8   # Her iki veri seti de bu boyuta indirgenir


# ─────────────────────────────────────────────────────────────────────────────
# Veri Hazırlama
# ─────────────────────────────────────────────────────────────────────────────

def prepare_skab(config):
    """SKAB: GroupKFold yerine basit son %20 test bolumu kullanilir (cross-dataset icin)."""
    from sklearn.model_selection import GroupShuffleSplit
    import glob, pandas as pd

    dfs = []
    groups = []
    valve1_path = config["datasets"]["skab"]["valve1_path"]
    valve2_path = config["datasets"]["skab"]["valve2_path"]
    for folder_path in [valve1_path, valve2_path]:
        csvs = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
        for csv in csvs:
            df = pd.read_csv(csv, sep=";", parse_dates=["datetime"], index_col="datetime")
            df["source"] = os.path.basename(csv)
            dfs.append(df)
            groups.extend([os.path.basename(csv)] * len(df))

    df_all = pd.concat(dfs).reset_index(drop=True)
    feature_cols = [c for c in df_all.columns if c not in
                    ["anomaly", "changepoint", "source", "datetime",
                     "source_group", "source_file"]]
    X = df_all[feature_cols].values.astype(np.float32)
    y = df_all["anomaly"].values.astype(np.float32)

    # Temporal split: son %20 test
    n = len(X)
    split = int(n * 0.80)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    return X_tr, X_te, y_tr, y_te


def prepare_batadal(config):
    import pandas as pd
    df = handle_missing(load_batadal(config))
    X, y = get_batadal_features_labels(df, config)
    X_tr, _, X_te, y_tr, _, y_te = get_batadal_temporal_split(X, y, config)
    return X_tr.astype(np.float32), X_te.astype(np.float32), \
           y_tr.astype(np.float32), y_te.astype(np.float32)


def align_to_common_dim(X_tr_src, X_te_src, X_tr_tgt, X_te_tgt, n_components=COMMON_DIM):
    """
    Kaynak ve hedef veri setlerini ayri ayri PCA ile ortak boyuta indirgir.
    Her PCA yalnizca kendi train verisiyle fit edilir (data leakage yok).
    """
    # Kaynak: src train fit, src test + tgt transform
    scaler_src = MinMaxScaler()
    X_tr_src_sc = scaler_src.fit_transform(X_tr_src)
    X_te_src_sc = scaler_src.transform(X_te_src)

    n_src = min(n_components, X_tr_src_sc.shape[1])
    pca_src = PCA(n_components=n_src, random_state=42)
    X_tr_src_pca = pca_src.fit_transform(X_tr_src_sc)
    X_te_src_pca = pca_src.transform(X_te_src_sc)

    # Hedef: tgt train fit, tgt test transform
    scaler_tgt = MinMaxScaler()
    X_tr_tgt_sc = scaler_tgt.fit_transform(X_tr_tgt)
    X_te_tgt_sc = scaler_tgt.transform(X_te_tgt)

    n_tgt = min(n_components, X_tr_tgt_sc.shape[1])
    pca_tgt = PCA(n_components=n_tgt, random_state=42)
    X_tr_tgt_pca = pca_tgt.fit_transform(X_tr_tgt_sc)
    X_te_tgt_pca = pca_tgt.transform(X_te_tgt_sc)

    # Ortak boyut: ikisinin minimumu
    final_dim = min(n_src, n_tgt)
    return (X_tr_src_pca[:, :final_dim], X_te_src_pca[:, :final_dim],
            X_tr_tgt_pca[:, :final_dim], X_te_tgt_pca[:, :final_dim],
            final_dim)


def make_loader(X, y, config, shuffle=False):
    ws = config["fixed"]["window_size"]
    seqs, labels = [], []
    for i in range(len(X) - ws):
        seqs.append(X[i:i + ws])
        labels.append(y[i + ws - 1])
    X_t = torch.tensor(np.array(seqs), dtype=torch.float32)
    y_t = torch.tensor(np.array(labels), dtype=torch.float32)
    ds  = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=config["training"]["batch_size"],
                      shuffle=shuffle)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Dataset Deneyi
# ─────────────────────────────────────────────────────────────────────────────

def cross_dl(X_tr_src, y_tr_src, X_te_tgt, y_te_tgt,
             X_tr_tgt, y_tr_tgt,
             model_name, config, seed):
    """
    Kaynak dataset'te egit, hedef dataset'te test et.
    Validation olarak hedef train setinin son %20'si kullanilir.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_feats = X_tr_src.shape[1]
    val_split = int(len(X_tr_tgt) * 0.8)
    X_val_tgt = X_tr_tgt[val_split:]
    y_val_tgt = y_tr_tgt[val_split:]

    tr_loader  = make_loader(X_tr_src, y_tr_src, config, shuffle=True)
    val_loader = make_loader(X_val_tgt, y_val_tgt, config, shuffle=False)
    te_loader  = make_loader(X_te_tgt, y_te_tgt, config, shuffle=False)

    builders = {"LSTM": build_lstm, "GRU": build_gru, "1D-CNN": build_cnn1d}
    model = builders[model_name](n_feats, config)

    t0 = time.time()
    model, _, _ = train_model(model, tr_loader, val_loader, config)
    train_time = time.time() - t0

    preds, y_true = predict_model(model, te_loader)
    f1 = f1_score(y_true, preds, zero_division=0)
    return f1, train_time


def cross_automata(X_tr_src, y_tr_src, X_te_tgt, y_te_tgt,
                   X_tr_tgt, config):
    """
    Automata: Gecis matrisi kaynak dataset PC1'i uzerinde ogrenilir.
    Test ise hedef dataset PC1'i uzerinde yapilir.
    Her dataset kendi scaler+PCA'si ile donusturulur (data leakage yok).
    """
    # Kaynak: PC1 ogren, Automata fit et
    scaler_src = MinMaxScaler()
    pca_src    = PCA(n_components=1, random_state=42)
    pc1_tr_src = pca_src.fit_transform(
        scaler_src.fit_transform(X_tr_src)
    ).ravel()

    # Hedef: kendi scaler+PCA ile PC1 al (bağımsız)
    scaler_tgt = MinMaxScaler()
    pca_tgt    = PCA(n_components=1, random_state=42)
    pc1_tr_tgt = pca_tgt.fit_transform(
        scaler_tgt.fit_transform(X_tr_tgt)
    ).ravel()
    pc1_te_tgt = pca_tgt.transform(
        scaler_tgt.transform(X_te_tgt)
    ).ravel()

    # Automata'yi KAYNAK PC1 ile fit et
    t0 = time.time()
    automata = ProbabilisticAutomata(config)
    automata.fit(pc1_tr_src)
    train_time = time.time() - t0

    # HEDEF PC1'i kaynak istatistiklerine gore normalize et
    mn, mx = pc1_tr_src.min(), pc1_tr_src.max()
    if mx > mn:
        pc1_te_norm = np.clip((pc1_te_tgt - mn) / (mx - mn), 0, 1)
    else:
        pc1_te_norm = pc1_te_tgt

    ws    = config["fixed"]["window_size"]
    preds = automata.predict(pc1_te_norm)

    n      = min(len(preds), len(y_te_tgt) - ws + 1)
    y_true = y_te_tgt[ws - 1: ws - 1 + n]
    preds  = preds[:n]

    f1 = f1_score(y_true, preds, zero_division=0)
    return f1, train_time



# ─────────────────────────────────────────────────────────────────────────────
# Ana Fonksiyon
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("Veriler yukleniyor...")
    X_tr_sk, X_te_sk, y_tr_sk, y_te_sk = prepare_skab(config)
    X_tr_bt, X_te_bt, y_tr_bt, y_te_bt = prepare_batadal(config)

    print(f"SKAB    train={X_tr_sk.shape}  test={X_te_sk.shape}")
    print(f"BATADAL train={X_tr_bt.shape}  test={X_te_bt.shape}")

    # Ortak boyut hizalama (DL icin)
    print(f"\nPCA ile {COMMON_DIM} boyutlu ortak uzay olusturuluyor...")
    (X_tr_sk_a, X_te_sk_a,
     X_tr_bt_a, X_te_bt_a, final_dim) = align_to_common_dim(
        X_tr_sk, X_te_sk, X_tr_bt, X_te_bt, COMMON_DIM
    )
    print(f"Ortak boyut: {final_dim}")

    DL_MODELS = ["LSTM", "GRU", "1D-CNN"]
    results = {
        "skab_train_batadal_test": {},
        "batadal_train_skab_test": {},
    }

    # ── Train SKAB → Test BATADAL ─────────────────────────────────────────
    print("\n[Train: SKAB -> Test: BATADAL]")
    for model_name in DL_MODELS:
        print(f"  {model_name}...", end=" ", flush=True)
        f1s = []
        for seed in SEEDS:
            f1, _ = cross_dl(
                X_tr_sk_a, y_tr_sk,
                X_te_bt_a, y_te_bt,
                X_tr_bt_a, y_tr_bt,
                model_name, config, seed
            )
            f1s.append(f1)
        mean_f1 = float(np.mean(f1s))
        std_f1  = float(np.std(f1s))
        results["skab_train_batadal_test"][model_name] = {
            "f1_mean": round(mean_f1, 4),
            "f1_std":  round(std_f1,  4),
        }
        print(f"F1={mean_f1:.4f} +/- {std_f1:.4f}")

    # Automata: SKAB → BATADAL
    print("  Automata...", end=" ", flush=True)
    f1_auto_sk_bt, _ = cross_automata(X_tr_sk, y_tr_sk, X_te_bt, y_te_bt, X_tr_bt, config)
    results["skab_train_batadal_test"]["Automata"] = {
        "f1_mean": round(f1_auto_sk_bt, 4),
        "f1_std":  0.0,
    }
    print(f"F1={f1_auto_sk_bt:.4f}")

    # ── Train BATADAL → Test SKAB ─────────────────────────────────────────
    print("\n[Train: BATADAL -> Test: SKAB]")
    for model_name in DL_MODELS:
        print(f"  {model_name}...", end=" ", flush=True)
        f1s = []
        for seed in SEEDS:
            f1, _ = cross_dl(
                X_tr_bt_a, y_tr_bt,
                X_te_sk_a, y_te_sk,
                X_tr_sk_a, y_tr_sk,
                model_name, config, seed
            )
            f1s.append(f1)
        mean_f1 = float(np.mean(f1s))
        std_f1  = float(np.std(f1s))
        results["batadal_train_skab_test"][model_name] = {
            "f1_mean": round(mean_f1, 4),
            "f1_std":  round(std_f1,  4),
        }
        print(f"F1={mean_f1:.4f} +/- {std_f1:.4f}")

    # Automata: BATADAL → SKAB
    print("  Automata...", end=" ", flush=True)
    f1_auto_bt_sk, _ = cross_automata(X_tr_bt, y_tr_bt, X_te_sk, y_te_sk, X_tr_sk, config)
    results["batadal_train_skab_test"]["Automata"] = {
        "f1_mean": round(f1_auto_bt_sk, 4),
        "f1_std":  0.0,
    }
    print(f"F1={f1_auto_bt_sk:.4f}")

    # ── Kaydet ───────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    out = "results/cross_dataset_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # ── Tablo 3 Cikti ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TABLO 3: Cross-Dataset Performans Karsilastirmasi (F1)")
    print("=" * 60)
    print(f"{'Model':<12} {'Train:SKAB->Test:BT':>20} {'Train:BT->Test:SKAB':>20}")
    print("-" * 54)
    for model in ["LSTM", "GRU", "1D-CNN", "Automata"]:
        sk_bt = results["skab_train_batadal_test"][model]
        bt_sk = results["batadal_train_skab_test"][model]
        print(f"{model:<12} "
              f"{sk_bt['f1_mean']:>8.4f} +/- {sk_bt['f1_std']:.4f}   "
              f"{bt_sk['f1_mean']:>8.4f} +/- {bt_sk['f1_std']:.4f}")
    print("=" * 60)
    print(f"\nSonuclar kaydedildi -> {out}")


if __name__ == "__main__":
    main()
