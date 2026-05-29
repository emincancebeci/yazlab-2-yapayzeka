"""
pipeline.py — Ana Çalıştırma Betiği
Tüm deneyler (SKAB + BATADAL, 5 seed, gürültü, unseen, sweep) burada orchestrate edilir.

Kullanım:
    py src/pipeline.py --config configs/experiments.yaml
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import (
    load_skab, get_skab_features_labels, get_skab_kfold_splits,
    load_batadal, get_batadal_features_labels, get_batadal_temporal_split,
    make_dataloader,
)
from src.data.preprocess import Preprocessor, handle_missing
from src.models.automata.automata import ProbabilisticAutomata
from src.models.deep_learning import train_model, predict_model
from src.models.deep_learning.lstm_model import build_lstm
from src.models.deep_learning.gru_model import build_gru
from src.models.deep_learning.cnn1d_model import build_cnn1d
from src.evaluate.metrics import compute_metrics, aggregate_seed_results


# ─────────────────────────────────────────────────────────────────────────────
# Yardımcılar
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dl_builders(n_features: int, config: dict) -> dict:
    return {
        "LSTM":   lambda: build_lstm(n_features, config),
        "GRU":    lambda: build_gru(n_features, config),
        "1D-CNN": lambda: build_cnn1d(n_features, config),
    }


def _align_labels(y: np.ndarray, preds, window_size: int):
    """Automata tahminlerini etiket dizisiyle hizalar."""
    n = len(preds)
    return y[window_size - 1: window_size - 1 + n]


# ─────────────────────────────────────────────────────────────────────────────
# SKAB — GroupKFold × 5 seed
# ─────────────────────────────────────────────────────────────────────────────

def run_skab(config: dict) -> dict:
    print("\n[SKAB] Veri yükleniyor...")
    df = handle_missing(load_skab(config))
    X, y, groups = get_skab_features_labels(df, config)
    n_features = X.shape[1]
    seeds = config["seeds"]
    w = config["fixed"]["window_size"]

    results = {}
    builders = dl_builders(n_features, config)
    all_models = list(builders.keys()) + ["Automata"]

    for model_name in all_models:
        print(f"  [{model_name}] başlatılıyor...")
        seed_metrics, train_times, infer_times = [], [], []

        for seed in seeds:
            set_seed(seed)
            fold_metrics = []

            for _, X_tr, X_va, y_tr, y_va, _, _ in get_skab_kfold_splits(X, y, groups, config):
                prep = Preprocessor(config)

                if model_name == "Automata":
                    pc1_tr = prep.fit_transform_pca(X_tr).ravel()
                    pc1_va = prep.transform_pca(X_va).ravel()

                    t0 = time.time()
                    model = ProbabilisticAutomata(config)
                    model.fit(pc1_tr)
                    train_times.append(time.time() - t0)

                    t0 = time.time()
                    preds = model.predict(pc1_va)
                    infer_times.append(time.time() - t0)

                    y_eval = _align_labels(y_va, preds, w)
                else:
                    X_tr_sc = prep.fit_transform(X_tr)
                    X_va_sc = prep.transform(X_va)
                    tr_loader = make_dataloader(X_tr_sc, y_tr, config, shuffle=True)
                    va_loader = make_dataloader(X_va_sc, y_va, config, shuffle=False)

                    mdl = builders[model_name]()
                    mdl, _, tr_t = train_model(mdl, tr_loader, va_loader, config)
                    train_times.append(tr_t)

                    t0 = time.time()
                    preds, y_eval = predict_model(mdl, va_loader)
                    infer_times.append(time.time() - t0)

                fold_metrics.append(compute_metrics(y_eval, preds))

            seed_metrics.append(
                {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
            )

        agg = aggregate_seed_results(seed_metrics)
        agg["train_time_mean"] = float(np.mean(train_times))
        agg["infer_time_mean"] = float(np.mean(infer_times))
        results[model_name] = agg
        print(f"    F1 = {agg['f1_mean']:.4f} ± {agg['f1_std']:.4f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# BATADAL — Temporal %60/%20/%20 × 5 seed
# ─────────────────────────────────────────────────────────────────────────────

def run_batadal(config: dict) -> dict:
    print("\n[BATADAL] Veri yükleniyor...")
    df = handle_missing(load_batadal(config))
    X, y = get_batadal_features_labels(df, config)
    X_tr, X_va, X_te, y_tr, y_va, y_te = get_batadal_temporal_split(X, y, config)
    n_features = X_tr.shape[1]
    seeds = config["seeds"]
    w = config["fixed"]["window_size"]

    # Ön işleme bir kez yapılır (seed'den bağımsız)
    prep = Preprocessor(config)
    X_tr_sc = prep.fit_transform(X_tr)
    X_va_sc = prep.transform(X_va)
    X_te_sc = prep.transform(X_te)

    prep_pca = Preprocessor(config)
    pc1_tr = prep_pca.fit_transform_pca(X_tr).ravel()
    pc1_te = prep_pca.transform_pca(X_te).ravel()

    results = {}
    builders = dl_builders(n_features, config)
    all_models = list(builders.keys()) + ["Automata"]

    for model_name in all_models:
        print(f"  [{model_name}] başlatılıyor...")
        seed_metrics, train_times, infer_times = [], [], []

        for seed in seeds:
            set_seed(seed)

            if model_name == "Automata":
                t0 = time.time()
                model = ProbabilisticAutomata(config)
                model.fit(pc1_tr)
                train_times.append(time.time() - t0)

                t0 = time.time()
                preds = model.predict(pc1_te)
                infer_times.append(time.time() - t0)

                y_eval = _align_labels(y_te, preds, w)
            else:
                tr_loader = make_dataloader(X_tr_sc, y_tr, config, shuffle=True)
                va_loader = make_dataloader(X_va_sc, y_va, config, shuffle=False)
                te_loader = make_dataloader(X_te_sc, y_te, config, shuffle=False)

                mdl = builders[model_name]()
                mdl, _, tr_t = train_model(mdl, tr_loader, va_loader, config)
                train_times.append(tr_t)

                t0 = time.time()
                preds, y_eval = predict_model(mdl, te_loader)
                infer_times.append(time.time() - t0)

            seed_metrics.append(compute_metrics(y_eval, preds))

        agg = aggregate_seed_results(seed_metrics)
        agg["train_time_mean"] = float(np.mean(train_times))
        agg["infer_time_mean"] = float(np.mean(infer_times))
        results[model_name] = agg
        print(f"    F1 = {agg['f1_mean']:.4f} ± {agg['f1_std']:.4f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# GÜRÜLTÜ DENEYİ — BATADAL test seti üzerinde Gaussian gürültü etkisi
# ─────────────────────────────────────────────────────────────────────────────

def run_noise(config: dict) -> dict:
    print("\n[NOISE] Gürültü deneyi...")
    df = handle_missing(load_batadal(config))
    X, y = get_batadal_features_labels(df, config)
    X_tr, X_va, X_te, y_tr, y_va, y_te = get_batadal_temporal_split(X, y, config)
    n_features = X_tr.shape[1]
    noise_std = config["noise"]["std"]
    w = config["fixed"]["window_size"]

    prep = Preprocessor(config)
    X_tr_sc  = prep.fit_transform(X_tr)
    X_va_sc  = prep.transform(X_va)
    X_te_sc  = prep.transform(X_te)
    X_te_nsy = prep.add_gaussian_noise(X_te_sc, noise_std)

    prep_pca = Preprocessor(config)
    pc1_tr   = prep_pca.fit_transform_pca(X_tr).ravel()
    pc1_te   = prep_pca.transform_pca(X_te).ravel()
    pc1_nsy  = pc1_te + np.random.normal(0, noise_std, pc1_te.shape)

    results = {}
    builders = dl_builders(n_features, config)
    set_seed(config["seeds"][0])

    for model_name in list(builders.keys()) + ["Automata"]:
        if model_name == "Automata":
            model = ProbabilisticAutomata(config)
            model.fit(pc1_tr)
            p_orig  = model.predict(pc1_te)
            p_noisy = model.predict(pc1_nsy)
            y_eval  = _align_labels(y_te, p_orig, w)
        else:
            tr_loader    = make_dataloader(X_tr_sc, y_tr, config, shuffle=True)
            va_loader    = make_dataloader(X_va_sc, y_va, config, shuffle=False)
            te_loader    = make_dataloader(X_te_sc,  y_te, config, shuffle=False)
            noisy_loader = make_dataloader(X_te_nsy, y_te, config, shuffle=False)

            mdl = builders[model_name]()
            mdl, _, _ = train_model(mdl, tr_loader, va_loader, config)
            p_orig, y_eval = predict_model(mdl, te_loader)
            p_noisy, _     = predict_model(mdl, noisy_loader)

        results[model_name] = {
            "original": compute_metrics(y_eval, p_orig),
            "noisy":    compute_metrics(y_eval, p_noisy),
        }
        o = results[model_name]["original"]["f1"]
        n = results[model_name]["noisy"]["f1"]
        print(f"    {model_name:8}  orig F1={o:.4f}  noisy F1={n:.4f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# UNSEEN ANALİZİ
# ─────────────────────────────────────────────────────────────────────────────

def run_unseen(config: dict) -> dict:
    print("\n[UNSEEN] Unseen analizi...")
    df = handle_missing(load_batadal(config))
    X, y = get_batadal_features_labels(df, config)
    X_tr, _, X_te, y_tr, _, y_te = get_batadal_temporal_split(X, y, config)
    w = config["fixed"]["window_size"]

    prep = Preprocessor(config)
    pc1_tr = prep.fit_transform_pca(X_tr).ravel()
    pc1_te = prep.transform_pca(X_te).ravel()

    model = ProbabilisticAutomata(config)
    model.fit(pc1_tr)

    seq    = model.predict_sequence(pc1_te)
    y_eval = _align_labels(y_te, seq, w)

    unseen_idx = [i for i, r in enumerate(seq) if r["status"] == "unseen"]
    total = len(seq)

    if unseen_idx:
        u_preds  = [1 if seq[i]["decision"] == "anomaly" else 0 for i in unseen_idx]
        u_labels = [int(y_eval[i]) for i in unseen_idx]
        map_acc  = compute_metrics(u_labels, u_preds)["accuracy"]
        det_rate = float(np.mean([p == 1 for p in u_preds]))
    else:
        map_acc = det_rate = 0.0

    result = {
        "total_patterns":   total,
        "unseen_count":     len(unseen_idx),
        "unseen_rate":      len(unseen_idx) / total if total > 0 else 0.0,
        "detection_rate":   det_rate,
        "mapping_accuracy": map_acc,
    }
    print(f"    Unseen: {result['unseen_count']}/{total} "
          f"({result['unseen_rate']:.2%})  "
          f"Det={det_rate:.4f}  MapAcc={map_acc:.4f}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# PARAMETRE SWEEP — window_size × alphabet_size (Automata)
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(config: dict) -> dict:
    print("\n[SWEEP] Parametre taraması (window × alphabet)...")
    df = handle_missing(load_batadal(config))
    X, y = get_batadal_features_labels(df, config)
    X_tr, _, X_te, y_tr, _, y_te = get_batadal_temporal_split(X, y, config)

    prep = Preprocessor(config)
    pc1_tr = prep.fit_transform_pca(X_tr).ravel()
    pc1_te = prep.transform_pca(X_te).ravel()

    results = {}
    for ws in config["sweep"]["window_sizes"]:
        results[str(ws)] = {}
        for ab in config["sweep"]["alphabet_sizes"]:
            model = ProbabilisticAutomata(config, window_size=ws, alphabet_size=ab)
            model.fit(pc1_tr)
            preds  = model.predict(pc1_te)
            y_eval = _align_labels(y_te, preds, ws)
            m      = compute_metrics(y_eval, preds)
            results[str(ws)][str(ab)] = {
                "f1":      m["f1"],
                "n_states": model.n_states,
            }
            print(f"    ws={ws} ab={ab}  F1={m['f1']:.4f}  states={model.n_states}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="YazLab-2 Deney Pipeline")
    parser.add_argument("--config", default="configs/experiments.yaml",
                        help="Konfigürasyon dosyası yolu")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    for d in [config["output"]["results_dir"],
              config["output"]["figures_dir"],
              config["output"]["log_dir"]]:
        os.makedirs(d, exist_ok=True)

    results = {}
    results["skab"]    = run_skab(config)
    results["batadal"] = run_batadal(config)
    results["noise"]   = run_noise(config)
    results["unseen"]  = run_unseen(config)
    results["sweep"]   = run_sweep(config)

    out = os.path.join(config["output"]["results_dir"], "results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nTüm sonuçlar kaydedildi → {out}")


if __name__ == "__main__":
    main()
