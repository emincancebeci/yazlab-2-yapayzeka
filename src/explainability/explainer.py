"""
explainer.py - Olasiliksal Aciklanabilirlik Modulu

Her Automata karari icin iki formatta cikti uretir:
  1. [SYSTEM DECISION] metin formatı (proje sartnamesi zorunlu)
  2. JSON formatı (config'de save_json=true ise dosyaya kaydedilir)

Kullanim:
    python src/explainability/explainer.py
"""

import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.loader import (
    load_batadal, get_batadal_features_labels, get_batadal_temporal_split,
)
from src.data.preprocess import Preprocessor, handle_missing
from src.models.automata.automata import ProbabilisticAutomata


# ─────────────────────────────────────────────────────────────────────────────
# Metin Formatı: [SYSTEM DECISION]
# ─────────────────────────────────────────────────────────────────────────────

def format_system_decision(record: dict) -> str:
    """
    Proje sartnamesi zorunlu formati:
    [SYSTEM DECISION]
    Time Step: ...
    Previous State: ...
    Incoming Pattern: ...
    Status: seen / unseen
    Nearest Pattern: ... (distance = X)   [yalnizca unseen ise]
    Transition Prob: ...
    Path Probability: ...
    Decision: NORMAL / ANOMALY
    Confidence Score: ... (High / Medium / Low)
    """
    t         = record["time_step"]
    prev      = record["state"]          or "START"
    pattern   = record["pattern"]
    status    = record["status"].upper()
    mapped    = record["mapped_to"]
    lev_dist  = record["levenshtein_dist"]
    trans_p   = record["transition_prob"]
    path_p    = record["path_probability"]
    decision  = record["decision"].upper()

    # Guven skoru
    if path_p >= 0.3:
        confidence_label = "High"
    elif path_p >= 0.05:
        confidence_label = "Medium"
    else:
        confidence_label = "Low"

    lines = [
        "[SYSTEM DECISION]",
        f"Time Step        : t = {t}",
        f"Previous State   : \"{prev}\"",
        f"Incoming Pattern : \"{pattern}\"",
        f"Status           : {status}",
    ]

    if status == "UNSEEN":
        nearest = mapped or pattern
        dist    = lev_dist or 0
        lines.append(f"Nearest Pattern  : \"{nearest}\" (distance = {dist})")

    if trans_p is not None:
        lines.append(f"Transition Prob  : {trans_p:.6f}")
    else:
        lines.append("Transition Prob  : N/A (first window)")

    lines += [
        f"Path Probability : {path_p:.6f}",
        f"Decision         : {decision}",
        f"Confidence Score : {path_p:.6f} ({confidence_label})",
        "-" * 50,
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# JSON Formatı
# ─────────────────────────────────────────────────────────────────────────────

def format_json(record: dict) -> dict:
    """
    Proje sartnamesi JSON formatı.
    """
    trans_p = record["transition_prob"]
    path_p  = record["path_probability"]

    if path_p >= 0.3:
        confidence_label = "High"
    elif path_p >= 0.05:
        confidence_label = "Medium"
    else:
        confidence_label = "Low"

    return {
        "time_step"        : record["time_step"],
        "state"            : record["state"] or "START",
        "pattern"          : record["pattern"],
        "status"           : record["status"],
        "mapped_to"        : record["mapped_to"],
        "levenshtein_dist" : record["levenshtein_dist"],
        "transition_prob"  : round(trans_p, 6) if trans_p is not None else None,
        "path_probability" : round(path_p, 6),
        "decision"         : record["decision"],
        "confidence"       : {
            "score" : round(path_p, 6),
            "label" : confidence_label,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Explainer Sinifi
# ─────────────────────────────────────────────────────────────────────────────

class AutomataExplainer:
    """
    ProbabilisticAutomata'nin predict_sequence() ciktisini
    [SYSTEM DECISION] ve JSON formatinda sunar.
    """

    def __init__(self, config: dict):
        self.config   = config
        self.save_json = config["output"].get("save_json", True)

    def explain(self, sequence: list, out_path: str = None) -> list:
        """
        sequence: automata.predict_sequence() ciktisi (list[dict])
        out_path: JSON cikti dosyasi yolu (None ise otomatik)

        Returns:
            json_records: list[dict]  — JSON formatinda kayitlar
        """
        text_blocks  = []
        json_records = []

        for record in sequence:
            text_blocks.append(format_system_decision(record))
            json_records.append(format_json(record))

        # Metin ciktisi
        full_text = "\n".join(text_blocks)

        # JSON kayit
        if self.save_json:
            if out_path is None:
                os.makedirs(self.config["output"]["log_dir"], exist_ok=True)
                out_path = os.path.join(
                    self.config["output"]["log_dir"],
                    "explainer_output.json"
                )
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(json_records, f, indent=2, ensure_ascii=False)

        return json_records, full_text

    def explain_single(self, record: dict) -> tuple:
        """Tek bir kayit icin aciklama uretir."""
        return format_json(record), format_system_decision(record)

    def summary(self, json_records: list) -> dict:
        """
        Tum sekans icin ozet istatistikler.
        """
        total   = len(json_records)
        anomaly = sum(1 for r in json_records if r["decision"] == "anomaly")
        unseen  = sum(1 for r in json_records if r["status"] == "unseen")
        probs   = [r["path_probability"] for r in json_records]

        return {
            "total_windows"   : total,
            "anomaly_count"   : anomaly,
            "normal_count"    : total - anomaly,
            "anomaly_rate"    : round(anomaly / total, 4) if total > 0 else 0.0,
            "unseen_count"    : unseen,
            "unseen_rate"     : round(unseen / total, 4) if total > 0 else 0.0,
            "mean_path_prob"  : round(float(np.mean(probs)), 6) if probs else 0.0,
            "min_path_prob"   : round(float(np.min(probs)), 6) if probs else 0.0,
            "max_path_prob"   : round(float(np.max(probs)), 6) if probs else 0.0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Demo / Ana Calistirma
# ─────────────────────────────────────────────────────────────────────────────

def run_explainer_demo(config: dict, n_examples: int = 5):
    """
    BATADAL uzerinde Automata'yi fit edip ilk n_examples karar icin
    [SYSTEM DECISION] ciktisi uretir.
    """
    print("Veri yukleniyor...")
    df = handle_missing(load_batadal(config))
    X, y = get_batadal_features_labels(df, config)
    X_tr, _, X_te, y_tr, _, y_te = get_batadal_temporal_split(X, y, config)

    prep   = Preprocessor(config)
    pc1_tr = prep.fit_transform_pca(X_tr).ravel()
    pc1_te = prep.transform_pca(X_te).ravel()

    print("Automata fit ediliyor...")
    model = ProbabilisticAutomata(config)
    model.fit(pc1_tr)

    print("Sekans tahminleniyor...")
    sequence = model.predict_sequence(pc1_te)

    explainer    = AutomataExplainer(config)
    json_records, full_text = explainer.explain(sequence)

    # Ozet
    summ = explainer.summary(json_records)
    print("\n=== EXPLAINER OZET ===")
    for k, v in summ.items():
        print(f"  {k:20}: {v}")

    # Ilk n ornek [SYSTEM DECISION] ciktisi
    print(f"\n=== ORNEK {n_examples} KARAR ===\n")
    sample_blocks = full_text.split("-" * 50)
    for block in sample_blocks[:n_examples]:
        if block.strip():
            print(block.strip())
            print("-" * 50)

    # JSON dosyasinin konumu
    log_dir = config["output"]["log_dir"]
    print(f"\nTam JSON cikti -> {log_dir}explainer_output.json")
    print(f"Toplam kayit   : {len(json_records)}")

    return json_records, full_text


if __name__ == "__main__":
    with open("configs/experiments.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    os.makedirs(config["output"]["log_dir"], exist_ok=True)
    run_explainer_demo(config, n_examples=5)
