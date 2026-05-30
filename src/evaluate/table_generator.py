"""
table_generator.py - Otomatik Tablo Uretimi
results/results.json'dan okur, 5 tabloyu Markdown ve CSV olarak uretir.

Tablo 1: Model Performansi ve Stabilitesi (F1 +/- Std, 5 seed)
Tablo 2: Gurultu Etkisi ve Unseen Senaryo Analizi
Tablo 3: Cross-Dataset Genellenebilirlik (sadece mevcut veriyle doldurulur)
Tablo 4: Automata Parametre Duyarlilik Analizi
Tablo 5: Modellerin Calisma Suresi (Training & Inference)
"""

import json
import os


def load_results(results_path: str) -> dict:
    with open(results_path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Tablo 1: Model Performansi ve Stabilitesi
# ---------------------------------------------------------------------------

def table1_performance(results: dict) -> str:
    """F1 +/- Std -- 5 seed x 4 model x 2 dataset"""
    models  = ["LSTM", "GRU", "1D-CNN", "Automata"]
    datasets = ["skab", "batadal"]
    ds_labels = {"skab": "SKAB", "batadal": "BATADAL"}

    lines = []
    lines.append("### Tablo 1: Model Performansi ve Stabilitesi (Ortalama F1-score +/- Standart Sapma)")
    lines.append("")
    lines.append("| Model | SKAB F1 | BATADAL F1 |")
    lines.append("|-------|---------|------------|")

    for model in models:
        row = f"| {model} |"
        for ds in datasets:
            v = results[ds].get(model, {})
            mean = v.get("f1_mean", 0.0)
            std  = v.get("f1_std",  0.0)
            row += f" {mean:.4f} ± {std:.4f} |"
        lines.append(row)

    lines.append("")
    lines.append("> *5 farkli random seed [42, 123, 2026, 7, 999] ile elde edilen ortalama ve standart sapma.*")
    lines.append("> *SKAB icin GroupKFold (k=5) fold ortalamasi alinmistir.*")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tablo 2: Gurultu Etkisi ve Unseen Analizi
# ---------------------------------------------------------------------------

def table2_noise_unseen(results: dict) -> str:
    """Orijinal vs Gurultulu F1 + Unseen Det.Rate ve Map.Acc"""
    models = ["LSTM", "GRU", "1D-CNN", "Automata"]
    noise  = results.get("noise", {})
    unseen = results.get("unseen", {})

    lines = []
    lines.append("### Tablo 2: Gurultu Etkisi ve Unseen Senaryo Analizi")
    lines.append("")
    lines.append("| Model | Orijinal F1 | Gurultulu F1 | F1 Dususu | Unseen Det. Rate | Unseen Map. Acc. |")
    lines.append("|-------|-------------|--------------|-----------|-----------------|-----------------|")

    for model in models:
        v = noise.get(model, {})
        orig  = v.get("original", {}).get("f1", 0.0)
        noisy = v.get("noisy",    {}).get("f1", 0.0)
        drop  = orig - noisy

        # Unseen sadece Automata icin anlamli, diger modeller icin N/A
        if model == "Automata":
            det_rate = unseen.get("detection_rate", 0.0)
            map_acc  = unseen.get("mapping_accuracy", 0.0)
            det_str  = f"{det_rate:.4f}"
            map_str  = f"{map_acc:.4f}"
        else:
            det_str = "N/A"
            map_str = "N/A"

        drop_str = f"{drop:+.4f}"
        lines.append(f"| {model} | {orig:.4f} | {noisy:.4f} | {drop_str} | {det_str} | {map_str} |")

    lines.append("")
    u_count = unseen.get("unseen_count", 0)
    u_total = unseen.get("total_patterns", 0)
    u_rate  = unseen.get("unseen_rate", 0.0)
    lines.append(f"> *Gurultu: Gaussian noise (std=0.1) BATADAL test setine eklenmistir.*")
    lines.append(f"> *Unseen: {u_count}/{u_total} pattern ({u_rate:.2%}) egitim SAX sozlugunde bulunmamistir.*")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tablo 3: Cross-Dataset Genellenebilirlik
# ---------------------------------------------------------------------------

def table3_cross_dataset(results: dict) -> str:
    """
    Cross-dataset matrisi. Pipeline'da cross-dataset deneyi ayri calistirilmadiysa
    mevcut sonuclar uzerinden kosegen doldurulur, diger hucreler '-' olarak kalir.
    """
    models  = ["LSTM", "GRU", "1D-CNN", "Automata"]
    lines   = []
    lines.append("### Tablo 3: Cross-Dataset Performans Karsilastirmasi (F1-score)")
    lines.append("")
    lines.append("| Train / Test | SKAB | BATADAL |")
    lines.append("|-------------|------|---------|")

    skab_row    = "| Train: SKAB |"
    batadal_row = "| Train: BATADAL |"

    for model in models:
        skab_f1    = results["skab"].get(model, {}).get("f1_mean", None)
        batadal_f1 = results["batadal"].get(model, {}).get("f1_mean", None)

        # Kosegen: ayni dataset
        skab_str    = f"{skab_f1:.4f}" if skab_f1 is not None else "-"
        batadal_str = f"{batadal_f1:.4f}" if batadal_f1 is not None else "-"

    # Model bazinda ayri satirlar
    lines.append("| | **SKAB (test)** | **BATADAL (test)** |")
    lines.append("|---|---|----|")
    for model in models:
        sk = results["skab"].get(model, {}).get("f1_mean", 0.0)
        bt = results["batadal"].get(model, {}).get("f1_mean", 0.0)
        lines.append(f"| {model} (train=kendi seti) | {sk:.4f} | {bt:.4f} |")

    lines.append("")
    lines.append("> *Her model kendi egitim setinde egitilip kendi test setinde degerlendirilmistir.*")
    lines.append("> *Capraz veri seti (Train:SKAB -> Test:BATADAL) deneyi ek calismada yapilabilir.*")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tablo 4: Automata Parametre Duyarlilik Analizi
# ---------------------------------------------------------------------------

def table4_sweep(results: dict) -> str:
    """window_size x alphabet_size sweep sonuclari"""
    sweep = results.get("sweep", {})
    window_sizes   = sorted(sweep.keys(), key=int)
    alphabet_sizes = sorted(next(iter(sweep.values())).keys(), key=int) if sweep else []

    lines = []
    lines.append("### Tablo 4: Automata Parametre Duyarlilik Analizi")
    lines.append("")

    # F1 Tablosu
    lines.append("#### 4a: F1-score")
    header = "| Parametre |" + "".join(f" Deger={a} |" for a in alphabet_sizes)
    sep    = "|-----------|" + "".join("---------|" for _ in alphabet_sizes)
    lines.append(header)
    lines.append(sep)

    for ws in window_sizes:
        row = f"| Window Size={ws} |"
        for ab in alphabet_sizes:
            f1 = sweep.get(ws, {}).get(ab, {}).get("f1", 0.0)
            row += f" {f1:.4f} |"
        lines.append(row)

    lines.append("")

    # State Sayisi Tablosu
    lines.append("#### 4b: State Sayisi")
    lines.append(header)
    lines.append(sep)

    for ws in window_sizes:
        row = f"| Window Size={ws} |"
        for ab in alphabet_sizes:
            n_states = sweep.get(ws, {}).get(ab, {}).get("n_states", 0)
            row += f" {n_states} |"
        lines.append(row)

    lines.append("")
    lines.append("> *Parametre degisiminin model performansi ve state sayisi uzerindeki etkisi.*")
    lines.append("> *Sabit karsilastirma parametreleri: window_size=4, alphabet_size=3.*")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tablo 5: Calisma Suresi
# ---------------------------------------------------------------------------

def table5_runtime(results: dict) -> str:
    """Training ve Inference sureleri"""
    models = ["LSTM", "GRU", "1D-CNN", "Automata"]
    lines  = []
    lines.append("### Tablo 5: Modellerin Calisma Suresi (Runtime) Karsilastirmasi")
    lines.append("")
    lines.append("| Model | SKAB Egitim (sn) | SKAB Inference (sn) | BATADAL Egitim (sn) | BATADAL Inference (sn) |")
    lines.append("|-------|-----------------|--------------------|--------------------|----------------------|")

    for model in models:
        sk = results["skab"].get(model, {})
        bt = results["batadal"].get(model, {})
        sk_tr = sk.get("train_time_mean", 0.0)
        sk_in = sk.get("infer_time_mean", 0.0)
        bt_tr = bt.get("train_time_mean", 0.0)
        bt_in = bt.get("infer_time_mean", 0.0)
        lines.append(f"| {model} | {sk_tr:.2f} | {sk_in:.4f} | {bt_tr:.2f} | {bt_in:.4f} |")

    lines.append("")
    lines.append("> *Sureler 5 seed ortalamasi olarak raporlanmistir. GPU: NVIDIA RTX 5070 Laptop.*")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ana Fonksiyon
# ---------------------------------------------------------------------------

def generate_all_tables(results_path: str, output_dir: str) -> str:
    """
    Tum 5 tabloyu uretir ve hem dosyaya yazar hem de string olarak dondurur.
    """
    results = load_results(results_path)
    os.makedirs(output_dir, exist_ok=True)

    sections = [
        "## Deney Sonuclari ve Karsilastirmali Analiz Tablolari\n",
        table1_performance(results),
        table2_noise_unseen(results),
        table3_cross_dataset(results),
        table4_sweep(results),
        table5_runtime(results),
    ]

    full_text = "\n".join(sections)

    out_path = os.path.join(output_dir, "tables.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    # Ayrica her tabloyu ayri dosyaya da yaz
    for i, (fn, content) in enumerate([
        ("tablo1_performans.md",  table1_performance(results)),
        ("tablo2_gurultu.md",     table2_noise_unseen(results)),
        ("tablo3_crossdataset.md",table3_cross_dataset(results)),
        ("tablo4_sweep.md",       table4_sweep(results)),
        ("tablo5_runtime.md",     table5_runtime(results)),
    ], start=1):
        with open(os.path.join(output_dir, fn), "w", encoding="utf-8") as f:
            f.write(content)

    print(f"Tablolar uretildi -> {out_path}")
    return full_text


if __name__ == "__main__":
    import yaml, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    tables = generate_all_tables(
        results_path="results/results.json",
        output_dir="results/tables/"
    )
    print(tables)
