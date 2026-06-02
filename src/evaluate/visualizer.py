"""
visualizer.py - Zorunlu Gorselleştirmeler (5 adet)

1. Confusion Matrix        - Her model icin
2. ROC / Precision-Recall  - Her model icin
3. Automata State Diagram  - networkx ile
4. Transition Probability Heatmap - seaborn ile
5. Parametre Duyarlilik Grafikleri - window x alphabet sweep

Kullanim:
    python src/evaluate/visualizer.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # GUI gerektirmez, dosyaya kaydeder
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ─────────────────────────────────────────────────────────────────────────────
# Stil Ayarlari
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "LSTM":     "#4C72B0",
    "GRU":      "#DD8452",
    "1D-CNN":   "#55A868",
    "Automata": "#C44E52",
}

plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi":   150,
})

MODELS   = ["LSTM", "GRU", "1D-CNN", "Automata"]
DATASETS = {"skab": "SKAB", "batadal": "BATADAL"}


# ─────────────────────────────────────────────────────────────────────────────
# Gorsel 1: Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────

def _make_confusion_matrix(tp, fp, fn, tn):
    return np.array([[tn, fp], [fn, tp]])


def plot_confusion_matrices(results: dict, out_dir: str):
    """
    Her model x her dataset icin Confusion Matrix.
    results.json'dan precision/recall/f1/accuracy kullanilarak
    yaklasik TP/FP/FN/TN degerleri hesaplanir.
    """
    fig, axes = plt.subplots(
        len(MODELS), len(DATASETS),
        figsize=(10, 14)
    )
    fig.suptitle("Confusion Matrix - Tum Modeller", fontsize=15, fontweight="bold", y=1.01)

    for col, (ds_key, ds_label) in enumerate(DATASETS.items()):
        for row, model in enumerate(MODELS):
            ax = axes[row][col]
            v  = results[ds_key].get(model, {})

            prec = v.get("precision_mean", 0.0)
            rec  = v.get("recall_mean",    0.0)
            acc  = v.get("accuracy_mean",  0.0)
            f1   = v.get("f1_mean",        0.0)

            # Yaklasik CM degerleri (N=1000 normalize)
            N  = 1000
            tp = int(rec * prec * N / max(f1, 1e-9)) if f1 > 0 else 0
            fp = int(tp / max(prec, 1e-9) - tp)      if prec > 0 else 0
            fn = int(tp / max(rec, 1e-9) - tp)       if rec  > 0 else 0
            tn = N - tp - fp - fn

            cm = np.array([[max(tn, 0), max(fp, 0)],
                           [max(fn, 0), max(tp, 0)]])

            sns.heatmap(
                cm, annot=True, fmt="d", ax=ax,
                cmap="Blues", cbar=False,
                xticklabels=["Normal", "Anomali"],
                yticklabels=["Normal", "Anomali"],
                linewidths=0.5, linecolor="gray",
            )
            ax.set_title(f"{model} — {ds_label}\nF1={f1:.4f}", fontsize=10)
            ax.set_ylabel("Gercek" if col == 0 else "")
            ax.set_xlabel("Tahmin")

    plt.tight_layout()
    path = os.path.join(out_dir, "gorsel1_confusion_matrix.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [1] Confusion Matrix kaydedildi -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Gorsel 2: ROC / Precision-Recall Egrisi (tahmini)
# ─────────────────────────────────────────────────────────────────────────────

def _approx_roc(precision, recall, f1):
    """
    Elimizdeki tek nokta (precision, recall) uzerinden
    yaklasik bir egri cizer; AUC tahmini icin kullanilir.
    Gercek egri icin ham skorlar gerekir.
    """
    # Basit 3 nokta: (0,1), (recall, precision), (1,0)
    rec_pts  = np.array([0.0, recall,    1.0])
    prec_pts = np.array([1.0, precision, 0.0])
    return rec_pts, prec_pts


def plot_roc_pr(results: dict, out_dir: str):
    """Precision-Recall egrisi (mevcut metrikler uzerinden tahmini)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Precision-Recall Egrisi", fontsize=14, fontweight="bold")

    for ax, (ds_key, ds_label) in zip(axes, DATASETS.items()):
        for model in MODELS:
            v    = results[ds_key].get(model, {})
            prec = v.get("precision_mean", 0.0)
            rec  = v.get("recall_mean",    0.0)
            f1   = v.get("f1_mean",        0.0)

            rec_pts, prec_pts = _approx_roc(prec, rec, f1)
            auc = np.trapezoid(prec_pts, rec_pts)
            auc = abs(auc)

            ax.plot(rec_pts, prec_pts,
                    color=COLORS[model], linewidth=2,
                    marker="o", markersize=5,
                    label=f"{model} (AUC~{auc:.3f})")

        ax.set_title(ds_label)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        # Cizgi referansi
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)

    plt.tight_layout()
    path = os.path.join(out_dir, "gorsel2_precision_recall.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [2] PR Egrisi kaydedildi -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Gorsel 3: Automata State Diagram (networkx)
# ─────────────────────────────────────────────────────────────────────────────

def plot_state_diagram(config: dict, out_dir: str):
    """
    Automata'yi BATADAL uzerinde fit edip state diagram'i cizer.
    En yuksek trafikli 20 gecisi gosterir (okunabilirlik icin).
    """
    try:
        import networkx as nx
        import yaml
    except ImportError:
        print("  [3] networkx/yaml kurulu degil, atlanıyor.")
        return

    sys.path.insert(0, ".")
    from src.data.loader import load_batadal, get_batadal_features_labels, get_batadal_temporal_split
    from src.data.preprocess import Preprocessor, handle_missing
    from src.models.automata.automata import ProbabilisticAutomata

    df = handle_missing(load_batadal(config))
    X, y = get_batadal_features_labels(df, config)
    X_tr, _, _, y_tr, _, _ = get_batadal_temporal_split(X, y, config)

    prep = Preprocessor(config)
    pc1  = prep.fit_transform_pca(X_tr).ravel()

    model = ProbabilisticAutomata(config)
    model.fit(pc1)

    # Gecis matrisini al
    trans = model.transition_matrix   # {state: {next_state: prob}}
    if not trans:
        print("  [3] Gecis matrisi bos, atlanıyor.")
        return

    # En yuksek 20 gecis
    edges = []
    for s, nexts in trans.items():
        for ns, prob in nexts.items():
            edges.append((s, ns, prob))
    edges = sorted(edges, key=lambda x: -x[2])[:20]

    G = nx.DiGraph()
    for s, ns, prob in edges:
        G.add_edge(s, ns, weight=prob)

    fig, ax = plt.subplots(figsize=(13, 9))
    pos = nx.spring_layout(G, seed=42, k=2.5)

    # Dugum rengi: anomali threshold'una gore
    threshold = config["automata"]["anomaly_threshold"]
    node_cols = []
    for node in G.nodes():
        # O node'dan cikan tum gecislerin min olasiligi
        out_probs = [d["weight"] for _, _, d in G.out_edges(node, data=True)]
        min_p = min(out_probs) if out_probs else 1.0
        node_cols.append("#C44E52" if min_p < threshold else "#4C72B0")

    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(edge_weights) if edge_weights else 1.0

    nx.draw_networkx_nodes(G, pos, node_color=node_cols,
                           node_size=800, alpha=0.9, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, font_color="white",
                            font_weight="bold", ax=ax)
    nx.draw_networkx_edges(G, pos,
                           width=[2 + 4 * w / max_w for w in edge_weights],
                           edge_color=edge_weights, edge_cmap=plt.cm.YlOrRd,
                           arrows=True, arrowsize=15,
                           connectionstyle="arc3,rad=0.1", ax=ax)
    edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6, ax=ax)

    legend_els = [
        mpatches.Patch(color="#4C72B0", label="Normal durum"),
        mpatches.Patch(color="#C44E52", label="Anomali riski (dusuk prob)"),
    ]
    ax.legend(handles=legend_els, loc="upper left", fontsize=9)
    ax.set_title("Automata State Diagram (En Yuksek 20 Gecis)", fontsize=13, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    path = os.path.join(out_dir, "gorsel3_state_diagram.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [3] State Diagram kaydedildi -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Gorsel 4: Transition Probability Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_transition_heatmap(config: dict, out_dir: str):
    """
    Automata gecis olasiliklari matrisi (ilk 15 state x ilk 15 state).
    """
    try:
        import yaml
    except ImportError:
        print("  [4] yaml kurulu degil, atlanıyor.")
        return

    sys.path.insert(0, ".")
    from src.data.loader import load_batadal, get_batadal_features_labels, get_batadal_temporal_split
    from src.data.preprocess import Preprocessor, handle_missing
    from src.models.automata.automata import ProbabilisticAutomata

    df = handle_missing(load_batadal(config))
    X, y = get_batadal_features_labels(df, config)
    X_tr, _, _, y_tr, _, _ = get_batadal_temporal_split(X, y, config)

    prep = Preprocessor(config)
    pc1  = prep.fit_transform_pca(X_tr).ravel()

    model = ProbabilisticAutomata(config)
    model.fit(pc1)

    trans = model.transition_matrix
    if not trans:
        print("  [4] Gecis matrisi bos, atlanıyor.")
        return

    states  = sorted(trans.keys())[:15]
    n       = len(states)
    matrix  = np.zeros((n, n))

    for i, s in enumerate(states):
        for j, ns in enumerate(states):
            matrix[i, j] = trans.get(s, {}).get(ns, 0.0)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        matrix, ax=ax,
        xticklabels=states, yticklabels=states,
        cmap="YlOrRd", annot=True, fmt=".2f",
        linewidths=0.3, linecolor="gray",
        cbar_kws={"label": "Gecis Olasiligi"},
        annot_kws={"size": 7},
    )
    ax.set_title("Automata Gecis Olasiligi Heatmap\n(Ilk 15 State)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Hedef State")
    ax.set_ylabel("Kaynak State")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)

    plt.tight_layout()
    path = os.path.join(out_dir, "gorsel4_transition_heatmap.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [4] Transition Heatmap kaydedildi -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Gorsel 5: Parametre Duyarlilik Grafikleri
# ─────────────────────────────────────────────────────────────────────────────

def plot_sensitivity(results: dict, out_dir: str):
    """
    window_size ve alphabet_size'in F1 ve state sayisi uzerine etkisi.
    2x2 subplot: F1 vs window, F1 vs alphabet, states vs window, states vs alphabet.
    """
    sweep = results.get("sweep", {})
    window_sizes   = sorted(sweep.keys(), key=int)
    alphabet_sizes = sorted(next(iter(sweep.values())).keys(), key=int) if sweep else []

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Automata Parametre Duyarlilik Analizi", fontsize=14, fontweight="bold")

    palette = plt.cm.tab10(np.linspace(0, 0.6, len(alphabet_sizes)))

    # ── Sol üst: F1 vs Window Size (alphabet sabit, her biri cizgi) ──────────
    ax = axes[0][0]
    for idx, ab in enumerate(alphabet_sizes):
        f1s = [sweep[ws][ab]["f1"] for ws in window_sizes]
        ax.plot([int(w) for w in window_sizes], f1s,
                marker="o", color=palette[idx], linewidth=2,
                label=f"Alphabet={ab}")
    ax.set_title("F1-score vs Window Size")
    ax.set_xlabel("Window Size")
    ax.set_ylabel("F1-score")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Sag üst: F1 vs Alphabet Size (window sabit) ──────────────────────────
    ax = axes[0][1]
    palette2 = plt.cm.tab10(np.linspace(0.3, 0.9, len(window_sizes)))
    for idx, ws in enumerate(window_sizes):
        f1s = [sweep[ws][ab]["f1"] for ab in alphabet_sizes]
        ax.plot([int(a) for a in alphabet_sizes], f1s,
                marker="s", color=palette2[idx], linewidth=2,
                label=f"Window={ws}")
    ax.set_title("F1-score vs Alphabet Size")
    ax.set_xlabel("Alphabet Size")
    ax.set_ylabel("F1-score")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Sol alt: State Sayisi vs Window Size ─────────────────────────────────
    ax = axes[1][0]
    for idx, ab in enumerate(alphabet_sizes):
        states = [sweep[ws][ab]["n_states"] for ws in window_sizes]
        ax.plot([int(w) for w in window_sizes], states,
                marker="o", color=palette[idx], linewidth=2,
                label=f"Alphabet={ab}")
    ax.set_title("State Sayisi vs Window Size")
    ax.set_xlabel("Window Size")
    ax.set_ylabel("State Sayisi")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Sag alt: State Sayisi vs Alphabet Size ───────────────────────────────
    ax = axes[1][1]
    for idx, ws in enumerate(window_sizes):
        states = [sweep[ws][ab]["n_states"] for ab in alphabet_sizes]
        ax.plot([int(a) for a in alphabet_sizes], states,
                marker="s", color=palette2[idx], linewidth=2,
                label=f"Window={ws}")
    ax.set_title("State Sayisi vs Alphabet Size")
    ax.set_xlabel("Alphabet Size")
    ax.set_ylabel("State Sayisi")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "gorsel5_sensitivity.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [5] Sensitivity Grafigi kaydedildi -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Ana Fonksiyon
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_figures(results_path: str, config_path: str, out_dir: str):
    import yaml
    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    os.makedirs(out_dir, exist_ok=True)
    print(f"\nGorseller uretiliyor -> {out_dir}\n")

    plot_confusion_matrices(results, out_dir)
    plot_roc_pr(results, out_dir)
    plot_state_diagram(config, out_dir)
    plot_transition_heatmap(config, out_dir)
    plot_sensitivity(results, out_dir)

    print("\nTum gorseller tamamlandi!")


if __name__ == "__main__":
    generate_all_figures(
        results_path="results/results.json",
        config_path="configs/experiments.yaml",
        out_dir="results/figures/",
    )
