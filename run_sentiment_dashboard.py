"""
run_sentiment_dashboard.py
===========================
Dashboard für Financial Sentiment Analysis.
4-Panel-Chart: Verteilung | Modellvergleich | Konfusionsmatrizen | Sentiment-Zeitreihe

Ausführungslogik strikt getrennt von Kernmodul (financial_sentiment_model.py).
Keine Finanzberechnungen hier — nur Datenaufbereitung und Visualisierung.

Zeitkonvention: kein Zeitindex (statischer Datensatz) — x-Achse = Sample-Index.
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split

from financial_sentiment_model import (
    load_phrasebank,
    vader_predict,
    vader_compound_scores,
    build_lr_pipeline,
    evaluate,
    sentiment_timeseries,
    LABEL_NAMES,
    TEST_SIZE,
    RANDOM_STATE,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Designkonstanten — Dashboard-Palette
# ---------------------------------------------------------------------------

CLR_NEG    = "#E05C5C"   # Rot — negativ
CLR_NEU    = "#8C8C8C"   # Grau — neutral
CLR_POS    = "#4CAF82"   # Grün — positiv
CLR_VADER  = "#5B8CCC"   # Blau — VADER
CLR_LR     = "#F0A500"   # Amber — TF-IDF+LR
CLR_BG     = "#0F1117"   # Fast-Schwarz Hintergrund
CLR_PANEL  = "#1A1D27"   # Panel-Hintergrund
CLR_TEXT   = "#E8EAF0"   # Haupttext
CLR_SUBTLE = "#4A4E5A"   # Subtiler Kontrast

LABEL_COLORS = [CLR_NEG, CLR_NEU, CLR_POS]

SAVE_PNG: bool = True
OUTPUT_PATH: str = "financial_sentiment_dashboard.png"

# ---------------------------------------------------------------------------
# Matplotlib Stil
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "figure.facecolor":    CLR_BG,
    "axes.facecolor":      CLR_PANEL,
    "axes.edgecolor":      CLR_SUBTLE,
    "axes.labelcolor":     CLR_TEXT,
    "axes.titlecolor":     CLR_TEXT,
    "xtick.color":         CLR_TEXT,
    "ytick.color":         CLR_TEXT,
    "text.color":          CLR_TEXT,
    "grid.color":          CLR_SUBTLE,
    "grid.alpha":          0.35,
    "font.family":         "monospace",
    "axes.spines.top":     False,
    "axes.spines.right":   False,
})

# ---------------------------------------------------------------------------
# Daten laden und Modelle trainieren
# ---------------------------------------------------------------------------

def run_pipeline():
    """Lädt Daten, trainiert beide Modelle, gibt alle Metriken zurück."""

    texts, labels = load_phrasebank()
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    # VADER (regelbasiert — kein Training)
    vader_preds    = vader_predict(X_test)
    vader_scores   = vader_compound_scores(X_test)
    vader_metrics  = evaluate(y_test, vader_preds, "VADER Baseline")

    # TF-IDF + Logistic Regression
    pipe = build_lr_pipeline()
    pipe.fit(X_train, y_train)
    lr_preds   = list(pipe.predict(X_test))
    lr_metrics = evaluate(y_test, lr_preds, "TF-IDF + LogReg")

    # Sentiment-Zeitreihe (gesamter Datensatz, sortiert nach True-Label)
    sort_idx   = np.argsort(labels)
    ts_data    = sentiment_timeseries(
        [texts[i] for i in sort_idx],
        [labels[i] for i in sort_idx],
        vader_predict(texts),          # VADER auf gesamtem Set
        list(pipe.predict(texts)),     # LR auf gesamtem Set
    )

    return {
        "y_test":        y_test,
        "vader_preds":   vader_preds,
        "vader_scores":  vader_scores,
        "lr_preds":      lr_preds,
        "vader_metrics": vader_metrics,
        "lr_metrics":    lr_metrics,
        "ts":            ts_data,
        "n_train":       len(X_train),
        "n_test":        len(X_test),
        "n_total":       len(texts),
    }

# ---------------------------------------------------------------------------
# Panel-Funktionen
# ---------------------------------------------------------------------------

def plot_label_distribution(ax, y_test: list, vader_preds: list, lr_preds: list) -> None:
    """
    Panel 1: Balkendiagramm — Label-Verteilung (True vs. VADER vs. LR).
    Zeigt wie jedes Modell die Klassen verteilt.
    """
    x      = np.arange(3)
    width  = 0.25

    def counts(preds):
        return [preds.count(k) for k in [0, 1, 2]]

    bars_true  = ax.bar(x - width,      counts(y_test),      width, color=CLR_SUBTLE,  label="True Labels",    alpha=0.9, zorder=3)
    bars_vader = ax.bar(x,              counts(vader_preds), width, color=CLR_VADER,   label="VADER",          alpha=0.9, zorder=3)
    bars_lr    = ax.bar(x + width,      counts(lr_preds),    width, color=CLR_LR,      label="TF-IDF + LogReg",alpha=0.9, zorder=3)

    for bars in [bars_true, bars_vader, bars_lr]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + 0.1,
                    str(int(h)), ha="center", va="bottom", fontsize=8, color=CLR_TEXT
                )

    ax.set_xticks(x)
    ax.set_xticklabels(["Negative", "Neutral", "Positive"], fontsize=9)
    ax.set_ylabel("Sample Count", fontsize=9)
    ax.set_title("Predicted Class Distribution (Test Set)", fontsize=10, pad=8)
    ax.legend(fontsize=8, framealpha=0.2)
    ax.grid(axis="y", zorder=0)
    ax.set_ylim(0, max(counts(y_test) + counts(vader_preds) + counts(lr_preds)) * 1.25)


def plot_model_comparison(ax, vader_metrics: dict, lr_metrics: dict) -> None:
    """
    Panel 2: Horizontaler Balkenvergleich — Accuracy und Macro-F1 beider Modelle.
    Majority-Class-Baseline als Referenzlinie.
    """
    metrics    = ["Accuracy", "Macro F1"]
    vader_vals = [vader_metrics["accuracy"], vader_metrics["f1_macro"]]
    lr_vals    = [lr_metrics["accuracy"],    lr_metrics["f1_macro"]]
    baseline   = vader_metrics["majority_baseline"]  # identisch für beide

    y     = np.arange(2)
    height = 0.3

    ax.barh(y + height / 2, vader_vals, height, color=CLR_VADER, label="VADER Baseline", alpha=0.9)
    ax.barh(y - height / 2, lr_vals,    height, color=CLR_LR,    label="TF-IDF + LogReg", alpha=0.9)

    # Majority-Class-Baseline
    ax.axvline(baseline, color=CLR_SUBTLE, linestyle="--", linewidth=1.2, label=f"Majority Class ({baseline:.0%})")

    # Werte-Annotation
    for val, y_pos in zip(vader_vals, y + height / 2):
        ax.text(val + 0.01, y_pos, f"{val:.3f}", va="center", fontsize=9, color=CLR_VADER)
    for val, y_pos in zip(lr_vals, y - height / 2):
        ax.text(val + 0.01, y_pos, f"{val:.3f}", va="center", fontsize=9, color=CLR_LR)

    ax.set_yticks(y)
    ax.set_yticklabels(metrics, fontsize=9)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Score", fontsize=9)
    ax.set_title("Model Performance Comparison", fontsize=10, pad=8)
    ax.legend(fontsize=8, framealpha=0.2)
    ax.grid(axis="x", zorder=0)


def plot_confusion_matrix(ax, cm: np.ndarray, title: str, color: str) -> None:
    """
    Panel 3a/3b: Konfusionsmatrix als Heatmap.

    cm: 3×3 int-Array — Zeilen = True, Spalten = Predicted
    """
    # Normierte Darstellung (zeilenweise)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    # Farbpalette: Panel-Hintergrund → Modellfarbe
    cmap = LinearSegmentedColormap.from_list("cm", [CLR_PANEL, color], N=256)
    ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    for i in range(3):
        for j in range(3):
            raw   = cm[i, j]
            normd = cm_norm[i, j]
            txt_color = CLR_TEXT if normd < 0.6 else CLR_BG
            ax.text(j, i, f"{raw}\n({normd:.0%})",
                    ha="center", va="center", fontsize=8, color=txt_color)

    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(["neg", "neu", "pos"], fontsize=8)
    ax.set_yticklabels(["neg", "neu", "pos"], fontsize=8)
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("True", fontsize=8)
    ax.set_title(title, fontsize=10, pad=8, color=color)


def plot_sentiment_timeseries(ax, ts: dict) -> None:
    """
    Panel 4: Sentiment-Zeitreihe über alle Samples (sortiert nach True Label).

    Numerische Kodierung: negative=-1, neutral=0, positive=+1
    Zeigt True Labels als Scatter + VADER/LR als gleitenden Mittelwert.
    """
    idx   = ts["indices"]
    true  = ts["true"]
    vader = ts["vader"]
    lr    = ts["lr"]

    # True Labels als farbige Scatter
    colors_true = [LABEL_COLORS[int(v + 1)] for v in true]
    ax.scatter(idx, true + np.random.default_rng(42).uniform(-0.05, 0.05, len(true)),
               c=colors_true, s=18, alpha=0.6, zorder=3, label="_nolegend_")

    # Gleitender Mittelwert (Fenster=7) für Modell-Prognosen
    window = 7

    def rolling_mean(arr, w):
        return np.convolve(arr, np.ones(w) / w, mode="valid")

    pad  = window // 2
    idx_r = idx[pad: pad + len(rolling_mean(vader, window))]
    ax.plot(idx_r, rolling_mean(vader, window), color=CLR_VADER,
            linewidth=1.8, alpha=0.85, label="VADER (rolling mean)")
    ax.plot(idx_r, rolling_mean(lr, window),    color=CLR_LR,
            linewidth=1.8, alpha=0.85, label="TF-IDF+LR (rolling mean)")

    # Trennlinien zwischen Klassen (sortierter Datensatz)
    n   = len(true)
    n_neg = int((true == -1).sum())
    n_neu = int((true ==  0).sum())
    for xpos, label in [(n_neg, "neutral →"), (n_neg + n_neu, "positive →")]:
        ax.axvline(xpos, color=CLR_SUBTLE, linestyle=":", linewidth=1.0)
        ax.text(xpos + 0.5, 1.12, label, fontsize=7, color=CLR_SUBTLE, va="top")

    ax.axhline(0, color=CLR_SUBTLE, linewidth=0.7, linestyle="--")
    ax.set_ylim(-1.3, 1.3)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["Negative", "Neutral", "Positive"], fontsize=8)
    ax.set_xlabel("Sample Index (sorted by true label)", fontsize=9)
    ax.set_title("Sentiment Time Series — All Samples", fontsize=10, pad=8)

    legend_patches = [
        mpatches.Patch(color=CLR_NEG, label="True: Negative"),
        mpatches.Patch(color=CLR_NEU, label="True: Neutral"),
        mpatches.Patch(color=CLR_POS, label="True: Positive"),
    ]
    ax.legend(handles=legend_patches + [
        plt.Line2D([0], [0], color=CLR_VADER, lw=2, label="VADER (rolling mean)"),
        plt.Line2D([0], [0], color=CLR_LR,    lw=2, label="TF-IDF+LR (rolling mean)"),
    ], fontsize=7.5, framealpha=0.2, ncol=2)


# ---------------------------------------------------------------------------
# Dashboard zusammenbauen
# ---------------------------------------------------------------------------

def build_dashboard(data: dict) -> plt.Figure:
    """
    Erstellt das 4-Panel-Dashboard.

    Layout:
        [Panel 1: Distribution] [Panel 2: Comparison  ]
        [Panel 3a: CM VADER   ] [Panel 3b: CM LR      ]
        [Panel 4: Timeseries — full width              ]
    """
    fig = plt.figure(figsize=(14, 12), facecolor=CLR_BG)
    gs  = gridspec.GridSpec(
        3, 2,
        figure=fig,
        hspace=0.55,
        wspace=0.35,
        height_ratios=[1.0, 1.0, 0.9],
    )

    ax1  = fig.add_subplot(gs[0, 0])  # Distribution
    ax2  = fig.add_subplot(gs[0, 1])  # Model Comparison
    ax3a = fig.add_subplot(gs[1, 0])  # Confusion Matrix VADER
    ax3b = fig.add_subplot(gs[1, 1])  # Confusion Matrix LR
    ax4  = fig.add_subplot(gs[2, :])  # Sentiment Timeseries

    plot_label_distribution(ax1, data["y_test"], data["vader_preds"], data["lr_preds"])
    plot_model_comparison(ax2, data["vader_metrics"], data["lr_metrics"])
    plot_confusion_matrix(ax3a, data["vader_metrics"]["conf_matrix"], "Confusion Matrix — VADER", CLR_VADER)
    plot_confusion_matrix(ax3b, data["lr_metrics"]["conf_matrix"],    "Confusion Matrix — TF-IDF+LR", CLR_LR)
    plot_sentiment_timeseries(ax4, data["ts"])

    # Haupt-Titel
    fig.suptitle(
        "Financial Sentiment Analysis  ·  VADER vs. TF-IDF + Logistic Regression",
        fontsize=13, fontweight="bold", color=CLR_TEXT, y=0.98,
    )

    # Footer mit Datensatz-Info
    footer = (
        f"Dataset: Financial PhraseBank (sentences_allagree)  "
        f"·  N={data['n_total']} samples  "
        f"·  Train/Test = {data['n_train']}/{data['n_test']}  "
        f"·  Labels: 0=negative, 1=neutral, 2=positive"
    )
    fig.text(0.5, 0.005, footer, ha="center", fontsize=7.5, color=CLR_SUBTLE)

    return fig


# ---------------------------------------------------------------------------
# Konsolenbericht
# ---------------------------------------------------------------------------

def print_report(data: dict) -> None:
    """Gibt Konsolenbericht mit Kernmetriken aus."""
    sep = "─" * 52

    print(f"\n{sep}")
    print("  Financial Sentiment Analysis — Report")
    print(sep)
    print(f"  Dataset   : Financial PhraseBank (sentences_allagree)")
    print(f"  Samples   : {data['n_total']} total | "
          f"Train {data['n_train']} | Test {data['n_test']}")
    print(sep)

    for metrics, color in [
        (data["vader_metrics"], "VADER Baseline   "),
        (data["lr_metrics"],    "TF-IDF + LogReg  "),
    ]:
        print(f"\n  {color}")
        print(f"    Accuracy  : {metrics['accuracy']:.4f}")
        print(f"    Macro F1  : {metrics['f1_macro']:.4f}")
        print(f"    Baseline  : {metrics['majority_baseline']:.4f} (majority class)")
        print()
        for line in metrics["report"].splitlines():
            print(f"    {line}")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Hauptausführung
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Lade Daten und trainiere Modelle...")
    data = run_pipeline()

    print_report(data)

    print("Erstelle Dashboard...")
    fig = build_dashboard(data)

    if SAVE_PNG:
        import os
        output_dir = os.path.dirname(OUTPUT_PATH)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight",
                    facecolor=CLR_BG, edgecolor="none")
        print(f"Dashboard gespeichert: {OUTPUT_PATH}")

    plt.close(fig)
    print("Fertig.")