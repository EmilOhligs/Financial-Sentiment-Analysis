"""
financial_sentiment_model.py
=============================
Finanz-Sentiment-Klassifikator: VADER (regelbasiert) vs. TF-IDF + Logistic Regression (supervised).

Datensatz: Financial PhraseBank (Malo et al. 2014) — Finanznachrichten und Analystenberichte,
           manuell gelabelt in negative / neutral / positive.

           Zwei Modi über USE_HUGGINGFACE (siehe unten):
           True  → echter Datensatz via Hugging Face (~2260 Sätze)
           False → eingebetteter Fallback-Datensatz (63 Sätze, offline-fähig)

Label-Mapping:  0 = negative  |  1 = neutral  |  2 = positive
Verteilung:     ~13% neg  |  ~28% neu  |  ~59% pos  (orientiert am Original)

Formel TF-IDF:
    TF(t, d)    = Häufigkeit von Term t in Dokument d
    IDF(t)      = log(N / df(t))   N = Anzahl Dokumente, df = Dokumente mit Term t
    TF-IDF(t,d) = TF(t,d) · IDF(t)

Formel Logistic Regression (Softmax für 3 Klassen):
    P(y=k | x) = exp(w_k · x) / Σ_j exp(w_j · x)

VADER Compound Score:
    Compound ∈ [-1, 1]
    Positiv:  compound >= 0.05
    Negativ:  compound <= -0.05
    Neutral:  sonst

Annahmen:
    - Sätze sind tokenisierbar mit Standard-Whitespace-Tokenisierung
    - Kein Stemming (Finanzterme sind morphologisch sensitiv)
    - Log-Return-Renditen werden nicht berechnet — rein NLP-Modul
"""

import warnings
import numpy as np
from collections import Counter
from typing import Tuple, Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
from nltk.sentiment.vader import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# SWITCH: Datensatz-Quelle
# ---------------------------------------------------------------------------
# True  → Hugging Face datasets (pip install datasets erforderlich, Netzwerk)
#         ~2260 Sätze — empfohlen für realistische Modellevaluation
# False → eingebetteter Fallback-Datensatz (63 Sätze, offline, kein pip)
#         geeignet für Offline-Entwicklung und schnelle Tests
USE_HUGGINGFACE: bool = True

# ---------------------------------------------------------------------------
# Konstanten — alle benannt, keine hart codierten Magic Numbers
# ---------------------------------------------------------------------------

# VADER Schwellwerte (Hutto & Gilbert 2014, Tab. 5)
VADER_POS_THRESHOLD: float = 0.05
VADER_NEG_THRESHOLD: float = -0.05

# TF-IDF Parameter
TFIDF_MAX_FEATURES: int = 5_000
TFIDF_NGRAM_RANGE: Tuple[int, int] = (1, 2)

# Logistic Regression
LR_MAX_ITER: int = 1_000
LR_C: float = 1.0

# Train/Test Split
TEST_SIZE: float = 0.20
RANDOM_STATE: int = 42

# Label-Definitionen
LABEL_NAMES: List[str] = ["negative", "neutral", "positive"]
LABEL_MAP: Dict[str, int] = {"negative": 0, "neutral": 1, "positive": 2}

# ---------------------------------------------------------------------------
# Eingebetteter Fallback-Datensatz (USE_HUGGINGFACE = False)
# ---------------------------------------------------------------------------

RAW_DATA: List[Tuple[str, str]] = [
    # POSITIVE
    ("Operating profit rose to EUR 13.1 mn from EUR 8.7 mn in the same period last year.", "positive"),
    ("Net sales of Tieto increased by 1.9 percent in 2008.", "positive"),
    ("The company said it expects its operating result to improve.", "positive"),
    ("The rights issue will strengthen the company balance sheet.", "positive"),
    ("Profit before taxes amounted to EUR 1.1 mn, up from EUR 0.6 mn.", "positive"),
    ("Sales increased by 10 percent in the second quarter.", "positive"),
    ("The acquisition of the company has been completed successfully.", "positive"),
    ("The company reported strong earnings growth in the third quarter.", "positive"),
    ("Revenue increased to EUR 5.4 bn from EUR 4.9 bn the year before.", "positive"),
    ("The group posted a net profit of EUR 2.1 mn for the period.", "positive"),
    ("Demand for the company products remained strong throughout the year.", "positive"),
    ("The new product launch exceeded market expectations significantly.", "positive"),
    ("Operating profit climbed 18 percent year on year.", "positive"),
    ("The company successfully completed refinancing of its debt.", "positive"),
    ("Earnings per share rose to EUR 1.47 from EUR 1.21.", "positive"),
    ("The company has won a significant contract worth EUR 50 mn.", "positive"),
    ("Net profit for the period totaled EUR 6.0 mn versus EUR 4.7 mn.", "positive"),
    ("The board of directors proposed an increase in dividend to EUR 0.50.", "positive"),
    ("Customer satisfaction scores improved markedly across all segments.", "positive"),
    ("The expansion into new markets has contributed positively to earnings.", "positive"),
    ("Cash flow from operations increased by 22 percent.", "positive"),
    ("The company achieved record sales in the fourth quarter.", "positive"),
    ("Profit margins improved due to cost optimization measures.", "positive"),
    ("The group has secured long-term financing on favorable terms.", "positive"),
    ("Order intake grew by 15 percent compared to the previous year.", "positive"),
    ("The company expects to return to profitability in the next quarter.", "positive"),
    ("Market share expanded in all key regions during the period.", "positive"),
    ("The strategic partnership is expected to drive future growth.", "positive"),
    ("Operating expenses declined as a proportion of revenue.", "positive"),
    ("The company reported better than expected results for the full year.", "positive"),
    ("Net interest income increased by EUR 12 mn year on year.", "positive"),
    ("The order backlog reached a record high of EUR 450 mn.", "positive"),
    ("EBITDA margin expanded by 2.3 percentage points.", "positive"),
    ("The company raised its full year earnings guidance.", "positive"),
    ("Free cash flow improved substantially in the second half.", "positive"),
    # NEUTRAL
    ("The company will publish its interim report on 12 February 2009.", "neutral"),
    ("Ahlstrom has operations in 22 countries.", "neutral"),
    ("The annual general meeting is scheduled for 25 March 2009.", "neutral"),
    ("According to the company the total amount of shares is 50 000 000.", "neutral"),
    ("The company is headquartered in Helsinki Finland.", "neutral"),
    ("The board of directors consists of five members.", "neutral"),
    ("The transaction is subject to regulatory approval.", "neutral"),
    ("The company employs approximately 12 000 people worldwide.", "neutral"),
    ("The financial year ends on 31 December.", "neutral"),
    ("The shares will be listed on the Helsinki Stock Exchange.", "neutral"),
    ("The company has operations in over 30 countries.", "neutral"),
    ("The chief executive officer will present the results on Thursday.", "neutral"),
    ("The merger is expected to close in the first half of 2009.", "neutral"),
    ("The company produces electronic components for the automotive industry.", "neutral"),
    ("The results will be announced after the market close.", "neutral"),
    ("The company has entered into a letter of intent.", "neutral"),
    ("The supervisory board approved the proposal.", "neutral"),
    ("The company has not disclosed the financial terms of the deal.", "neutral"),
    # NEGATIVE
    ("Operating profit declined to EUR 1.2 mn from EUR 5.5 mn.", "negative"),
    ("The company reported a net loss of EUR 2.4 mn for the period.", "negative"),
    ("Revenue fell by 8 percent compared to the previous year.", "negative"),
    ("The company faces significant headwinds in its core markets.", "negative"),
    ("Earnings per share decreased to EUR 0.12 from EUR 0.34.", "negative"),
    ("The group posted a loss before taxes of EUR 3.1 mn.", "negative"),
    ("The company announced a restructuring program affecting 500 jobs.", "negative"),
    ("Market conditions deteriorated sharply in the second half.", "negative"),
    ("Net sales fell by 12 percent due to weaker demand.", "negative"),
    ("The company lowered its full year earnings guidance.", "negative"),
]


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _validate_texts(texts: List[str], name: str = "texts") -> None:
    if not isinstance(texts, list):
        raise TypeError(f"{name} muss eine Liste sein, erhalten: {type(texts)}")
    if len(texts) == 0:
        raise ValueError(f"{name} darf nicht leer sein.")
    for i, t in enumerate(texts):
        if not isinstance(t, str):
            raise TypeError(f"{name}[{i}] muss str sein, erhalten: {type(t)}")
        if len(t.strip()) == 0:
            raise ValueError(f"{name}[{i}] ist ein leerer String.")


def _validate_labels(labels: List[int], name: str = "labels") -> None:
    valid = {0, 1, 2}
    for i, lbl in enumerate(labels):
        if lbl not in valid:
            raise ValueError(
                f"{name}[{i}] = {lbl} ist kein gültiges Label. Erlaubt: {valid}"
            )


# ---------------------------------------------------------------------------
# Datensatz laden
# ---------------------------------------------------------------------------

def load_phrasebank() -> Tuple[List[str], List[int]]:
    """
    Gibt den Financial PhraseBank-Datensatz zurück.

    Switch über USE_HUGGINGFACE am Anfang der Datei:
        True  → _load_from_huggingface() — ~2260 Sätze, Netzwerk erforderlich
        False → _load_fallback()         — 63 Sätze, offline

    Rückgabe ist in beiden Fällen identisch:
        texts  : List[str]
        labels : List[int] {0=negative, 1=neutral, 2=positive}
    """
    if USE_HUGGINGFACE:
        return _load_from_huggingface()
    else:
        return _load_fallback()


def _load_from_huggingface() -> Tuple[List[str], List[int]]:
    """
    Lädt Financial PhraseBank direkt als Parquet via pandas.
    Umgeht das veraltete Script-Format — kein load_dataset() nötig.

    Label-Mapping laut Datensatz-Dokumentation:
        0 = negative  |  1 = neutral  |  2 = positive
    """
    try:
        import pandas as pd
        url = (
            "https://huggingface.co/datasets/FinanceMTEB/financial_phrasebank"
            "/resolve/main/data/train-00000-of-00001.parquet"
            )
        df = pd.read_parquet(url)
        print("Spalten:", list(df.columns))
        print("Erste Zeile:", df.iloc[0].to_dict())

        if "text" not in df.columns or "label" not in df.columns:
            raise RuntimeError(f"Unerwartete Spalten: {list(df.columns)}")

        texts  = df["text"].tolist()
        labels = df["label"].tolist()

    except Exception as e:
        raise RuntimeError(
            f"Datensatz konnte nicht geladen werden: {e}\n"
            f"Alternativ: USE_HUGGINGFACE = False setzen."
        )

    if len(texts) < 100:
        raise RuntimeError(f"Datensatz zu klein: {len(texts)} Samples.")

    dist = Counter(labels)
    for k in [0, 1, 2]:
        if dist[k] == 0:
            raise RuntimeError(f"Klasse '{LABEL_NAMES[k]}' hat 0 Samples.")

    return texts, labels


def _load_fallback() -> Tuple[List[str], List[int]]:
    """Eingebetteter Fallback-Datensatz (63 Sätze, offline)."""
    texts  = [s for s, _ in RAW_DATA]
    labels = [LABEL_MAP[lbl] for _, lbl in RAW_DATA]

    dist = Counter(labels)
    for k, name in enumerate(LABEL_NAMES):
        if dist[k] == 0:
            raise RuntimeError(f"Klasse '{name}' hat 0 Samples im Fallback.")

    return texts, labels


# ---------------------------------------------------------------------------
# VADER Baseline
# ---------------------------------------------------------------------------

def vader_predict(texts: List[str]) -> List[int]:
    """
    Regelbasierter Sentiment-Klassifikator via VADER.

    Formel Compound Score:
        compound = Σ(adjusted word scores) / √(Σ(adjusted word scores)² + 15)

    Schwellwerte:
        compound >= 0.05  → 2 (positiv)
        compound <= -0.05 → 0 (negativ)
        sonst             → 1 (neutral)
    """
    _validate_texts(texts)

    sia   = SentimentIntensityAnalyzer()
    preds = []

    for text in texts:
        compound = sia.polarity_scores(text)["compound"]
        if compound >= VADER_POS_THRESHOLD:
            preds.append(2)
        elif compound <= VADER_NEG_THRESHOLD:
            preds.append(0)
        else:
            preds.append(1)

    if len(set(preds)) < 2:
        raise RuntimeError(
            f"VADER gibt nur Klasse {set(preds)} aus — Lexikon unvollständig?"
        )

    return preds


def vader_compound_scores(texts: List[str]) -> List[float]:
    """Rohe VADER Compound Scores ∈ [-1, 1] für Zeitreihen-Visualisierung."""
    _validate_texts(texts)
    sia = SentimentIntensityAnalyzer()
    return [sia.polarity_scores(t)["compound"] for t in texts]


# ---------------------------------------------------------------------------
# TF-IDF + Logistic Regression
# ---------------------------------------------------------------------------

def build_lr_pipeline() -> Pipeline:
    """
    TF-IDF Vektorisierung → Logistic Regression (Softmax, 3 Klassen).

    TF-IDF: max_features=5000, ngram_range=(1,2), sublinear_tf=True
    LR:     C=1.0, solver=lbfgs, max_iter=1000
    """
    tfidf = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )
    lr = LogisticRegression(
        C=LR_C,
        max_iter=LR_MAX_ITER,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    return Pipeline([("tfidf", tfidf), ("lr", lr)])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(y_true: List[int], y_pred: List[int], model_name: str = "Modell") -> Dict:
    """
    Accuracy, Macro-F1, Confusion Matrix, Classification Report.
    Macro-F1 bevorzugt weil Datensatz class-imbalanced ist.
    """
    _validate_labels(y_true, "y_true")
    _validate_labels(y_pred, "y_pred")

    if len(y_true) != len(y_pred):
        raise ValueError(f"Längen: y_true={len(y_true)}, y_pred={len(y_pred)}")

    acc    = accuracy_score(y_true, y_pred)
    f1     = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm     = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    report = classification_report(y_true, y_pred, target_names=LABEL_NAMES, zero_division=0)
    majority_baseline = max(Counter(y_true).values()) / len(y_true)

    if acc < 0.30:
        raise RuntimeError(f"{model_name}: Accuracy={acc:.3f} unter 30%.")

    return {
        "accuracy":          acc,
        "f1_macro":          f1,
        "conf_matrix":       cm,
        "report":            report,
        "model_name":        model_name,
        "majority_baseline": majority_baseline,
    }


# ---------------------------------------------------------------------------
# Sentiment-Zeitreihe
# ---------------------------------------------------------------------------

def sentiment_timeseries(
    texts: List[str],
    labels_true: List[int],
    labels_pred_vader: List[int],
    labels_pred_lr: List[int],
) -> Dict:
    """Sentiment-Zeitreihe für Dashboard. Kodierung: neg=-1, neu=0, pos=+1."""
    _validate_texts(texts)
    n = len(texts)
    for arr, name in [
        (labels_true,       "labels_true"),
        (labels_pred_vader, "labels_pred_vader"),
        (labels_pred_lr,    "labels_pred_lr"),
    ]:
        if len(arr) != n:
            raise ValueError(f"{name} hat Länge {len(arr)}, erwartet {n}")

    def encode(labels):
        return np.array([l - 1 for l in labels], dtype=float)

    return {
        "indices": np.arange(n),
        "true":    encode(labels_true),
        "vader":   encode(labels_pred_vader),
        "lr":      encode(labels_pred_lr),
        "texts":   texts,
    }


# ---------------------------------------------------------------------------
# Validierungsbeispiel
# ---------------------------------------------------------------------------

def _run_validation_example() -> None:
    test_cases = [
        ("Operating profit rose strongly", 2),
        ("Net loss reported for the year", 0),
        ("The report is scheduled Monday", 1),
    ]
    preds = vader_predict([t for t, _ in test_cases])
    for (text, exp), pred in zip(test_cases, preds):
        status = "OK" if pred == exp else f"FEHLER (erwartet {LABEL_NAMES[exp]})"
        print(f"  [{status}] '{text}' → {LABEL_NAMES[pred]}")


if __name__ == "__main__":
    print(f"=== Datensatz-Modus: {'Hugging Face' if USE_HUGGINGFACE else 'Fallback'} ===")
    print("\n=== Validierungsbeispiel (VADER) ===")
    _run_validation_example()

    print("\n=== Vollständiger Datensatz-Test ===")
    texts, labels = load_phrasebank()
    print(f"Samples geladen: {len(texts)}")

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels,
    )

    vader_preds   = vader_predict(X_test)
    vader_metrics = evaluate(y_test, vader_preds, "VADER")
    print(f"VADER     — Accuracy: {vader_metrics['accuracy']:.3f} | Macro-F1: {vader_metrics['f1_macro']:.3f}")

    pipe = build_lr_pipeline()
    pipe.fit(X_train, y_train)
    lr_preds   = list(pipe.predict(X_test))
    lr_metrics = evaluate(y_test, lr_preds, "TF-IDF + LR")
    print(f"TF-IDF+LR — Accuracy: {lr_metrics['accuracy']:.3f} | Macro-F1: {lr_metrics['f1_macro']:.3f}")