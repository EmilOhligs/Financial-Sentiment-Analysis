"""
vader_demo.py
=============
VADER Sentiment Demo — Video 2
Zeigt strukturelles Versagen auf einem Finanz-Satz.
"""

from nltk.sentiment.vader import SentimentIntensityAnalyzer


# ── Konfiguration ────────────────────────────────────────────────────────────

SENTENCE = "The company successfully avoided a larger writedown."
TRUE_LABEL = "negative"

VADER_POS_THRESHOLD =  0.05
VADER_NEG_THRESHOLD = -0.05

WIDTH = 58


# ── Hilfsfunktionen ──────────────────────────────────────────────────────────

def classify(compound: float) -> str:
    if compound >= VADER_POS_THRESHOLD:
        return "positive"
    elif compound <= VADER_NEG_THRESHOLD:
        return "negative"
    else:
        return "neutral"


def score_bar(score: float, width: int = 20) -> str:
    """Visuelle Skala von -4 bis +4 für Wort-Scores."""
    norm = (score + 4) / 8
    pos  = int(norm * width)
    pos  = max(0, min(width - 1, pos))
    bar  = ["-"] * width
    bar[width // 2] = "|"
    bar[pos]        = "█"
    return "".join(bar)


def compound_bar(compound: float, width: int = 30) -> str:
    """Visuelle Skala von -1 bis +1 für Compound Score."""
    norm = (compound + 1) / 2
    pos  = int(norm * width)
    pos  = max(0, min(width - 1, pos))
    bar  = [" "] * width
    bar[width // 2] = "·"
    bar[pos]        = "█"
    neg_zone = "░" * (width // 2 - 1)
    pos_zone = "░" * (width - width // 2 - 1)
    marker   = "·"
    result   = list(neg_zone + marker + pos_zone)
    result[pos] = "█"
    return "".join(result)


def divider(char: str = "─") -> str:
    return char * WIDTH


# ── Hauptlogik ───────────────────────────────────────────────────────────────

def run():
    sia    = SentimentIntensityAnalyzer()
    tokens = SENTENCE.rstrip(".").split()

    # Wort-Scores aus dem VADER-Lexikon
    word_scores = {}
    for token in tokens:
        clean = token.strip(".,;:!?\"'").lower()
        if clean in sia.lexicon:
            word_scores[token] = sia.lexicon[clean]

    # Compound Score
    scores   = sia.polarity_scores(SENTENCE)
    compound = scores["compound"]
    label    = classify(compound)

    # ── Ausgabe ──────────────────────────────────────────────────────────────

    print()
    print(divider("═"))
    print("  VADER  ·  Financial Sentiment Demo")
    print(divider("═"))

    print()
    print(f"  Sentence")
    print(f"  {SENTENCE}")

    print()
    print(divider())
    print("  Word Scores  (from VADER lexicon)")
    print(divider())

    for token in tokens:
        clean = token.strip(".,;:!?\"'").lower()
        if clean in sia.lexicon:
            s   = sia.lexicon[clean]
            bar = score_bar(s)
            print(f"  {token:<18}  {s:+.2f}  [{bar}]")
        else:
            print(f"  {token:<18}   n/a   [{'·':^20}]  (not in lexicon)")

    print()
    print(divider())
    print("  Compound Score")
    print(divider())

    bar = compound_bar(compound)
    print(f"  [{bar}]")
    print(f"   -1.0                  0                 +1.0")
    print(f"                compound = {compound:+.3f}")

    print()
    print(divider())
    print("  Classification")
    print(divider())

    match = "✓" if label == TRUE_LABEL else "✗"

    print(f"  VADER label   →  {label.upper()}")
    print(f"  True label    →  {TRUE_LABEL.upper()}")
    print(f"  Correct       →  {match}")

    if label != TRUE_LABEL:
        print()
        print(f"  \"successfully\" scored {word_scores.get('successfully', 'n/a'):+}  and dominated.")
        print(f"  \"writedown\" is not in the lexicon.  Score: 0.")
        print(f"  The model read success.  The sentence describes a loss.")

    print()
    print(divider("═"))
    print()


if __name__ == "__main__":
    run()