
"""Compare unigram and unigram-plus-bigram TF-IDF models."""

import json
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data/processed/uci_recipe_reviews_labeled.csv"
REPORT_FILE = ROOT / "reports/tfidf_model_comparison.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20


def metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }


def main():
    df = pd.read_csv(DATA_FILE).dropna(
        subset=["text", "proxy_sentiment"]
    )
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["proxy_sentiment"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["proxy_sentiment"],
    )

    results = {}

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    results["majority_baseline"] = metrics(
        y_test, dummy.predict(X_test)
    )

    for name, ngram_range in [
        ("unigrams_only", (1, 1)),
        ("unigrams_and_bigrams", (1, 2)),
    ]:
        model = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        strip_accents="unicode",
                        ngram_range=ngram_range,
                        min_df=2,
                        max_features=100_000,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

        model.fit(X_train, y_train)
        results[name] = metrics(y_test, model.predict(X_test))

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(
        json.dumps(
            {
                "dataset": "UCI Recipe Reviews and User Feedback",
                "dataset_is_freecash_data": False,
                "random_state": RANDOM_STATE,
                "test_size": TEST_SIZE,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nTF-IDF MODEL COMPARISON")
    print("=" * 70)
    print(pd.DataFrame(results).T.round(4).to_string())
    print(f"\nSaved report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
