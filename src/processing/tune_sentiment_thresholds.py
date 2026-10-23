
"""Tune minority-class decision thresholds without test-set leakage.

Uses UCI recipe reviews with rating-derived proxy labels.
The held-out test set is evaluated only after validation tuning.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data/processed/uci_recipe_reviews_labeled.csv"
REPORT_FILE = ROOT / "reports/sentiment_threshold_tuning.json"

RANDOM_STATE = 42
LABELS = ["negative", "neutral", "positive"]


def calculate_metrics(y_true, predictions):
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "macro_f1": float(
            f1_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(
                y_true, predictions, average="weighted", zero_division=0
            )
        ),
        "classification_report": classification_report(
            y_true,
            predictions,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        ),
    }


def apply_thresholds(probabilities, classes, negative_threshold, neutral_threshold):
    """Apply tuned thresholds, otherwise choose the highest probability.

    A negative or neutral prediction is made only if its probability
    reaches its class threshold. If both qualify, choose the one with
    the larger probability. Otherwise predict the highest-probability
    class among the remaining eligible choices, defaulting to positive.
    """
    class_index = {label: i for i, label in enumerate(classes)}
    output = []

    for row in probabilities:
        negative_p = row[class_index["negative"]]
        neutral_p = row[class_index["neutral"]]

        negative_ok = negative_p >= negative_threshold
        neutral_ok = neutral_p >= neutral_threshold

        if negative_ok and neutral_ok:
            if negative_p >= neutral_p:
                output.append("negative")
            else:
                output.append("neutral")
        elif negative_ok:
            output.append("negative")
        elif neutral_ok:
            output.append("neutral")
        else:
            output.append("positive")

    return np.asarray(output)


def main():
    df = pd.read_csv(DATA_FILE).dropna(
        subset=["text", "proxy_sentiment"]
    )
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].copy()

    X = df["text"]
    y = df["proxy_sentiment"]

    # 60% training, 20% validation, 20% final test.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
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

    # Choose thresholds on validation data only.
    validation_probabilities = model.predict_proba(X_validation)
    classes = model.named_steps["classifier"].classes_

    default_validation_predictions = model.predict(X_validation)
    default_validation_metrics = calculate_metrics(
        y_validation, default_validation_predictions
    )

    # Coarse grid first to limit overfitting and computation.
    threshold_values = np.arange(0.15, 0.81, 0.05)
    best = None

    for negative_threshold in threshold_values:
        for neutral_threshold in threshold_values:
            predictions = apply_thresholds(
                validation_probabilities,
                classes,
                float(negative_threshold),
                float(neutral_threshold),
            )
            score = f1_score(
                y_validation,
                predictions,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )

            candidate = {
                "negative_threshold": float(negative_threshold),
                "neutral_threshold": float(neutral_threshold),
                "validation_macro_f1": float(score),
                "predictions": predictions,
            }

            if best is None or score > best["validation_macro_f1"]:
                best = candidate

    # Final evaluation happens only after the thresholds are chosen.
    test_probabilities = model.predict_proba(X_test)
    default_test_predictions = model.predict(X_test)
    tuned_test_predictions = apply_thresholds(
        test_probabilities,
        classes,
        best["negative_threshold"],
        best["neutral_threshold"],
    )

    default_test_metrics = calculate_metrics(
        y_test, default_test_predictions
    )
    tuned_test_metrics = calculate_metrics(
        y_test, tuned_test_predictions
    )

    report = {
        "dataset": "UCI Recipe Reviews and User Feedback",
        "dataset_is_freecash_data": False,
        "label_type": "rating-derived proxy sentiment",
        "random_state": RANDOM_STATE,
        "split_sizes": {
            "train": len(X_train),
            "validation": len(X_validation),
            "test": len(X_test),
        },
        "validation_default_macro_f1": default_validation_metrics["macro_f1"],
        "selected_thresholds": {
            "negative": best["negative_threshold"],
            "neutral": best["neutral_threshold"],
        },
        "validation_tuned_macro_f1": best["validation_macro_f1"],
        "test_default_argmax": default_test_metrics,
        "test_tuned_thresholds": tuned_test_metrics,
        "test_class_counts": y_test.value_counts().to_dict(),
        "notes": [
            "Thresholds were selected using validation data only.",
            "The test set was not used to select thresholds.",
            "Threshold search is coarse and can overfit validation data.",
            "Proxy labels are derived from ratings, not human annotations.",
        ],
    }

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("=" * 65)
    print("THRESHOLD TUNING RESULTS")
    print("=" * 65)
    print(f"Train rows:      {len(X_train):,}")
    print(f"Validation rows: {len(X_validation):,}")
    print(f"Test rows:       {len(X_test):,}")

    print("\nSelected thresholds (validation set):")
    print(f"Negative: {best['negative_threshold']:.2f}")
    print(f"Neutral:  {best['neutral_threshold']:.2f}")
    print(f"Validation macro F1: {best['validation_macro_f1']:.4f}")

    for name, metrics in [
        ("Default argmax — test set", default_test_metrics),
        ("Tuned thresholds — test set", tuned_test_metrics),
    ]:
        print(f"\n{name}")
        print("-" * 45)
        for key in [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
        ]:
            print(f"{key:20s}: {metrics[key]:.4f}")

        print("\nPer-class results:")
        print(
            pd.DataFrame(metrics["classification_report"])
            .T.round(3).to_string()
        )

    print(f"\nSaved report: {REPORT_FILE}")
    print("\nReminder: these results use recipe-review data.")


if __name__ == "__main__":
    main()
