
"""Train and evaluate baseline sentiment classifiers.

Labels are rating-derived proxies from the UCI recipe-review dataset,
not human-annotated sentiment ground truth.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "uci_recipe_reviews_labeled.csv"
)
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.20
LABELS = ["negative", "neutral", "positive"]


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Labeled dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    required = {"text", "proxy_sentiment"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["text", "proxy_sentiment"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].copy()

    X = df["text"]
    y = df["proxy_sentiment"]

    if set(y.unique()) != set(LABELS):
        raise ValueError(
            f"Expected classes {LABELS}; found {sorted(y.unique())}"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Baseline: always predict the majority class from training data.
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_predictions = dummy.predict(X_test)

    # TF-IDF is fitted only on the training text, through the pipeline.
    model = Pipeline(
        steps=[
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
    predictions = model.predict(X_test)

    def evaluate(actual, predicted) -> dict:
        return {
            "accuracy": float(accuracy_score(actual, predicted)),
            "balanced_accuracy": float(
                balanced_accuracy_score(actual, predicted)
            ),
            "macro_f1": float(
                f1_score(
                    actual, predicted, average="macro", zero_division=0
                )
            ),
            "weighted_f1": float(
                f1_score(
                    actual, predicted, average="weighted", zero_division=0
                )
            ),
            "classification_report": classification_report(
                actual,
                predicted,
                labels=LABELS,
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(
                actual, predicted, labels=LABELS
            ).tolist(),
        }

    baseline_metrics = evaluate(y_test, dummy_predictions)
    model_metrics = evaluate(y_test, predictions)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "tfidf_logistic_regression.joblib"
    report_path = REPORT_DIR / "sentiment_baseline_evaluation.json"

    joblib.dump(model, model_path)

    report = {
        "experiment": "TF-IDF + Logistic Regression baseline",
        "dataset": "UCI Recipe Reviews and User Feedback",
        "source_url": (
            "https://archive.ics.uci.edu/dataset/911/"
            "recipe+reviews+and+user+feedback+dataset"
        ),
        "dataset_is_freecash_data": False,
        "dataset_is_trustpilot_data": False,
        "label_definition": {
            "negative": "1–2 stars",
            "neutral": "3 stars",
            "positive": "4–5 stars",
            "zero_star_reviews": "excluded because they are unrated",
            "caveat": (
                "Ratings are proxy labels, not human-annotated text sentiment."
            ),
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "training_class_counts": y_train.value_counts().to_dict(),
        "test_class_counts": y_test.value_counts().to_dict(),
        "baseline": baseline_metrics,
        "tfidf_logistic_regression": model_metrics,
        "confusion_matrix_label_order": LABELS,
        "model_file": str(model_path.relative_to(PROJECT_ROOT)),
    }

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("=" * 65)
    print("SENTIMENT BASELINE EVALUATION")
    print("=" * 65)
    print(f"Training rows: {len(X_train):,}")
    print(f"Test rows:     {len(X_test):,}")
    print("\nTraining class counts:")
    print(y_train.value_counts().to_string())
    print("\nTest class counts:")
    print(y_test.value_counts().to_string())

    for name, metrics in [
        ("Majority-class baseline", baseline_metrics),
        ("TF-IDF + Logistic Regression", model_metrics),
    ]:
        print(f"\n{name}")
        print("-" * 40)
        for metric in [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
        ]:
            print(f"{metric:20s}: {metrics[metric]:.4f}")

        print("\nPer-class results:")
        print(
            pd.DataFrame(metrics["classification_report"])
            .T.round(3).to_string()
        )

        print("\nConfusion matrix (rows=actual, columns=predicted)")
        print(pd.DataFrame(
            metrics["confusion_matrix"],
            index=LABELS,
            columns=LABELS,
        ).to_string())

    print(f"\nSaved model:  {model_path}")
    print(f"Saved report: {report_path}")
    print("\nReminder: these are recipe-review development results.")


if __name__ == "__main__":
    main()
