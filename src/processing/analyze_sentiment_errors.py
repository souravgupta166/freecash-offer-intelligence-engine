
"""Analyze errors from the original 80/20 sentiment baseline.

Recreates the original split and model so results are comparable.
Labels are rating-derived proxies from UCI recipe reviews.
"""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = ROOT / "data/processed/uci_recipe_reviews_labeled.csv"
OUTPUT_DIR = ROOT / "reports"
ERROR_FILE = OUTPUT_DIR / "sentiment_misclassified_reviews.csv"
SUMMARY_FILE = OUTPUT_DIR / "sentiment_error_summary.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE).dropna(
        subset=["text", "proxy_sentiment"]
    ).copy()

    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].copy()
    df = df.reset_index(drop=True)

    X = df["text"]
    y = df["proxy_sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
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

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    # Keep original test indices so every error can be traced to its row.
    results = pd.DataFrame(
        {
            "text": X_test.to_numpy(),
            "actual": y_test.to_numpy(),
            "predicted": predictions,
        },
        index=X_test.index,
    )

    results["correct"] = results["actual"] == results["predicted"]
    results["text_length"] = results["text"].str.len()

    classifier_classes = model.named_steps["classifier"].classes_
    for column_index, label in enumerate(classifier_classes):
        results[f"probability_{label}"] = probabilities[:, column_index]

    results["error_type"] = results.apply(
        lambda row: (
            "correct"
            if row["correct"]
            else f"actual_{row['actual']}_predicted_{row['predicted']}"
        ),
        axis=1,
    )

    errors = results.loc[~results["correct"]].copy()
    errors = errors.sort_values(
        by="text_length",
        ascending=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    errors.to_csv(ERROR_FILE, index=False, encoding="utf-8")

    summary = (
        results.groupby(["actual", "predicted"], observed=False)
        .size()
        .rename("count")
        .reset_index()
    )
    summary.to_csv(SUMMARY_FILE, index=False, encoding="utf-8")

    print("=" * 65)
    print("ERROR ANALYSIS")
    print("=" * 65)
    print(f"Test records: {len(results):,}")
    print(f"Correct predictions: {results['correct'].sum():,}")
    print(f"Misclassified records: {len(errors):,}")
    print(f"Error rate: {(~results['correct']).mean():.2%}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    print("\nConfusion counts (actual -> predicted):")
    print(
        summary.sort_values(
            ["actual", "count"],
            ascending=[True, False],
        ).to_string(index=False)
    )

    print("\nError counts by actual class:")
    print(errors["actual"].value_counts().to_string())

    print("\nError counts by prediction:")
    print(errors["predicted"].value_counts().to_string())

    print("\nMedian review text length:")
    print(f"Correct: {results.loc[results['correct'], 'text_length'].median():.0f}")
    print(f"Errors:  {errors['text_length'].median():.0f}")

    print("\nExamples of misclassified reviews (maximum 10):")
    for _, row in errors.head(10).iterrows():
        print("-" * 50)
        print(f"Actual: {row['actual']} | Predicted: {row['predicted']}")
        print(f"Length: {row['text_length']}")
        print(f"Text: {row['text'][:400].replace(chr(10), ' ')}")

    print(f"\nSaved errors: {ERROR_FILE}")
    print(f"Saved confusion counts: {SUMMARY_FILE}")
    print("\nThese are recipe reviews, not Freecash or Trustpilot reviews.")


if __name__ == "__main__":
    main()
