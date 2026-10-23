
"""Measure classification errors by review text length."""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data/processed/uci_recipe_reviews_labeled.csv"
OUTPUT_FILE = ROOT / "reports/error_metrics_by_text_length.csv"

RANDOM_STATE = 42


def main():
    df = pd.read_csv(DATA_FILE).dropna(
        subset=["text", "proxy_sentiment"]
    ).copy()

    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].copy()
    df = df.reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["proxy_sentiment"],
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df["proxy_sentiment"],
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_features=100_000,
            sublinear_tf=True,
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results = pd.DataFrame({
        "text": X_test.to_numpy(),
        "actual": y_test.to_numpy(),
        "predicted": predictions,
    })
    results["text_length"] = results["text"].str.len()
    results["correct"] = results["actual"] == results["predicted"]

    # Fixed, interpretable length bands.
    results["length_group"] = pd.cut(
        results["text_length"],
        bins=[0, 20, 80, 200, float("inf")],
        labels=["1-20", "21-80", "81-200", "201+"],
        include_lowest=True,
    )

    summary = (
        results.groupby("length_group", observed=True)
        .apply(
            lambda group: pd.Series({
                "reviews": len(group),
                "accuracy": accuracy_score(
                    group["actual"], group["predicted"]
                ),
                "macro_f1": f1_score(
                    group["actual"],
                    group["predicted"],
                    average="macro",
                    zero_division=0,
                ),
                "error_rate": 1 - group["correct"].mean(),
            }),
            include_groups=False,
        )
        .reset_index()
    )

    summary["reviews"] = summary["reviews"].astype(int)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print("ERROR METRICS BY REVIEW LENGTH")
    print("=" * 65)
    print(summary.round(4).to_string(index=False))
    print(f"\nSaved report: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
