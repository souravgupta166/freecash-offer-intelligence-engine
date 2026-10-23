
"""Evaluate abstention rules based on review text length.

Short texts can be flagged for manual review rather than automatically
assigned a sentiment. Cutoffs are selected using validation data only.
"""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data/processed/uci_recipe_reviews_labeled.csv"
REPORT_FILE = ROOT / "reports/selective_prediction_evaluation.csv"

RANDOM_STATE = 42
CUTOFFS = [0, 10, 15, 20, 25, 30, 40, 50]


def evaluate_coverage_and_risk(frame, cutoff):
    """Measure automatic prediction coverage and errors above a cutoff."""
    accepted = frame[frame["text_length"] >= cutoff]

    if accepted.empty:
        return {
            "minimum_characters": cutoff,
            "total_reviews": len(frame),
            "automatically_predicted": 0,
            "coverage": 0.0,
            "selective_error_rate": None,
            "selective_accuracy": None,
        }

    error_rate = (accepted["actual"] != accepted["predicted"]).mean()

    return {
        "minimum_characters": cutoff,
        "total_reviews": len(frame),
        "automatically_predicted": len(accepted),
        "coverage": len(accepted) / len(frame),
        "selective_error_rate": error_rate,
        "selective_accuracy": accuracy_score(
            accepted["actual"], accepted["predicted"]
        ),
    }


def main():
    df = pd.read_csv(DATA_FILE).dropna(
        subset=["text", "proxy_sentiment"]
    ).copy()

    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    X_train, X_temp, y_train, y_temp = train_test_split(
        df["text"],
        df["proxy_sentiment"],
        test_size=0.40,
        random_state=RANDOM_STATE,
        stratify=df["proxy_sentiment"],
    )

    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
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

    validation = pd.DataFrame({
        "text": X_validation.to_numpy(),
        "actual": y_validation.to_numpy(),
        "predicted": model.predict(X_validation),
    })
    validation["text_length"] = validation["text"].str.len()

    test = pd.DataFrame({
        "text": X_test.to_numpy(),
        "actual": y_test.to_numpy(),
        "predicted": model.predict(X_test),
    })
    test["text_length"] = test["text"].str.len()

    # Evaluate all candidate cutoffs on validation data.
    validation_results = pd.DataFrame([
        evaluate_coverage_and_risk(validation, cutoff)
        for cutoff in CUTOFFS
    ])

    # Selection rule: maximize validation coverage while achieving
    # a selective error rate <= 8%. If no cutoff qualifies, use 0.
    eligible = validation_results[
        validation_results["selective_error_rate"].notna()
        & (validation_results["selective_error_rate"] <= 0.08)
    ]

    if eligible.empty:
        selected_cutoff = 0
    else:
        selected_cutoff = int(
            eligible.sort_values(
                ["coverage", "minimum_characters"],
                ascending=[False, True],
            ).iloc[0]["minimum_characters"]
        )

    # Test the selected cutoff once on the held-out test data.
    test_results = pd.DataFrame([
        evaluate_coverage_and_risk(test, cutoff)
        for cutoff in CUTOFFS
    ])
    test_results["partition"] = "test"
    validation_results["partition"] = "validation"

    all_results = pd.concat(
        [validation_results, test_results],
        ignore_index=True,
    )
    all_results["selected_on_validation"] = (
        all_results["minimum_characters"] == selected_cutoff
    )

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    all_results.to_csv(REPORT_FILE, index=False, encoding="utf-8")

    print("=" * 65)
    print("SELECTIVE PREDICTION EVALUATION")
    print("=" * 65)
    print(f"Training rows:   {len(X_train):,}")
    print(f"Validation rows: {len(X_validation):,}")
    print(f"Test rows:       {len(X_test):,}")
    print(f"Selected cutoff: {selected_cutoff} characters")
    print("Validation selection rule: error rate <= 8%, maximize coverage")

    for partition, results in [
        ("VALIDATION", validation_results),
        ("HELD-OUT TEST", test_results),
    ]:
        print(f"\n{partition}")
        print("-" * 65)
        print(results[[
            "minimum_characters",
            "automatically_predicted",
            "coverage",
            "selective_error_rate",
            "selective_accuracy",
        ]].round(4).to_string(index=False))

    print(f"\nSaved report: {REPORT_FILE}")
    print(
        "\nReminder: this evaluates recipe-review proxy labels, "
        "not Freecash or Trustpilot data."
    )


if __name__ == "__main__":
    main()
