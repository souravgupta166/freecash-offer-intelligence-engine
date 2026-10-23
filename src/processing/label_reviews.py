
"""Create transparent proxy sentiment labels from numeric ratings.

These are heuristic labels, not human-annotated sentiment ground truth.
Zero-star records are unrated and are excluded from rating-based training.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "uci_recipe_reviews_development.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "uci_recipe_reviews_labeled.csv"
)


def assign_proxy_sentiment(stars: int) -> str | None:
    """Map a valid rating to a documented proxy sentiment label."""
    if pd.isna(stars):
        return None

    if stars == 0:
        return None
    if stars in (1, 2):
        return "negative"
    if stars == 3:
        return "neutral"
    if stars in (4, 5):
        return "positive"

    raise ValueError(f"Unexpected star rating: {stars}")


def build_labeled_dataset(
    input_file: Path = INPUT_FILE,
    output_file: Path = OUTPUT_FILE,
) -> pd.DataFrame:
    """Validate ratings, label rated reviews, and save the result."""
    if not input_file.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_file}")

    df = pd.read_csv(input_file)

    required_columns = {"text", "stars"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {sorted(missing_columns)}"
        )

    df["stars"] = pd.to_numeric(df["stars"], errors="raise")

    if not df["stars"].isin([0, 1, 2, 3, 4, 5]).all():
        raise ValueError("Ratings must be integers between 0 and 5.")

    if not (df["stars"] % 1 == 0).all():
        raise ValueError("Ratings must be whole numbers.")

    df["proxy_sentiment"] = df["stars"].map(assign_proxy_sentiment)

    # Unrated records remain in the source development dataset,
    # but are excluded from this rating-labeled training dataset.
    labeled = df.dropna(subset=["proxy_sentiment"]).copy()
    labeled["proxy_sentiment"] = labeled["proxy_sentiment"].astype(str)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(output_file, index=False, encoding="utf-8")

    print(f"Input rows: {len(df):,}")
    print(f"Unrated rows excluded: {df['proxy_sentiment'].isna().sum():,}")
    print(f"Labeled rows: {len(labeled):,}")
    print("\nProxy sentiment distribution:")
    print(labeled["proxy_sentiment"].value_counts().to_string())
    print("\nClass proportions (%):")
    print(
        (labeled["proxy_sentiment"].value_counts(normalize=True) * 100)
        .round(2)
        .to_string()
    )
    print(f"\nSaved labeled dataset: {output_file}")

    return labeled


if __name__ == "__main__":
    build_labeled_dataset()
