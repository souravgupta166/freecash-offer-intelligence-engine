
"""Ingest a licensed UCI review dataset for NLP development.

This dataset contains recipe reviews. It is NOT Freecash or Trustpilot data.
Source: UCI Machine Learning Repository, dataset 911.
Licence: Creative Commons Attribution 4.0 International (CC BY 4.0).
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "reviews"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

SOURCE_URL = (
    "https://archive.ics.uci.edu/dataset/911/"
    "recipe+reviews+and+user+feedback+dataset"
)
SOURCE_DOI = "10.24432/C5FG95"

# Exclude usernames, user IDs, and comment IDs.
SELECTED_COLUMNS = [
    "text",
    "stars",
    "created_at",
    "thumbs_up",
    "thumbs_down",
    "reply_count",
]


def ingest_uci_reviews() -> dict:
    """Download, validate, and save the selected review fields."""
    for directory in (RAW_DIR, PROCESSED_DIR, METADATA_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    print("Fetching dataset 911 from the UCI repository...")
    dataset = fetch_ucirepo(id=911)

    features = dataset.data.features

    missing_columns = [
        column for column in SELECTED_COLUMNS
        if column not in features.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Expected columns missing from source: {missing_columns}"
        )

    source_rows = len(features)

    # Keep only fields needed for review text analysis.
    raw = features[SELECTED_COLUMNS].copy()

    # Convert relevant fields to consistent types.
    raw["text"] = raw["text"].astype("string").str.strip()
    raw["stars"] = pd.to_numeric(raw["stars"], errors="coerce")
    raw["created_at"] = pd.to_numeric(
        raw["created_at"], errors="coerce"
    )
    raw["thumbs_up"] = pd.to_numeric(
        raw["thumbs_up"], errors="coerce"
    )
    raw["thumbs_down"] = pd.to_numeric(
        raw["thumbs_down"], errors="coerce"
    )
    raw["reply_count"] = pd.to_numeric(
        raw["reply_count"], errors="coerce"
    )

    # Preserve the selected source fields before text cleaning.
    raw_path = RAW_DIR / "uci_recipe_reviews_selected.csv"
    raw.to_csv(raw_path, index=False, encoding="utf-8")

    # Remove records with missing or empty review text.
    cleaned = raw.dropna(subset=["text"]).copy()
    cleaned = cleaned[cleaned["text"].str.len() > 0].copy()

    # Keep only valid rating values. Rating 0 means unrated in this dataset.
    cleaned = cleaned[
        cleaned["stars"].between(0, 5, inclusive="both")
        & (cleaned["stars"] % 1 == 0)
    ].copy()

    cleaned["stars"] = cleaned["stars"].astype("int64")

    # Add a parsed UTC timestamp. Invalid timestamps become missing.
    cleaned["created_at_utc"] = pd.to_datetime(
        cleaned["created_at"],
        unit="s",
        utc=True,
        errors="coerce",
    )

    # Remove exact duplicate rows from the analysis dataset.
    before_deduplication = len(cleaned)
    cleaned = cleaned.drop_duplicates(
        subset=["text", "stars", "created_at"]
    ).copy()
    duplicate_rows_removed = before_deduplication - len(cleaned)

    processed_path = (
        PROCESSED_DIR / "uci_recipe_reviews_development.csv"
    )
    cleaned.to_csv(
        processed_path,
        index=False,
        encoding="utf-8",
    )

    manifest = {
        "dataset_name": "Recipe Reviews and User Feedback",
        "source_type": "UCI research dataset",
        "source_url": SOURCE_URL,
        "doi": SOURCE_DOI,
        "license": "CC BY 4.0",
        "attribution": (
            "Ali, A., Matuszewski, S., & Czupyt, J. (2023). "
            "Recipe Reviews and User Feedback [Dataset]. "
            "UCI Machine Learning Repository. "
            "https://doi.org/10.24432/C5FG95"
        ),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_role": "NLP development and testing only",
        "is_freecash_data": False,
        "is_trustpilot_data": False,
        "source_rows": int(source_rows),
        "rows_with_nonempty_text": int(
            raw["text"].notna().sum()
        ),
        "rows_after_validation_and_deduplication": int(len(cleaned)),
        "duplicate_rows_removed": int(duplicate_rows_removed),
        "selected_columns": SELECTED_COLUMNS,
        "excluded_personal_identifiers": [
            "user_id",
            "user_name",
            "comment_id",
        ],
        "raw_selected_file": str(raw_path.relative_to(PROJECT_ROOT)),
        "processed_file": str(
            processed_path.relative_to(PROJECT_ROOT)
        ),
    }

    manifest_path = METADATA_DIR / "uci_recipe_reviews_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nIngestion completed.")
    print(f"Source rows: {source_rows:,}")
    print(f"Rows saved for analysis: {len(cleaned):,}")
    print(f"Duplicate rows removed: {duplicate_rows_removed:,}")
    print(f"Selected source data: {raw_path}")
    print(f"Processed data: {processed_path}")
    print(f"Dataset manifest: {manifest_path}")
    print("\nThis is recipe review data, not Freecash data.")

    return manifest


if __name__ == "__main__":
    ingest_uci_reviews()
