
"""Track dataset provenance and collection authorization."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "metadata"


def create_provenance_record(
    *,
    source_id: str,
    source_url: str,
    collection_method: str,
    permission_confirmed: bool,
    record_count: int,
    dataset_type: str,
) -> dict[str, Any]:
    """Create auditable metadata for a dataset collection run.

    permission_confirmed must reflect actual documented authorization,
    not merely that a URL is publicly accessible.
    """
    if not source_id.strip():
        raise ValueError("source_id must not be empty.")

    parsed = urlparse(source_url)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise ValueError("source_url must be a valid HTTPS URL.")

    if not collection_method.strip():
        raise ValueError("collection_method must not be empty.")

    if not dataset_type.strip():
        raise ValueError("dataset_type must not be empty.")

    if isinstance(record_count, bool) or not isinstance(record_count, int):
        raise ValueError("record_count must be a non-negative integer.")

    if record_count < 0:
        raise ValueError("record_count must be a non-negative integer.")

    if not isinstance(permission_confirmed, bool):
        raise ValueError("permission_confirmed must be True or False.")

    return {
        "source_id": source_id.strip(),
        "source_url": source_url,
        "collection_method": collection_method.strip(),
        "permission_confirmed": permission_confirmed,
        "record_count": record_count,
        "dataset_type": dataset_type.strip(),
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": "Freecash Offer Intelligence Engine",
        "data_status": (
            "authorized_collection"
            if permission_confirmed
            else "permission_not_confirmed"
        ),
    }


def save_provenance_record(
    record: dict[str, Any],
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Save one provenance record as a timestamped JSON file."""
    if record.get("permission_confirmed") is not True:
        raise PermissionError(
            "Cannot register this run as authorized: permission is not confirmed."
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    filename = f"provenance_{timestamp}.json"
    destination = output_path / filename

    with destination.open("x", encoding="utf-8") as file:
        json.dump(record, file, indent=2, ensure_ascii=False)

    return destination
