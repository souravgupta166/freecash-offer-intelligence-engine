
import json

import pytest

from src.processing.provenance import (
    create_provenance_record,
    save_provenance_record,
)


def valid_record():
    return create_provenance_record(
        source_id="example_source",
        source_url="https://example.com/",
        collection_method="authorized_api",
        permission_confirmed=True,
        record_count=10,
        dataset_type="reviews",
    )


def test_creates_provenance_record():
    record = valid_record()

    assert record["source_id"] == "example_source"
    assert record["permission_confirmed"] is True
    assert record["record_count"] == 10
    assert record["data_status"] == "authorized_collection"
    assert record["collected_at_utc"]


def test_rejects_invalid_source_url():
    with pytest.raises(ValueError, match="valid HTTPS URL"):
        create_provenance_record(
            source_id="example",
            source_url="http://example.com",
            collection_method="authorized_api",
            permission_confirmed=True,
            record_count=1,
            dataset_type="reviews",
        )


def test_rejects_negative_record_count():
    with pytest.raises(ValueError, match="non-negative integer"):
        create_provenance_record(
            source_id="example",
            source_url="https://example.com",
            collection_method="authorized_api",
            permission_confirmed=True,
            record_count=-1,
            dataset_type="reviews",
        )


def test_refuses_to_save_without_permission(tmp_path):
    record = create_provenance_record(
        source_id="example",
        source_url="https://example.com",
        collection_method="manual",
        permission_confirmed=False,
        record_count=0,
        dataset_type="offers",
    )

    with pytest.raises(PermissionError, match="permission is not confirmed"):
        save_provenance_record(record, tmp_path)


def test_saves_authorized_record_as_json(tmp_path):
    record = valid_record()
    path = save_provenance_record(record, tmp_path)

    assert path.exists()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["source_id"] == "example_source"
    assert saved["permission_confirmed"] is True
