from __future__ import annotations

import pytest

from src.ocr.lineage import (
    LineageRecord,
    ensure_allowed_roles,
    validate_family_isolation,
)


SHA_A = "a" * 64
SHA_B = "b" * 64


def _record(**overrides: object) -> LineageRecord:
    values: dict[str, object] = {
        "sample_id": "page-1",
        "document_id": "doc-1",
        "document_family_id": "family-1",
        "split_role": "train",
        "source_dataset": "funsd",
        "source_hash": SHA_A,
        "license_id": "FUNSD-CDLA-Permissive",
        "annotation_version": "1.1",
    }
    values.update(overrides)
    return LineageRecord(**values)


def test_lineage_rejects_unknown_split_role() -> None:
    with pytest.raises(ValueError, match="unsupported split role"):
        _record(split_role="validation")


def test_derived_lineage_inherits_family_and_appends_transform() -> None:
    source = _record(transformation_chain=("render:200dpi",))

    derived = source.derive(
        sample_id="page-1-tile-0",
        transformation="tile:x0=0,y0=0,w=512,h=512",
    )

    assert derived.parent_sample_id == "page-1"
    assert derived.document_family_id == source.document_family_id
    assert derived.split_role == source.split_role
    assert derived.source_hash == source.source_hash
    assert derived.transformation_chain == (
        "render:200dpi",
        "tile:x0=0,y0=0,w=512,h=512",
    )


def test_role_guard_rejects_train_record_in_dev_selection() -> None:
    with pytest.raises(ValueError, match="role train is not allowed"):
        ensure_allowed_roles([_record()], {"dev_select"})


def test_family_isolation_rejects_cross_split_family() -> None:
    records = [
        _record(sample_id="page-1", split_role="train"),
        _record(
            sample_id="page-2",
            document_id="doc-2",
            split_role="dev_select",
            source_hash=SHA_B,
        ),
    ]

    with pytest.raises(ValueError, match="family-1"):
        validate_family_isolation(records)
