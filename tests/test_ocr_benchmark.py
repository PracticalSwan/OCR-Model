from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from src.ocr.benchmark import (
    analyze_benchmark_page,
    select_benchmark_rows,
    validate_benchmark_records,
)


def _selection_row(dataset: str, index: int, **overrides: str) -> dict[str, str]:
    row = {
        "document_id": f"{dataset}-doc-{index}",
        "page_id": f"{dataset}-page-{index:04d}",
        "dataset": dataset,
        "dataset_component": "component",
        "document_type": {
            "fatura": "invoice",
            "funsd": "form",
            "sroie": "receipt",
        }[dataset],
        "language": "tr" if dataset == "fatura" else "en",
        "image_path": f"data/raw/public/{dataset}/page-{index}.png",
        "normalized_annotation_path": (
            f"data/processed/normalized_ie_annotations/{dataset}/page-{index}.json"
        ),
        "project_split": "dev_select",
        "split_group_id": f"{dataset}-family-{index % 10}",
        "duplicate_group_id": f"{dataset}-duplicate-{index}",
        "sha256": hashlib.sha256(f"{dataset}-{index}".encode()).hexdigest(),
        "is_private": "false",
        "is_usable": "true",
        "quality_bucket": "degraded" if index % 3 == 0 else "clean",
        "text_density_bucket": ("low", "medium", "high")[index % 3],
        "character_size_bucket": ("small", "medium", "large")[index % 3],
        "resolution_bucket": ("low", "medium", "high")[index % 3],
        "has_table": "true" if index % 2 else "false",
        "has_amount": "true",
        "has_date": "true" if index % 2 else "false",
        "has_identifier": "true" if index % 5 else "false",
    }
    row.update(overrides)
    return row


def test_benchmark_selection_includes_all_small_datasets_and_283_fatura() -> None:
    rows = (
        [_selection_row("funsd", index) for index in range(20)]
        + [_selection_row("sroie", index) for index in range(97)]
        + [_selection_row("fatura", index) for index in range(400)]
    )

    selected = select_benchmark_rows(rows, fatura_count=283, seed=42)

    counts = {
        dataset: sum(row["dataset"] == dataset for row in selected)
        for dataset in ("funsd", "sroie", "fatura")
    }
    assert counts == {"funsd": 20, "sroie": 97, "fatura": 283}
    assert len(selected) == 400


def test_benchmark_selection_is_deterministic() -> None:
    rows = (
        [_selection_row("funsd", index) for index in range(20)]
        + [_selection_row("sroie", index) for index in range(97)]
        + [_selection_row("fatura", index) for index in range(400)]
    )

    first = select_benchmark_rows(rows, fatura_count=283, seed=42)
    second = select_benchmark_rows(list(reversed(rows)), fatura_count=283, seed=42)

    assert [row["page_id"] for row in first] == [row["page_id"] for row in second]


def test_benchmark_selection_rejects_private_input() -> None:
    rows = [_selection_row("funsd", 1, is_private="true")]

    with pytest.raises(ValueError, match="private"):
        select_benchmark_rows(rows, fatura_count=0)


def test_benchmark_selection_prefers_unique_analyzed_image_hashes() -> None:
    common = {
        "split": "dev_select",
        "is_private": "false",
        "is_usable": "true",
        "document_type": "invoice",
        "split_group_id": "fatura-family",
        "quality_bucket": "clean",
        "text_density_bucket": "medium",
        "character_size_bucket": "medium",
        "resolution_bucket": "medium",
        "has_table": "true",
        "has_amount": "true",
        "has_date": "true",
        "has_identifier": "true",
    }
    rows = [
        _selection_row("funsd", 0),
        _selection_row("sroie", 0),
        {
            **_selection_row("fatura", 0),
            **common,
            "source_image_sha256": "b" * 64,
            "sha256": "",
        },
        {
            **_selection_row("fatura", 1),
            **common,
            "source_image_sha256": "a" * 64,
            "sha256": "",
        },
        {
            **_selection_row("fatura", 2),
            **common,
            "source_image_sha256": "a" * 64,
            "sha256": "",
        },
    ]

    selected = select_benchmark_rows(rows, fatura_count=2, seed=42)
    fatura_hashes = [
        row["source_image_sha256"] for row in selected if row["dataset"] == "fatura"
    ]

    assert sorted(fatura_hashes) == ["a" * 64, "b" * 64]


def test_page_analysis_derives_geometry_and_field_strata(tmp_path: Path) -> None:
    image_path = tmp_path / "data" / "raw" / "public" / "funsd" / "page.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (120, 80), "white").save(image_path)
    image_sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
    annotation_path = (
        tmp_path
        / "data"
        / "processed"
        / "normalized_ie_annotations"
        / "funsd"
        / "page.json"
    )
    annotation_path.parent.mkdir(parents=True)
    annotation = {
        "schema_version": "1.1",
        "document_id": "doc-1",
        "page_id": "page-1",
        "dataset": "funsd",
        "page": {"width": 120, "height": 80},
        "tokens": [
            {
                "id": "token-1",
                "text": "Invoice 2026-0042",
                "polygon": [[5, 5], [90, 5], [90, 20], [5, 20]],
                "bbox": [5, 5, 90, 20],
                "entity_label": "QUESTION",
            },
            {
                "id": "token-2",
                "text": "TOTAL 1,250.00",
                "polygon": [[5, 30], [100, 30], [100, 50], [5, 50]],
                "bbox": [5, 30, 100, 50],
                "entity_label": "VALUE",
            },
        ],
        "entities": [],
        "relations": [],
        "canonical_fields": {
            "invoice_number": {"value": "2026-0042"},
            "total_amount": {"value": "1,250.00"},
        },
        "is_private": False,
    }
    annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
    row = {
        "document_id": "doc-1",
        "page_id": "page-1",
        "dataset": "funsd",
        "dataset_component": "dataset",
        "document_type": "form",
        "language": "en",
        "image_path": image_path.relative_to(tmp_path).as_posix(),
        "normalized_annotation_path": annotation_path.relative_to(tmp_path).as_posix(),
        "project_split": "dev_select",
        "split_group_id": "family-1",
        "duplicate_group_id": "duplicate-1",
        "sha256": image_sha,
        "is_private": "false",
        "is_usable": "true",
    }

    record = analyze_benchmark_page(
        row,
        project_root=tmp_path,
        license_id="FUNSD-test-license",
    )

    assert record["width"] == 120
    assert record["height"] == 80
    assert record["median_text_height"] == 17.5
    assert record["text_region_count"] == 2
    assert record["has_amount"] == "true"
    assert record["has_identifier"] == "true"
    assert record["ground_truth_text_available"] == "true"
    assert record["ground_truth_polygons_available"] == "true"
    assert record["is_private"] == "false"


def test_page_analysis_rejects_polygon_outside_image(tmp_path: Path) -> None:
    image_path = tmp_path / "data" / "raw" / "public" / "funsd" / "page.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (20, 20), "white").save(image_path)
    annotation_path = (
        tmp_path
        / "data"
        / "processed"
        / "normalized_ie_annotations"
        / "funsd"
        / "page.json"
    )
    annotation_path.parent.mkdir(parents=True)
    annotation_path.write_text(
        json.dumps(
            {
                "schema_version": "1.1",
                "document_id": "doc-1",
                "page_id": "page-1",
                "dataset": "funsd",
                "page": {"width": 20, "height": 20},
                "tokens": [
                    {
                        "id": "token-1",
                        "text": "text",
                        "polygon": [[0, 0], [21, 0], [21, 5], [0, 5]],
                    }
                ],
                "canonical_fields": {},
                "is_private": False,
            }
        ),
        encoding="utf-8",
    )
    row = {
        "document_id": "doc-1",
        "page_id": "page-1",
        "dataset": "funsd",
        "document_type": "form",
        "language": "en",
        "image_path": image_path.relative_to(tmp_path).as_posix(),
        "normalized_annotation_path": annotation_path.relative_to(tmp_path).as_posix(),
        "project_split": "dev_select",
        "split_group_id": "family-1",
        "duplicate_group_id": "duplicate-1",
        "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "is_private": "false",
        "is_usable": "true",
    }

    with pytest.raises(ValueError, match="outside page bounds"):
        analyze_benchmark_page(
            row,
            project_root=tmp_path,
            license_id="FUNSD-test-license",
        )


def test_page_analysis_allows_approved_external_annotation_junction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "data" / "raw" / "public" / "funsd" / "page.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (40, 30), "white").save(image_path)
    lexical_annotation = (
        tmp_path
        / "data"
        / "processed"
        / "normalized_ie_annotations"
        / "funsd"
        / "page.json"
    )
    external_annotation = tmp_path.parent / f"{tmp_path.name}-assets" / "page.json"
    external_annotation.parent.mkdir(parents=True)
    external_annotation.write_text(
        json.dumps(
            {
                "schema_version": "1.1",
                "document_id": "doc-1",
                "page_id": "page-1",
                "dataset": "funsd",
                "page": {"width": 40, "height": 30},
                "tokens": [
                    {
                        "id": "token-1",
                        "text": "TOTAL 1.00",
                        "polygon": [[1, 1], [30, 1], [30, 10], [1, 10]],
                    }
                ],
                "canonical_fields": {"total_amount": {"value": "1.00"}},
                "is_private": False,
            }
        ),
        encoding="utf-8",
    )
    original_resolve = Path.resolve

    def resolve_annotation_link(self: Path, *args: object, **kwargs: object) -> Path:
        if self == lexical_annotation.parents[1]:
            return external_annotation.parent
        if self == lexical_annotation:
            return external_annotation
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve_annotation_link)
    row = {
        "document_id": "doc-1",
        "page_id": "page-1",
        "dataset": "funsd",
        "document_type": "form",
        "language": "en",
        "image_path": image_path.relative_to(tmp_path).as_posix(),
        "normalized_annotation_path": lexical_annotation.relative_to(tmp_path).as_posix(),
        "project_split": "dev_select",
        "split_group_id": "family-1",
        "duplicate_group_id": "duplicate-1",
        "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "is_private": "false",
        "is_usable": "true",
    }

    record = analyze_benchmark_page(
        row,
        project_root=tmp_path,
        license_id="FUNSD-test-license",
    )

    assert record["normalized_annotation_path"] == (
        "data/processed/normalized_ie_annotations/funsd/page.json"
    )
    assert record["annotation_sha256"] == hashlib.sha256(
        external_annotation.read_bytes()
    ).hexdigest()


def test_benchmark_validation_rejects_duplicate_page() -> None:
    record = {
        "benchmark_id": "benchmark-1",
        "page_id": "page-1",
        "dataset": "funsd",
        "split": "dev_select",
        "source_image_path": "data/raw/public/funsd/page.png",
        "source_image_sha256": "a" * 64,
        "width": "100",
        "height": "100",
        "median_text_height": "12",
        "text_region_count": "1",
        "document_type": "form",
        "has_table": "false",
        "has_amount": "false",
        "has_date": "false",
        "has_identifier": "false",
        "ground_truth_text_available": "true",
        "ground_truth_polygons_available": "true",
        "selected_reason": "all_funsd_dev_select",
        "is_private": "false",
    }

    with pytest.raises(ValueError, match="duplicate page_id"):
        validate_benchmark_records(
            [record, dict(record, benchmark_id="benchmark-2")],
            minimum_total=0,
            minimum_fatura=0,
        )
