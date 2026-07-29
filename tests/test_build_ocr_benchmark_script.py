from __future__ import annotations

import csv
import json
from pathlib import Path

import build_ocr_benchmark

from src.ocr.benchmark import BENCHMARK_COLUMNS
from src.rotation_common import sha256_file


def _record(dataset: str, index: int) -> dict[str, object]:
    record: dict[str, object] = {column: "" for column in BENCHMARK_COLUMNS}
    record.update(
        {
            "benchmark_id": f"ocrbench-{dataset}-{index}",
            "page_id": f"{dataset}-page-{index}",
            "document_id": f"{dataset}-doc-{index}",
            "document_family_id": f"{dataset}-family-{index}",
            "dataset": dataset,
            "split": "dev_select",
            "source_image_path": f"data/raw/public/{dataset}/page-{index}.png",
            "source_image_sha256": f"{index + 1:064x}",
            "normalized_annotation_path": (
                f"data/processed/normalized_ie_annotations/{dataset}/page-{index}.json"
            ),
            "annotation_sha256": f"{index + 4:064x}",
            "annotation_version": "1.1",
            "license_id": "test-license",
            "width": 100,
            "height": 100,
            "median_text_height": 12.0,
            "text_region_count": 1,
            "document_type": "invoice" if dataset == "fatura" else "form",
            "has_table": "false",
            "has_amount": "true",
            "has_date": "true",
            "has_identifier": "true",
            "ground_truth_text_available": "true",
            "ground_truth_polygons_available": "true",
            "selected_reason": f"all_{dataset}_dev_select",
            "is_private": "false",
        }
    )
    return record


def test_write_benchmark_outputs_binds_manifest_and_summary(tmp_path: Path) -> None:
    manifest = tmp_path / "ocr_benchmark_manifest.csv"
    summary = tmp_path / "benchmark_summary.json"
    selection = tmp_path / "benchmark_selection.md"
    records = [_record("funsd", 0), _record("sroie", 1), _record("fatura", 2)]

    result = build_ocr_benchmark.write_benchmark_outputs(
        records,
        manifest_path=manifest,
        summary_path=summary,
        selection_path=selection,
        seed=42,
        source_commit="a" * 40,
        available_counts={"funsd": 1, "sroie": 1, "fatura": 5},
        storage_gate={"passed": True, "c_free_gib": 40.0, "asset_free_gib": 300.0},
        minimum_total=3,
        minimum_fatura=1,
    )

    with manifest.open(encoding="utf-8", newline="") as handle:
        written = list(csv.DictReader(handle))
    persisted_summary = json.loads(summary.read_text(encoding="utf-8"))
    assert {row["build_id"] for row in written} == {result["build_id"]}
    assert persisted_summary["manifest_sha256"] == sha256_file(manifest)
    assert persisted_summary["counts_by_dataset"] == {
        "fatura": 1,
        "funsd": 1,
        "sroie": 1,
    }
    assert persisted_summary["private_row_count"] == 0
    assert "# OCR Benchmark Selection" in selection.read_text(encoding="utf-8")
