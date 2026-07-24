from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.compile_layout_stream_trials import summarize_stream_manifest


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _row(
    stream: str,
    *,
    split: str = "train",
    private: str = "false",
) -> dict[str, str]:
    return {
        "build_id": "final-ocr-v2",
        "project_split": split,
        "token_source": stream,
        "is_private": private,
        "is_usable": "true",
        "alignment_coverage": "0.8" if stream != "ground_truth" else "",
        "entity_retention_rate": "0.7" if stream != "ground_truth" else "",
        "relation_retention_rate": "0.6" if stream != "ground_truth" else "",
        "canonical_retention_rate": "0.9" if stream != "ground_truth" else "",
    }


def test_stream_trial_reports_ratios_and_rejects_nonselection_variants(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.csv"
    rows = [
        *[_row("ground_truth") for _ in range(8)],
        _row("paddleocr"),
        _row("hybrid"),
        _row("ground_truth", split="dev_select"),
        _row("paddleocr", split="dev_select"),
    ]
    _write_manifest(manifest, rows)
    stack = {
        "detector_sha256": "a" * 64,
        "recognizer_sha256": "b" * 64,
    }

    report = summarize_stream_manifest(
        manifest,
        trial_id="A",
        ocr_profile="adaptive",
        selected=True,
        ocr_stack=stack,
        checkpoint_sha256="c" * 64,
        source_commit="d" * 40,
        device="gpu:0",
        duration_seconds=1.5,
    )

    assert report["train_ground_truth_ratio"] == pytest.approx(0.8)
    assert report["train_paddleocr_ratio"] == pytest.approx(0.1)
    assert report["train_hybrid_ratio"] == pytest.approx(0.1)
    assert report["mean_alignment_coverage"] == pytest.approx(0.8)
    assert report["nonselection_ocr_variant_count"] == 0
    assert report["private_row_count"] == 0


def test_stream_trial_refuses_private_rows(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [_row("ground_truth", private="true")])

    with pytest.raises(ValueError, match="private"):
        summarize_stream_manifest(
            manifest,
            trial_id="A",
            ocr_profile="adaptive",
            selected=False,
            ocr_stack={
                "detector_sha256": "a" * 64,
                "recognizer_sha256": "b" * 64,
            },
            checkpoint_sha256="c" * 64,
            source_commit="d" * 40,
            device="cpu",
            duration_seconds=0.0,
        )
