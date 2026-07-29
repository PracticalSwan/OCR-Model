from __future__ import annotations

import pytest

from src.evaluation.ocr_reporting import build_metric_report


SHA = "a" * 64


def _report(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "build_id": "ocr-benchmark-deadbeef",
        "split": "dev_select",
        "manifest_sha256": SHA,
        "detector_sha256": SHA,
        "recognizer_sha256": SHA,
        "checkpoint_sha256": SHA,
        "calibration_sha256": None,
        "configuration_sha256": SHA,
        "source_commit": "1" * 40,
        "device": "gpu:0",
        "sample_count": 400,
        "failure_count": 0,
        "duration_seconds": 12.5,
        "private_row_count": 0,
        "metrics": {"polygon_f1": 0.5},
    }
    values.update(overrides)
    return build_metric_report(**values)


def test_metric_report_contains_required_provenance() -> None:
    report = _report()

    assert report["schema_version"] == "1.0"
    assert report["split"] == "dev_select"
    assert report["manifest_sha256"] == SHA
    assert report["private_row_count"] == 0
    assert report["metrics"] == {"polygon_f1": 0.5}


def test_metric_report_rejects_private_rows() -> None:
    with pytest.raises(ValueError, match="private_row_count must be zero"):
        _report(private_row_count=1)


def test_metric_report_rejects_invalid_hash() -> None:
    with pytest.raises(ValueError, match="manifest_sha256"):
        _report(manifest_sha256="not-a-sha")


def test_metric_report_rejects_failures_above_sample_count() -> None:
    with pytest.raises(ValueError, match="failure_count"):
        _report(sample_count=3, failure_count=4)
