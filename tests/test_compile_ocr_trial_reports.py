from __future__ import annotations

from pathlib import Path

import pytest

from scripts.compile_ocr_trial_reports import (
    _recognition_row,
    _validate_metric_rows,
)


SHA = "a" * 64


def test_recognition_ledger_row_keeps_required_metric_provenance() -> None:
    report = {
        "build_id": "ocr-recognizer-eval-deadbeef",
        "status": "passed",
        "split": "dev_select",
        "manifest_sha256": SHA,
        "detector_sha256": SHA,
        "recognizer_sha256": SHA,
        "checkpoint_sha256": SHA,
        "calibration_sha256": None,
        "configuration_sha256": SHA,
        "source_commit": "1" * 40,
        "device": "gpu:0",
        "sample_count": 100,
        "failure_count": 0,
        "duration_seconds": 12.5,
        "private_row_count": 0,
        "model_name": "recognizer",
        "metrics": {
            "trial_id": "trial",
            "track": "general",
            "cer": 0.1,
            "wer": 0.2,
        },
    }

    row = _recognition_row(
        Path("trial.json"),
        report,
        selected=True,
        accepted=True,
    )

    _validate_metric_rows([row])
    for field in (
        "build_id",
        "manifest_sha256",
        "detector_sha256",
        "recognizer_sha256",
        "checkpoint_sha256",
        "calibration_sha256",
        "configuration_sha256",
        "source_commit",
        "device",
        "sample_count",
        "failure_count",
        "duration_seconds",
        "private_row_count",
    ):
        assert field in row


def test_metric_ledger_validation_rejects_missing_provenance() -> None:
    with pytest.raises(ValueError, match="duration_seconds"):
        _validate_metric_rows(
            [
                {
                    "build_id": "build",
                    "split": "dev_select",
                    "manifest_sha256": SHA,
                    "detector_sha256": SHA,
                    "recognizer_sha256": SHA,
                    "checkpoint_sha256": SHA,
                    "calibration_sha256": None,
                    "configuration_sha256": SHA,
                    "source_commit": "1" * 40,
                    "device": "gpu:0",
                    "sample_count": 1,
                    "failure_count": 0,
                    "private_row_count": 0,
                }
            ]
        )
