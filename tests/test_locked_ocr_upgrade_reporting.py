from __future__ import annotations

from scripts.evaluate_locked_ocr_upgrade import (
    _locked_reference_binding_errors,
)


def _report() -> dict:
    return {
        "split": "test_in_domain",
        "public_only": True,
        "private_row_count": 0,
        "failure_count": 0,
        "token_sources": ["ground_truth"],
        "checkpoint_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
        "calibration_sha256": "c" * 64,
        "evaluation_build_id": "final-ocr-v2",
        "sample_count": 10,
        "detector_sha256": "d" * 64,
        "recognizer_sha256": "e" * 64,
        "preprocessing_sha256": "f" * 64,
    }


def _errors(report: dict) -> list[str]:
    return _locked_reference_binding_errors(
        report,
        expected_checkpoint_sha256="a" * 64,
        expected_manifest_sha256="b" * 64,
        expected_calibration_sha256="c" * 64,
        expected_build_id="final-ocr-v2",
        expected_stack={
            "detector_sha256": "d" * 64,
            "recognizer_sha256": "e" * 64,
            "preprocessing_sha256": "f" * 64,
        },
        expected_sample_count=10,
    )


def test_locked_reference_binding_accepts_exact_frozen_stack() -> None:
    assert _errors(_report()) == []


def test_locked_reference_binding_rejects_stale_manifest_and_ocr_stack() -> None:
    report = _report()
    report["manifest_sha256"] = "0" * 64
    report["preprocessing_sha256"] = "1" * 64

    assert _errors(report) == [
        "manifest_sha256",
        "preprocessing_sha256",
    ]
