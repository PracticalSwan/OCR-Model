"""Shared provenance envelope for OCR upgrade metric reports."""
from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{7,64}$")
_PUBLIC_SPLITS = {
    "train",
    "dev_select",
    "dev_calibration",
    "test_in_domain",
    "unseen_coru",
    "unseen_domain_test",
}
_PRIVATE_SPLITS = {"private_operational", "private_test"}


def build_metric_report(
    *,
    build_id: str,
    split: str,
    manifest_sha256: str,
    detector_sha256: str,
    recognizer_sha256: str,
    checkpoint_sha256: str | None,
    calibration_sha256: str | None,
    configuration_sha256: str,
    source_commit: str,
    device: str,
    sample_count: int,
    failure_count: int,
    duration_seconds: float,
    private_row_count: int,
    metrics: Mapping[str, Any],
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build and validate the mandatory report metadata shared by all metrics."""
    if not str(build_id).strip():
        raise ValueError("build_id must be nonempty")
    if split not in _PUBLIC_SPLITS | _PRIVATE_SPLITS:
        raise ValueError(f"unsupported report split: {split}")
    required_hashes = {
        "manifest_sha256": manifest_sha256,
        "detector_sha256": detector_sha256,
        "recognizer_sha256": recognizer_sha256,
        "configuration_sha256": configuration_sha256,
    }
    optional_hashes = {
        "checkpoint_sha256": checkpoint_sha256,
        "calibration_sha256": calibration_sha256,
    }
    for name, value in {**required_hashes, **optional_hashes}.items():
        if value is None and name in optional_hashes:
            continue
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value.casefold()):
            raise ValueError(f"{name} must be a SHA-256 digest or null when optional")
    if not isinstance(source_commit, str) or not _GIT_SHA_RE.fullmatch(
        source_commit.casefold()
    ):
        raise ValueError("source_commit must be a hexadecimal Git object ID")
    if not str(device).strip():
        raise ValueError("device must be nonempty")
    if sample_count < 0:
        raise ValueError("sample_count must be non-negative")
    if failure_count < 0 or failure_count > sample_count:
        raise ValueError("failure_count must be between zero and sample_count")
    if duration_seconds < 0:
        raise ValueError("duration_seconds must be non-negative")
    if private_row_count < 0:
        raise ValueError("private_row_count must be non-negative")
    if private_row_count and split not in _PRIVATE_SPLITS:
        raise ValueError("private_row_count must be zero for public reports")

    timestamp = generated_at_utc or datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": "1.0",
        "build_id": build_id,
        "generated_at_utc": timestamp,
        "split": split,
        "manifest_sha256": manifest_sha256.casefold(),
        "detector_sha256": detector_sha256.casefold(),
        "recognizer_sha256": recognizer_sha256.casefold(),
        "checkpoint_sha256": (
            checkpoint_sha256.casefold() if checkpoint_sha256 is not None else None
        ),
        "calibration_sha256": (
            calibration_sha256.casefold() if calibration_sha256 is not None else None
        ),
        "configuration_sha256": configuration_sha256.casefold(),
        "source_commit": source_commit.casefold(),
        "device": device,
        "sample_count": int(sample_count),
        "failure_count": int(failure_count),
        "duration_seconds": float(duration_seconds),
        "private_row_count": int(private_row_count),
        "metrics": dict(metrics),
    }
