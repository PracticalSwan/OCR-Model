"""Deterministic binding for the OCR models and preprocessing policy."""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from src.ocr.model_registry import ModelRegistry
from src.rotation_common import configuration_hash


def build_ocr_stack_binding(
    cfg: Mapping[str, Any],
    registry: ModelRegistry,
    *,
    ocr_profile: str,
) -> dict[str, Any]:
    """Return the exact OCR model and preprocessing hashes used by a profile."""
    profile = str(ocr_profile).casefold()
    if profile not in {"original", "custom", "adaptive"}:
        raise ValueError(f"unsupported OCR profile: {ocr_profile!r}")

    detector, general = registry.route_models("general")
    thai_detector, thai = registry.route_models("thai")
    if detector.artifact_hash != thai_detector.artifact_hash:
        raise ValueError("general and Thai OCR routes must use the same detector")

    recognizer_sha256 = hashlib.sha256(
        (general.artifact_hash + thai.artifact_hash).encode("ascii")
    ).hexdigest()
    ocr_cfg = dict(cfg.get("ocr") or {})
    adaptive = profile == "adaptive"
    preprocessing_policy = {
        "schema_version": "1.0",
        "ocr_profile": profile,
        "preprocessing_version": str(
            ocr_cfg.get("preprocessing_version", "1.0")
        ),
        "preprocessing_profile": (
            str(ocr_cfg.get("adaptive_preprocessing_profile", "quality_auto"))
            if adaptive
            else str(ocr_cfg.get("preprocessing_profile", "original"))
        ),
        "orientation_candidates": list(
            ocr_cfg.get("orientation_candidates", [0, 90, 180, 270])
        ),
        "enable_continuous_deskew": bool(
            ocr_cfg.get("enable_continuous_deskew", True)
        ),
        "enable_document_orientation_classifier": bool(
            ocr_cfg.get("enable_document_orientation_classifier", False)
        ),
        "enable_document_unwarping": bool(
            ocr_cfg.get("enable_document_unwarping", False)
        ),
        "enable_textline_orientation": bool(
            ocr_cfg.get("enable_textline_orientation", False)
        ),
        "adaptive_rendering": _profile_component(
            ocr_cfg.get("adaptive_rendering"), enabled=adaptive
        ),
        "tiling": _profile_component(
            ocr_cfg.get("tiling"), enabled=adaptive
        ),
        "recognition_retries": _profile_component(
            ocr_cfg.get("recognition_retries"), enabled=adaptive
        ),
        "kmeans_controls_ocr": False,
    }
    return {
        "schema_version": "1.0",
        "ocr_profile": profile,
        "detector_model_id": detector.model_id or detector.name,
        "detector_sha256": detector.artifact_hash,
        "general_recognizer_model_id": general.model_id or general.name,
        "general_recognizer_sha256": general.artifact_hash,
        "thai_recognizer_model_id": thai.model_id or thai.name,
        "thai_recognizer_sha256": thai.artifact_hash,
        "recognizer_sha256": recognizer_sha256,
        "preprocessing_sha256": configuration_hash(preprocessing_policy),
        "preprocessing_policy": preprocessing_policy,
    }


def validate_ocr_stack_binding(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    """Reject a calibration artifact bound to a different OCR stack."""
    for key in (
        "detector_sha256",
        "recognizer_sha256",
        "preprocessing_sha256",
    ):
        actual_value = str(actual.get(key, "")).casefold()
        expected_value = str(expected.get(key, "")).casefold()
        if not actual_value or actual_value != expected_value:
            raise ValueError(
                f"calibration OCR stack binding mismatch for {key}"
            )


def _profile_component(value: Any, *, enabled: bool) -> dict[str, Any]:
    component = dict(value or {})
    component["enabled"] = bool(enabled and component.get("enabled", False))
    return component
