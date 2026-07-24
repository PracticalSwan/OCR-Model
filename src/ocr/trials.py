"""Pure metric, acceptance, and artifact helpers for bounded OCR trials."""
from __future__ import annotations

import hashlib
import math
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.evaluation.metrics import edit_distance, normalized_text, ocr_text_metrics
from src.rotation_common import sha256_file


_AMOUNT_FIELDS = frozenset({"subtotal", "tax", "total_amount"})
_DATE_FIELDS = frozenset({"date"})
_IDENTIFIER_FIELDS = frozenset(
    {"invoice_number", "receipt_number", "reference_number", "tax_identifier"}
)
_CURRENCY_FIELDS = frozenset({"currency"})
_EMAIL_FIELDS = frozenset({"email"})
_TURKISH_CHARACTERS = frozenset("çğıöşüÇĞİÖŞÜ")


def detector_page_result(
    reference_regions: Sequence[Mapping[str, Any]],
    predicted_regions: Sequence[Mapping[str, Any]],
    *,
    page_width: int,
    page_height: int,
    critical_token_ids: set[str] | frozenset[str],
    iou_threshold: float = 0.5,
    duration_seconds: float = 0.0,
) -> dict[str, Any]:
    """Match one page and retain the reference slices needed for acceptance."""
    if page_width <= 0 or page_height <= 0:
        raise ValueError("page dimensions must be positive")
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")
    references = [_polygon(region) for region in reference_regions]
    predictions = [_polygon(region) for region in predicted_regions]
    valid_references = [
        (index, polygon)
        for index, polygon in enumerate(references)
        if polygon is not None
    ]
    valid_predictions = [
        (index, polygon)
        for index, polygon in enumerate(predictions)
        if polygon is not None
    ]
    candidates: list[tuple[float, int, int]] = []
    for reference_index, reference in valid_references:
        for prediction_index, prediction in valid_predictions:
            iou = _polygon_iou(reference, prediction)
            if iou >= iou_threshold:
                candidates.append((iou, reference_index, prediction_index))
    matched_references: set[int] = set()
    matched_predictions: set[int] = set()
    matched_ious: list[float] = []
    for iou, reference_index, prediction_index in sorted(candidates, reverse=True):
        if (
            reference_index in matched_references
            or prediction_index in matched_predictions
        ):
            continue
        matched_references.add(reference_index)
        matched_predictions.add(prediction_index)
        matched_ious.append(iou)

    size_counts = {
        "small_expected": 0,
        "small_matched": 0,
        "medium_expected": 0,
        "medium_matched": 0,
        "large_expected": 0,
        "large_matched": 0,
    }
    critical_expected = 0
    critical_matched = 0
    for reference_index, polygon in valid_references:
        height_ratio = (
            float(np.max(polygon[:, 1]) - np.min(polygon[:, 1])) / page_height
        )
        # Relative bands keep the metric comparable across invoice scans and
        # lower-resolution forms without relying on a fixed pixel height.
        size = (
            "small"
            if height_ratio <= 0.05
            else "medium"
            if height_ratio <= 0.20
            else "large"
        )
        size_counts[f"{size}_expected"] += 1
        if reference_index in matched_references:
            size_counts[f"{size}_matched"] += 1
        token_ids = {
            str(value)
            for value in reference_regions[reference_index].get("token_ids", ())
        }
        if token_ids & set(critical_token_ids):
            critical_expected += 1
            if reference_index in matched_references:
                critical_matched += 1

    return {
        "true_positive": len(matched_references),
        "expected": len(valid_references),
        "predicted": len(valid_predictions),
        **size_counts,
        "critical_expected": critical_expected,
        "critical_matched": critical_matched,
        "mean_matched_iou": (
            sum(matched_ious) / len(matched_ious) if matched_ious else None
        ),
        "duration_seconds": float(duration_seconds),
        "failed": False,
    }


def aggregate_detector_results(
    page_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate page-level detector counts without averaging page ratios."""
    count_fields = (
        "true_positive",
        "expected",
        "predicted",
        "small_expected",
        "small_matched",
        "medium_expected",
        "medium_matched",
        "large_expected",
        "large_matched",
        "critical_expected",
        "critical_matched",
    )
    counts = {
        field: sum(int(row.get(field, 0)) for row in page_results)
        for field in count_fields
    }
    precision = _ratio(counts["true_positive"], counts["predicted"], default=0.0)
    recall = _ratio(counts["true_positive"], counts["expected"], default=0.0)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    duration = sum(float(row.get("duration_seconds", 0.0)) for row in page_results)
    failures = sum(bool(row.get("failed", False)) for row in page_results)
    return {
        **counts,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "small_text_recall": _ratio(
            counts["small_matched"], counts["small_expected"]
        ),
        "medium_text_recall": _ratio(
            counts["medium_matched"], counts["medium_expected"]
        ),
        "large_text_recall": _ratio(
            counts["large_matched"], counts["large_expected"]
        ),
        "critical_region_recall": _ratio(
            counts["critical_matched"], counts["critical_expected"]
        ),
        "sample_count": len(page_results),
        "failure_count": failures,
        "duration_seconds": duration,
        "time_per_page_seconds": duration / len(page_results)
        if page_results
        else None,
    }


def detector_acceptance(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    export_reload_passed: bool,
    adapter_contract_passed: bool,
    rotated_page_passed: bool,
    blank_low_text_passed: bool,
    private_row_count: int,
    maximum_time_per_page_seconds: float,
) -> dict[str, Any]:
    """Apply the detector replacement rules without discretionary tuning."""
    baseline_f1 = float(baseline["f1"])
    candidate_f1 = float(candidate["f1"])
    baseline_critical = _finite_float(baseline.get("critical_region_recall"))
    candidate_critical = _finite_float(candidate.get("critical_region_recall"))
    time_per_page = _finite_float(candidate.get("time_per_page_seconds"))
    criteria = {
        "f1_improvement_or_floor": (
            candidate_f1 - baseline_f1 >= 0.10 - 1e-12
            or candidate_f1 >= 0.60 - 1e-12
        ),
        "critical_recall_no_regression": (
            baseline_critical is not None
            and candidate_critical is not None
            and candidate_critical + 1e-12 >= baseline_critical
        ),
        "no_new_failures": int(candidate.get("failure_count", 0)) == 0,
        "processing_time_acceptable": (
            time_per_page is not None
            and time_per_page <= maximum_time_per_page_seconds
        ),
        "export_reload_passed": bool(export_reload_passed),
        "adapter_contract_passed": bool(adapter_contract_passed),
        "rotated_page_passed": bool(rotated_page_passed),
        "blank_low_text_passed": bool(blank_low_text_passed),
        "no_private_data": int(private_row_count) == 0,
    }
    return {
        "accepted": all(criteria.values()),
        "criteria": criteria,
        "f1_absolute_gain": candidate_f1 - baseline_f1,
    }


def recognition_sample_result(
    reference: str,
    prediction: str,
    *,
    confidence: float,
    critical_fields: Iterable[str] = (),
    category: str = "",
    duration_seconds: float = 0.0,
) -> dict[str, Any]:
    """Compute one ground-truth-crop result and its exact-match slices."""
    text = ocr_text_metrics(reference, prediction)
    expected = normalized_text(reference)
    actual = normalized_text(prediction)
    exact = expected == actual
    fields = {str(value).casefold() for value in critical_fields}
    normalized_category = str(category).casefold()
    has_numeric = any(character.isdigit() for character in reference)
    has_amount = bool(fields & _AMOUNT_FIELDS) or "amount" in normalized_category
    has_date = bool(fields & _DATE_FIELDS) or "date" in normalized_category
    has_identifier = (
        bool(fields & _IDENTIFIER_FIELDS)
        or "identifier" in normalized_category
        or normalized_category in {"invoice", "receipt", "reference"}
    )
    has_currency = (
        bool(fields & _CURRENCY_FIELDS)
        or "currency" in normalized_category
        or any(value in reference for value in ("$", "€", "£", "฿", "USD", "EUR", "THB"))
    )
    has_email = bool(fields & _EMAIL_FIELDS) or "email" in normalized_category
    turkish_reference = _turkish_characters(reference)
    turkish_prediction = _turkish_characters(prediction)
    turkish_accuracy = (
        max(
            0.0,
            1.0
            - edit_distance(turkish_reference, turkish_prediction)
            / max(1, len(turkish_reference)),
        )
        if turkish_reference
        else None
    )
    has_english = any("a" <= value.casefold() <= "z" for value in reference)
    finite_confidence = math.isfinite(float(confidence))
    return {
        **text,
        "reference_words": len(expected.split()),
        "exact": exact,
        "numeric_exact": exact if has_numeric else None,
        "amount_exact": exact if has_amount else None,
        "date_exact": exact if has_date else None,
        "identifier_exact": exact if has_identifier else None,
        "currency_exact": exact if has_currency else None,
        "email_exact": exact if has_email else None,
        "turkish_character_accuracy": turkish_accuracy,
        "english_exact": exact if has_english else None,
        "confidence": float(confidence),
        "confidence_finite": finite_confidence,
        "duration_seconds": float(duration_seconds),
    }


def aggregate_recognition_results(
    sample_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate OCR recognition metrics at corpus grain."""
    reference_characters = sum(
        int(row.get("reference_characters", 0)) for row in sample_results
    )
    character_errors = sum(
        int(row.get("character_errors", 0)) for row in sample_results
    )
    reference_words = sum(
        len(normalized_text(row.get("reference", "")).split())
        if "reference" in row
        else int(row.get("reference_words", 0))
        for row in sample_results
    )
    # `ocr_text_metrics` does not expose word count, so callers may supply it.
    if not reference_words:
        reference_words = sum(
            max(1, int(row.get("word_errors", 0)))
            for row in sample_results
            if int(row.get("reference_characters", 0)) > 0
        )
    word_errors = sum(int(row.get("word_errors", 0)) for row in sample_results)
    duration = sum(float(row.get("duration_seconds", 0.0)) for row in sample_results)
    confidences = [
        float(row["confidence"])
        for row in sample_results
        if bool(row.get("confidence_finite", False))
    ]
    labels = [
        float(bool(row.get("exact", False)))
        for row in sample_results
        if bool(row.get("confidence_finite", False))
    ]
    return {
        "sample_count": len(sample_results),
        "reference_characters": reference_characters,
        "character_errors": character_errors,
        "cer": character_errors / max(1, reference_characters),
        "reference_words": reference_words,
        "word_errors": word_errors,
        "wer": word_errors / max(1, reference_words),
        "exact_line_accuracy": _mean_boolean(sample_results, "exact"),
        "numeric_exact_match": _mean_optional(sample_results, "numeric_exact"),
        "amount_exact_match": _mean_optional(sample_results, "amount_exact"),
        "date_exact_match": _mean_optional(sample_results, "date_exact"),
        "identifier_exact_match": _mean_optional(
            sample_results, "identifier_exact"
        ),
        "currency_exact_match": _mean_optional(sample_results, "currency_exact"),
        "email_exact_match": _mean_optional(sample_results, "email_exact"),
        "turkish_character_accuracy": _mean_optional(
            sample_results, "turkish_character_accuracy"
        ),
        "english_exact_match": _mean_optional(sample_results, "english_exact"),
        "confidence_all_finite": len(confidences) == len(sample_results),
        "confidence_ece": _expected_calibration_error(confidences, labels),
        "confidence_brier": _brier_score(confidences, labels),
        "duration_seconds": duration,
        "time_per_sample_seconds": duration / len(sample_results)
        if sample_results
        else None,
        "failure_count": sum(bool(row.get("failed", False)) for row in sample_results),
    }


def recognizer_selection_score(metrics: Mapping[str, Any]) -> float:
    """Return the frozen weighted DEV_SELECT recognizer score."""
    cer = _clamp(_finite_float(metrics.get("cer")) or 0.0)
    wer = _clamp(_finite_float(metrics.get("wer")) or 0.0)
    return (
        0.25 * (1.0 - cer)
        + 0.20 * (1.0 - wer)
        + 0.20 * _metric(metrics, "amount_exact_match")
        + 0.15 * _metric(metrics, "identifier_exact_match")
        + 0.10 * _metric(metrics, "date_exact_match")
        + 0.05 * _metric(metrics, "currency_exact_match")
        + 0.05 * _metric(metrics, "turkish_character_accuracy")
    )


def recognizer_acceptance(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    export_reload_passed: bool,
    dictionary_match_passed: bool,
    private_row_count: int,
    maximum_time_per_sample_seconds: float,
) -> dict[str, Any]:
    """Apply the general recognizer replacement criteria."""
    baseline_wer = float(baseline["wer"])
    candidate_wer = float(candidate["wer"])
    relative_wer_gain = (
        (baseline_wer - candidate_wer) / baseline_wer
        if baseline_wer > 0
        else 0.0
    )
    baseline_critical = _critical_exact_score(baseline)
    candidate_critical = _critical_exact_score(candidate)
    critical_gain = candidate_critical - baseline_critical
    candidate_time = _finite_float(candidate.get("time_per_sample_seconds"))
    candidate_turkish = _finite_float(
        candidate.get("turkish_character_accuracy")
    )
    criteria = {
        "wer_improvement_or_floor": (
            relative_wer_gain >= 0.15 - 1e-12
            or candidate_wer <= 0.45 + 1e-12
        ),
        "critical_field_exact_gain": critical_gain >= 0.10 - 1e-12,
        "numeric_and_amount_no_regression": (
            _metric(candidate, "numeric_exact_match")
            + 1e-12
            >= _metric(baseline, "numeric_exact_match")
            and _metric(candidate, "amount_exact_match")
            + 1e-12
            >= _metric(baseline, "amount_exact_match")
        ),
        "english_no_material_regression": (
            _metric(candidate, "english_exact_match") + 0.02 + 1e-12
            >= _metric(baseline, "english_exact_match")
        ),
        "turkish_valid": (
            candidate_turkish is not None and 0.0 <= candidate_turkish <= 1.0
        ),
        "confidence_finite": bool(candidate.get("confidence_all_finite", False)),
        "export_reload_passed": bool(export_reload_passed),
        "dictionary_match_passed": bool(dictionary_match_passed),
        "processing_time_acceptable": (
            candidate_time is not None
            and candidate_time <= maximum_time_per_sample_seconds
        ),
        "no_private_data": int(private_row_count) == 0,
    }
    return {
        "accepted": all(criteria.values()),
        "criteria": criteria,
        "wer_relative_gain": relative_wer_gain,
        "critical_field_exact_gain": critical_gain,
        "baseline_selection_score": recognizer_selection_score(baseline),
        "candidate_selection_score": recognizer_selection_score(candidate),
    }


def sha256_tree(root: str | Path) -> str:
    """Hash every relative path and file digest in a deterministic tree."""
    path = Path(root)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    files = sorted(value for value in path.rglob("*") if value.is_file())
    if not files:
        raise ValueError(f"artifact tree contains no files: {path}")
    for file_path in files:
        relative = file_path.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(file_path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _polygon(region: Mapping[str, Any]) -> np.ndarray | None:
    value = region.get("polygon")
    try:
        points = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or len(points) < 3
        or not np.isfinite(points).all()
    ):
        return None
    hull = cv2.convexHull(points).reshape(-1, 2)
    return hull if len(hull) >= 3 and cv2.contourArea(hull) > 0 else None


def _polygon_iou(left: np.ndarray, right: np.ndarray) -> float:
    left_area = float(cv2.contourArea(left))
    right_area = float(cv2.contourArea(right))
    intersection, _ = cv2.intersectConvexConvex(left, right)
    union = left_area + right_area - float(intersection)
    return _clamp(float(intersection) / union) if union > 0 else 0.0


def _ratio(numerator: int, denominator: int, *, default: float | None = None) -> float | None:
    return numerator / denominator if denominator else default


def _mean_boolean(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    return (
        sum(bool(row.get(field, False)) for row in rows) / len(rows)
        if rows
        else None
    )


def _mean_optional(
    rows: Sequence[Mapping[str, Any]], field: str
) -> float | None:
    values = [
        float(row[field])
        for row in rows
        if row.get(field) is not None and math.isfinite(float(row[field]))
    ]
    return sum(values) / len(values) if values else None


def _turkish_characters(value: str) -> list[str]:
    return [
        character.casefold()
        for character in unicodedata.normalize("NFC", str(value))
        if character in _TURKISH_CHARACTERS
    ]


def _expected_calibration_error(
    confidences: Sequence[float], labels: Sequence[float], *, bins: int = 10
) -> float | None:
    if not confidences:
        return None
    total = len(confidences)
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            sample
            for sample, confidence in enumerate(confidences)
            if lower <= _clamp(confidence) < upper
            or (index == bins - 1 and _clamp(confidence) == 1.0)
        ]
        if not members:
            continue
        mean_confidence = sum(_clamp(confidences[item]) for item in members) / len(members)
        mean_accuracy = sum(labels[item] for item in members) / len(members)
        error += len(members) / total * abs(mean_confidence - mean_accuracy)
    return error


def _brier_score(
    confidences: Sequence[float], labels: Sequence[float]
) -> float | None:
    return (
        sum((_clamp(confidence) - label) ** 2 for confidence, label in zip(confidences, labels))
        / len(confidences)
        if confidences
        else None
    )


def _critical_exact_score(metrics: Mapping[str, Any]) -> float:
    values = [
        _finite_float(metrics.get(field))
        for field in (
            "amount_exact_match",
            "date_exact_match",
            "identifier_exact_match",
            "currency_exact_match",
            "email_exact_match",
        )
    ]
    available = [value for value in values if value is not None]
    return sum(available) / len(available) if available else 0.0


def _metric(metrics: Mapping[str, Any], field: str) -> float:
    value = _finite_float(metrics.get(field))
    return _clamp(value) if value is not None else 0.0


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
