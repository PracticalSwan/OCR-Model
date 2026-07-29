"""Quality signals and deterministic selection for adaptive PDF rendering."""
from __future__ import annotations

import math
import re
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from src.ocr.scoring import score_ocr_candidate


_CRITICAL_PATTERN = re.compile(
    r"(?:\d[\d\s.,:/-]{2,}|(?:inv|invoice|receipt|ref(?:erence)?)\s*[-:#]?\s*\w+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AdaptiveRenderingConfig:
    """Conservative pass-one quality thresholds selected on development data."""

    base_dpi: int = 200
    rerender_dpi: int = 300
    minimum_box_count: int = 3
    minimum_median_text_height_ratio: float = 0.007
    minimum_text_coverage: float = 0.0015
    minimum_mean_confidence: float = 0.55
    minimum_median_confidence: float = 0.55
    minimum_text_characters: int = 16
    sparse_table_box_count: int = 8
    minimum_score_gain: float = 0.0

    def __post_init__(self) -> None:
        if self.base_dpi < 72 or self.rerender_dpi > 600:
            raise ValueError("adaptive PDF DPI values must be within [72, 600]")
        if self.rerender_dpi <= self.base_dpi:
            raise ValueError("rerender DPI must be greater than base DPI")
        for name in (
            "minimum_median_text_height_ratio",
            "minimum_text_coverage",
            "minimum_mean_confidence",
            "minimum_median_confidence",
            "minimum_score_gain",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def page_quality_signals(
    result: Mapping[str, Any],
    *,
    image_width: int,
    image_height: int,
    critical_fields_expected: bool = False,
    table_like: bool = False,
) -> dict[str, Any]:
    """Summarize page-level OCR quality without consulting held-out labels."""
    words = [
        word
        for word in (result.get("words") or [])
        if isinstance(word, Mapping) and str(word.get("text", "")).strip()
    ]
    confidences = [
        value
        for word in words
        if (value := _finite_confidence(word.get("confidence"))) is not None
    ]
    heights = [
        max(0.0, float(word["bbox"][3]) - float(word["bbox"][1]))
        for word in words
        if _valid_bbox(word.get("bbox"))
    ]
    page_area = max(1.0, float(image_width) * float(image_height))
    detected_area = sum(_polygon_area(word.get("polygon")) for word in words)
    full_text = str(result.get("full_text") or "\n".join(str(w["text"]) for w in words))
    compact_text = "".join(full_text.split())
    critical_found = bool(_CRITICAL_PATTERN.search(full_text))
    median_height = statistics.median(heights) if heights else 0.0
    mean_confidence = statistics.fmean(confidences) if confidences else 0.0
    median_confidence = statistics.median(confidences) if confidences else 0.0
    return {
        "box_count": len(words),
        "median_text_height_pixels": float(median_height),
        "median_text_height_ratio": float(median_height / max(1, image_height)),
        "detected_text_coverage": float(min(1.0, detected_area / page_area)),
        "mean_confidence": float(mean_confidence),
        "median_confidence": float(median_confidence),
        "text_character_count": len(compact_text),
        "critical_fields_expected": bool(critical_fields_expected),
        "critical_pattern_found": critical_found,
        "table_like": bool(table_like),
        "table_sparse": bool(table_like and len(words) < 8),
        "page_width": int(image_width),
        "page_height": int(image_height),
    }


def rerender_reasons(
    signals: Mapping[str, Any],
    config: AdaptiveRenderingConfig,
) -> list[str]:
    """Return stable reason codes; an empty list means the 200-DPI pass is kept."""
    reasons: list[str] = []
    if float(signals.get("median_text_height_ratio", 0.0)) < config.minimum_median_text_height_ratio:
        reasons.append("tiny_median_text_height")
    if int(signals.get("box_count", 0)) < config.minimum_box_count:
        reasons.append("suspiciously_few_text_boxes")
    if float(signals.get("detected_text_coverage", 0.0)) < config.minimum_text_coverage:
        reasons.append("low_text_coverage")
    if (
        float(signals.get("mean_confidence", 0.0)) < config.minimum_mean_confidence
        or float(signals.get("median_confidence", 0.0)) < config.minimum_median_confidence
    ):
        reasons.append("low_recognition_confidence")
    if int(signals.get("text_character_count", 0)) < config.minimum_text_characters:
        reasons.append("unusually_short_ocr_text")
    if (
        bool(signals.get("critical_fields_expected"))
        and not bool(signals.get("critical_pattern_found"))
    ):
        reasons.append("critical_text_missing")
    if (
        bool(signals.get("table_like"))
        and int(signals.get("box_count", 0)) < config.sparse_table_box_count
    ):
        reasons.append("sparse_table_detection")
    return reasons


def render_candidate_score(
    result: Mapping[str, Any], image_size: tuple[int, int]
) -> float:
    """Use the same bounded OCR candidate score for both render passes."""
    width, height = image_size
    return float(score_ocr_candidate(result, width, height)["total"])


def select_render_candidate(
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    first_size: tuple[int, int],
    second_size: tuple[int, int],
    first_dpi: int,
    second_dpi: int,
    reasons: Sequence[str],
    minimum_score_gain: float = 0.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select one render result and return complete auditable provenance."""
    first_score = render_candidate_score(first, first_size)
    second_score = render_candidate_score(second, second_size)
    use_second = second_score > first_score + float(minimum_score_gain)
    selected = second if use_second else first
    provenance = {
        "enabled": True,
        "first_pass_dpi": int(first_dpi),
        "second_pass_dpi": int(second_dpi),
        "selected_dpi": int(second_dpi if use_second else first_dpi),
        "rerender_reasons": list(reasons),
        "first_pass_score": first_score,
        "second_pass_score": second_score,
        "first_pass_duration_seconds": max(
            0.0, float(first.get("duration_seconds", 0.0))
        ),
        "second_pass_duration_seconds": max(
            0.0, float(second.get("duration_seconds", 0.0))
        ),
        "processing_time_impact_seconds": max(
            0.0, float(second.get("duration_seconds", 0.0))
        ),
        "selection_reason": (
            "second_pass_quality_gain" if use_second else "first_pass_not_worse"
        ),
    }
    return selected, provenance


def no_rerender_provenance(
    result: Mapping[str, Any],
    *,
    image_size: tuple[int, int],
    dpi: int,
) -> dict[str, Any]:
    score = render_candidate_score(result, image_size)
    return {
        "enabled": True,
        "first_pass_dpi": int(dpi),
        "second_pass_dpi": None,
        "selected_dpi": int(dpi),
        "rerender_reasons": [],
        "first_pass_score": score,
        "second_pass_score": None,
        "first_pass_duration_seconds": max(
            0.0, float(result.get("duration_seconds", 0.0))
        ),
        "second_pass_duration_seconds": 0.0,
        "processing_time_impact_seconds": 0.0,
        "selection_reason": "quality_gate_passed",
    }


def _finite_confidence(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _valid_bbox(value: Any) -> bool:
    try:
        x0, y0, x1, y1 = map(float, value)
    except (TypeError, ValueError):
        return False
    return all(math.isfinite(v) for v in (x0, y0, x1, y1)) and x1 > x0 and y1 > y0


def _polygon_area(value: Any) -> float:
    try:
        points = [(float(point[0]), float(point[1])) for point in value]
    except (TypeError, ValueError, IndexError):
        return 0.0
    if len(points) < 3 or not all(math.isfinite(v) for point in points for v in point):
        return 0.0
    return abs(
        sum(
            x0 * y1 - x1 * y0
            for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1])
        )
    ) / 2.0
