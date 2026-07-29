"""Perspective-correct OCR crops and bounded evidence-preserving retries."""
from __future__ import annotations

import math
import re
import statistics
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Pattern, Sequence

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps


@dataclass(frozen=True)
class PaddingProfile:
    name: str
    horizontal: float
    vertical: float


PADDING_PROFILES = {
    "A": PaddingProfile("A", 0.03, 0.08),
    "B": PaddingProfile("B", 0.05, 0.12),
    "C": PaddingProfile("C", 0.07, 0.15),
}


def rectify_polygon_crop(
    image: Image.Image,
    polygon: Sequence[Sequence[float]],
    profile: PaddingProfile = PADDING_PROFILES["B"],
    *,
    minimum_character_height: int = 24,
    maximum_upscale: float = 3.0,
) -> tuple[Image.Image, dict[str, Any]]:
    """Rectify a four-point region after proportional source-space expansion."""
    points = np.asarray(polygon, dtype=np.float64)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("OCR crop polygon must contain four finite points")
    ordered = _order_quad(points)
    center = ordered.mean(axis=0)
    expanded = ordered.copy()
    for index, point in enumerate(ordered):
        horizontal = profile.horizontal
        vertical = profile.vertical
        delta = point - center
        expanded[index] = center + np.array(
            [delta[0] * (1.0 + 2.0 * horizontal), delta[1] * (1.0 + 2.0 * vertical)]
        )
    expanded[:, 0] = np.clip(expanded[:, 0], 0, image.width - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, image.height - 1)
    width = max(
        np.linalg.norm(expanded[1] - expanded[0]),
        np.linalg.norm(expanded[2] - expanded[3]),
    )
    height = max(
        np.linalg.norm(expanded[3] - expanded[0]),
        np.linalg.norm(expanded[2] - expanded[1]),
    )
    target_width = int(round(width))
    target_height = int(round(height))
    if target_width < 2 or target_height < 2:
        raise ValueError("OCR crop is empty or too small after rectification")
    destination = np.array(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(
        expanded.astype(np.float32), destination
    )
    rgb = np.asarray(image.convert("RGB"))
    warped = cv2.warpPerspective(
        rgb,
        matrix,
        (target_width, target_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    if warped.size == 0:
        raise ValueError("OCR crop is empty after perspective transform")
    upscale = min(
        float(maximum_upscale),
        max(1.0, float(minimum_character_height) / max(1.0, target_height)),
    )
    crop = Image.fromarray(warped, mode="RGB")
    if upscale > 1.001:
        crop = crop.resize(
            (
                max(2, round(crop.width * upscale)),
                max(2, round(crop.height * upscale)),
            ),
            Image.Resampling.LANCZOS,
        )
    return crop, {
        "padding_profile": profile.name,
        "horizontal_padding_ratio": profile.horizontal,
        "vertical_padding_ratio": profile.vertical,
        "rectified_width": target_width,
        "rectified_height": target_height,
        "upscale_factor": float(upscale),
        "source_polygon": points.tolist(),
        "expanded_polygon": expanded.tolist(),
    }


def retry_crop_variants(
    image: Image.Image,
    polygon: Sequence[Sequence[float]],
    *,
    initial_profile: str = "B",
    maximum_candidates: int = 5,
) -> list[tuple[str, Image.Image, dict[str, Any]]]:
    """Create at most five documented alternatives in deterministic order."""
    if initial_profile not in PADDING_PROFILES:
        raise ValueError(f"unknown padding profile: {initial_profile}")
    if not 1 <= int(maximum_candidates) <= 5:
        raise ValueError("maximum retry candidates must be within [1, 5]")
    ordered_names = list(PADDING_PROFILES)
    initial_index = ordered_names.index(initial_profile)
    next_name = ordered_names[min(len(ordered_names) - 1, initial_index + 1)]
    base, base_meta = rectify_polygon_crop(image, polygon, PADDING_PROFILES[initial_profile])
    variants: list[tuple[str, Image.Image, dict[str, Any]]] = [
        ("original_rectified", base, base_meta)
    ]
    if next_name != initial_profile:
        larger, larger_meta = rectify_polygon_crop(
            image, polygon, PADDING_PROFILES[next_name]
        )
        variants.append(("larger_padding", larger, larger_meta))
    upscaled = base.resize(
        (max(2, base.width * 2), max(2, base.height * 2)),
        Image.Resampling.LANCZOS,
    )
    variants.append(("upscaled", upscaled, {**base_meta, "retry_upscale": 2.0}))
    variants.append(
        (
            "grayscale",
            ImageOps.grayscale(base).convert("RGB"),
            {**base_meta, "color_transform": "grayscale"},
        )
    )
    variants.append(
        (
            "local_contrast",
            ImageEnhance.Contrast(ImageOps.autocontrast(base, cutoff=1)).enhance(1.15),
            {**base_meta, "color_transform": "local_contrast"},
        )
    )
    return variants[: int(maximum_candidates)]


def recognize_with_retries(
    image: Image.Image,
    polygon: Sequence[Sequence[float]],
    recognizer: Callable[[Image.Image], Mapping[str, Any]],
    *,
    route: str,
    expected_pattern: str | Pattern[str] | None = None,
    max_candidates: int = 5,
    confidence_threshold: float = 0.65,
    initial_profile: str = "B",
    initial_prediction: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run retries only after a low-confidence first result and retain all evidence."""
    if not 0.0 <= float(confidence_threshold) <= 1.0:
        raise ValueError("confidence threshold must be in [0, 1]")
    pattern = re.compile(expected_pattern) if isinstance(expected_pattern, str) else expected_pattern
    variants = retry_crop_variants(
        image,
        polygon,
        initial_profile=initial_profile,
        maximum_candidates=max_candidates,
    )
    candidates: list[dict[str, Any]] = []
    first_variant, first_crop, first_meta = variants[0]
    first = _normalize_prediction(
        initial_prediction if initial_prediction is not None else recognizer(first_crop)
    )
    candidates.append(
        _candidate_record(first, first_variant, first_meta, route, pattern, [])
    )
    retry = first["confidence"] < float(confidence_threshold)
    if retry:
        for variant, crop, metadata in variants[1:]:
            prediction = _normalize_prediction(recognizer(crop))
            candidates.append(
                _candidate_record(
                    prediction,
                    variant,
                    metadata,
                    route,
                    pattern,
                    [item["text"] for item in candidates],
                )
            )
    selected = max(
        candidates,
        key=lambda item: (
            float(item["score"]),
            float(item["confidence"]),
            -int(item["ordinal"]),
        ),
    )
    return {
        "text": selected["text"],
        "confidence": selected["confidence"],
        "selected_variant": selected["variant"],
        "selection_reason": selected["selection_reason"],
        "score": selected["score"],
        "retry_performed": retry,
        "candidate_count": len(candidates),
        "original_candidate": {
            "text": candidates[0]["text"],
            "confidence": candidates[0]["confidence"],
        },
        "candidates": candidates,
    }


def _normalize_prediction(value: Mapping[str, Any]) -> dict[str, Any]:
    text = str(value.get("text", "")).strip()
    try:
        confidence = float(value.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    if not math.isfinite(confidence):
        confidence = 0.0
    return {"text": text, "confidence": max(0.0, min(1.0, confidence))}


def _candidate_record(
    prediction: Mapping[str, Any],
    variant: str,
    metadata: Mapping[str, Any],
    route: str,
    pattern: Pattern[str] | None,
    previous_texts: Sequence[str],
) -> dict[str, Any]:
    text = str(prediction["text"])
    confidence = float(prediction["confidence"])
    printable = sum(
        character.isprintable()
        and not unicodedata.category(character).startswith("C")
        for character in text
    )
    valid_ratio = printable / max(1, len(text))
    alpha = [character for character in text if character.isalpha()]
    thai_ratio = (
        sum("\u0e00" <= character <= "\u0e7f" for character in alpha)
        / max(1, len(alpha))
    )
    script_consistency = thai_ratio if route == "thai" else 1.0 - thai_ratio
    pattern_match = 1.0 if pattern is not None and pattern.search(text) else (
        0.5 if pattern is None else 0.0
    )
    normalized = " ".join(text.casefold().split())
    agreements = sum(
        normalized == " ".join(value.casefold().split()) for value in previous_texts
    )
    agreement_score = agreements / max(1, len(previous_texts))
    score = (
        0.55 * confidence
        + 0.15 * valid_ratio
        + 0.10 * script_consistency
        + 0.15 * pattern_match
        + 0.05 * agreement_score
    )
    reasons = ["recognizer_confidence", "valid_character_ratio", "script_consistency"]
    if pattern is not None:
        reasons.append("expected_pattern_match" if pattern_match else "expected_pattern_miss")
    if agreements:
        reasons.append("candidate_agreement")
    return {
        "ordinal": len(previous_texts),
        "variant": variant,
        "text": text,
        "confidence": confidence,
        "valid_character_ratio": valid_ratio,
        "script_consistency": script_consistency,
        "expected_pattern_score": pattern_match,
        "agreement_score": agreement_score,
        "score": float(max(0.0, min(1.0, score))),
        "selection_reason": "+".join(reasons),
        "crop": dict(metadata),
    }


def _order_quad(points: np.ndarray) -> np.ndarray:
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    ordered = np.empty((4, 2), dtype=np.float64)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    if len({tuple(point) for point in ordered}) != 4:
        raise ValueError("OCR crop polygon is degenerate")
    return ordered
