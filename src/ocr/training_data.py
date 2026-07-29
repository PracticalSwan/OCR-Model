"""Public OCR detector and recognizer training-data conversion helpers."""
from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from PIL import Image


SUPPORTED_PUBLIC_DATASETS = frozenset({"fatura", "funsd", "sroie"})
TRAINING_SPLITS = frozenset({"train", "dev_select"})
CRITICAL_FIELDS = frozenset(
    {
        "date",
        "invoice_number",
        "receipt_number",
        "reference_number",
        "subtotal",
        "tax",
        "total_amount",
        "currency",
        "email",
        "phone_number",
    }
)


def convert_fatura_detection(annotation: Mapping[str, Any]) -> list[dict[str, Any]]:
    return _convert_detection_annotation(annotation, expected_dataset="fatura")


def convert_funsd_detection(annotation: Mapping[str, Any]) -> list[dict[str, Any]]:
    return _convert_detection_annotation(annotation, expected_dataset="funsd")


def convert_sroie_detection(annotation: Mapping[str, Any]) -> list[dict[str, Any]]:
    return _convert_detection_annotation(annotation, expected_dataset="sroie")


def convert_detection_annotation(
    annotation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Dispatch a normalized public annotation to its dataset converter."""
    dataset = str(annotation.get("dataset", "")).casefold()
    converters = {
        "fatura": convert_fatura_detection,
        "funsd": convert_funsd_detection,
        "sroie": convert_sroie_detection,
    }
    try:
        converter = converters[dataset]
    except KeyError as exc:
        raise ValueError(f"unsupported OCR training dataset: {dataset!r}") from exc
    return converter(annotation)


def _convert_detection_annotation(
    annotation: Mapping[str, Any],
    *,
    expected_dataset: str,
) -> list[dict[str, Any]]:
    dataset = str(annotation.get("dataset", "")).casefold()
    if dataset != expected_dataset:
        raise ValueError(
            f"dataset converter mismatch: expected {expected_dataset}, found {dataset}"
        )
    if _truth(annotation.get("is_private")):
        raise ValueError("private annotation is refused")
    page = annotation.get("page")
    if not isinstance(page, Mapping):
        raise ValueError("annotation page metadata is missing")
    width = _positive_int(page.get("width"), "page width")
    height = _positive_int(page.get("height"), "page height")
    if not str(annotation.get("image_path", "")).startswith(
        f"data/raw/public/{expected_dataset}/"
    ):
        raise ValueError("annotation source provenance is not public")

    grouped: dict[tuple[tuple[float, float], ...], dict[str, Any]] = {}
    for token in repaired_annotation_tokens(annotation):
        if not isinstance(token, Mapping):
            raise ValueError("annotation token must be an object")
        points = _token_polygon(token)
        _validate_polygon(
            points,
            width=width,
            height=height,
            token_id=str(token.get("id", "<missing>")),
        )
        ignored = _truth(token.get("ignore")) or _truth(token.get("ignored"))
        text = str(token.get("text", ""))
        if ignored and not text.strip():
            text = "###"
        else:
            validate_transcription(text)
        key = tuple((round(x, 6), round(y, 6)) for x, y in points)
        region = grouped.setdefault(
            key,
            {
                "transcriptions": [],
                "points": [[float(x), float(y)] for x, y in points],
                "token_ids": [],
                "source_ids": [],
                "ignore": ignored,
            },
        )
        region["transcriptions"].append(text)
        region["token_ids"].append(str(token.get("id", "")))
        source_id = str(token.get("source_id", ""))
        if source_id and source_id not in region["source_ids"]:
            region["source_ids"].append(source_id)
        region["ignore"] = bool(region["ignore"] and ignored)

    converted: list[dict[str, Any]] = []
    for region in grouped.values():
        transcription = " ".join(region.pop("transcriptions"))
        validate_transcription(transcription)
        converted.append({"transcription": transcription, **region})
    if not converted:
        raise ValueError("annotation contains no valid text-detection regions")
    return converted


def repaired_annotation_tokens(
    annotation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Recover physical SROIE rows swallowed by an unmatched source quote."""
    tokens = [dict(token) for token in annotation.get("tokens", [])]
    if str(annotation.get("dataset", "")).casefold() != "sroie":
        return tokens
    repaired: list[dict[str, Any]] = []
    coordinate_pattern = re.compile(
        r"^\s*(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?),"
        r"(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?),"
        r"(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?),"
        r"(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?),(.*)$"
    )
    for token in tokens:
        text = str(token.get("text", ""))
        physical_lines = text.splitlines()
        if len(physical_lines) <= 1:
            repaired.append(token)
            continue
        first_text = _strip_unmatched_quote(physical_lines[0].strip())
        first = dict(token)
        first["text"] = first_text
        repaired.append(first)
        for line_index, line in enumerate(physical_lines[1:], start=1):
            match = coordinate_pattern.fullmatch(line)
            if match is None:
                raise ValueError(
                    "SROIE multiline transcription is not a recoverable physical row"
                )
            coordinates = [float(value) for value in match.groups()[:8]]
            continuation_text = _strip_unmatched_quote(match.group(9).strip())
            validate_transcription(continuation_text)
            points = [
                [coordinates[0], coordinates[1]],
                [coordinates[2], coordinates[3]],
                [coordinates[4], coordinates[5]],
                [coordinates[6], coordinates[7]],
            ]
            repaired.append(
                {
                    **token,
                    "id": f"{token.get('id', 'token')}:physical:{line_index}",
                    "text": continuation_text,
                    "polygon": points,
                    "bbox": [
                        min(point[0] for point in points),
                        min(point[1] for point in points),
                        max(point[0] for point in points),
                        max(point[1] for point in points),
                    ],
                    "source_id": f"{token.get('source_id', '')}:physical:{line_index}",
                    "source_repair": "sroie_unmatched_quote_physical_row_v1",
                }
            )
    return repaired


def assert_split_isolation(rows: Sequence[Mapping[str, Any]]) -> None:
    """Refuse duplicate families or exact source bytes across train and dev."""
    group_roles: dict[str, set[str]] = defaultdict(set)
    hash_roles: dict[str, set[str]] = defaultdict(set)
    build_ids = {
        str(row.get("build_id", "")).strip()
        for row in rows
        if str(row.get("build_id", "")).strip()
    }
    if len(build_ids) > 1:
        raise ValueError(f"mixed build IDs are refused: {sorted(build_ids)}")
    for row in rows:
        role = _normalized_role(row.get("project_split"))
        for field in ("duplicate_group_id", "split_group_id"):
            value = str(row.get(field, "")).strip()
            if value:
                group_roles[f"{field}:{value}"].add(role)
        digest = str(row.get("sha256", "")).casefold()
        if digest:
            hash_roles[digest].add(role)
    leaked_groups = sorted(key for key, roles in group_roles.items() if len(roles) > 1)
    if leaked_groups:
        kind = leaked_groups[0].split(":", 1)[0]
        raise ValueError(
            f"{kind} leakage across train and validation: {leaked_groups[:5]}"
        )
    leaked_hashes = sorted(key for key, roles in hash_roles.items() if len(roles) > 1)
    if leaked_hashes:
        raise ValueError(
            f"source hash leakage across train and validation: {leaked_hashes[:5]}"
        )


def build_line_groups(tokens: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Group normalized regions into deterministic horizontal text lines."""
    regions = _collapse_token_regions(tokens)
    ordered = sorted(regions, key=lambda row: (row["top"], row["left"], row["text"]))
    lines: list[dict[str, Any]] = []
    for region in ordered:
        best: dict[str, Any] | None = None
        best_score = -math.inf
        for line in lines:
            overlap = max(
                0.0,
                min(line["bottom"], region["bottom"])
                - max(line["top"], region["top"]),
            )
            minimum_height = min(line["height"], region["height"])
            overlap_ratio = overlap / minimum_height if minimum_height > 0 else 0.0
            baseline_delta = abs(line["bottom"] - region["bottom"])
            horizontal_gap = max(0.0, region["left"] - line["right"])
            same_baseline = baseline_delta <= max(line["height"], region["height"]) * 0.55
            reasonable_gap = horizontal_gap <= max(
                line["height"], region["height"]
            ) * 12.0
            if (overlap_ratio >= 0.35 or same_baseline) and reasonable_gap:
                score = overlap_ratio - baseline_delta / max(
                    line["height"], region["height"], 1.0
                )
                if score > best_score:
                    best = line
                    best_score = score
        if best is None:
            lines.append(
                {
                    "regions": [region],
                    "left": region["left"],
                    "top": region["top"],
                    "right": region["right"],
                    "bottom": region["bottom"],
                    "height": region["height"],
                }
            )
        else:
            best["regions"].append(region)
            best["left"] = min(best["left"], region["left"])
            best["top"] = min(best["top"], region["top"])
            best["right"] = max(best["right"], region["right"])
            best["bottom"] = max(best["bottom"], region["bottom"])
            best["height"] = best["bottom"] - best["top"]

    results: list[dict[str, Any]] = []
    for line in lines:
        line_regions = sorted(line["regions"], key=lambda row: (row["left"], row["top"]))
        transcription = " ".join(row["text"] for row in line_regions)
        validate_transcription(transcription)
        results.append(
            {
                "transcription": transcription,
                "token_ids": [
                    token_id
                    for row in line_regions
                    for token_id in row["token_ids"]
                ],
                "source_ids": [
                    source_id
                    for row in line_regions
                    for source_id in row["source_ids"]
                ],
                "polygon": [
                    [float(line["left"]), float(line["top"])],
                    [float(line["right"]), float(line["top"])],
                    [float(line["right"]), float(line["bottom"])],
                    [float(line["left"]), float(line["bottom"])],
                ],
            }
        )
    return sorted(results, key=lambda row: (row["polygon"][0][1], row["polygon"][0][0]))


def build_word_regions(tokens: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Collapse duplicate source regions into recognition-ready word regions."""
    regions = _collapse_token_regions(tokens)
    return [
        {
            "transcription": region["text"],
            "token_ids": list(region["token_ids"]),
            "source_ids": list(region["source_ids"]),
            "polygon": [list(point) for point in region["polygon"]],
        }
        for region in sorted(
            regions,
            key=lambda row: (row["top"], row["left"], row["text"]),
        )
    ]


def critical_fields_by_token(
    annotation: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    token_fields: dict[str, set[str]] = defaultdict(set)
    fields = annotation.get("canonical_fields", {})
    if not isinstance(fields, Mapping):
        return {}
    for field, value in fields.items():
        normalized_field = str(field).casefold()
        if normalized_field not in CRITICAL_FIELDS or not isinstance(value, Mapping):
            continue
        for token_id in value.get("token_ids", []):
            if str(token_id):
                token_fields[str(token_id)].add(normalized_field)
    return {
        token_id: tuple(sorted(field_names))
        for token_id, field_names in token_fields.items()
    }


def rectify_polygon_crop(
    image: Image.Image,
    polygon: Sequence[Sequence[float]],
    *,
    padding_ratio: float,
) -> Image.Image:
    """Perspective-rectify a four-point crop with proportional padding."""
    if not 0.0 <= padding_ratio <= 1.0:
        raise ValueError("padding_ratio must be between 0 and 1")
    points = np.asarray(polygon, dtype=np.float64)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("crop polygon must contain four finite points")
    ordered = _ordered_quad(points)
    center = ordered.mean(axis=0)
    expanded = center + (ordered - center) * (1.0 + 2.0 * padding_ratio)
    expanded[:, 0] = np.clip(expanded[:, 0], 0, image.width - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, image.height - 1)
    tl, tr, br, bl = expanded
    width = max(
        int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))),
        1,
    )
    height = max(
        int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))),
        1,
    )
    if width < 2 or height < 2:
        raise ValueError("rectified crop dimensions are too small")
    return image.transform(
        (width, height),
        Image.Transform.QUAD,
        (
            float(tl[0]),
            float(tl[1]),
            float(bl[0]),
            float(bl[1]),
            float(br[0]),
            float(br[1]),
            float(tr[0]),
            float(tr[1]),
        ),
        resample=Image.Resampling.BICUBIC,
    )


def validate_transcription(value: Any) -> str:
    text = str(value)
    if not text.strip() or any(character in text for character in ("\t", "\r", "\n")):
        raise ValueError("transcription must be nonempty and contain no tab/newline")
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("transcription is not valid UTF-8") from exc
    return text


def _collapse_token_regions(
    tokens: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    regions: dict[tuple[tuple[float, float], ...], dict[str, Any]] = {}
    for token in tokens:
        points = _token_polygon(token)
        key = tuple((round(x, 6), round(y, 6)) for x, y in points)
        text = validate_transcription(token.get("text", ""))
        left = min(point[0] for point in points)
        right = max(point[0] for point in points)
        top = min(point[1] for point in points)
        bottom = max(point[1] for point in points)
        region = regions.setdefault(
            key,
            {
                "texts": [],
                "token_ids": [],
                "source_ids": [],
                "polygon": [[float(x), float(y)] for x, y in points],
                "left": left,
                "right": right,
                "top": top,
                "bottom": bottom,
                "height": bottom - top,
            },
        )
        region["texts"].append(text)
        region["token_ids"].append(str(token.get("id", "")))
        source_id = str(token.get("source_id", ""))
        if source_id and source_id not in region["source_ids"]:
            region["source_ids"].append(source_id)
    return [
        {
            **region,
            "text": " ".join(region.pop("texts")),
        }
        for region in regions.values()
    ]


def _token_polygon(token: Mapping[str, Any]) -> list[tuple[float, float]]:
    raw_polygon = token.get("polygon")
    if raw_polygon:
        try:
            return [(float(point[0]), float(point[1])) for point in raw_polygon]
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError("token polygon is malformed") from exc
    bbox = token.get("bbox")
    if not isinstance(bbox, Sequence) or len(bbox) != 4:
        raise ValueError("token must include a polygon or [x1,y1,x2,y2] bbox")
    try:
        x1, y1, x2, y2 = (float(value) for value in bbox)
    except (TypeError, ValueError) as exc:
        raise ValueError("token bbox is malformed") from exc
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def _validate_polygon(
    points: Sequence[tuple[float, float]],
    *,
    width: int,
    height: int,
    token_id: str,
) -> None:
    if len(points) < 4:
        raise ValueError(f"token {token_id} polygon requires at least four points")
    if any(not math.isfinite(value) for point in points for value in point):
        raise ValueError(f"token {token_id} polygon contains non-finite coordinates")
    if any(
        x < 0 or y < 0 or x > width or y > height
        for x, y in points
    ):
        raise ValueError(f"token {token_id} polygon is outside image bounds")
    for first in range(len(points)):
        first_next = (first + 1) % len(points)
        for second in range(first + 1, len(points)):
            second_next = (second + 1) % len(points)
            if (
                first == second
                or first_next == second
                or second_next == first
            ):
                continue
            if _segments_intersect(
                points[first],
                points[first_next],
                points[second],
                points[second_next],
            ):
                raise ValueError(
                    f"token {token_id} polygon has a self-intersection"
                )
    area = abs(
        sum(
            points[index][0] * points[(index + 1) % len(points)][1]
            - points[(index + 1) % len(points)][0] * points[index][1]
            for index in range(len(points))
        )
        / 2.0
    )
    if area <= 1e-6:
        raise ValueError(f"token {token_id} polygon has non-positive area")


def _segments_intersect(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    def orientation(
        first: tuple[float, float],
        second: tuple[float, float],
        third: tuple[float, float],
    ) -> float:
        return (second[0] - first[0]) * (third[1] - first[1]) - (
            second[1] - first[1]
        ) * (third[0] - first[0])

    first = orientation(a, b, c)
    second = orientation(a, b, d)
    third = orientation(c, d, a)
    fourth = orientation(c, d, b)
    return (
        (first > 0 > second or first < 0 < second)
        and (third > 0 > fourth or third < 0 < fourth)
    )


def _ordered_quad(points: np.ndarray) -> np.ndarray:
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    return np.asarray(
        [
            points[np.argmin(sums)],
            points[np.argmin(differences)],
            points[np.argmax(sums)],
            points[np.argmax(differences)],
        ],
        dtype=np.float64,
    )


def _positive_int(value: Any, label: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if number <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return number


def _normalized_role(value: Any) -> str:
    role = str(value).casefold()
    if role == "train":
        return "train"
    if role == "dev_select":
        return "validation"
    raise ValueError(f"unsupported OCR training split: {value!r}")


def _truth(value: Any) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def _strip_unmatched_quote(value: str) -> str:
    if value.startswith('"') and not value.endswith('"'):
        return value[1:]
    if value.endswith('"') and not value.startswith('"'):
        return value[:-1]
    return value
