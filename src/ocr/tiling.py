"""Overlapping OCR tiles with source-coordinate restoration and deduplication."""
from __future__ import annotations

import copy
import hashlib
import math
import statistics
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.information_extraction.geometry import polygon_to_bbox
from src.rotation_common import canonical_json, stable_id


@dataclass(frozen=True)
class Tile:
    index: int
    row: int
    column: int
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def as_dict(self) -> dict[str, int]:
        return {
            "index": self.index,
            "row": self.row,
            "column": self.column,
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
            "width": self.width,
            "height": self.height,
        }


def generate_tiles(
    width: int,
    height: int,
    *,
    grid: tuple[int, int] = (2, 2),
    overlap: float = 0.15,
) -> list[Tile]:
    """Return deterministic overlapping tiles that exactly cover the page."""
    columns, rows = map(int, grid)
    if width < 2 or height < 2:
        raise ValueError("page dimensions must be at least 2x2")
    if columns < 1 or rows < 1:
        raise ValueError("tile grid dimensions must be positive")
    if not 0.0 <= float(overlap) < 0.5:
        raise ValueError("tile overlap must be in [0, 0.5)")
    base_width = math.ceil(width / columns)
    base_height = math.ceil(height / rows)
    overlap_x = round(base_width * float(overlap))
    overlap_y = round(base_height * float(overlap))
    tiles: list[Tile] = []
    for row in range(rows):
        for column in range(columns):
            base_x0 = column * base_width
            base_y0 = row * base_height
            base_x1 = min(width, (column + 1) * base_width)
            base_y1 = min(height, (row + 1) * base_height)
            x0 = max(0, base_x0 - (overlap_x if column else 0))
            y0 = max(0, base_y0 - (overlap_y if row else 0))
            x1 = min(width, base_x1 + (overlap_x if column < columns - 1 else 0))
            y1 = min(height, base_y1 + (overlap_y if row < rows - 1 else 0))
            tiles.append(Tile(len(tiles), row, column, x0, y0, x1, y1))
    return tiles


def tiling_trigger_reasons(
    signals: Mapping[str, Any],
    *,
    image_width: int,
    image_height: int,
    minimum_dimension: int = 700,
) -> list[str]:
    """Activate only on weak pages where a tile can materially enlarge text."""
    if max(image_width, image_height) < int(minimum_dimension):
        return []
    reasons: list[str] = []
    if int(signals.get("box_count", 0)) < 3:
        reasons.append("weak_full_page_box_count")
    if float(signals.get("detected_text_coverage", 0.0)) < 0.0015:
        reasons.append("weak_full_page_coverage")
    if float(signals.get("median_text_height_ratio", 0.0)) < 0.007:
        reasons.append("small_text")
    if float(signals.get("mean_confidence", 0.0)) < 0.50:
        reasons.append("low_full_page_confidence")
    return reasons


def map_tile_result_to_page(
    result: Mapping[str, Any],
    tile: Tile,
    *,
    upscale: float = 1.0,
) -> dict[str, Any]:
    """Map one normalized tile result into the untouched page coordinate system."""
    scale = float(upscale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("tile upscale must be finite and positive")
    mapped = copy.deepcopy(dict(result))
    words: list[dict[str, Any]] = []
    for ordinal, source_word in enumerate(result.get("words") or []):
        word = copy.deepcopy(dict(source_word))
        polygon = [
            [
                tile.x0 + float(point[0]) / scale,
                tile.y0 + float(point[1]) / scale,
            ]
            for point in word.get("polygon") or []
        ]
        if len(polygon) < 4:
            continue
        word["polygon"] = polygon
        word["bbox"] = polygon_to_bbox(polygon)
        word["id"] = stable_id(
            "tile-word",
            tile.index,
            ordinal,
            word.get("text", ""),
            word["bbox"],
        )
        word["tile_index"] = tile.index
        words.append(word)
    mapped["words"] = words
    mapped["tile"] = {**tile.as_dict(), "upscale": scale}
    mapped["lines"] = _lines_from_words(_reading_order(words))
    mapped["full_text"] = "\n".join(line["text"] for line in mapped["lines"])
    return mapped


def merge_tiled_results(
    results: Sequence[Mapping[str, Any]],
    *,
    image_width: int,
    image_height: int,
    polygon_iou_threshold: float = 0.50,
) -> dict[str, Any]:
    """Confidence-aware NMS for already-recognized tile words."""
    if not results:
        raise ValueError("at least one tile result is required")
    if not 0.0 <= polygon_iou_threshold <= 1.0:
        raise ValueError("polygon IoU threshold must be in [0, 1]")
    all_words = [
        copy.deepcopy(dict(word))
        for result in results
        for word in (result.get("words") or [])
        if str(word.get("text", "")).strip()
    ]
    ranked = sorted(
        all_words,
        key=lambda word: (
            _confidence(word),
            _polygon_area(word.get("polygon")),
            str(word.get("text", "")).casefold(),
        ),
        reverse=True,
    )
    kept: list[dict[str, Any]] = []
    duplicates = 0
    for candidate in ranked:
        duplicate = next(
            (
                existing
                for existing in kept
                if _duplicate(candidate, existing, polygon_iou_threshold)
            ),
            None,
        )
        if duplicate is None:
            kept.append(candidate)
        else:
            duplicates += 1
    kept = _reading_order(kept)
    for index, word in enumerate(kept):
        word["id"] = stable_id("merged-word", index, word.get("text", ""), word["bbox"])
    lines = _lines_from_words(kept)
    confidence_values = [_confidence(word) for word in kept]
    template = copy.deepcopy(dict(results[0]))
    template.update(
        {
            "full_text": "\n".join(line["text"] for line in lines),
            "words": kept,
            "lines": lines,
            "mean_confidence": (
                statistics.fmean(confidence_values) if confidence_values else None
            ),
            "orientation": 0.0,
            "source_width": int(image_width),
            "source_height": int(image_height),
            "duration_seconds": sum(
                max(0.0, float(result.get("duration_seconds", 0.0)))
                for result in results
            ),
            "tiling": {
                "enabled": True,
                "tile_count": len(results),
                "input_word_count": len(all_words),
                "output_word_count": len(kept),
                "duplicates_removed": duplicates,
                "duplicate_rate": duplicates / max(1, len(all_words)),
                "polygon_iou_threshold": float(polygon_iou_threshold),
            },
        }
    )
    template["provenance_hash"] = hashlib.sha256(
        canonical_json(
            {
                "detector_model": template.get("detector_model"),
                "recognizer_model": template.get("recognizer_model"),
                "words": kept,
                "tiling": template["tiling"],
            }
        ).encode("utf-8")
    ).hexdigest()
    return template


def _duplicate(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    threshold: float,
) -> bool:
    overlap = polygon_iou(first.get("polygon"), second.get("polygon"))
    if overlap < threshold:
        return False
    first_text = " ".join(str(first.get("text", "")).casefold().split())
    second_text = " ".join(str(second.get("text", "")).casefold().split())
    return first_text == second_text or overlap >= max(0.80, threshold)


def polygon_iou(first: Any, second: Any) -> float:
    """Convex-polygon IoU using Sutherland-Hodgman clipping."""
    first_points = _points(first)
    second_points = _points(second)
    if len(first_points) < 3 or len(second_points) < 3:
        return 0.0
    subject = _ensure_ccw(first_points)
    clip = _ensure_ccw(second_points)
    intersection = subject
    for index, edge_start in enumerate(clip):
        edge_end = clip[(index + 1) % len(clip)]
        output: list[tuple[float, float]] = []
        if not intersection:
            break
        previous = intersection[-1]
        for current in intersection:
            current_inside = _inside(current, edge_start, edge_end)
            previous_inside = _inside(previous, edge_start, edge_end)
            if current_inside:
                if not previous_inside:
                    output.append(_line_intersection(previous, current, edge_start, edge_end))
                output.append(current)
            elif previous_inside:
                output.append(_line_intersection(previous, current, edge_start, edge_end))
            previous = current
        intersection = output
    intersection_area = _area(intersection)
    union = _area(subject) + _area(clip) - intersection_area
    return float(intersection_area / union) if union > 0 else 0.0


def _reading_order(words: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    values = [copy.deepcopy(dict(word)) for word in words]
    if not values:
        return []
    heights = [
        max(1.0, float(word["bbox"][3]) - float(word["bbox"][1]))
        for word in values
    ]
    tolerance = max(2.0, statistics.median(heights) * 0.60)
    values.sort(key=lambda word: (float(word["bbox"][1]), float(word["bbox"][0])))
    rows: list[list[dict[str, Any]]] = []
    row_centers: list[float] = []
    for word in values:
        center = (float(word["bbox"][1]) + float(word["bbox"][3])) / 2.0
        target = next(
            (index for index, value in enumerate(row_centers) if abs(center - value) <= tolerance),
            None,
        )
        if target is None:
            rows.append([word])
            row_centers.append(center)
        else:
            rows[target].append(word)
            row_centers[target] = statistics.fmean(
                (float(item["bbox"][1]) + float(item["bbox"][3])) / 2.0
                for item in rows[target]
            )
    ordered: list[dict[str, Any]] = []
    for _, row in sorted(zip(row_centers, rows), key=lambda item: item[0]):
        ordered.extend(sorted(row, key=lambda word: float(word["bbox"][0])))
    return ordered


def _lines_from_words(words: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": stable_id("tile-line", index, word["id"]),
            "text": str(word["text"]),
            "word_ids": [word["id"]],
            "polygon": copy.deepcopy(word["polygon"]),
            "bbox": list(word["bbox"]),
            "confidence": word.get("confidence"),
        }
        for index, word in enumerate(words)
    ]


def _confidence(word: Mapping[str, Any]) -> float:
    try:
        value = float(word.get("confidence"))
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def _points(value: Any) -> list[tuple[float, float]]:
    try:
        points = [(float(point[0]), float(point[1])) for point in value]
    except (TypeError, ValueError, IndexError):
        return []
    return points if all(math.isfinite(v) for point in points for v in point) else []


def _area(points: Sequence[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    return abs(
        sum(
            x0 * y1 - x1 * y0
            for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1])
        )
    ) / 2.0


def _polygon_area(value: Any) -> float:
    return _area(_points(value))


def _ensure_ccw(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    signed = sum(
        x0 * y1 - x1 * y0
        for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1])
    )
    return points if signed >= 0 else list(reversed(points))


def _inside(
    point: tuple[float, float],
    edge_start: tuple[float, float],
    edge_end: tuple[float, float],
) -> bool:
    return (
        (edge_end[0] - edge_start[0]) * (point[1] - edge_start[1])
        - (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0])
    ) >= -1e-9


def _line_intersection(
    first: tuple[float, float],
    second: tuple[float, float],
    edge_start: tuple[float, float],
    edge_end: tuple[float, float],
) -> tuple[float, float]:
    x1, y1 = first
    x2, y2 = second
    x3, y3 = edge_start
    x4, y4 = edge_end
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-12:
        return second
    first_det = x1 * y2 - y1 * x2
    second_det = x3 * y4 - y3 * x4
    return (
        (first_det * (x3 - x4) - (x1 - x2) * second_det) / denominator,
        (first_det * (y3 - y4) - (y1 - y2) * second_det) / denominator,
    )
