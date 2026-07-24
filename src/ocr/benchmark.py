"""Deterministic public DEV_SELECT OCR benchmark construction."""
from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageStat

from src.rotation_common import deterministic_rank, sha256_file


BENCHMARK_COLUMNS = (
    "benchmark_id",
    "build_id",
    "page_id",
    "document_id",
    "document_family_id",
    "dataset",
    "dataset_component",
    "split",
    "source_image_path",
    "source_image_sha256",
    "normalized_annotation_path",
    "annotation_sha256",
    "annotation_version",
    "license_id",
    "width",
    "height",
    "median_text_height",
    "text_region_count",
    "document_type",
    "language",
    "split_group_id",
    "duplicate_group_id",
    "has_table",
    "has_amount",
    "has_date",
    "has_identifier",
    "ground_truth_text_available",
    "ground_truth_polygons_available",
    "text_density",
    "text_density_bucket",
    "character_size_bucket",
    "resolution_bucket",
    "quality_bucket",
    "image_contrast_stddev",
    "image_edge_energy",
    "diagnostic_rotation_degrees",
    "selected_reason",
    "is_private",
)

_REQUIRED_COLUMNS = {
    "benchmark_id",
    "page_id",
    "dataset",
    "split",
    "source_image_path",
    "source_image_sha256",
    "width",
    "height",
    "median_text_height",
    "text_region_count",
    "document_type",
    "has_table",
    "has_amount",
    "has_date",
    "has_identifier",
    "ground_truth_text_available",
    "ground_truth_polygons_available",
    "selected_reason",
    "is_private",
}
_ALLOWED_DATASETS = {"fatura", "funsd", "sroie"}
_AMOUNT_FIELDS = {"subtotal", "tax", "total_amount", "currency"}
_IDENTIFIER_FIELDS = {"invoice_number", "receipt_number", "reference_number"}


def select_benchmark_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    fatura_count: int = 283,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Select every FUNSD/SROIE row plus stratified, deduplicated FATURA rows."""
    if fatura_count < 0:
        raise ValueError("fatura_count must be non-negative")
    copied = [dict(row) for row in rows]
    page_ids = [str(row.get("page_id", "")) for row in copied]
    if not all(page_ids) or len(page_ids) != len(set(page_ids)):
        raise ValueError("input page_id values must be nonempty and unique")
    for row in copied:
        dataset = str(row.get("dataset", "")).casefold()
        if dataset not in _ALLOWED_DATASETS:
            raise ValueError(f"unsupported benchmark dataset: {dataset}")
        if str(row.get("project_split", row.get("split", ""))).casefold() != "dev_select":
            raise ValueError(f"benchmark row is not DEV_SELECT: {row['page_id']}")
        if _truth(row.get("is_private")):
            raise ValueError(f"private benchmark row refused: {row['page_id']}")
        if not _truth(row.get("is_usable", "true")):
            raise ValueError(f"unusable benchmark row refused: {row['page_id']}")

    selected: list[dict[str, Any]] = []
    for dataset in ("funsd", "sroie"):
        required = sorted(
            (row for row in copied if str(row["dataset"]).casefold() == dataset),
            key=lambda row: str(row["page_id"]),
        )
        if not required:
            raise ValueError(f"no {dataset} DEV_SELECT rows are available")
        for row in required:
            row["selected_reason"] = f"all_{dataset}_dev_select"
        selected.extend(required)

    fatura_rows = [
        row for row in copied if str(row["dataset"]).casefold() == "fatura"
    ]
    if len(fatura_rows) < fatura_count:
        raise ValueError(
            f"only {len(fatura_rows)} FATURA DEV_SELECT rows; {fatura_count} required"
        )
    fatura_rows = _prefer_unique_hashes(fatura_rows, seed=seed)
    unique_rows: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    for row in fatura_rows:
        is_duplicate = bool(row.pop("_benchmark_exact_duplicate", False))
        (duplicate_rows if is_duplicate else unique_rows).append(row)
    unique_count = min(fatura_count, len(unique_rows))
    chosen_fatura = _round_robin_stratified(
        unique_rows,
        count=unique_count,
        seed=seed,
    )
    if len(chosen_fatura) < fatura_count:
        chosen_fatura.extend(
            _round_robin_stratified(
                duplicate_rows,
                count=fatura_count - len(chosen_fatura),
                seed=seed,
            )
        )
    for row in chosen_fatura:
        row["selected_reason"] = "deterministic_stratified_fatura_dev_select"
    selected.extend(chosen_fatura)
    dataset_order = {"funsd": 0, "sroie": 1, "fatura": 2}
    return sorted(
        selected,
        key=lambda row: (
            dataset_order[str(row["dataset"]).casefold()],
            str(row["page_id"]),
        ),
    )


def analyze_benchmark_page(
    row: Mapping[str, Any],
    *,
    project_root: str | Path,
    license_id: str,
) -> dict[str, Any]:
    """Validate a normalized page and derive reproducible benchmark strata."""
    root = Path(project_root).resolve()
    if str(row.get("project_split", "")).casefold() != "dev_select":
        raise ValueError("benchmark analysis accepts DEV_SELECT only")
    if _truth(row.get("is_private")):
        raise ValueError("private benchmark rows are refused")
    if not str(license_id).strip():
        raise ValueError("license_id must be nonempty")
    image_lexical, image_path = resolve_project_input(
        root,
        row.get("image_path"),
        label="source image",
        required_prefix="data/raw/public",
    )
    annotation_lexical, annotation_path = resolve_project_input(
        root,
        row.get("normalized_annotation_path"),
        label="normalized annotation",
        required_prefix="data/processed/normalized_ie_annotations",
        allow_prefix_junction=True,
    )
    expected_image_hash = str(row.get("sha256", "")).casefold()
    actual_image_hash = sha256_file(image_path)
    if expected_image_hash != actual_image_hash:
        raise ValueError(
            f"source image hash mismatch for {row.get('page_id', '<unknown>')}"
        )
    try:
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid normalized annotation: {annotation_path}") from exc
    if _truth(annotation.get("is_private")):
        raise ValueError("private normalized annotation is refused")
    for identity in ("document_id", "page_id", "dataset"):
        if str(annotation.get(identity, "")).casefold() != str(
            row.get(identity, "")
        ).casefold():
            raise ValueError(f"{identity} mismatch between manifest and annotation")

    with Image.open(image_path) as opened:
        opened.load()
        image = opened.convert("L")
        width, height = opened.size
    page = annotation.get("page", {})
    if int(page.get("width", 0)) != width or int(page.get("height", 0)) != height:
        raise ValueError(
            f"annotation page dimensions do not match image for {row['page_id']}"
        )
    tokens = list(annotation.get("tokens", []))
    if not tokens:
        raise ValueError(f"benchmark page has no text regions: {row['page_id']}")

    heights: list[float] = []
    total_area = 0.0
    all_text = True
    all_polygons = True
    has_table = False
    text_fragments: list[str] = []
    for token in tokens:
        text = str(token.get("text", ""))
        if not text.strip():
            all_text = False
        else:
            text_fragments.append(text)
        polygon = token.get("polygon")
        if not polygon:
            all_polygons = False
            raise ValueError(f"token {token.get('id', '<unknown>')} has no polygon")
        area, token_height = _validate_polygon(
            polygon,
            width=width,
            height=height,
            token_id=str(token.get("id", "<unknown>")),
        )
        heights.append(token_height)
        total_area += area
        label = str(token.get("entity_label", "")).casefold()
        has_table = has_table or label == "table_cell"

    canonical_fields = {
        str(name).casefold(): value
        for name, value in dict(annotation.get("canonical_fields", {})).items()
    }
    joined_text = " ".join(text_fragments)
    has_amount = bool(set(canonical_fields) & _AMOUNT_FIELDS) or _looks_like_amount(
        joined_text
    )
    has_date = "date" in canonical_fields or _looks_like_date(joined_text)
    has_identifier = bool(
        set(canonical_fields) & _IDENTIFIER_FIELDS
    ) or _looks_like_identifier(joined_text)
    median_height = float(statistics.median(heights))
    density = min(1.0, total_area / float(width * height))
    contrast, edge_energy = _image_quality(image)
    page_id = str(row["page_id"])
    diagnostic_angle = (
        37
        if int(hashlib.sha256(page_id.encode("utf-8")).hexdigest()[:8], 16) % 4
        == 0
        else 0
    )
    family = (
        str(row.get("split_group_id", "")).strip()
        or str(row.get("duplicate_group_id", "")).strip()
        or str(row["document_id"])
    )
    benchmark_digest = hashlib.sha256(
        f"{page_id}|{actual_image_hash}|{diagnostic_angle}".encode("utf-8")
    ).hexdigest()
    return {
        "benchmark_id": f"ocrbench_{benchmark_digest[:16]}",
        "build_id": "",
        "page_id": page_id,
        "document_id": str(row["document_id"]),
        "document_family_id": family,
        "dataset": str(row["dataset"]).casefold(),
        "dataset_component": str(row.get("dataset_component", "")),
        "split": "dev_select",
        "source_image_path": image_lexical.relative_to(root).as_posix(),
        "source_image_sha256": actual_image_hash,
        "normalized_annotation_path": annotation_lexical.relative_to(root).as_posix(),
        "annotation_sha256": sha256_file(annotation_path),
        "annotation_version": str(annotation.get("schema_version", "")),
        "license_id": str(license_id),
        "width": width,
        "height": height,
        "median_text_height": median_height,
        "text_region_count": len(tokens),
        "document_type": str(row.get("document_type", "unknown")).casefold(),
        "language": str(row.get("language", "unknown")).casefold(),
        "split_group_id": str(row.get("split_group_id", "")),
        "duplicate_group_id": str(row.get("duplicate_group_id", "")),
        "has_table": _flag(has_table),
        "has_amount": _flag(has_amount),
        "has_date": _flag(has_date),
        "has_identifier": _flag(has_identifier),
        "ground_truth_text_available": _flag(all_text),
        "ground_truth_polygons_available": _flag(all_polygons),
        "text_density": density,
        "text_density_bucket": _bucket(
            density,
            low_boundary=0.03,
            high_boundary=0.12,
        ),
        "character_size_bucket": _bucket(
            median_height,
            low_boundary=12.0,
            high_boundary=24.0,
        ),
        "resolution_bucket": _bucket(
            float(width * height),
            low_boundary=500_000.0,
            high_boundary=2_000_000.0,
        ),
        "quality_bucket": (
            "degraded" if contrast < 35.0 or edge_energy < 7.0 else "clean"
        ),
        "image_contrast_stddev": contrast,
        "image_edge_energy": edge_energy,
        "diagnostic_rotation_degrees": diagnostic_angle,
        "selected_reason": str(row.get("selected_reason", "")),
        "is_private": "false",
    }


def benchmark_build_id(records: Sequence[Mapping[str, Any]], *, seed: int) -> str:
    """Return a stable content ID that excludes its own build_id field."""
    material = [
        {
            key: str(record.get(key, ""))
            for key in BENCHMARK_COLUMNS
            if key != "build_id"
        }
        for record in sorted(records, key=lambda value: str(value.get("page_id", "")))
    ]
    payload = {"schema_version": "1.0", "seed": int(seed), "records": material}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"ocr-benchmark-{digest[:16]}"


def validate_benchmark_records(
    records: Sequence[Mapping[str, Any]],
    *,
    minimum_total: int = 400,
    minimum_fatura: int = 283,
    expected_all_counts: Mapping[str, int] | None = None,
) -> dict[str, int]:
    """Validate minimum coverage, uniqueness, split role, and public-only state."""
    if len(records) < minimum_total:
        raise ValueError(
            f"benchmark has {len(records)} records; at least {minimum_total} required"
        )
    page_ids = [str(record.get("page_id", "")) for record in records]
    if len(page_ids) != len(set(page_ids)):
        raise ValueError("duplicate page_id in OCR benchmark")
    benchmark_ids = [str(record.get("benchmark_id", "")) for record in records]
    if len(benchmark_ids) != len(set(benchmark_ids)):
        raise ValueError("duplicate benchmark_id in OCR benchmark")
    counts: Counter[str] = Counter()
    for record in records:
        missing = sorted(_REQUIRED_COLUMNS - set(record))
        if missing:
            raise ValueError(f"benchmark record missing columns: {missing}")
        if str(record.get("split", "")).casefold() != "dev_select":
            raise ValueError(f"non-DEV_SELECT benchmark row: {record.get('page_id')}")
        if _truth(record.get("is_private")):
            raise ValueError(f"private benchmark row: {record.get('page_id')}")
        if not _truth(record.get("ground_truth_text_available")):
            raise ValueError(f"benchmark row has no ground-truth text: {record.get('page_id')}")
        if not _truth(record.get("ground_truth_polygons_available")):
            raise ValueError(
                f"benchmark row has no ground-truth polygons: {record.get('page_id')}"
            )
        if int(float(record.get("width", 0))) <= 0 or int(
            float(record.get("height", 0))
        ) <= 0:
            raise ValueError(f"invalid page dimensions: {record.get('page_id')}")
        if not _is_sha256(str(record.get("source_image_sha256", ""))):
            raise ValueError(f"invalid source hash: {record.get('page_id')}")
        counts[str(record["dataset"]).casefold()] += 1
    if counts["fatura"] < minimum_fatura:
        raise ValueError(
            f"benchmark has {counts['fatura']} FATURA rows; {minimum_fatura} required"
        )
    if expected_all_counts:
        for dataset in ("funsd", "sroie"):
            expected = int(expected_all_counts.get(dataset, 0))
            if counts[dataset] != expected:
                raise ValueError(
                    f"benchmark must contain all {dataset} rows: "
                    f"{counts[dataset]} selected, {expected} available"
                )
    return dict(sorted(counts.items()))


def _prefer_unique_hashes(
    rows: list[dict[str, Any]], *, seed: int
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        source_hash = str(
            row.get("sha256")
            or row.get("source_image_sha256")
            or f"missing:{row.get('page_id', '')}"
        ).casefold()
        grouped[source_hash].append(row)
    unique_first: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for source_hash in sorted(grouped):
        candidates = sorted(
            grouped[source_hash],
            key=lambda row: deterministic_rank(str(row["page_id"]), seed),
        )
        candidates[0]["_benchmark_exact_duplicate"] = False
        unique_first.append(candidates[0])
        for duplicate in candidates[1:]:
            duplicate["_benchmark_exact_duplicate"] = True
            duplicates.append(duplicate)
    return unique_first + sorted(
        duplicates,
        key=lambda row: deterministic_rank(str(row["page_id"]), seed),
    )


def _round_robin_stratified(
    rows: Sequence[dict[str, Any]], *, count: int, seed: int
) -> list[dict[str, Any]]:
    if count == 0:
        return []
    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("split_group_id", "")),
            str(row.get("document_type", "")),
            str(row.get("quality_bucket", "")),
            str(row.get("text_density_bucket", "")),
            str(row.get("character_size_bucket", "")),
            str(row.get("resolution_bucket", "")),
            str(row.get("has_table", "")),
            str(row.get("has_amount", "")),
            str(row.get("has_date", "")),
            str(row.get("has_identifier", "")),
        )
        buckets[key].append(row)
    for key in buckets:
        buckets[key].sort(
            key=lambda row: deterministic_rank(str(row["page_id"]), seed)
        )
    ordered_keys = sorted(
        buckets,
        key=lambda key: deterministic_rank("|".join(key), seed),
    )
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        progress = False
        for key in ordered_keys:
            if buckets[key]:
                selected.append(buckets[key].pop(0))
                progress = True
                if len(selected) == count:
                    break
        if not progress:
            raise ValueError("stratified benchmark pool exhausted")
    return selected


def resolve_project_input(
    root: str | Path,
    value: Any,
    *,
    label: str,
    required_prefix: str | Path,
    allow_prefix_junction: bool = False,
) -> tuple[Path, Path]:
    """Resolve a project input while preserving and validating its lexical path.

    ``allow_prefix_junction`` permits the required prefix itself to resolve
    outside the project, but still refuses nested links that escape that
    prefix's resolved target.
    """
    project_root = Path(root).resolve()
    path = Path(str(value or ""))
    candidate = path if path.is_absolute() else project_root / path
    lexical = Path(os.path.abspath(candidate))
    lexical_prefix = Path(os.path.abspath(project_root / required_prefix))
    try:
        lexical.relative_to(lexical_prefix)
    except ValueError as exc:
        raise ValueError(
            f"{label} escapes required project path {required_prefix}: {candidate}"
        ) from exc
    resolved_prefix = lexical_prefix.resolve()
    resolved = lexical.resolve()
    required_resolved_root = resolved_prefix if allow_prefix_junction else lexical_prefix
    try:
        resolved.relative_to(required_resolved_root)
    except ValueError as exc:
        raise ValueError(
            f"{label} escapes resolved project path {required_prefix}: {candidate}"
        ) from exc
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return lexical, resolved


def _validate_polygon(
    polygon: Any,
    *,
    width: int,
    height: int,
    token_id: str,
) -> tuple[float, float]:
    points = np.asarray(polygon, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 4:
        raise ValueError(f"token {token_id} polygon must have at least four points")
    if not np.isfinite(points).all():
        raise ValueError(f"token {token_id} polygon contains non-finite coordinates")
    if (
        np.any(points[:, 0] < 0)
        or np.any(points[:, 0] > width)
        or np.any(points[:, 1] < 0)
        or np.any(points[:, 1] > height)
    ):
        raise ValueError(f"token {token_id} polygon is outside page bounds")
    area = abs(
        float(
            np.dot(points[:, 0], np.roll(points[:, 1], -1))
            - np.dot(points[:, 1], np.roll(points[:, 0], -1))
        )
    ) / 2.0
    if area <= 1e-8:
        raise ValueError(f"token {token_id} polygon has non-positive area")
    if _self_intersects(points):
        raise ValueError(f"token {token_id} polygon is self-intersecting")
    return area, float(points[:, 1].max() - points[:, 1].min())


def _self_intersects(points: np.ndarray) -> bool:
    count = len(points)
    for first in range(count):
        a1 = points[first]
        a2 = points[(first + 1) % count]
        for second in range(first + 1, count):
            if second in {first, (first + 1) % count}:
                continue
            if first == 0 and second == count - 1:
                continue
            b1 = points[second]
            b2 = points[(second + 1) % count]
            if _segments_intersect(a1, a2, b1, b2):
                return True
    return False


def _segments_intersect(
    a1: np.ndarray,
    a2: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
) -> bool:
    def cross(origin: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
        return float(np.cross(left - origin, right - origin))

    first = cross(a1, a2, b1)
    second = cross(a1, a2, b2)
    third = cross(b1, b2, a1)
    fourth = cross(b1, b2, a2)
    return first * second < 0 and third * fourth < 0


def _image_quality(image: Image.Image) -> tuple[float, float]:
    sample = image.copy()
    sample.thumbnail((256, 256))
    contrast = float(ImageStat.Stat(sample).stddev[0])
    values = np.asarray(sample, dtype=np.float32)
    differences = []
    if values.shape[1] > 1:
        differences.append(np.abs(np.diff(values, axis=1)).mean())
    if values.shape[0] > 1:
        differences.append(np.abs(np.diff(values, axis=0)).mean())
    edge_energy = float(statistics.mean(differences)) if differences else 0.0
    return contrast, edge_energy


def _looks_like_amount(text: str) -> bool:
    import re

    return bool(
        re.search(
            r"(?i)(?:total|subtotal|tax|vat|amount|thb|usd|eur|gbp|[$€£฿¥])"
            r".{0,24}[-+]?\d[\d.,]*",
            text,
        )
    )


def _looks_like_date(text: str) -> bool:
    import re

    return bool(
        re.search(
            r"\b(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"
            r"|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})\b",
            text,
        )
    )


def _looks_like_identifier(text: str) -> bool:
    import re

    return bool(
        re.search(
            r"(?i)\b(?:invoice|receipt|reference|ref|inv|tax\s*id)"
            r"(?:\s*(?:no\.?|number|#|:))?\s*[A-Z0-9][A-Z0-9/-]{3,}\b",
            text,
        )
    )


def _bucket(value: float, *, low_boundary: float, high_boundary: float) -> str:
    if value < low_boundary:
        return "low"
    if value < high_boundary:
        return "medium"
    return "high"


def _truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).casefold() in {"1", "true", "yes"}


def _flag(value: bool) -> str:
    return "true" if value else "false"


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)
