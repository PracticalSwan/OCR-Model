#!/usr/bin/env python3
"""Evaluate OCR orientation and fine deskew on public DEV_SELECT angle evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image  # noqa: E402

from src.evaluation.critical_field_ocr import evaluate_critical_fields  # noqa: E402
from src.evaluation.metrics import normalized_text, ocr_text_metrics  # noqa: E402
from src.information_extraction.geometry import rotate_image_and_annotation  # noqa: E402
from src.ocr.environment import configure_external_environment, require_storage_gate  # noqa: E402
from src.ocr.model_registry import ModelRegistry  # noqa: E402
from src.ocr.pipeline import MultilingualOCR  # noqa: E402
from src.ocr.training_data import convert_detection_annotation, critical_fields_by_token  # noqa: E402
from src.ocr.trials import aggregate_detector_results, detector_page_result  # noqa: E402
from src.rotation_common import atomic_write_json, canonical_json, sha256_file  # noqa: E402
from scripts.evaluate_ocr_component_ablations import _balanced_rows  # noqa: E402
from scripts.run_large_ocr_ablation import (  # noqa: E402
    _best_field_candidates,
    _load_page,
    _load_rows,
)


ANGLES = (0, 1, 15, 30, 37, 45, 60, 89, 90, 91, 135, 179, 180, 225, 269, 270, 315, 359)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-manifest",
        default=str(PROJECT_ROOT / "data/metadata/ocr_benchmark_manifest.csv"),
    )
    parser.add_argument(
        "--model-setup",
        default=str(PROJECT_ROOT / "reports/ocr/model_setup.json"),
    )
    parser.add_argument(
        "--model-registry",
        default=str(PROJECT_ROOT / "reports/ocr_upgrade/model_registry.json"),
    )
    parser.add_argument("--device", choices=("cpu", "gpu:0"), default="gpu:0")
    parser.add_argument("--pages", type=int, default=9)
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "reports/ocr_upgrade/orientation_metrics.json"),
    )
    args = parser.parse_args()
    if not 3 <= args.pages <= 60:
        parser.error("--pages must be in [3, 60]")

    configure_external_environment()
    asset_root = Path("D:/CSX4201/vision-info-extraction-assets")
    require_storage_gate(
        asset_root,
        operation="public DEV_SELECT OCR orientation evaluation",
        anticipated_c_gib=0.1,
        anticipated_asset_gib=3.0,
    )
    manifest_path = Path(args.benchmark_manifest).resolve()
    rows = _balanced_rows(_load_rows(manifest_path), args.pages)
    registry = ModelRegistry.from_setup(
        Path(args.model_setup).resolve(),
        upgrade_registry=Path(args.model_registry).resolve(),
    )
    detector, general = registry.route_models("general")
    _, thai = registry.route_models("thai")
    pipeline = MultilingualOCR(
        registry,
        device=args.device,
        cardinal_angles=(0, 90, 180, 270),
        preprocessing_version="3.0-orientation-dev-select",
        preprocessing_profile="quality_auto",
        enable_fine_deskew=True,
        maximum_fine_candidates=2,
        enable_tiling=False,
    )
    observations = []
    started = time.perf_counter()
    for angle in ANGLES:
        for row in rows:
            image_path, annotation = _load_page(row)
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            rotated_image, rotated_annotation, _ = rotate_image_and_annotation(
                image,
                annotation,
                float(angle),
            )
            page_started = time.perf_counter()
            result = pipeline.extract_page(
                rotated_image,
                language_mode="general",
                metadata_language=row.get("language"),
            )
            observations.append(
                _evaluate(
                    row,
                    rotated_annotation,
                    result,
                    image_size=rotated_image.size,
                    input_angle=float(angle),
                    duration_seconds=time.perf_counter() - page_started,
                )
            )
        print(f"orientation angle {angle}: {len(rows)} pages", flush=True)

    by_angle = {
        str(angle): _aggregate(
            [row for row in observations if row["input_angle"] == float(angle)]
        )
        for angle in ANGLES
    }
    aggregate = _aggregate(observations)
    upright = by_angle["0"]
    for value in by_angle.values():
        value["retention_vs_upright"] = {
            field: (
                float(value[field]) / float(upright[field])
                if float(upright.get(field, 0.0) or 0.0) > 0
                else None
            )
            for field in (
                "polygon_f1",
                "recognized_text_coverage",
                "critical_field_exact_match",
            )
        }
    report = {
        "schema_version": "1.0",
        "status": "passed",
        "build_id": "ocr-orientation-"
        + hashlib.sha256(
            canonical_json(
                {
                    "manifest": sha256_file(manifest_path),
                    "pages": [row["page_id"] for row in rows],
                    "angles": ANGLES,
                }
            ).encode("utf-8")
        ).hexdigest()[:16],
        "split": "dev_select",
        "manifest_sha256": sha256_file(manifest_path),
        "detector_sha256": detector.artifact_hash,
        "recognizer_sha256": hashlib.sha256(
            (general.artifact_hash + thai.artifact_hash).encode("ascii")
        ).hexdigest(),
        "checkpoint_sha256": None,
        "calibration_sha256": None,
        "configuration_sha256": hashlib.sha256(
            canonical_json(
                {
                    "cardinal_angles": [0, 90, 180, 270],
                    "fine_deskew": True,
                    "maximum_fine_candidates": 2,
                    "preprocessing": "quality_auto",
                }
            ).encode("utf-8")
        ).hexdigest(),
        "source_commit": _git_commit(),
        "device": args.device,
        "sample_count": len(observations),
        "failure_count": 0,
        "duration_seconds": time.perf_counter() - started,
        "private_row_count": 0,
        "source_page_count": len(rows),
        "datasets": sorted({row["dataset"] for row in rows}),
        "angles": list(ANGLES),
        "metrics": aggregate,
        "by_angle": by_angle,
        "orientation_tolerance_degrees": 3.0,
        "kmeans_instantiated": False,
        "kmeans_controls_ocr": False,
        "test_private_tuning_rows": 0,
    }
    atomic_write_json(Path(args.output), report)
    print(json.dumps(report, indent=2))
    return 0


def _evaluate(
    row: Mapping[str, str],
    annotation: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    image_size: tuple[int, int],
    input_angle: float,
    duration_seconds: float,
) -> dict[str, Any]:
    width, height = image_size
    references = [
        {"polygon": region["points"], "token_ids": region.get("token_ids", [])}
        for region in convert_detection_annotation(annotation)
    ]
    critical_ids = {
        token_id
        for token_id, fields in critical_fields_by_token(annotation).items()
        if fields
    }
    detection = detector_page_result(
        references,
        [{"polygon": word.get("polygon")} for word in result.get("words") or []],
        page_width=width,
        page_height=height,
        critical_token_ids=critical_ids,
        duration_seconds=duration_seconds,
    )
    reference_text = " ".join(
        str(token.get("text", ""))
        for token in annotation.get("tokens") or []
        if str(token.get("text", "")).strip()
    )
    text = ocr_text_metrics(reference_text, str(result.get("full_text", "")))
    reference_fields = {
        field: str(value.get("raw_text") or value.get("value") or "")
        for field, value in (annotation.get("canonical_fields") or {}).items()
        if isinstance(value, Mapping)
        and str(value.get("raw_text") or value.get("value") or "").strip()
    }
    predicted_fields = _best_field_candidates(
        reference_fields,
        [str(line.get("text", "")) for line in result.get("lines") or []],
    )
    expected_correction = (-input_angle) % 360.0
    selected_orientation = float(result.get("orientation", 0.0)) % 360.0
    error = circular_error(selected_orientation, expected_correction)
    return {
        "dataset": row["dataset"],
        "input_angle": input_angle,
        "selected_orientation": selected_orientation,
        "orientation_error": error,
        "orientation_within_tolerance": error <= 3.0,
        "detection": detection,
        "text": text,
        "reference_words": len(normalized_text(reference_text).split()),
        "reference_fields": reference_fields,
        "predicted_fields": predicted_fields,
        "duration_seconds": duration_seconds,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    detector = aggregate_detector_results([row["detection"] for row in rows])
    reference_characters = sum(
        int(row["text"]["reference_characters"]) for row in rows
    )
    character_errors = sum(int(row["text"]["character_errors"]) for row in rows)
    reference_words = sum(int(row["reference_words"]) for row in rows)
    word_errors = sum(int(row["text"]["word_errors"]) for row in rows)
    critical = evaluate_critical_fields(
        [row["reference_fields"] for row in rows],
        [row["predicted_fields"] for row in rows],
    )
    return {
        "page_count": len(rows),
        "polygon_f1": detector["f1"],
        "recognized_text_coverage": max(
            0.0, 1.0 - character_errors / max(1, reference_characters)
        ),
        "cer": character_errors / max(1, reference_characters),
        "wer": word_errors / max(1, reference_words),
        "critical_field_exact_match": critical["aggregate"][
            "critical_exact_match"
        ],
        "orientation_selection_accuracy": sum(
            bool(row["orientation_within_tolerance"]) for row in rows
        )
        / max(1, len(rows)),
        "mean_orientation_error_degrees": statistics.fmean(
            float(row["orientation_error"]) for row in rows
        )
        if rows
        else None,
        "time_per_page_seconds": statistics.fmean(
            float(row["duration_seconds"]) for row in rows
        )
        if rows
        else None,
    }


def circular_error(left: float, right: float) -> float:
    delta = abs((float(left) - float(right)) % 360.0)
    return min(delta, 360.0 - delta)


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()


if __name__ == "__main__":
    raise SystemExit(main())
