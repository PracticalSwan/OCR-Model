#!/usr/bin/env python3
"""Evaluate preprocessing profiles on the frozen 400-page public DEV_SELECT set."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from difflib import SequenceMatcher
from io import StringIO
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image  # noqa: E402

from src.evaluation.critical_field_ocr import evaluate_critical_fields  # noqa: E402
from src.evaluation.metrics import normalized_text, ocr_text_metrics  # noqa: E402
from src.ocr.benchmark import resolve_project_input  # noqa: E402
from src.ocr.environment import configure_external_environment, require_storage_gate  # noqa: E402
from src.ocr.model_registry import ModelRegistry  # noqa: E402
from src.ocr.pipeline import MultilingualOCR  # noqa: E402
from src.ocr.training_data import convert_detection_annotation, critical_fields_by_token  # noqa: E402
from src.ocr.trials import aggregate_detector_results, detector_page_result  # noqa: E402
from src.rotation_common import (  # noqa: E402
    atomic_write_json,
    atomic_write_text,
    canonical_json,
    sha256_file,
)


PROFILES = (
    "original",
    "grayscale_normalized",
    "adaptive_contrast",
    "background_normalized",
    "sharpen",
    "denoise",
    "quality_auto",
)


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
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=PROFILES,
        default=list(PROFILES),
    )
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument(
        "--detector-model",
        choices=("original", "custom", "auto"),
        default="auto",
    )
    parser.add_argument(
        "--general-recognizer",
        choices=("original", "custom", "auto"),
        default="auto",
    )
    parser.add_argument(
        "--thai-recognizer",
        choices=("original", "custom", "auto"),
        default="auto",
    )
    parser.add_argument(
        "--output-csv",
        default=str(PROJECT_ROOT / "reports/ocr_upgrade/preprocessing_ablation.csv"),
    )
    parser.add_argument(
        "--selection-output",
        default=str(PROJECT_ROOT / "reports/ocr_upgrade/preprocessing_selection.json"),
    )
    args = parser.parse_args()
    if not 1 <= args.limit <= 400:
        parser.error("--limit must be in [1, 400]")
    manifest_path = Path(args.benchmark_manifest)
    rows = _load_rows(manifest_path)[: args.limit]
    configure_external_environment()
    require_storage_gate(
        Path("D:/CSX4201/vision-info-extraction-assets"),
        operation="large OCR preprocessing ablation",
        anticipated_c_gib=0.1,
        anticipated_asset_gib=1.0,
    )
    registry = ModelRegistry.from_setup(
        args.model_setup,
        upgrade_registry=args.model_registry,
        detector_choice=args.detector_model,
        general_choice=args.general_recognizer,
        thai_choice=args.thai_recognizer,
    )
    detector, general = registry.route_models("general")
    _, thai = registry.route_models("thai")
    source_commit = _git_commit()
    manifest_sha = sha256_file(manifest_path)
    summaries: list[dict[str, Any]] = []
    for profile in args.profiles:
        started = time.perf_counter()
        try:
            pipeline = MultilingualOCR(
                registry,
                device=args.device,
                cardinal_angles=(0,),
                preprocessing_version=f"3.0-large-ablation-{profile}",
                preprocessing_profile=profile,
                enable_fine_deskew=False,
                enable_tiling=False,
            )
            observations = []
            for index, row in enumerate(rows, start=1):
                observations.append(_evaluate_page(pipeline, row))
                if index % 25 == 0 or index == len(rows):
                    print(
                        f"{profile}: {index}/{len(rows)} public DEV_SELECT pages",
                        flush=True,
                    )
            summary = _aggregate_profile(
                profile,
                observations,
                duration_seconds=time.perf_counter() - started,
            )
            summary.update(
                _provenance(
                    profile=profile,
                    manifest_sha=manifest_sha,
                    detector_hash=detector.artifact_hash,
                    general_hash=general.artifact_hash,
                    thai_hash=thai.artifact_hash,
                    source_commit=source_commit,
                    device=args.device,
                    sample_count=len(rows),
                    failure_count=summary["page_failure_count"],
                    duration_seconds=summary["duration_seconds"],
                )
            )
            summaries.append(summary)
            del pipeline
            gc.collect()
        except Exception as exc:
            summaries.append(
                {
                    "profile": profile,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                    **_provenance(
                        profile=profile,
                        manifest_sha=manifest_sha,
                        detector_hash=detector.artifact_hash,
                        general_hash=general.artifact_hash,
                        thai_hash=thai.artifact_hash,
                        source_commit=source_commit,
                        device=args.device,
                        sample_count=len(rows),
                        failure_count=len(rows),
                        duration_seconds=time.perf_counter() - started,
                    ),
                }
            )
    successful = [row for row in summaries if row.get("status") == "passed"]
    if not successful:
        raise RuntimeError("every large OCR preprocessing profile failed")
    fastest = min(float(row["time_per_page_seconds"]) for row in successful)
    for row in successful:
        efficiency = min(
            1.0, fastest / max(1e-9, float(row["time_per_page_seconds"]))
        )
        row["normalized_efficiency_score"] = efficiency
        row["selection_score"] = preprocessing_selection_score(
            detector_f1=float(row["detection_f1"]),
            coverage=float(row["recognized_text_coverage"]),
            wer=float(row["wer"]),
            critical_exact=float(row["critical_field_exact_match"]),
            efficiency=efficiency,
        )
    chosen = max(
        successful,
        key=lambda row: (
            float(row["selection_score"]),
            row["profile"] == "original",
            -float(row["time_per_page_seconds"]),
        ),
    )
    _write_csv(Path(args.output_csv), summaries)
    selection = {
        "schema_version": "1.0",
        "status": "passed",
        "build_id": "ocr-preprocessing-selection-"
        + hashlib.sha256(
            canonical_json(summaries).encode("utf-8")
        ).hexdigest()[:16],
        "split": "dev_select",
        "manifest_sha256": manifest_sha,
        "detector_sha256": detector.artifact_hash,
        "recognizer_sha256": hashlib.sha256(
            (general.artifact_hash + thai.artifact_hash).encode("ascii")
        ).hexdigest(),
        "checkpoint_sha256": None,
        "calibration_sha256": None,
        "configuration_hash": chosen["configuration_hash"],
        "source_commit": source_commit,
        "device": args.device,
        "sample_count": len(rows),
        "failure_count": sum(int(row.get("page_failure_count", 0)) for row in successful),
        "duration_seconds": sum(float(row["duration_seconds"]) for row in successful),
        "private_row_count": 0,
        "chosen_profile": chosen["profile"],
        "chosen_selection_score": chosen["selection_score"],
        "profiles": summaries,
        "selection_formula": (
            "0.30*detection_f1 + 0.25*coverage + 0.15*(1-clamped_wer) "
            "+ 0.20*critical_exact + 0.10*efficiency"
        ),
        "selection_rule": (
            "highest composite score; exact ties prefer original, then lower time"
        ),
        "test_private_tuning_rows": 0,
    }
    atomic_write_json(Path(args.selection_output), selection)
    print(json.dumps(selection, indent=2, ensure_ascii=False))
    return 0


def preprocessing_selection_score(
    *,
    detector_f1: float,
    coverage: float,
    wer: float,
    critical_exact: float,
    efficiency: float,
) -> float:
    return (
        0.30 * _clamp(detector_f1)
        + 0.25 * _clamp(coverage)
        + 0.15 * (1.0 - _clamp(wer))
        + 0.20 * _clamp(critical_exact)
        + 0.10 * _clamp(efficiency)
    )


def _evaluate_page(
    pipeline: MultilingualOCR,
    row: Mapping[str, str],
) -> dict[str, Any]:
    image_path, annotation = _load_page(row)
    started = time.perf_counter()
    try:
        with Image.open(image_path) as image:
            result = pipeline.extract_page(
                image.convert("RGB"),
                language_mode="general",
                metadata_language=row.get("language"),
            )
        failed = False
        error = None
    except Exception as exc:
        result = {
            "words": [],
            "lines": [],
            "full_text": "",
            "mean_confidence": None,
            "duration_seconds": time.perf_counter() - started,
        }
        failed = True
        error = f"{type(exc).__name__}: {exc}"
    references = [
        {"polygon": region["points"], "token_ids": region.get("token_ids", [])}
        for region in convert_detection_annotation(annotation)
    ]
    critical_map = critical_fields_by_token(annotation)
    critical_ids = {
        token_id for token_id, fields in critical_map.items() if fields
    }
    detection = detector_page_result(
        references,
        [{"polygon": word["polygon"]} for word in result.get("words") or []],
        page_width=int(row["width"]),
        page_height=int(row["height"]),
        critical_token_ids=critical_ids,
        duration_seconds=time.perf_counter() - started,
    )
    detection["failed"] = failed
    reference_text = " ".join(
        str(token.get("text", ""))
        for token in annotation.get("tokens") or []
        if str(token.get("text", "")).strip()
    )
    text = ocr_text_metrics(reference_text, str(result.get("full_text", "")))
    references_fields = {
        field: str(value.get("raw_text") or value.get("value") or "")
        for field, value in (annotation.get("canonical_fields") or {}).items()
        if isinstance(value, Mapping)
        and str(value.get("raw_text") or value.get("value") or "").strip()
    }
    predictions_fields = _best_field_candidates(
        references_fields,
        [str(line.get("text", "")) for line in result.get("lines") or []],
    )
    return {
        "page_id": row["page_id"],
        "dataset": row["dataset"],
        "failed": failed,
        "error": error,
        "detection": detection,
        "text": text,
        "reference_word_count": len(normalized_text(reference_text).split()),
        "critical_reference": references_fields,
        "critical_prediction": predictions_fields,
        "mean_confidence": result.get("mean_confidence"),
        "duration_seconds": time.perf_counter() - started,
    }


def _aggregate_profile(
    profile: str,
    observations: list[dict[str, Any]],
    *,
    duration_seconds: float,
) -> dict[str, Any]:
    detector = aggregate_detector_results(
        [item["detection"] for item in observations]
    )
    reference_characters = sum(
        int(item["text"]["reference_characters"]) for item in observations
    )
    character_errors = sum(
        int(item["text"]["character_errors"]) for item in observations
    )
    total_word_errors = sum(int(item["text"]["word_errors"]) for item in observations)
    reference_words = sum(
        int(item["reference_word_count"]) for item in observations
    )
    critical = evaluate_critical_fields(
        [item["critical_reference"] for item in observations],
        [item["critical_prediction"] for item in observations],
    )
    failures = sum(bool(item["failed"]) for item in observations)
    return {
        "profile": profile,
        "status": "passed",
        "detection_precision": detector["precision"],
        "detection_recall": detector["recall"],
        "detection_f1": detector["f1"],
        "recognized_text_coverage": max(
            0.0, 1.0 - character_errors / max(1, reference_characters)
        ),
        "cer": character_errors / max(1, reference_characters),
        "wer": total_word_errors / max(1, reference_words),
        "critical_field_exact_match": critical["aggregate"][
            "critical_exact_match"
        ],
        "critical_field_count": critical["evaluated_field_count"],
        "time_per_page_seconds": duration_seconds / max(1, len(observations)),
        "page_failure_count": failures,
        "page_failure_rate": failures / max(1, len(observations)),
        "duration_seconds": duration_seconds,
        "normalized_efficiency_score": None,
        "selection_score": None,
    }


def _best_field_candidates(
    references: Mapping[str, str],
    lines: list[str],
) -> dict[str, str]:
    values = [line for line in lines if line.strip()]
    predictions: dict[str, str] = {}
    for field, reference in references.items():
        if not values:
            predictions[field] = ""
            continue
        normalized_reference = normalized_text(reference)
        predictions[field] = max(
            values,
            key=lambda line: SequenceMatcher(
                None, normalized_reference, normalized_text(line)
            ).ratio(),
        )
    return predictions


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 400:
        raise ValueError(
            f"frozen OCR benchmark must contain 400 pages, found {len(rows)}"
        )
    if any(row.get("split") != "dev_select" for row in rows):
        raise ValueError("large OCR ablation accepts DEV_SELECT only")
    if any(str(row.get("is_private", "")).casefold() != "false" for row in rows):
        raise ValueError("private row found in large OCR ablation manifest")
    return rows


def _load_page(
    row: Mapping[str, str],
) -> tuple[Path, dict[str, Any]]:
    _, image_path = resolve_project_input(
        PROJECT_ROOT,
        row["source_image_path"],
        label="benchmark image",
        required_prefix="data/raw/public",
    )
    _, annotation_path = resolve_project_input(
        PROJECT_ROOT,
        row["normalized_annotation_path"],
        label="benchmark annotation",
        required_prefix="data/processed/normalized_ie_annotations",
        allow_prefix_junction=True,
    )
    if sha256_file(image_path) != row["source_image_sha256"].casefold():
        raise ValueError(f"benchmark image hash drift: {row['page_id']}")
    if sha256_file(annotation_path) != row["annotation_sha256"].casefold():
        raise ValueError(f"benchmark annotation hash drift: {row['page_id']}")
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    return image_path, annotation


def _provenance(
    *,
    profile: str,
    manifest_sha: str,
    detector_hash: str,
    general_hash: str,
    thai_hash: str,
    source_commit: str,
    device: str,
    sample_count: int,
    failure_count: int,
    duration_seconds: float,
) -> dict[str, Any]:
    configuration_hash = hashlib.sha256(
        canonical_json(
            {
                "profile": profile,
                "cardinal_angles": [0],
                "fine_deskew": False,
                "tiling": False,
                "retry": False,
            }
        ).encode("utf-8")
    ).hexdigest()
    return {
        "split": "dev_select",
        "manifest_sha256": manifest_sha,
        "detector_sha256": detector_hash,
        "recognizer_sha256": hashlib.sha256(
            (general_hash + thai_hash).encode("ascii")
        ).hexdigest(),
        "checkpoint_sha256": None,
        "calibration_sha256": None,
        "configuration_hash": configuration_hash,
        "source_commit": source_commit,
        "device": device,
        "sample_count": sample_count,
        "failure_count": failure_count,
        "duration_seconds": duration_seconds,
        "private_row_count": 0,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    stream = StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=fields,
        lineterminator="\n",
        extrasaction="ignore",
    )
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, stream.getvalue())


def _clamp(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()


if __name__ == "__main__":
    raise SystemExit(main())
