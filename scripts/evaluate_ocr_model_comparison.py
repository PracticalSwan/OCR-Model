#!/usr/bin/env python3
"""Compare the frozen A-F OCR configurations on public DEV_SELECT only."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image  # noqa: E402

from src import config as cfgmod  # noqa: E402
from src.evaluation.critical_field_ocr import evaluate_critical_fields  # noqa: E402
from src.evaluation.metrics import extraction_metrics, normalized_text, ocr_text_metrics  # noqa: E402
from src.inference.document_io import DocumentPage  # noqa: E402
from src.inference.document_pipeline import DocumentPipeline  # noqa: E402
from src.ocr.benchmark import resolve_project_input  # noqa: E402
from src.ocr.environment import configure_external_environment, require_storage_gate  # noqa: E402
from src.ocr.errors import OCRModelUnavailable  # noqa: E402
from src.ocr.training_data import convert_detection_annotation, critical_fields_by_token  # noqa: E402
from src.ocr.trials import aggregate_detector_results, detector_page_result  # noqa: E402
from src.rotation_common import (  # noqa: E402
    atomic_write_json,
    atomic_write_text,
    canonical_json,
    sha256_file,
)


CONFIGURATIONS: dict[str, dict[str, str]] = {
    "A": {
        "description": "original detector + original recognizers",
        "ocr_profile": "original",
        "detector": "original",
        "general": "original",
        "thai": "original",
    },
    "B": {
        "description": "custom detector + original recognizers",
        "ocr_profile": "custom",
        "detector": "custom",
        "general": "original",
        "thai": "original",
    },
    "C": {
        "description": "original detector + accepted custom recognizers",
        "ocr_profile": "custom",
        "detector": "original",
        "general": "custom",
        "thai": "auto",
    },
    "D": {
        "description": "custom detector + accepted custom recognizers",
        "ocr_profile": "custom",
        "detector": "custom",
        "general": "custom",
        "thai": "auto",
    },
    "E": {
        "description": "registry-selected detector and recognizers",
        "ocr_profile": "custom",
        "detector": "auto",
        "general": "auto",
        "thai": "auto",
    },
    "F": {
        "description": "adaptive preprocessing, tiling, and recognition retries",
        "ocr_profile": "adaptive",
        "detector": "auto",
        "general": "auto",
        "thai": "auto",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument(
        "--benchmark-manifest",
        default=str(PROJECT_ROOT / "data/metadata/ocr_benchmark_manifest.csv"),
    )
    parser.add_argument(
        "--model-setup",
        default=str(PROJECT_ROOT / "reports/ocr/model_setup.json"),
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--calibration",
        default=str(PROJECT_ROOT / "models/multitask_calibration.json"),
    )
    parser.add_argument(
        "--disable-calibration",
        action="store_true",
        help=(
            "Use the checkpoint's conservative default thresholds for this "
            "pre-selection comparison. Final evaluation still requires a "
            "fresh OCR-stack-bound calibration."
        ),
    )
    parser.add_argument("--device", choices=("cpu", "gpu:0"), default="gpu:0")
    parser.add_argument(
        "--configurations",
        nargs="+",
        choices=tuple(CONFIGURATIONS),
        default=list(CONFIGURATIONS),
    )
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "reports/ocr_upgrade/ocr_model_comparison.csv"),
    )
    parser.add_argument(
        "--selection-output",
        default=str(PROJECT_ROOT / "reports/ocr_upgrade/ocr_model_selection.json"),
    )
    args = parser.parse_args()
    if not 1 <= args.limit <= 400:
        parser.error("--limit must be in [1, 400]")

    cfg = cfgmod.load_config(args.config)
    configure_external_environment(cfgmod.resolve_path(cfg, "external_assets"))
    require_storage_gate(
        cfgmod.resolve_path(cfg, "external_assets"),
        operation="public DEV_SELECT OCR A-F comparison",
        anticipated_c_gib=0.25,
        anticipated_asset_gib=2.0,
    )
    manifest_path = Path(args.benchmark_manifest).resolve()
    rows = _load_rows(manifest_path)[: args.limit]
    checkpoint = Path(args.checkpoint).resolve()
    calibration = (
        None if args.disable_calibration else Path(args.calibration).resolve()
    )
    for required in (
        checkpoint / "model.safetensors",
        checkpoint / "training_state.json",
    ):
        if not required.is_file():
            raise SystemExit(f"required comparison artifact is missing: {required}")
    if calibration is not None and not calibration.is_file():
        raise SystemExit(
            f"required comparison artifact is missing: {calibration}"
        )

    source_commit = _git_commit()
    manifest_sha = sha256_file(manifest_path)
    checkpoint_sha = sha256_file(checkpoint / "model.safetensors")
    calibration_sha = sha256_file(calibration) if calibration is not None else None
    registry_payload = json.loads(
        Path(cfgmod.resolve_path(cfg, "reports") / "ocr_upgrade/model_registry.json")
        .read_text(encoding="utf-8")
    )
    summaries: list[dict[str, Any]] = []
    for label in args.configurations:
        definition = CONFIGURATIONS[label]
        started = time.perf_counter()
        print(f"configuration {label}: {definition['description']}", flush=True)
        try:
            run_cfg = copy.deepcopy(cfg)
            run_cfg.setdefault("ocr", {})["orientation_candidates"] = [0]
            run_cfg["ocr"]["cache_enabled"] = False
            pipeline = DocumentPipeline.from_config(
                run_cfg,
                device=args.device,
                model_setup=args.model_setup,
                layout_checkpoint=checkpoint,
                calibration_path=calibration,
                use_layout_calibration=calibration is not None,
                enable_kmeans_display=False,
                require_layout_model=True,
                ocr_profile=definition["ocr_profile"],
                detector_model=definition["detector"],
                general_recognizer=definition["general"],
                thai_recognizer=definition["thai"],
            )
        except OCRModelUnavailable as exc:
            summaries.append(
                _unavailable_row(
                    label,
                    definition,
                    exc,
                    manifest_sha=manifest_sha,
                    checkpoint_sha=checkpoint_sha,
                    calibration_sha=calibration_sha,
                    source_commit=source_commit,
                    device=args.device,
                    sample_count=len(rows),
                    duration_seconds=time.perf_counter() - started,
                )
            )
            continue

        observations = []
        try:
            for index, row in enumerate(rows, start=1):
                observations.append(_evaluate_page(pipeline, row))
                if index % 25 == 0 or index == len(rows):
                    print(
                        f"configuration {label}: {index}/{len(rows)}",
                        flush=True,
                    )
        finally:
            pipeline.close()
        summary = _aggregate(observations)
        config_hash = hashlib.sha256(
            canonical_json(definition).encode("utf-8")
        ).hexdigest()
        summary.update(
            {
                "configuration": label,
                "description": definition["description"],
                "status": "passed",
                "selected": False,
                "ocr_profile": definition["ocr_profile"],
                "detector_choice": definition["detector"],
                "general_recognizer_choice": definition["general"],
                "thai_recognizer_choice": definition["thai"],
                "split": "dev_select",
                "manifest_sha256": manifest_sha,
                "checkpoint_sha256": checkpoint_sha,
                "calibration_sha256": calibration_sha,
                "configuration_sha256": config_hash,
                "source_commit": source_commit,
                "device": args.device,
                "sample_count": len(rows),
                "failure_count": sum(bool(item["failed"]) for item in observations),
                "duration_seconds": time.perf_counter() - started,
                "private_row_count": 0,
                **configuration_eligibility(definition, registry_payload),
            }
        )
        summaries.append(summary)

    successful = [row for row in summaries if row.get("status") == "passed"]
    if not successful:
        raise RuntimeError("all requested OCR comparison configurations were unavailable")
    fastest = min(float(row["time_per_page_seconds"]) for row in successful)
    for row in successful:
        efficiency = min(
            1.0, fastest / max(1e-9, float(row["time_per_page_seconds"]))
        )
        row["normalized_efficiency_score"] = efficiency
        row["selection_score"] = ocr_model_selection_score(
            detection_f1=float(row["polygon_f1"]),
            coverage=float(row["recognized_text_coverage"]),
            wer=float(row["wer"]),
            critical_exact=float(row["critical_field_exact_match"]),
            entity_f1=float(row["end_to_end_entity_f1"]),
            canonical_accuracy=float(row["end_to_end_canonical_accuracy"]),
            relation_f1=float(row["end_to_end_relation_f1"]),
            efficiency=efficiency,
            failure_rate=float(row["page_failure_rate"]),
        )
    eligible = [
        row
        for row in successful
        if bool(row.get("eligible_for_default", False))
    ]
    if not eligible:
        raise RuntimeError("no accepted OCR configuration is eligible for default")
    selected = select_simplest_material_configuration(eligible)
    selected["selected"] = True
    _write_csv(Path(args.output), summaries)
    selection = {
        "schema_version": "1.0",
        "status": "passed",
        "build_id": "ocr-model-selection-"
        + hashlib.sha256(canonical_json(summaries).encode("utf-8")).hexdigest()[:16],
        "split": "dev_select",
        "manifest_sha256": manifest_sha,
        "checkpoint_sha256": checkpoint_sha,
        "calibration_sha256": calibration_sha,
        "source_commit": source_commit,
        "device": args.device,
        "sample_count": sum(int(row.get("sample_count", 0)) for row in successful),
        "failure_count": sum(int(row.get("failure_count", 0)) for row in successful),
        "duration_seconds": sum(float(row["duration_seconds"]) for row in successful),
        "private_row_count": 0,
        "selected_configuration": selected["configuration"],
        "selected_ocr_profile": selected["ocr_profile"],
        "selected_score": selected["selection_score"],
        "configurations": summaries,
        "selection_formula": (
            "0.20*polygon_f1 + 0.15*coverage + 0.10*(1-clamped_wer) + "
            "0.15*critical_exact + 0.15*entity_f1 + 0.10*canonical_accuracy + "
            "0.05*relation_f1 + 0.05*efficiency + 0.05*(1-failure_rate)"
        ),
        "selection_rule": (
            "select the lowest-complexity configuration within 0.01 of the best "
            "score; keep A when the best gain is below 0.02"
        ),
        "kmeans_controls_ocr": False,
        "calibration_mode": (
            "checkpoint_default_thresholds"
            if calibration is None
            else "explicit_calibration_artifact"
        ),
        "test_private_tuning_rows": 0,
        "limitations": [
            "DEV_SELECT contains raster images, so PDF 200-to-300-DPI rerendering is measured in the separate adaptive-rendering report.",
            "Configurations B and D remain unavailable when no custom detector passed its hardware-bounded training and acceptance gate.",
        ],
    }
    atomic_write_json(Path(args.selection_output), selection)
    print(json.dumps(selection, indent=2, ensure_ascii=False))
    return 0


def ocr_model_selection_score(
    *,
    detection_f1: float,
    coverage: float,
    wer: float,
    critical_exact: float,
    entity_f1: float,
    canonical_accuracy: float,
    relation_f1: float,
    efficiency: float,
    failure_rate: float,
) -> float:
    return (
        0.20 * _clamp(detection_f1)
        + 0.15 * _clamp(coverage)
        + 0.10 * (1.0 - _clamp(wer))
        + 0.15 * _clamp(critical_exact)
        + 0.15 * _clamp(entity_f1)
        + 0.10 * _clamp(canonical_accuracy)
        + 0.05 * _clamp(relation_f1)
        + 0.05 * _clamp(efficiency)
        + 0.05 * (1.0 - _clamp(failure_rate))
    )


def select_simplest_material_configuration(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("no successful OCR configurations")
    complexity = {"A": 0, "C": 1, "E": 2, "B": 3, "D": 4, "F": 5}
    best = max(rows, key=lambda row: float(row["selection_score"]))
    baseline = next(
        (row for row in rows if row.get("configuration") == "A"),
        None,
    )
    if (
        baseline is not None
        and float(best["selection_score"]) - float(baseline["selection_score"]) < 0.02
    ):
        return baseline
    material = [
        row
        for row in rows
        if float(row["selection_score"]) + 0.01 >= float(best["selection_score"])
    ]
    return min(
        material,
        key=lambda row: (
            complexity.get(str(row.get("configuration")), 99),
            -float(row["selection_score"]),
        ),
    )


def configuration_eligibility(
    definition: Mapping[str, str],
    registry_payload: Mapping[str, Any],
) -> dict[str, Any]:
    models = registry_payload.get("models")
    if not isinstance(models, Mapping):
        raise ValueError("OCR registry has no models mapping")
    requirements = (
        ("detector", definition.get("detector")),
        ("general", definition.get("general")),
        ("thai", definition.get("thai")),
    )
    reasons = []
    for language, choice in requirements:
        if choice != "custom":
            continue
        candidates = [
            value
            for value in models.values()
            if isinstance(value, Mapping)
            and value.get("variant") == "custom"
            and (
                value.get("role") == "detector"
                if language == "detector"
                else value.get("role") == "recognizer"
                and str(value.get("language", "")).casefold().startswith(language)
            )
        ]
        if not candidates or not any(
            bool(value.get("available", False))
            and bool(value.get("accepted", False))
            for value in candidates
        ):
            reasons.append(f"{language}_custom_not_accepted")
    return {
        "eligible_for_default": not reasons,
        "default_ineligibility_reason": ";".join(reasons),
    }


def _evaluate_page(
    pipeline: DocumentPipeline,
    row: Mapping[str, str],
) -> dict[str, Any]:
    image_path, annotation = _load_page(row)
    started = time.perf_counter()
    try:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        result = pipeline.extract_pages(
            document_id=f"ocr_compare_{row['page_id']}",
            source_type="image",
            pages=[DocumentPage(1, image)],
            language="auto",
            language_hint=row.get("language"),
        )
        page = result["pages"][0]
        ocr_evidence = page.get("ocr") or {}
        failed = False
        error = None
    except Exception as exc:
        page = {
            "ocr": {"words": []},
            "full_text": "",
            "entities": [],
            "key_value_pairs": [],
        }
        result = {"fields": {}, "document_type": {"label": "unknown"}}
        ocr_evidence = {}
        failed = True
        error = f"{type(exc).__name__}: {exc}"
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
        [
            {"polygon": word.get("polygon")}
            for word in page.get("ocr", {}).get("words", [])
        ],
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
    text = ocr_text_metrics(reference_text, str(page.get("full_text", "")))
    extraction = extraction_metrics(annotation, page, result.get("fields") or {})
    reference_fields = {
        field: str(value.get("raw_text") or value.get("value") or "")
        for field, value in (annotation.get("canonical_fields") or {}).items()
        if isinstance(value, Mapping)
        and str(value.get("raw_text") or value.get("value") or "").strip()
    }
    predicted_fields = {
        field: str(value.get("value") or "")
        for field, value in (result.get("fields") or {}).items()
        if isinstance(value, Mapping)
    }
    error_counts = _error_categories(
        row,
        annotation,
        page,
        result,
        detection=detection,
        text=text,
        extraction=extraction,
        reference_text=reference_text,
        reference_fields=reference_fields,
        predicted_fields=predicted_fields,
    )
    return {
        "failed": failed,
        "error": error,
        "detection": detection,
        "text": text,
        "reference_words": len(normalized_text(reference_text).split()),
        "extraction": extraction,
        "reference_fields": reference_fields,
        "predicted_fields": predicted_fields,
        "tiling": dict(ocr_evidence.get("tiling") or {}),
        "recognition_retries": dict(
            ocr_evidence.get("recognition_retries") or {}
        ),
        "nonempty_output": bool(str(page.get("full_text", "")).strip()),
        "recognized_word_count": len(
            page.get("ocr", {}).get("words", [])
        ),
        "table_available": bool(page.get("tables") or []),
        "language_route": str(
            page.get("ocr", {}).get("language_route", "unknown")
        ),
        "document_type_correct": (
            str(result.get("document_type", {}).get("label", "")).casefold()
            == str(row.get("document_type", "")).casefold()
        ),
        "duration_seconds": time.perf_counter() - started,
        "error_counts": error_counts,
    }


def _aggregate(observations: list[dict[str, Any]]) -> dict[str, Any]:
    detector = aggregate_detector_results(
        [item["detection"] for item in observations]
    )
    reference_characters = sum(
        int(item["text"]["reference_characters"]) for item in observations
    )
    character_errors = sum(
        int(item["text"]["character_errors"]) for item in observations
    )
    reference_words = sum(int(item["reference_words"]) for item in observations)
    word_errors = sum(int(item["text"]["word_errors"]) for item in observations)
    entity_tp = sum(
        int(item["extraction"]["entity"]["true_positive"]) for item in observations
    )
    entity_expected = sum(
        int(item["extraction"]["entity"]["expected"]) for item in observations
    )
    entity_predicted = sum(
        int(item["extraction"]["entity"]["predicted"]) for item in observations
    )
    relation_tp = sum(
        int(item["extraction"]["relation"]["true_positive"]) for item in observations
    )
    relation_expected = sum(
        int(item["extraction"]["relation"]["expected"]) for item in observations
    )
    relation_predicted = sum(
        int(item["extraction"]["relation"]["predicted"]) for item in observations
    )
    field_applicable = sum(
        int(item["extraction"]["canonical_fields"]["applicable"])
        for item in observations
    )
    field_correct = sum(
        int(item["extraction"]["canonical_fields"]["correct"])
        for item in observations
    )
    critical = evaluate_critical_fields(
        [item["reference_fields"] for item in observations],
        [item["predicted_fields"] for item in observations],
    )
    failures = sum(bool(item["failed"]) for item in observations)
    duration = sum(float(item["duration_seconds"]) for item in observations)
    tiling_triggered = [
        item["tiling"]
        for item in observations
        if bool(item["tiling"].get("triggered", False))
    ]
    retries = [
        item["recognition_retries"]
        for item in observations
        if item["recognition_retries"]
    ]
    retry_items = [
        value
        for retry in retries
        for value in (retry.get("items") or [])
        if isinstance(value, Mapping)
    ]
    error_names = sorted(
        {
            name
            for item in observations
            for name in (item.get("error_counts") or {})
        }
    )
    return {
        "polygon_precision": detector["precision"],
        "polygon_recall": detector["recall"],
        "polygon_f1": detector["f1"],
        "small_text_recall": detector["small_text_recall"],
        "critical_region_recall": detector["critical_region_recall"],
        "recognized_text_coverage": max(
            0.0, 1.0 - character_errors / max(1, reference_characters)
        ),
        "cer": character_errors / max(1, reference_characters),
        "wer": word_errors / max(1, reference_words),
        "critical_field_exact_match": critical["aggregate"][
            "critical_exact_match"
        ],
        "critical_field_count": critical["evaluated_field_count"],
        "end_to_end_entity_f1": _f1(
            entity_tp, entity_predicted, entity_expected
        ),
        "end_to_end_canonical_accuracy": field_correct
        / max(1, field_applicable),
        "end_to_end_relation_f1": _f1(
            relation_tp, relation_predicted, relation_expected
        ),
        "document_type_accuracy": sum(
            bool(item["document_type_correct"]) for item in observations
        )
        / max(1, len(observations)),
        "time_per_page_seconds": duration / max(1, len(observations)),
        "page_failure_rate": failures / max(1, len(observations)),
        "nonempty_output_rate": sum(
            bool(item.get("nonempty_output", False))
            for item in observations
        )
        / max(1, len(observations)),
        "table_availability_rate": sum(
            bool(item.get("table_available", False))
            for item in observations
        )
        / max(1, len(observations)),
        "tiling_triggered_page_count": len(tiling_triggered),
        "tiling_selected_page_count": sum(
            bool(value.get("selected", False)) for value in tiling_triggered
        ),
        "tiling_duplicate_count": sum(
            int(value.get("duplicates_removed", 0) or 0)
            for value in tiling_triggered
        ),
        "tiling_duplicate_rate": (
            sum(
                int(value.get("duplicates_removed", 0) or 0)
                for value in tiling_triggered
            )
            / max(
                1,
                sum(
                    int(value.get("input_word_count", 0) or 0)
                    for value in tiling_triggered
                ),
            )
        ),
        "retried_page_count": len(retries),
        "retried_word_count": sum(
            int(value.get("retried_word_count", 0) or 0) for value in retries
        ),
        "retry_candidate_count": sum(
            int(value.get("candidate_count", 0) or 0) for value in retry_items
        ),
        "retry_non_original_selection_count": sum(
            str(value.get("selected_variant", "")) != "original_rectified"
            for value in retry_items
        ),
        "error_counts": {
            name: sum(
                int((item.get("error_counts") or {}).get(name, 0))
                for item in observations
            )
            for name in error_names
        },
        "normalized_efficiency_score": None,
        "selection_score": None,
    }


def _error_categories(
    row: Mapping[str, str],
    annotation: Mapping[str, Any],
    page: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    detection: Mapping[str, Any],
    text: Mapping[str, Any],
    extraction: Mapping[str, Any],
    reference_text: str,
    reference_fields: Mapping[str, str],
    predicted_fields: Mapping[str, str],
) -> dict[str, int]:
    """Return aggregate-only, reproducible public error-category signals."""
    predicted_text = str(page.get("full_text", ""))
    reference_normalized = normalized_text(reference_text)
    predicted_normalized = normalized_text(predicted_text)
    expected_regions = int(detection.get("expected", 0))
    predicted_regions = int(detection.get("predicted", 0))
    true_positive = int(detection.get("true_positive", 0))
    retries = dict((page.get("ocr") or {}).get("recognition_retries") or {})
    retry_items = [
        value
        for value in (retries.get("items") or [])
        if isinstance(value, Mapping)
    ]
    reference_numbers = re.findall(r"\d+(?:[.,]\d+)*", reference_normalized)
    predicted_numbers = re.findall(r"\d+(?:[.,]\d+)*", predicted_normalized)
    reference_identifiers = {
        value
        for value in re.findall(r"\b[\w/-]{5,}\b", reference_normalized)
        if any(character.isdigit() for character in value)
    }
    predicted_identifiers = set(
        re.findall(r"\b[\w/-]{5,}\b", predicted_normalized)
    )
    turkish_characters = "çğıöşü"
    table_expected = str(row.get("has_table", "")).casefold() == "true"
    table_available = bool(page.get("tables") or [])
    entity = dict(extraction.get("entity") or {})
    entity_wrong = (
        int(entity.get("true_positive", 0))
        < max(
            int(entity.get("expected", 0)),
            int(entity.get("predicted", 0)),
        )
    )
    ocr_wrong = int(text.get("character_errors", 0)) > 0
    fields_found_but_absent = sum(
        bool(normalized_text(value))
        and normalized_text(value) in predicted_normalized
        and not normalized_text(predicted_fields.get(field, ""))
        for field, value in reference_fields.items()
    )
    warnings = " ".join(
        str(value)
        for value in (
            list(page.get("warnings") or [])
            + list(result.get("warnings") or [])
        )
    ).casefold()
    low_confidence = page.get("ocr", {}).get("mean_confidence")
    low_contrast_signal = (
        low_confidence is not None
        and float(low_confidence) < 0.55
        and expected_regions > true_positive
    )
    reference_punctuation = re.findall(r"[^\w\s]", reference_text)
    predicted_punctuation = re.findall(r"[^\w\s]", predicted_text)
    return {
        "detection_missed_small_text": max(
            0,
            int(detection.get("small_expected", 0))
            - int(detection.get("small_matched", 0)),
        ),
        "detection_merged_region_signals": max(
            0, expected_regions - predicted_regions
        ),
        "detection_split_region_signals": max(
            0, predicted_regions - expected_regions
        ),
        "detection_false_positives": max(
            0, predicted_regions - true_positive
        ),
        "detection_table_miss_pages": int(
            table_expected and not table_available
        ),
        "detection_rotated_miss_pages": int(
            float(row.get("input_angle", 0.0) or 0.0) % 360.0 != 0.0
            and expected_regions > true_positive
        ),
        "detection_low_contrast_miss_pages": int(low_contrast_signal),
        "recognition_number_confusions": max(
            0,
            len(reference_numbers)
            - sum(value in predicted_numbers for value in reference_numbers),
        ),
        "recognition_decimal_confusions": sum(
            ("." in value or "," in value) and value not in predicted_numbers
            for value in reference_numbers
        ),
        "recognition_identifier_corruptions": sum(
            value not in predicted_identifiers for value in reference_identifiers
        ),
        "recognition_turkish_character_errors": sum(
            max(
                0,
                reference_normalized.count(character)
                - predicted_normalized.count(character),
            )
            for character in turkish_characters
        ),
        "recognition_spacing_error_pages": int(
            reference_normalized != predicted_normalized
            and reference_normalized.replace(" ", "")
            == predicted_normalized.replace(" ", "")
        ),
        "recognition_punctuation_losses": max(
            0, len(reference_punctuation) - len(predicted_punctuation)
        ),
        "recognition_crop_tight_retry_signals": sum(
            str(value.get("selected_variant", ""))
            not in {"", "original_rectified"}
            for value in retry_items
        ),
        "recognition_low_resolution_error_pages": int(
            min(int(row["width"]), int(row["height"])) <= 1000
            and float(text.get("cer", 0.0) or 0.0) > 0.30
        ),
        "recognition_language_route_errors": int(
            str(row.get("language", "")).casefold() in {"th", "thai"}
            and str(
                page.get("ocr", {}).get("language_route", "")
            ).casefold()
            != "thai"
        ),
        "downstream_ocr_correct_entity_wrong_pages": int(
            not ocr_wrong and entity_wrong
        ),
        "downstream_ocr_wrong_entity_wrong_pages": int(
            ocr_wrong and entity_wrong
        ),
        "downstream_field_evidence_found_but_abstained": int(
            fields_found_but_absent
        ),
        "downstream_relation_misses": max(
            0,
            int(extraction.get("relation", {}).get("expected", 0))
            - int(extraction.get("relation", {}).get("true_positive", 0)),
        ),
        "downstream_arithmetic_inconsistency_pages": int(
            "arithmetic" in warnings and "inconsisten" in warnings
        ),
        "downstream_table_grouping_failure_pages": int(
            table_expected and not table_available
        ),
    }


def _unavailable_row(
    label: str,
    definition: Mapping[str, str],
    exc: Exception,
    *,
    manifest_sha: str,
    checkpoint_sha: str,
    calibration_sha: str | None,
    source_commit: str,
    device: str,
    sample_count: int,
    duration_seconds: float,
) -> dict[str, Any]:
    return {
        "configuration": label,
        "description": definition["description"],
        "status": "unavailable",
        "selected": False,
        "ocr_profile": definition["ocr_profile"],
        "detector_choice": definition["detector"],
        "general_recognizer_choice": definition["general"],
        "thai_recognizer_choice": definition["thai"],
        "split": "dev_select",
        "manifest_sha256": manifest_sha,
        "checkpoint_sha256": checkpoint_sha,
        "calibration_sha256": calibration_sha,
        "configuration_sha256": hashlib.sha256(
            canonical_json(definition).encode("utf-8")
        ).hexdigest(),
        "source_commit": source_commit,
        "device": device,
        "sample_count": 0,
        "requested_sample_count": sample_count,
        "failure_count": 0,
        "duration_seconds": duration_seconds,
        "private_row_count": 0,
        "unavailable_reason": f"{type(exc).__name__}: {exc}",
    }


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 400:
        raise ValueError(f"frozen OCR benchmark must contain 400 rows, found {len(rows)}")
    if any(row.get("split") != "dev_select" for row in rows):
        raise ValueError("OCR model comparison accepts DEV_SELECT only")
    if any(str(row.get("is_private", "")).casefold() != "false" for row in rows):
        raise ValueError("private row found in OCR model comparison manifest")
    return rows


def _load_page(row: Mapping[str, str]) -> tuple[Path, dict[str, Any]]:
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
    return image_path, json.loads(annotation_path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    from io import StringIO

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


def _f1(true_positive: int, predicted: int, expected: int) -> float:
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / expected if expected else 0.0
    return (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )


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
