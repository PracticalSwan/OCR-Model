#!/usr/bin/env python3
"""Evaluate final inference on a deterministic CORU unseen-domain QA sample."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image  # noqa: E402

from src import config as cfgmod  # noqa: E402
from src.evaluation.metrics import normalized_text  # noqa: E402
from src.inference.document_io import DocumentPage  # noqa: E402
from src.inference.document_pipeline import DocumentPipeline  # noqa: E402
from src.ocr.environment import configure_external_environment, require_storage_gate  # noqa: E402
from src.ocr.model_registry import ModelRegistry  # noqa: E402
from src.ocr.stack_binding import build_ocr_stack_binding  # noqa: E402
from src.rotation_common import atomic_write_json, canonical_json, deterministic_rank, read_csv_rows, sha256_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", choices=("cpu", "gpu:0"), default="gpu:0")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--model-setup", default=str(PROJECT_ROOT / "reports" / "ocr" / "model_setup.json"))
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    cfg = cfgmod.load_config(args.config)
    asset_root = cfgmod.resolve_path(cfg, "external_assets")
    configure_external_environment(asset_root)
    require_storage_gate(
        asset_root, operation="CORU unseen-domain evaluation",
        anticipated_c_gib=0.25, anticipated_asset_gib=8.0,
    )
    checkpoint = Path(args.checkpoint).resolve()
    calibration = cfgmod.project_root(cfg) / "models" / "multitask_calibration.json"
    model_manifest = (
        cfgmod.resolve_path(cfg, "metadata")
        / "final_model_dataset_manifest_ocr_v2.csv"
    )
    profile = str(cfg.get("ocr", {}).get("default_profile", "original"))
    choice = "original" if profile == "original" else "auto"
    registry = ModelRegistry.from_setup(
        args.model_setup,
        upgrade_registry=cfgmod.resolve_path(cfg, "reports")
        / "ocr_upgrade"
        / "model_registry.json",
        detector_choice=choice,
        general_choice=choice,
        thai_choice=choice,
    )
    stack = build_ocr_stack_binding(cfg, registry, ocr_profile=profile)
    population = [
        row for row in read_csv_rows(cfgmod.resolve_path(cfg, "metadata") / "information_extraction_split_manifest.csv")
        if row.get("dataset") == "coru"
        and row.get("project_split") == "unseen_domain_test"
        and row.get("is_private") == "false"
    ]
    rows = sorted(population, key=lambda row: deterministic_rank(row["page_id"], 6060))[: args.limit]
    pipeline = DocumentPipeline.from_config(
        cfg, device=args.device, model_setup=args.model_setup,
        layout_checkpoint=checkpoint, calibration_path=calibration,
        enable_kmeans_display=False,
        require_layout_model=True,
    )
    counts: Counter[str] = Counter()
    field_applicable = 0
    field_correct = 0
    durations = []
    started = time.perf_counter()
    try:
        for row in rows:
            try:
                annotation = json.loads(
                    (PROJECT_ROOT / row["normalized_annotation_path"]).read_text(encoding="utf-8")
                )
                with Image.open(PROJECT_ROOT / row["image_path"]) as source:
                    image = source.convert("RGB")
                result = pipeline.extract_pages(
                    document_id="coru_unseen", source_type="image",
                    pages=[DocumentPage(1, image)], language="auto",
                )
                page = result["pages"][0]
                text = normalized_text(page["full_text"])
                answers = [
                    normalized_text(item.get("answer", ""))
                    for item in annotation.get("source_qa", [])
                    if normalized_text(item.get("answer", ""))
                ]
                counts["qa_answers"] += len(answers)
                counts["qa_answers_found_in_ocr"] += sum(answer in text for answer in answers)
                for field, expected in (annotation.get("canonical_fields") or {}).items():
                    if not isinstance(expected, dict) or not normalized_text(expected.get("value", "")):
                        continue
                    field_applicable += 1
                    predicted = result["fields"].get(field)
                    if isinstance(predicted, dict) and normalized_text(predicted.get("value", "")) == normalized_text(expected["value"]):
                        field_correct += 1
                counts["successful_pages"] += 1
                counts["nonempty_pages"] += bool(text)
                counts["entities"] += len(page["entities"])
                counts["relations"] += len(page["key_value_pairs"])
                counts["non_null_fields"] += sum(
                    isinstance(value, dict) and value.get("value") is not None
                    for value in result["fields"].values()
                )
                counts[f"route:{page['ocr']['language_route']}"] += 1
                counts[f"document_type:{result['document_type']['label']}"] += 1
                durations.append(float(result["processing"]["duration_seconds"]))
            except Exception as exc:
                counts["failed_pages"] += 1
                counts[f"error:{type(exc).__name__}"] += 1
    finally:
        pipeline.close()
    successful = counts["successful_pages"]
    report = {
        "schema_version": "1.0",
        "status": "passed" if counts["failed_pages"] == 0 else "completed_with_failures",
        "build_id": "unseen-coru-"
        + hashlib.sha256(
            canonical_json(
                {
                    "manifest_sha256": sha256_file(model_manifest),
                    "checkpoint_sha256": sha256_file(
                        checkpoint / "model.safetensors"
                    ),
                    "calibration_sha256": sha256_file(calibration),
                    "ocr_stack": stack,
                    "sample_pages": len(rows),
                }
            ).encode("utf-8")
        ).hexdigest()[:16],
        "dataset": "coru",
        "split": "unseen_domain_test",
        "manifest_sha256": sha256_file(model_manifest),
        "detector_sha256": stack["detector_sha256"],
        "recognizer_sha256": stack["recognizer_sha256"],
        "checkpoint_sha256": sha256_file(
            checkpoint / "model.safetensors"
        ),
        "configuration_sha256": stack["preprocessing_sha256"],
        "source_commit": _git_commit(),
        "device": args.device,
        "sample_count": len(rows),
        "failure_count": counts["failed_pages"],
        "duration_seconds": time.perf_counter() - started,
        "private_row_count": 0,
        "public_only": True,
        "private_page_count": 0,
        "population_pages": len(population),
        "sample_pages": len(rows),
        "sample_strategy": "deterministic bounded sample; never used for model or threshold selection",
        "successful_pages": successful,
        "failed_pages": counts["failed_pages"],
        "nonempty_rate": counts["nonempty_pages"] / max(1, successful),
        "qa_answer_text_recall": counts["qa_answers_found_in_ocr"] / max(1, counts["qa_answers"]),
        "qa_answers": counts["qa_answers"],
        "canonical_field_accuracy": field_correct / field_applicable if field_applicable else None,
        "canonical_fields_applicable": field_applicable,
        "mean_entities": counts["entities"] / max(1, successful),
        "mean_relations": counts["relations"] / max(1, successful),
        "mean_non_null_fields": counts["non_null_fields"] / max(1, successful),
        "mean_duration_seconds": statistics.fmean(durations) if durations else None,
        "route_counts": {
            key.split(":", 1)[1]: value for key, value in sorted(counts.items()) if key.startswith("route:")
        },
        "document_type_counts": {
            key.split(":", 1)[1]: value for key, value in sorted(counts.items()) if key.startswith("document_type:")
        },
        "error_type_counts": {
            key.split(":", 1)[1]: value for key, value in sorted(counts.items()) if key.startswith("error:")
        },
        "checkpoint_model_sha256": sha256_file(checkpoint / "model.safetensors"),
        "calibration_sha256": sha256_file(calibration),
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "CORU QA has answer text but no compatible token polygons, so entity and relation F1 are not defined.",
            "This bounded unseen-domain sample reports OCR answer coverage and canonical exact-match accuracy rather than pretending QA supervision is token labeling.",
        ],
    }
    output = cfgmod.resolve_path(cfg, "reports") / "final_model" / "unseen_domain_metrics.json"
    atomic_write_json(output, report)
    atomic_write_json(
        cfgmod.resolve_path(cfg, "reports")
        / "ocr_upgrade"
        / "unseen_coru_metrics.json",
        report,
    )
    print(json.dumps(report, indent=2))
    return 0 if successful else 1


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()


if __name__ == "__main__":
    raise SystemExit(main())
