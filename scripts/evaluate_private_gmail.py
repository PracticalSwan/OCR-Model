#!/usr/bin/env python3
"""Run local Gmail private-test inference and write aggregate-only metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image  # noqa: E402

from src import config as cfgmod  # noqa: E402
from src.inference.document_io import DocumentPage  # noqa: E402
from src.inference.document_pipeline import DocumentPipeline  # noqa: E402
from src.ocr.environment import configure_external_environment, require_storage_gate  # noqa: E402
from src.ocr.model_registry import ModelRegistry  # noqa: E402
from src.ocr.stack_binding import build_ocr_stack_binding  # noqa: E402
from src.rotation_common import atomic_write_json, canonical_json, deterministic_rank, read_csv_rows, sha256_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "gpu:0"), default="gpu:0")
    parser.add_argument("--layout-checkpoint")
    parser.add_argument("--model-setup", default=str(PROJECT_ROOT / "reports" / "ocr" / "model_setup.json"))
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    cfg = cfgmod.load_config(args.config)
    asset_root = cfgmod.resolve_path(cfg, "external_assets")
    configure_external_environment(asset_root)
    require_storage_gate(
        asset_root,
        operation="aggregate-only private Gmail operational evaluation",
        anticipated_c_gib=0.25,
        anticipated_asset_gib=4.0,
    )
    private_manifest = cfgmod.resolve_path(cfg, "metadata") / "private_information_extraction_manifest.csv"
    rows = [
        row for row in read_csv_rows(private_manifest)
        if row.get("is_private") == "true" and row.get("project_split") == "private_test"
        and row.get("is_usable") == "true" and row.get("image_path")
    ]
    rows.sort(key=lambda row: deterministic_rank(row["page_id"], 42))
    rows = rows[: args.limit]
    checkpoint = (
        Path(args.layout_checkpoint).resolve()
        if args.layout_checkpoint
        else Path(str(cfg.get("layout_model", {}).get("inference_checkpoint", ""))).resolve()
    )
    calibration = PROJECT_ROOT / "models" / "multitask_calibration.json"
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
    pipeline = DocumentPipeline.from_config(
        cfg,
        device=args.device,
        model_setup=args.model_setup,
        layout_checkpoint=args.layout_checkpoint,
        enable_kmeans_display=False,
    )
    counts: Counter[str] = Counter()
    confidences: list[float] = []
    durations: list[float] = []
    error_types: Counter[str] = Counter()
    started = time.perf_counter()
    try:
        for row in rows:
            try:
                with Image.open(PROJECT_ROOT / row["image_path"]) as source:
                    image = source.convert("RGB")
                result = pipeline.extract_pages(
                    document_id="private_document",
                    source_type="image",
                    pages=[DocumentPage(1, image)],
                    language="auto",
                    private_output=True,
                )
                page = result["pages"][0]
                counts["successful_pages"] += 1
                counts["ocr_words"] += len(page["ocr"]["words"])
                counts["entities"] += len(page["entities"])
                counts["relations"] += len(page["key_value_pairs"])
                counts["non_null_fields"] += sum(value is not None for value in result["fields"].values())
                counts[f"route:{page['ocr']['language_route']}"] += 1
                if page["ocr"]["mean_confidence"] is not None:
                    confidences.append(float(page["ocr"]["mean_confidence"]))
                durations.append(float(result["processing"]["duration_seconds"]))
            except Exception as exc:
                counts["failed_pages"] += 1
                error_types[type(exc).__name__] += 1
    finally:
        pipeline.close()
    attempted = len(rows)
    report = {
        "schema_version": "1.0",
        "status": "private_test_aggregate",
        "build_id": "private-operational-"
        + hashlib.sha256(
            canonical_json(
                {
                    "private_manifest_sha256": sha256_file(private_manifest),
                    "checkpoint_sha256": sha256_file(
                        checkpoint / "model.safetensors"
                    ),
                    "calibration_sha256": sha256_file(calibration),
                    "ocr_stack": stack,
                    "sample_count": attempted,
                }
            ).encode("utf-8")
        ).hexdigest()[:16],
        "split": "private_operational",
        "manifest_sha256": sha256_file(private_manifest),
        "detector_sha256": stack["detector_sha256"],
        "recognizer_sha256": stack["recognizer_sha256"],
        "checkpoint_sha256": sha256_file(
            checkpoint / "model.safetensors"
        ),
        "checkpoint_model_sha256": sha256_file(
            checkpoint / "model.safetensors"
        ),
        "calibration_sha256": sha256_file(calibration),
        "configuration_sha256": stack["preprocessing_sha256"],
        "source_commit": _git_commit(),
        "device": args.device,
        "sample_count": attempted,
        "failure_count": counts["failed_pages"],
        "duration_seconds": time.perf_counter() - started,
        "private_row_count": attempted,
        "attempted_pages": attempted,
        "successful_pages": counts["successful_pages"],
        "failed_pages": counts["failed_pages"],
        "attempted_documents": attempted,
        "successful_documents": counts["successful_pages"],
        "failed_documents": counts["failed_pages"],
        "mean_ocr_words": counts["ocr_words"] / max(1, counts["successful_pages"]),
        "mean_entities": counts["entities"] / max(1, counts["successful_pages"]),
        "mean_relations": counts["relations"] / max(1, counts["successful_pages"]),
        "mean_non_null_fields": counts["non_null_fields"] / max(1, counts["successful_pages"]),
        "mean_ocr_confidence": sum(confidences) / len(confidences) if confidences else None,
        "mean_duration_seconds": sum(durations) / len(durations) if durations else None,
        "route_counts": {key.split(":", 1)[1]: value for key, value in counts.items() if key.startswith("route:")},
        "error_type_counts": dict(error_types),
        "gmail_fit_rows": 0,
        "local_processing_only": True,
        "contains_filenames": False,
        "contains_ocr_text": False,
        "contains_images": False,
        "contains_per_document_predictions": False,
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": ["No private ground truth is used; this is operational aggregate testing, not accuracy evaluation."],
    }
    atomic_write_json(
        cfgmod.resolve_path(cfg, "reports") / "model_evaluation" / "private_gmail_aggregate.json",
        report,
    )
    atomic_write_json(
        cfgmod.resolve_path(cfg, "reports")
        / "final_model"
        / "private_test_aggregate.json",
        report,
    )
    atomic_write_json(
        cfgmod.resolve_path(cfg, "reports")
        / "ocr_upgrade"
        / "private_aggregate.json",
        report,
    )
    print(json.dumps(report, indent=2))
    return 0 if attempted and counts["successful_pages"] else 1


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()


if __name__ == "__main__":
    raise SystemExit(main())
