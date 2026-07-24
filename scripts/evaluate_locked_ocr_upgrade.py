#!/usr/bin/env python3
"""Run the selected OCR plus LayoutXLM stack on locked TEST_IN_DOMAIN once."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config as cfgmod  # noqa: E402
from src.evaluation.critical_field_ocr import evaluate_critical_fields  # noqa: E402
from src.inference.document_pipeline import DocumentPipeline  # noqa: E402
from src.ocr.environment import configure_external_environment, require_storage_gate  # noqa: E402
from src.ocr.model_registry import ModelRegistry  # noqa: E402
from src.ocr.stack_binding import build_ocr_stack_binding  # noqa: E402
from src.rotation_common import (  # noqa: E402
    atomic_write_json,
    canonical_json,
    read_csv_rows,
    sha256_file,
)
from scripts.evaluate_ocr_model_comparison import (  # noqa: E402
    _aggregate,
    _evaluate_page,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--model-manifest",
        default=str(
            PROJECT_ROOT
            / "data"
            / "metadata"
            / "final_model_dataset_manifest_ocr_v2.csv"
        ),
    )
    parser.add_argument(
        "--reference-layout-report",
        default=str(
            PROJECT_ROOT
            / "reports"
            / "final_model"
            / "evaluations"
            / "ocr_upgrade_locked_test_ground_truth.json"
        ),
    )
    parser.add_argument(
        "--model-setup",
        default=str(PROJECT_ROOT / "reports" / "ocr" / "model_setup.json"),
    )
    parser.add_argument("--device", choices=("cpu", "gpu:0"), default="gpu:0")
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "reports" / "ocr_upgrade"),
    )
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    outputs = {
        "ocr": output_root / "locked_test_ocr_metrics.json",
        "end_to_end": output_root / "locked_test_end_to_end_metrics.json",
        "critical": output_root / "critical_field_metrics.json",
        "sentinel": output_root / "locked_test_run.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise SystemExit(
            "locked TEST evaluation already has output; rerun refused: "
            + ", ".join(existing)
        )

    cfg = cfgmod.load_config(args.config)
    asset_root = cfgmod.resolve_path(cfg, "external_assets")
    configure_external_environment(asset_root)
    require_storage_gate(
        asset_root,
        operation="one-time locked OCR upgrade TEST_IN_DOMAIN evaluation",
        anticipated_c_gib=0.5,
        anticipated_asset_gib=12.0,
    )
    checkpoint = Path(args.checkpoint).resolve()
    checkpoint_model = checkpoint / "model.safetensors"
    calibration = PROJECT_ROOT / "models" / "multitask_calibration.json"
    model_manifest = Path(args.model_manifest).resolve()
    reference_report_path = Path(args.reference_layout_report).resolve()
    for required in (
        checkpoint_model,
        checkpoint / "training_state.json",
        calibration,
        model_manifest,
        reference_report_path,
    ):
        if not required.is_file():
            raise SystemExit(f"required locked-test input is missing: {required}")

    reference_layout = _json(reference_report_path)
    checkpoint_sha = sha256_file(checkpoint_model)
    if (
        reference_layout.get("split") != "test_in_domain"
        or reference_layout.get("checkpoint_sha256")
        != checkpoint_sha
        or int(reference_layout.get("private_row_count", -1)) != 0
    ):
        raise SystemExit(
            "reference-token locked-test report is not bound to this public checkpoint"
        )

    model_rows = read_csv_rows(model_manifest)
    model_test_ids = {
        row["page_id"]
        for row in model_rows
        if row.get("project_split") == "test_in_domain"
        and row.get("token_source") == "ground_truth"
        and row.get("is_usable") == "true"
        and row.get("is_private") == "false"
    }
    split_manifest = (
        cfgmod.resolve_path(cfg, "metadata")
        / "information_extraction_split_manifest.csv"
    )
    split_rows = [
        row
        for row in read_csv_rows(split_manifest)
        if row.get("project_split") == "test_in_domain"
        and row.get("is_usable") == "true"
        and row.get("is_private") == "false"
    ]
    if not split_rows or {row["page_id"] for row in split_rows} != model_test_ids:
        raise SystemExit(
            "selected OCR-v2 manifest and locked test split do not contain the same public pages"
        )
    if any(
        row.get("token_source") in {"paddleocr", "hybrid", "ocr_noise"}
        and row.get("project_split") == "test_in_domain"
        for row in model_rows
    ):
        raise SystemExit("OCR-derived TEST rows found in the selected training manifest")

    profile = str(cfg.get("ocr", {}).get("default_profile", "")).casefold()
    if profile not in {"original", "custom", "adaptive"}:
        raise SystemExit("final OCR default profile is not frozen")
    choice = "original" if profile == "original" else "auto"
    registry = ModelRegistry.from_setup(
        args.model_setup,
        upgrade_registry=cfgmod.resolve_path(
            cfg, "reports"
        )
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
        layout_checkpoint=checkpoint,
        calibration_path=calibration,
        enable_kmeans_display=False,
        require_layout_model=True,
        ocr_profile=profile,
    )
    observations: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for index, row in enumerate(split_rows, start=1):
            benchmark_row = _benchmark_row(row)
            observation = _evaluate_page(pipeline, benchmark_row)
            observation["slice"] = {
                "dataset": row.get("dataset", "unknown"),
                "document_type": row.get("document_type", "unknown"),
                "language": row.get("language", "unknown"),
            }
            observations.append(observation)
            if index % 25 == 0 or index == len(split_rows):
                print(
                    f"locked TEST_IN_DOMAIN: {index}/{len(split_rows)}",
                    flush=True,
                )
    finally:
        pipeline.close()
    duration = time.perf_counter() - started
    aggregate = _aggregate(observations)
    failure_count = sum(bool(row["failed"]) for row in observations)
    source_commit = _git_commit()
    common = {
        "schema_version": "1.0",
        "status": (
            "passed" if failure_count == 0 else "completed_with_failures"
        ),
        "build_id": "locked-test-"
        + hashlib.sha256(
            canonical_json(
                {
                    "manifest_sha256": sha256_file(model_manifest),
                    "checkpoint_sha256": checkpoint_sha,
                    "calibration_sha256": sha256_file(calibration),
                    "ocr_stack": stack,
                }
            ).encode("utf-8")
        ).hexdigest()[:16],
        "split": "test_in_domain",
        "manifest_sha256": sha256_file(model_manifest),
        "split_manifest_sha256": sha256_file(split_manifest),
        "detector_sha256": stack["detector_sha256"],
        "recognizer_sha256": stack["recognizer_sha256"],
        "checkpoint_sha256": checkpoint_sha,
        "calibration_sha256": sha256_file(calibration),
        "configuration_sha256": stack["preprocessing_sha256"],
        "source_commit": source_commit,
        "device": args.device,
        "sample_count": len(observations),
        "failure_count": failure_count,
        "duration_seconds": duration,
        "private_row_count": 0,
        "gmail_fit_rows": 0,
        "ocr_profile": profile,
        "test_used_for_selection": False,
        "coru_used_for_selection": False,
        "gmail_used_for_selection": False,
    }
    by_dataset = _grouped(observations, "dataset")
    by_document_type = _grouped(observations, "document_type")
    by_language = _grouped(observations, "language")
    detector = aggregate
    ocr_metrics = {
        **common,
        "metrics": {
            key: detector.get(key)
            for key in (
                "polygon_precision",
                "polygon_recall",
                "polygon_f1",
                "recognized_text_coverage",
                "cer",
                "wer",
                "critical_field_exact_match",
                "critical_field_count",
                "nonempty_output_rate",
                "page_failure_rate",
                "time_per_page_seconds",
            )
        },
        "text_size_recall": _text_size_recall(observations),
        "error_counts": aggregate.get("error_counts", {}),
        "by_dataset": by_dataset,
        "by_document_type": by_document_type,
        "by_language": by_language,
        "angle": 0,
    }
    reference_metrics = dict(reference_layout.get("metrics") or {})
    end_to_end_metrics = {
        **common,
        "reference_layout_report_sha256": sha256_file(
            reference_report_path
        ),
        "metrics": {
            "entity_f1": aggregate["end_to_end_entity_f1"],
            "relation_f1": aggregate["end_to_end_relation_f1"],
            "canonical_field_accuracy": aggregate[
                "end_to_end_canonical_accuracy"
            ],
            "critical_field_exact_match": aggregate[
                "critical_field_exact_match"
            ],
            "document_type_accuracy": aggregate[
                "document_type_accuracy"
            ],
            "table_availability_rate": aggregate[
                "table_availability_rate"
            ],
            "processing_time_per_page_seconds": aggregate[
                "time_per_page_seconds"
            ],
            "reference_token_entity_f1": reference_metrics.get(
                "entity_token_f1"
            ),
            "ocr_to_layout_degradation": (
                float(reference_metrics["entity_token_f1"])
                - float(aggregate["end_to_end_entity_f1"])
            ),
        },
        "error_counts": aggregate.get("error_counts", {}),
        "by_dataset": by_dataset,
        "by_document_type": by_document_type,
        "by_language": by_language,
        "angle": 0,
    }
    critical_metrics = {
        **common,
        "metrics": _critical_metrics(observations),
    }
    atomic_write_json(outputs["ocr"], ocr_metrics)
    atomic_write_json(outputs["end_to_end"], end_to_end_metrics)
    atomic_write_json(outputs["critical"], critical_metrics)
    atomic_write_json(
        outputs["sentinel"],
        {
            **common,
            "completed_outputs": {
                name: {
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
                for name, path in outputs.items()
                if name != "sentinel"
            },
            "rerun_policy": "refused after this sentinel is written",
        },
    )
    print(
        json.dumps(
            {
                "status": common["status"],
                "sample_count": len(observations),
                "failure_count": failure_count,
                "ocr_output": str(outputs["ocr"]),
                "end_to_end_output": str(outputs["end_to_end"]),
                "critical_output": str(outputs["critical"]),
                "sentinel": str(outputs["sentinel"]),
            },
            indent=2,
        )
    )
    return 0 if failure_count == 0 else 1


def _benchmark_row(row: Mapping[str, str]) -> dict[str, str]:
    image_path = PROJECT_ROOT / row["image_path"]
    annotation_path = PROJECT_ROOT / row["normalized_annotation_path"]
    if sha256_file(image_path) != str(row.get("sha256", "")).casefold():
        raise ValueError(f"locked image hash drift: {row['page_id']}")
    annotation = _json(annotation_path)
    page = dict(annotation.get("page") or {})
    return {
        **dict(row),
        "source_image_path": row["image_path"],
        "source_image_sha256": sha256_file(image_path),
        "annotation_sha256": sha256_file(annotation_path),
        "width": str(int(page["width"])),
        "height": str(int(page["height"])),
    }


def _grouped(
    observations: list[dict[str, Any]],
    field: str,
) -> dict[str, dict[str, Any]]:
    values = sorted(
        {
            str(row.get("slice", {}).get(field, "unknown"))
            for row in observations
        }
    )
    return {
        value: _aggregate(
            [
                row
                for row in observations
                if str(row.get("slice", {}).get(field, "unknown"))
                == value
            ]
        )
        for value in values
    }


def _text_size_recall(
    observations: list[dict[str, Any]],
) -> dict[str, float | None]:
    counts = {
        key: sum(int(row["detection"].get(key, 0)) for row in observations)
        for key in (
            "small_expected",
            "small_matched",
            "medium_expected",
            "medium_matched",
            "large_expected",
            "large_matched",
        )
    }
    return {
        size: (
            counts[f"{size}_matched"] / counts[f"{size}_expected"]
            if counts[f"{size}_expected"]
            else None
        )
        for size in ("small", "medium", "large")
    }


def _critical_metrics(
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    return evaluate_critical_fields(
        [row["reference_fields"] for row in observations],
        [row["predicted_fields"] for row in observations],
    )


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()


if __name__ == "__main__":
    raise SystemExit(main())
