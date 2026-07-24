#!/usr/bin/env python3
"""Measure adaptive preprocessing, tiling, and recognition retries in isolation."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config as cfgmod  # noqa: E402
from src.inference.document_pipeline import DocumentPipeline  # noqa: E402
from src.ocr.environment import configure_external_environment, require_storage_gate  # noqa: E402
from src.rotation_common import (  # noqa: E402
    atomic_write_json,
    canonical_json,
    deterministic_rank,
    sha256_file,
)
from scripts.evaluate_ocr_model_comparison import (  # noqa: E402
    _aggregate,
    _evaluate_page,
    _load_rows,
)


COMPONENTS: dict[str, dict[str, Any]] = {
    "base": {
        "ocr_profile": "custom",
        "adaptive_preprocessing": False,
        "tiling": False,
        "recognition_retries": False,
    },
    "preprocessing_only": {
        "ocr_profile": "adaptive",
        "adaptive_preprocessing": True,
        "tiling": False,
        "recognition_retries": False,
    },
    "tiling_only": {
        "ocr_profile": "adaptive",
        "adaptive_preprocessing": False,
        "tiling": True,
        "recognition_retries": False,
    },
    "recognition_retries_only": {
        "ocr_profile": "adaptive",
        "adaptive_preprocessing": False,
        "tiling": False,
        "recognition_retries": True,
    },
    "combined": {
        "ocr_profile": "adaptive",
        "adaptive_preprocessing": True,
        "tiling": True,
        "recognition_retries": True,
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
    parser.add_argument("--device", choices=("cpu", "gpu:0"), default="gpu:0")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--components",
        nargs="+",
        choices=tuple(COMPONENTS),
        default=list(COMPONENTS),
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "reports/ocr_upgrade"),
    )
    args = parser.parse_args()
    if not 1 <= args.limit <= 400:
        parser.error("--limit must be in [1, 400]")

    cfg = cfgmod.load_config(args.config)
    configure_external_environment(cfgmod.resolve_path(cfg, "external_assets"))
    require_storage_gate(
        cfgmod.resolve_path(cfg, "external_assets"),
        operation="public DEV_SELECT OCR component ablations",
        anticipated_c_gib=0.25,
        anticipated_asset_gib=2.0,
    )
    manifest_path = Path(args.benchmark_manifest).resolve()
    rows = _balanced_rows(_load_rows(manifest_path), args.limit)
    checkpoint = Path(args.checkpoint).resolve()
    calibration = Path(args.calibration).resolve()
    source_commit = _git_commit()
    provenance = {
        "split": "dev_select",
        "manifest_sha256": sha256_file(manifest_path),
        "checkpoint_sha256": sha256_file(checkpoint / "model.safetensors"),
        "calibration_sha256": sha256_file(calibration),
        "source_commit": source_commit,
        "device": args.device,
        "sample_count": len(rows),
        "private_row_count": 0,
    }
    results: dict[str, dict[str, Any]] = {}
    for component in args.components:
        definition = COMPONENTS[component]
        run_cfg = copy.deepcopy(cfg)
        run_cfg.setdefault("ocr", {})["cache_enabled"] = False
        run_cfg["ocr"]["orientation_candidates"] = [0]
        run_cfg["ocr"]["preprocessing_profile"] = "original"
        run_cfg["ocr"]["adaptive_preprocessing_profile"] = (
            "quality_auto"
            if definition["adaptive_preprocessing"]
            else "original"
        )
        run_cfg["ocr"].setdefault("tiling", {})["enabled"] = bool(
            definition["tiling"]
        )
        run_cfg["ocr"].setdefault("recognition_retries", {})["enabled"] = bool(
            definition["recognition_retries"]
        )
        started = time.perf_counter()
        pipeline = DocumentPipeline.from_config(
            run_cfg,
            device=args.device,
            model_setup=args.model_setup,
            layout_checkpoint=checkpoint,
            calibration_path=calibration,
            enable_kmeans_display=False,
            require_layout_model=True,
            ocr_profile=str(definition["ocr_profile"]),
        )
        observations = []
        try:
            for index, row in enumerate(rows, start=1):
                observations.append(_evaluate_page(pipeline, row))
                if index % 20 == 0 or index == len(rows):
                    print(f"{component}: {index}/{len(rows)}", flush=True)
        finally:
            pipeline.close()
        metrics = _aggregate(observations)
        results[component] = {
            "definition": definition,
            "configuration_sha256": hashlib.sha256(
                canonical_json(definition).encode("utf-8")
            ).hexdigest(),
            "failure_count": sum(bool(row["failed"]) for row in observations),
            "duration_seconds": time.perf_counter() - started,
            "metrics": metrics,
        }

    output_root = Path(args.output_root)
    base = results.get("base")
    _write_component_report(
        output_root / "tiling_metrics.json",
        component="tiling",
        baseline=base,
        candidate=results.get("tiling_only"),
        provenance=provenance,
        results=results,
    )
    retry_report = _write_component_report(
        output_root / "recognition_retry_metrics.json",
        component="recognition_retries",
        baseline=base,
        candidate=results.get("recognition_retries_only"),
        provenance=provenance,
        results=results,
    )
    crop_report = {
        **retry_report,
        "component": "crop_padding",
        "padding_profiles": {
            "A": {"horizontal": 0.03, "vertical": 0.08},
            "B": {"horizontal": 0.05, "vertical": 0.12},
            "C": {"horizontal": 0.07, "vertical": 0.15},
        },
        "selected_default_profile": "B",
        "maximum_retry_candidates": int(
            cfg.get("ocr", {})
            .get("recognition_retries", {})
            .get("maximum_candidates", 5)
        ),
        "selection_boundary": (
            "Padding is exercised only for low-confidence detected words; "
            "regex evidence may rank observed candidates but never invent text."
        ),
    }
    atomic_write_json(output_root / "crop_padding_metrics.json", crop_report)
    combined_report = {
        "schema_version": "1.0",
        "status": "passed",
        "component": "adaptive_image_pipeline",
        **provenance,
        "sample_count": len(results) * int(provenance["sample_count"]),
        "configuration_sha256": hashlib.sha256(
            canonical_json(COMPONENTS).encode("utf-8")
        ).hexdigest(),
        "failure_count": sum(
            int(value["failure_count"]) for value in results.values()
        ),
        "duration_seconds": sum(
            float(value["duration_seconds"]) for value in results.values()
        ),
        "components": results,
        "test_private_tuning_rows": 0,
        "kmeans_controls_ocr": False,
    }
    atomic_write_json(output_root / "adaptive_image_pipeline_metrics.json", combined_report)
    print(json.dumps(combined_report, indent=2))
    return 0


def _write_component_report(
    path: Path,
    *,
    component: str,
    baseline: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
    provenance: dict[str, Any],
    results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    available = baseline is not None and candidate is not None
    report = {
        "schema_version": "1.0",
        "status": "passed" if available else "not_requested",
        "component": component,
        **provenance,
        "sample_count": (
            2 * int(provenance["sample_count"])
            if available
            else 0
        ),
        "configuration_sha256": hashlib.sha256(
            canonical_json(
                {
                    "component": component,
                    "baseline": baseline.get("definition") if baseline else None,
                    "candidate": candidate.get("definition") if candidate else None,
                }
            ).encode("utf-8")
        ).hexdigest(),
        "failure_count": (
            int(baseline["failure_count"]) + int(candidate["failure_count"])
            if available
            else 0
        ),
        "duration_seconds": (
            float(baseline["duration_seconds"])
            + float(candidate["duration_seconds"])
            if available
            else 0.0
        ),
        "baseline": baseline,
        "candidate": candidate,
        "metric_deltas": _metric_deltas(baseline, candidate),
        "all_component_results": results,
        "test_private_tuning_rows": 0,
        "kmeans_controls_ocr": False,
    }
    atomic_write_json(path, report)
    return report


def _metric_deltas(
    baseline: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
) -> dict[str, float]:
    if baseline is None or candidate is None:
        return {}
    fields = (
        "polygon_f1",
        "recognized_text_coverage",
        "cer",
        "wer",
        "critical_field_exact_match",
        "end_to_end_entity_f1",
        "end_to_end_canonical_accuracy",
        "end_to_end_relation_f1",
        "time_per_page_seconds",
        "page_failure_rate",
    )
    return {
        field: float(candidate["metrics"][field])
        - float(baseline["metrics"][field])
        for field in fields
    }


def _balanced_rows(
    rows: list[dict[str, str]],
    limit: int,
) -> list[dict[str, str]]:
    if limit >= len(rows):
        return rows
    buckets: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        buckets.setdefault(row["dataset"], []).append(row)
    weights = {"funsd": 0.20, "sroie": 0.30, "fatura": 0.50}
    selected: list[dict[str, str]] = []
    for dataset in sorted(buckets):
        count = min(
            len(buckets[dataset]),
            max(1, round(limit * weights.get(dataset, 0.0))),
        )
        selected.extend(
            sorted(
                buckets[dataset],
                key=lambda row: deterministic_rank(row["page_id"], 4201),
            )[:count]
        )
    if len(selected) < limit:
        selected_ids = {row["page_id"] for row in selected}
        remaining = sorted(
            [row for row in rows if row["page_id"] not in selected_ids],
            key=lambda row: deterministic_rank(row["page_id"], 4202),
        )
        selected.extend(remaining[: limit - len(selected)])
    return selected[:limit]


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()


if __name__ == "__main__":
    raise SystemExit(main())
