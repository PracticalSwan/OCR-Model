#!/usr/bin/env python3
"""Run local Gmail private-test inference and write aggregate-only metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    parser.add_argument(
        "--limit",
        type=int,
        default=2,
        help="maximum anonymous source documents, not page rows",
    )
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
    documents = _group_private_documents(rows, limit=args.limit)
    private_output_root = (
        asset_root / "private-evaluation" / "ocr-upgrade-final"
    )
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
        layout_checkpoint=checkpoint,
        calibration_path=calibration,
        enable_kmeans_display=False,
        require_layout_model=True,
        ocr_profile=profile,
    )
    results: list[dict[str, Any]] = []
    local_records: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for index, document_rows in enumerate(documents, start=1):
            pages: list[DocumentPage] = []
            try:
                for page_number, row in enumerate(document_rows, start=1):
                    with Image.open(
                        PROJECT_ROOT / row["image_path"]
                    ) as source:
                        image = source.convert("RGB")
                    pages.append(DocumentPage(page_number, image))
                result = pipeline.extract_pages(
                    document_id=f"private_{index:06d}",
                    source_type="pdf" if len(pages) > 1 else "image",
                    pages=pages,
                    language="auto",
                    private_output=True,
                )
                if len(result.get("pages") or []) != len(document_rows):
                    raise RuntimeError(
                        "private multipage result did not preserve page count"
                    )
                results.append(result)
                local_records.append(
                    {
                        "anonymous_document_id": f"private_{index:06d}",
                        "page_count": len(document_rows),
                        "status": "passed",
                        "error_type": None,
                    }
                )
            except Exception as exc:
                local_records.append(
                    {
                        "anonymous_document_id": f"private_{index:06d}",
                        "page_count": len(document_rows),
                        "status": "failed",
                        "error_type": type(exc).__name__,
                    }
                )
            finally:
                for page in pages:
                    page.image.close()
    finally:
        pipeline.close()
    elapsed = time.perf_counter() - started
    attempted_documents = len(documents)
    attempted_pages = sum(len(rows) for rows in documents)
    aggregate = _private_operation_aggregate(
        results,
        attempted_documents=attempted_documents,
        attempted_pages=attempted_pages,
        elapsed_seconds=elapsed,
    )
    checkpoint_sha256 = sha256_file(
        checkpoint / "model.safetensors"
    )
    calibration_sha256 = sha256_file(calibration)
    atomic_write_json(
        private_output_root / "operational_status.json",
        {
            "schema_version": "1.0",
            "local_private_detail": True,
            "contains_source_filenames": False,
            "contains_ocr_text": False,
            "contains_images": False,
            "manifest_sha256": sha256_file(private_manifest),
            "checkpoint_sha256": checkpoint_sha256,
            "calibration_sha256": calibration_sha256,
            "ocr_stack": stack,
            "attempted_documents": attempted_documents,
            "attempted_pages": attempted_pages,
            "records": local_records,
        },
    )
    report = {
        "schema_version": "1.0",
        "status": "private_test_aggregate",
        "build_id": "private-operational-"
        + hashlib.sha256(
            canonical_json(
                {
                    "private_manifest_sha256": sha256_file(private_manifest),
                    "checkpoint_sha256": checkpoint_sha256,
                    "calibration_sha256": calibration_sha256,
                    "ocr_stack": stack,
                    "attempted_documents": attempted_documents,
                    "attempted_pages": attempted_pages,
                }
            ).encode("utf-8")
        ).hexdigest()[:16],
        "split": "private_operational",
        "manifest_sha256": sha256_file(private_manifest),
        "detector_sha256": stack["detector_sha256"],
        "recognizer_sha256": stack["recognizer_sha256"],
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_model_sha256": checkpoint_sha256,
        "calibration_sha256": calibration_sha256,
        "configuration_sha256": stack["preprocessing_sha256"],
        "source_commit": _git_commit(),
        "device": args.device,
        "sample_count": attempted_pages,
        "failure_count": aggregate["failed_pages"],
        "duration_seconds": elapsed,
        "private_row_count": attempted_pages,
        **aggregate,
        "gmail_fit_rows": 0,
        "local_processing_only": True,
        "contains_filenames": False,
        "contains_ocr_text": False,
        "contains_images": False,
        "contains_per_document_predictions": False,
        "limitations": [
            "No private ground truth is used; this is operational aggregate testing, not accuracy evaluation.",
            "The public report contains only anonymous counts and aggregate processing time.",
        ],
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
    return 0 if attempted_documents and results else 1


def _group_private_documents(
    rows: Sequence[Mapping[str, str]],
    *,
    limit: int,
) -> list[list[dict[str, str]]]:
    """Group private page rows without exposing source identity in reports."""
    if limit < 1:
        raise ValueError("private document limit must be positive")
    grouped: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        document_id = str(row.get("document_id", "")).strip()
        page_id = str(row.get("page_id", "")).strip()
        if not document_id or not page_id:
            raise ValueError("private manifest row lacks document/page identity")
        grouped[document_id].append(dict(row))
    ordered_ids = sorted(
        grouped,
        key=lambda value: deterministic_rank(value, 42),
    )[:limit]
    return [
        sorted(grouped[document_id], key=lambda row: row["page_id"])
        for document_id in ordered_ids
    ]


def _private_operation_aggregate(
    results: Sequence[Mapping[str, Any]],
    *,
    attempted_documents: int,
    attempted_pages: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Return only the private aggregate fields permitted for publication."""
    successful_documents = len(results)
    successful_pages = sum(
        len(result.get("pages") or []) for result in results
    )
    nonempty_output_count = sum(
        bool(
            str(page.get("full_text", "")).strip()
            or (page.get("ocr") or {}).get("words")
        )
        for result in results
        for page in (result.get("pages") or [])
    )
    aggregate_processing_time = sum(
        float((result.get("processing") or {}).get("duration_seconds", 0.0))
        for result in results
    )
    return {
        "attempted_documents": int(attempted_documents),
        "successful_documents": successful_documents,
        "failed_documents": max(
            0, int(attempted_documents) - successful_documents
        ),
        "attempted_pages": int(attempted_pages),
        "successful_pages": successful_pages,
        "failed_pages": max(0, int(attempted_pages) - successful_pages),
        "processed_pages": successful_pages,
        "nonempty_output_count": nonempty_output_count,
        "aggregate_processing_time_seconds": aggregate_processing_time,
        "aggregate_wall_time_seconds": float(elapsed_seconds),
    }


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()


if __name__ == "__main__":
    raise SystemExit(main())
