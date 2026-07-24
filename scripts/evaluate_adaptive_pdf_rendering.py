#!/usr/bin/env python3
"""Evaluate 200-DPI, 300-DPI, and adaptive PDF rendering on public DEV_SELECT."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import normalized_text, ocr_text_metrics  # noqa: E402
from src.ocr.adaptive import (  # noqa: E402
    AdaptiveRenderingConfig,
    no_rerender_provenance,
    page_quality_signals,
    rerender_reasons,
    select_render_candidate,
)
from src.ocr.environment import configure_external_environment, require_storage_gate  # noqa: E402
from src.ocr.model_registry import ModelRegistry  # noqa: E402
from src.ocr.pipeline import MultilingualOCR  # noqa: E402
from src.ocr.training_data import convert_detection_annotation, critical_fields_by_token  # noqa: E402
from src.ocr.trials import aggregate_detector_results, detector_page_result  # noqa: E402
from src.rotation_common import atomic_write_json, canonical_json, sha256_file  # noqa: E402
from src.inference.document_io import load_document_pages  # noqa: E402
from scripts.evaluate_ocr_component_ablations import _balanced_rows  # noqa: E402
from scripts.run_large_ocr_ablation import (  # noqa: E402
    _best_field_candidates,
    _load_page,
    _load_rows,
)
from src.evaluation.critical_field_ocr import evaluate_critical_fields  # noqa: E402


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
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--source-dpi", type=int, default=300)
    parser.add_argument(
        "--derived-pdf",
        default=(
            "D:/CSX4201/vision-info-extraction-assets/evaluations/"
            "ocr_upgrade/adaptive_rendering/public_dev_select.pdf"
        ),
    )
    parser.add_argument(
        "--output",
        default=str(
            PROJECT_ROOT / "reports/ocr_upgrade/adaptive_rendering_metrics.json"
        ),
    )
    args = parser.parse_args()
    if not 1 <= args.limit <= 100:
        parser.error("--limit must be in [1, 100]")
    if not 150 <= args.source_dpi <= 600:
        parser.error("--source-dpi must be in [150, 600]")

    configure_external_environment()
    asset_root = Path("D:/CSX4201/vision-info-extraction-assets")
    require_storage_gate(
        asset_root,
        operation="public adaptive PDF rendering evaluation",
        anticipated_c_gib=0.1,
        anticipated_asset_gib=2.0,
    )
    manifest_path = Path(args.benchmark_manifest).resolve()
    rows = _balanced_rows(_load_rows(manifest_path), args.limit)
    derived_pdf = Path(args.derived_pdf)
    derived_pdf.parent.mkdir(parents=True, exist_ok=True)
    _create_public_pdf(rows, derived_pdf, source_dpi=args.source_dpi)
    _, _, pages_200 = load_document_pages(
        derived_pdf, pdf_dpi=200, max_pixels=60_000_000
    )
    _, _, pages_300 = load_document_pages(
        derived_pdf, pdf_dpi=300, max_pixels=60_000_000
    )
    if len(pages_200) != len(rows) or len(pages_300) != len(rows):
        raise RuntimeError("derived public PDF page count drifted")

    registry = ModelRegistry.from_setup(
        Path(args.model_setup).resolve(),
        upgrade_registry=Path(args.model_registry).resolve(),
    )
    detector, general = registry.route_models("general")
    _, thai = registry.route_models("thai")
    pipeline = MultilingualOCR(
        registry,
        device=args.device,
        cardinal_angles=(0,),
        preprocessing_version="3.0-adaptive-pdf-evaluation",
        preprocessing_profile="original",
        enable_fine_deskew=False,
        enable_tiling=False,
    )
    config = AdaptiveRenderingConfig()
    observations: dict[str, list[dict[str, Any]]] = {
        "dpi_200": [],
        "dpi_300": [],
        "adaptive": [],
    }
    provenance_rows = []
    started = time.perf_counter()
    for index, (row, page_200, page_300) in enumerate(
        zip(rows, pages_200, pages_300),
        start=1,
    ):
        _, annotation = _load_page(row)
        first = pipeline.extract_page(
            page_200.image,
            language_mode="general",
            metadata_language=row.get("language"),
        )
        second = pipeline.extract_page(
            page_300.image,
            language_mode="general",
            metadata_language=row.get("language"),
        )
        signals = page_quality_signals(
            first,
            image_width=page_200.image.width,
            image_height=page_200.image.height,
            critical_fields_expected=(
                str(row.get("has_amount", "")).casefold() == "true"
                or str(row.get("has_date", "")).casefold() == "true"
                or str(row.get("has_identifier", "")).casefold() == "true"
            ),
            table_like=str(row.get("has_table", "")).casefold() == "true",
        )
        reasons = rerender_reasons(signals, config)
        if reasons:
            selected, render_provenance = select_render_candidate(
                first,
                second,
                first_size=page_200.image.size,
                second_size=page_300.image.size,
                first_dpi=200,
                second_dpi=300,
                reasons=reasons,
                minimum_score_gain=config.minimum_score_gain,
            )
        else:
            selected = first
            render_provenance = no_rerender_provenance(
                first,
                image_size=page_200.image.size,
                dpi=200,
            )
        selected_dpi = int(render_provenance["selected_dpi"])
        observations["dpi_200"].append(
            _evaluate_result(row, annotation, first, page_200.image.size)
        )
        observations["dpi_300"].append(
            _evaluate_result(row, annotation, second, page_300.image.size)
        )
        observations["adaptive"].append(
            _evaluate_result(
                row,
                annotation,
                selected,
                page_300.image.size if selected_dpi == 300 else page_200.image.size,
            )
        )
        provenance_rows.append(
            {
                "page_id_hash": hashlib.sha256(
                    row["page_id"].encode("utf-8")
                ).hexdigest(),
                "triggered": bool(reasons),
                "selected_dpi": selected_dpi,
                "reasons": reasons,
                "first_pass_score": render_provenance["first_pass_score"],
                "second_pass_score": render_provenance.get("second_pass_score"),
            }
        )
        if index % 10 == 0 or index == len(rows):
            print(f"adaptive PDF rendering: {index}/{len(rows)}", flush=True)

    aggregates = {
        mode: _aggregate(values)
        for mode, values in observations.items()
    }
    trigger_count = sum(row["triggered"] for row in provenance_rows)
    selected_300 = sum(row["selected_dpi"] == 300 for row in provenance_rows)
    report = {
        "schema_version": "1.0",
        "status": "passed",
        "build_id": "adaptive-rendering-"
        + hashlib.sha256(
            canonical_json(
                {
                    "manifest": sha256_file(manifest_path),
                    "rows": [row["page_id"] for row in rows],
                    "config": config.as_dict(),
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
            canonical_json(config.as_dict()).encode("utf-8")
        ).hexdigest(),
        "source_commit": _git_commit(),
        "device": args.device,
        "sample_count": len(rows),
        "failure_count": sum(
            item["failed"]
            for values in observations.values()
            for item in values
        ),
        "duration_seconds": time.perf_counter() - started,
        "private_row_count": 0,
        "source_pdf": {
            "path": derived_pdf.as_posix(),
            "sha256": sha256_file(derived_pdf),
            "page_count": len(rows),
            "source_dpi": args.source_dpi,
            "public_only": True,
        },
        "rendering_config": config.as_dict(),
        "rerender_trigger_count": trigger_count,
        "rerender_trigger_rate": trigger_count / max(1, len(rows)),
        "selected_300_dpi_count": selected_300,
        "selected_300_dpi_rate": selected_300 / max(1, len(rows)),
        "reason_counts": dict(
            sorted(
                Counter(
                    reason
                    for row in provenance_rows
                    for reason in row["reasons"]
                ).items()
            )
        ),
        "metrics": aggregates,
        "adaptive_vs_200_deltas": _metric_deltas(
            aggregates["dpi_200"], aggregates["adaptive"]
        ),
        "dpi_300_vs_200_deltas": _metric_deltas(
            aggregates["dpi_200"], aggregates["dpi_300"]
        ),
        "page_selection_evidence": provenance_rows,
        "test_private_tuning_rows": 0,
        "kmeans_controls_ocr": False,
        "limitations": [
            "The PDF is derived only from public DEV_SELECT images at a declared 300-DPI physical scale.",
            "This isolates raster render resolution; vector-PDF behavior is covered by integration tests.",
        ],
    }
    atomic_write_json(Path(args.output), report)
    print(json.dumps(report, indent=2))
    return 0


def _create_public_pdf(
    rows: list[dict[str, str]],
    output: Path,
    *,
    source_dpi: int,
) -> None:
    import fitz

    document = fitz.open()
    try:
        for row in rows:
            image_path, _ = _load_page(row)
            width = int(row["width"])
            height = int(row["height"])
            page = document.new_page(
                width=width * 72.0 / source_dpi,
                height=height * 72.0 / source_dpi,
            )
            page.insert_image(page.rect, filename=str(image_path))
        document.save(output, garbage=4, deflate=True)
    finally:
        document.close()


def _evaluate_result(
    row: Mapping[str, str],
    annotation: Mapping[str, Any],
    result: Mapping[str, Any],
    image_size: tuple[int, int],
) -> dict[str, Any]:
    width, height = image_size
    source_width = int(row["width"])
    source_height = int(row["height"])
    scale_x = width / max(1, source_width)
    scale_y = height / max(1, source_height)
    references = [
        {
            "polygon": [
                [float(x) * scale_x, float(y) * scale_y]
                for x, y in region["points"]
            ],
            "token_ids": region.get("token_ids", []),
        }
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
            for word in result.get("words") or []
        ],
        page_width=width,
        page_height=height,
        critical_token_ids=critical_ids,
        duration_seconds=float(result.get("duration_seconds", 0.0)),
    )
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
        "failed": False,
        "detection": detection,
        "text": text,
        "reference_words": len(normalized_text(reference_text).split()),
        "reference_fields": references_fields,
        "predicted_fields": predictions_fields,
        "duration_seconds": float(result.get("duration_seconds", 0.0)),
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
    duration = sum(float(row["duration_seconds"]) for row in rows)
    return {
        "polygon_precision": detector["precision"],
        "polygon_recall": detector["recall"],
        "polygon_f1": detector["f1"],
        "recognized_text_coverage": max(
            0.0, 1.0 - character_errors / max(1, reference_characters)
        ),
        "cer": character_errors / max(1, reference_characters),
        "wer": word_errors / max(1, reference_words),
        "critical_field_exact_match": critical["aggregate"][
            "critical_exact_match"
        ],
        "critical_field_count": critical["evaluated_field_count"],
        "nonempty_output_rate": sum(
            not bool(row["text"]["empty_output"]) for row in rows
        )
        / max(1, len(rows)),
        "time_per_page_seconds": duration / max(1, len(rows)),
        "page_failure_rate": 0.0,
    }


def _metric_deltas(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, float]:
    return {
        field: float(candidate[field]) - float(baseline[field])
        for field in (
            "polygon_f1",
            "recognized_text_coverage",
            "cer",
            "wer",
            "critical_field_exact_match",
            "nonempty_output_rate",
            "time_per_page_seconds",
            "page_failure_rate",
        )
    }


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()


if __name__ == "__main__":
    raise SystemExit(main())
