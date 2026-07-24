#!/usr/bin/env python3
"""Evaluate one PaddleOCR detector on the frozen 400-page DEV_SELECT benchmark."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.ocr_reporting import build_metric_report  # noqa: E402
from src.ocr.benchmark import resolve_project_input  # noqa: E402
from src.ocr.training_data import (  # noqa: E402
    convert_detection_annotation,
    critical_fields_by_token,
)
from src.ocr.trials import (  # noqa: E402
    aggregate_detector_results,
    detector_page_result,
    sha256_tree,
)
from src.rotation_common import (  # noqa: E402
    atomic_write_json,
    atomic_write_text,
    canonical_json,
    sha256_file,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-name", default="PP-OCRv6_medium_det")
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--benchmark-manifest",
        default=str(PROJECT_ROOT / "data/metadata/ocr_benchmark_manifest.csv"),
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--recognizer-sha256", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    manifest_path = Path(args.benchmark_manifest)
    config_path = Path(args.config)
    model_dir = Path(args.model_dir)
    rows = _load_benchmark(manifest_path)
    detector_sha = sha256_tree(model_dir)
    configuration_sha = sha256_file(config_path)
    source_commit = _git_commit()
    build_material = canonical_json(
        {
            "trial_id": args.trial_id,
            "manifest": sha256_file(manifest_path),
            "detector": detector_sha,
            "config": configuration_sha,
        }
    )
    build_id = "ocr-detector-eval-" + hashlib.sha256(
        build_material.encode("utf-8")
    ).hexdigest()[:16]

    from paddleocr import TextDetection

    model = TextDetection(
        model_name=args.model_name,
        model_dir=str(model_dir),
        device=args.device,
    )
    _reset_peak_memory()
    page_results: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for offset in range(0, len(rows), args.batch_size):
        group = rows[offset : offset + args.batch_size]
        prepared = [_prepare_page(row) for row in group]
        paths = [str(item["image_path"]) for item in prepared]
        started = time.perf_counter()
        try:
            outputs = list(model.predict(input=paths, batch_size=len(paths)))
            if len(outputs) != len(group):
                raise RuntimeError(
                    f"detector returned {len(outputs)} results for {len(group)} pages"
                )
            elapsed_per_page = (time.perf_counter() - started) / len(group)
            for item, output in zip(prepared, outputs):
                polygons, scores = _detector_payload(output)
                _append_page(
                    item,
                    polygons=polygons,
                    scores=scores,
                    duration_seconds=elapsed_per_page,
                    error=None,
                    page_results=page_results,
                    prediction_rows=prediction_rows,
                )
        except Exception as batch_error:
            for item in prepared:
                single_started = time.perf_counter()
                try:
                    output = list(
                        model.predict(input=str(item["image_path"]), batch_size=1)
                    )
                    if len(output) != 1:
                        raise RuntimeError(
                            f"detector returned {len(output)} results for one page"
                        )
                    polygons, scores = _detector_payload(output[0])
                    error = None
                except Exception as single_error:
                    polygons, scores = [], []
                    error = (
                        f"batch={type(batch_error).__name__}: {batch_error}; "
                        f"single={type(single_error).__name__}: {single_error}"
                    )
                _append_page(
                    item,
                    polygons=polygons,
                    scores=scores,
                    duration_seconds=time.perf_counter() - single_started,
                    error=error,
                    page_results=page_results,
                    prediction_rows=prediction_rows,
                )
        completed = min(offset + len(group), len(rows))
        print(f"detector DEV_SELECT: {completed}/{len(rows)}", flush=True)

    metrics = aggregate_detector_results(page_results)
    prediction_text = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in prediction_rows
    )
    prediction_path = Path(args.predictions)
    atomic_write_text(prediction_path, prediction_text)
    prediction_sha = sha256_file(prediction_path)
    metrics.update(
        {
            "trial_id": args.trial_id,
            "iou_threshold": 0.5,
            "small_text_definition": "polygon height <= 5% of page height",
            "medium_text_definition": "5% < polygon height <= 20% of page height",
            "large_text_definition": "polygon height > 20% of page height",
            "predictions_path": prediction_path.as_posix(),
            "predictions_sha256": prediction_sha,
            "model_directory": model_dir.as_posix(),
            "adapter_contract_passed": all(
                _valid_prediction_row(row) for row in prediction_rows
            ),
            "peak_gpu_memory_bytes": _peak_memory_bytes(),
        }
    )
    report = build_metric_report(
        build_id=build_id,
        split="dev_select",
        manifest_sha256=sha256_file(manifest_path),
        detector_sha256=detector_sha,
        recognizer_sha256=args.recognizer_sha256,
        checkpoint_sha256=args.checkpoint_sha256,
        calibration_sha256=None,
        configuration_sha256=configuration_sha,
        source_commit=source_commit,
        device=args.device,
        sample_count=len(rows),
        failure_count=int(metrics["failure_count"]),
        duration_seconds=float(metrics["duration_seconds"]),
        private_row_count=0,
        metrics=metrics,
    )
    report["model_name"] = args.model_name
    report["environment"] = _environment_versions()
    atomic_write_json(Path(args.output), report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if int(metrics["failure_count"]) == 0 else 1


def _load_benchmark(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 400:
        raise ValueError(f"frozen detector benchmark must contain 400 pages, found {len(rows)}")
    for row in rows:
        if row.get("split") != "dev_select":
            raise ValueError(f"non-DEV_SELECT benchmark row: {row.get('page_id')}")
        if str(row.get("is_private", "")).casefold() != "false":
            raise ValueError(f"private benchmark row refused: {row.get('page_id')}")
    return rows


def _prepare_page(row: dict[str, str]) -> dict[str, Any]:
    _, image_path = resolve_project_input(
        PROJECT_ROOT, row["source_image_path"], label="benchmark image"
    )
    _, annotation_path = resolve_project_input(
        PROJECT_ROOT,
        row["normalized_annotation_path"],
        label="benchmark annotation",
    )
    if sha256_file(image_path) != row["source_image_sha256"].casefold():
        raise ValueError(f"benchmark image hash drift: {row['page_id']}")
    if sha256_file(annotation_path) != row["annotation_sha256"].casefold():
        raise ValueError(f"benchmark annotation hash drift: {row['page_id']}")
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    if annotation.get("page_id") != row["page_id"]:
        raise ValueError(f"benchmark annotation page mismatch: {row['page_id']}")
    regions = convert_detection_annotation(annotation)
    references = [
        {
            "polygon": region["points"],
            "token_ids": region.get("token_ids", []),
        }
        for region in regions
    ]
    critical_map = critical_fields_by_token(annotation)
    critical_ids = {
        token_id for token_id, fields in critical_map.items() if fields
    }
    return {
        "row": row,
        "image_path": image_path,
        "references": references,
        "critical_ids": critical_ids,
    }


def _detector_payload(output: Any) -> tuple[list[list[list[float]]], list[float]]:
    payload = getattr(output, "json", output)
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise TypeError(f"unsupported detector result: {type(payload).__name__}")
    result = payload.get("res", payload)
    polygons = result.get("dt_polys", [])
    scores = result.get("dt_scores", [])
    normalized = [
        [[float(point[0]), float(point[1])] for point in polygon]
        for polygon in polygons
    ]
    normalized_scores = [float(score) for score in scores]
    if normalized_scores and len(normalized_scores) != len(normalized):
        raise ValueError("detector polygon and score counts differ")
    if not normalized_scores:
        normalized_scores = [0.0] * len(normalized)
    return normalized, normalized_scores


def _append_page(
    item: dict[str, Any],
    *,
    polygons: list[list[list[float]]],
    scores: list[float],
    duration_seconds: float,
    error: str | None,
    page_results: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> None:
    predictions = [{"polygon": polygon} for polygon in polygons]
    result = detector_page_result(
        item["references"],
        predictions,
        page_width=int(item["row"]["width"]),
        page_height=int(item["row"]["height"]),
        critical_token_ids=item["critical_ids"],
        duration_seconds=duration_seconds,
    )
    result["failed"] = error is not None
    page_results.append(result)
    prediction_rows.append(
        {
            "page_id": item["row"]["page_id"],
            "dataset": item["row"]["dataset"],
            "polygons": polygons,
            "scores": scores,
            "duration_seconds": duration_seconds,
            "error": error,
            "private": False,
        }
    )


def _valid_prediction_row(row: dict[str, Any]) -> bool:
    if row["error"] is not None or len(row["polygons"]) != len(row["scores"]):
        return False
    for polygon in row["polygons"]:
        if len(polygon) < 4:
            return False
        for point in polygon:
            if len(point) != 2 or not all(
                isinstance(value, (int, float)) for value in point
            ):
                return False
    return True


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()


def _environment_versions() -> dict[str, str]:
    import importlib.metadata

    return {
        "python": sys.version.split()[0],
        "paddle": _package_version("paddlepaddle-gpu", importlib.metadata),
        "paddleocr": _package_version("paddleocr", importlib.metadata),
        "paddlex": _package_version("paddlex", importlib.metadata),
    }


def _package_version(name: str, metadata: Any) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unavailable"


def _reset_peak_memory() -> None:
    try:
        import paddle

        paddle.device.cuda.reset_max_memory_allocated()
        paddle.device.cuda.reset_max_memory_reserved()
    except Exception:
        pass


def _peak_memory_bytes() -> int | None:
    try:
        import paddle

        return int(paddle.device.cuda.max_memory_allocated())
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
