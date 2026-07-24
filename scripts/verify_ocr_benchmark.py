#!/usr/bin/env python3
"""Verify the hash-bound public OCR benchmark without reading private data."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config as cfgmod  # noqa: E402
from src.ocr.benchmark import (  # noqa: E402
    benchmark_build_id,
    resolve_project_input,
    validate_benchmark_records,
)
from src.rotation_common import read_csv_rows, sha256_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument(
        "--manifest",
        default="data/metadata/ocr_benchmark_manifest.csv",
    )
    parser.add_argument("--minimum-total", type=int, default=400)
    parser.add_argument("--minimum-fatura", type=int, default=283)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = cfgmod.load_config(args.config)
    root = cfgmod.project_root(cfg)
    metadata = cfgmod.resolve_path(cfg, "metadata")
    manifest_path = _resolve(root, args.manifest)
    records = read_csv_rows(manifest_path)
    available = Counter(
        row["dataset"].casefold()
        for row in read_csv_rows(
            metadata / "information_extraction_split_manifest.csv"
        )
        if row.get("project_split") == "dev_select"
        and row.get("dataset", "").casefold() in {"fatura", "funsd", "sroie"}
        and row.get("is_private", "").casefold() == "false"
        and row.get("is_usable", "").casefold() == "true"
    )
    counts = validate_benchmark_records(
        records,
        minimum_total=args.minimum_total,
        minimum_fatura=args.minimum_fatura,
        expected_all_counts=available,
    )
    build_ids = {row.get("build_id", "") for row in records}
    if len(build_ids) != 1 or "" in build_ids:
        raise ValueError(f"benchmark build IDs are missing or mixed: {build_ids}")
    expected_build = benchmark_build_id(records, seed=args.seed)
    if build_ids != {expected_build}:
        raise ValueError(
            f"benchmark build ID mismatch: {build_ids} != {expected_build}"
        )

    errors: list[str] = []
    for row in records:
        _, image_path = resolve_project_input(
            root,
            row["source_image_path"],
            label="source image",
            required_prefix="data/raw/public",
        )
        _, annotation_path = resolve_project_input(
            root,
            row["normalized_annotation_path"],
            label="normalized annotation",
            required_prefix="data/processed/normalized_ie_annotations",
            allow_prefix_junction=True,
        )
        if sha256_file(image_path) != row["source_image_sha256"]:
            errors.append(f"source image hash mismatch: {row['page_id']}")
        if sha256_file(annotation_path) != row["annotation_sha256"]:
            errors.append(f"annotation hash mismatch: {row['page_id']}")
        if int(float(row.get("diagnostic_rotation_degrees", 0))) not in {0, 37}:
            errors.append(f"unexpected diagnostic rotation: {row['page_id']}")
    if errors:
        raise ValueError("; ".join(errors[:20]))

    result = {
        "schema_version": "1.0",
        "status": "passed",
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "build_id": expected_build,
        "split": "dev_select",
        "sample_count": len(records),
        "failure_count": 0,
        "private_row_count": 0,
        "counts_by_dataset": counts,
        "available_dev_select_counts": dict(sorted(available.items())),
        "source_files_rehashed": len(records),
        "annotation_files_rehashed": len(records),
    }
    print(json.dumps(result, indent=2))
    return 0


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    resolved = (path if path.is_absolute() else root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path escapes project root: {value}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


if __name__ == "__main__":
    raise SystemExit(main())
