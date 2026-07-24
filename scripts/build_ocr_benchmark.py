#!/usr/bin/env python3
"""Build the deterministic public DEV_SELECT OCR benchmark."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config as cfgmod  # noqa: E402
from src.ocr.benchmark import (  # noqa: E402
    BENCHMARK_COLUMNS,
    analyze_benchmark_page,
    benchmark_build_id,
    select_benchmark_rows,
    validate_benchmark_records,
)
from src.ocr.environment import require_storage_gate  # noqa: E402
from src.rotation_common import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    read_csv_rows,
    sha256_file,
)


DATASET_LICENSES = {
    "fatura": {
        "id": "CC-BY-4.0-Zenodo-8261508",
        "source": "https://zenodo.org/records/8261508",
        "note": "Zenodo record declares Creative Commons Attribution 4.0.",
    },
    "funsd": {
        "id": "FUNSD-EPFL-NC-RESEARCH-EDUCATION",
        "source": "https://guillaumejaume.github.io/FUNSD/work/",
        "note": "Official terms restrict use to non-commercial research and education.",
    },
    "sroie": {
        "id": "MIT-ZZZDAVID-ICDAR2019-SROIE-REPOSITORY",
        "source": "https://github.com/zzzDavid/ICDAR-2019-SROIE",
        "note": (
            "The public source repository containing the corrected dataset declares MIT; "
            "the original challenge may retain additional underlying rights."
        ),
    },
}


def write_benchmark_outputs(
    records: Sequence[Mapping[str, Any]],
    *,
    manifest_path: Path,
    summary_path: Path,
    selection_path: Path,
    seed: int,
    source_commit: str,
    available_counts: Mapping[str, int],
    storage_gate: Mapping[str, Any],
    minimum_total: int = 400,
    minimum_fatura: int = 283,
    duration_seconds: float = 0.0,
) -> dict[str, Any]:
    """Atomically persist a hash-bound manifest and its public-safe reports."""
    selected = [dict(record) for record in records]
    counts = validate_benchmark_records(
        selected,
        minimum_total=minimum_total,
        minimum_fatura=minimum_fatura,
        expected_all_counts=available_counts,
    )
    build_id = benchmark_build_id(selected, seed=seed)
    for record in selected:
        record["build_id"] = build_id
    atomic_write_csv(manifest_path, selected, BENCHMARK_COLUMNS)
    manifest_hash = sha256_file(manifest_path)
    summary = {
        "schema_version": "1.0",
        "status": "passed",
        "build_id": build_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "dev_select",
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": manifest_hash,
        "source_commit": source_commit,
        "selection_seed": int(seed),
        "sample_count": len(selected),
        "failure_count": 0,
        "duration_seconds": float(duration_seconds),
        "private_row_count": 0,
        "counts_by_dataset": counts,
        "available_dev_select_counts": {
            key: int(value) for key, value in sorted(available_counts.items())
        },
        "counts_by_document_type": _counts(selected, "document_type"),
        "counts_by_language": _counts(selected, "language"),
        "counts_by_quality": _counts(selected, "quality_bucket"),
        "counts_by_text_density": _counts(selected, "text_density_bucket"),
        "counts_by_character_size": _counts(selected, "character_size_bucket"),
        "counts_by_resolution": _counts(selected, "resolution_bucket"),
        "counts_by_diagnostic_rotation": _counts(
            selected, "diagnostic_rotation_degrees"
        ),
        "field_presence": {
            field: sum(str(row.get(field, "")).casefold() == "true" for row in selected)
            for field in ("has_table", "has_amount", "has_date", "has_identifier")
        },
        "document_family_count": len(
            {str(row.get("document_family_id", "")) for row in selected}
        ),
        "exact_source_hash_duplicate_groups": _duplicate_group_count(
            selected, "source_image_sha256"
        ),
        "license_sources": DATASET_LICENSES,
        "storage_gate": dict(storage_gate),
        "selection_policy": {
            "all_funsd_dev_select": True,
            "all_sroie_dev_select": True,
            "minimum_fatura": int(minimum_fatura),
            "minimum_total": int(minimum_total),
            "fatura_strategy": (
                "exact-hash unique first, then deterministic round-robin across "
                "family/document/quality/density/size/resolution/table/field strata"
            ),
        },
    }
    atomic_write_json(summary_path, summary)
    atomic_write_text(selection_path, _selection_markdown(summary))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--fatura-count", type=int, default=283)
    parser.add_argument("--minimum-total", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="data/metadata/ocr_benchmark_manifest.csv",
    )
    parser.add_argument(
        "--summary",
        default="reports/ocr_upgrade/benchmark_summary.json",
    )
    parser.add_argument(
        "--selection-report",
        default="reports/ocr_upgrade/benchmark_selection.md",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()

    cfg = cfgmod.load_config(args.config)
    root = cfgmod.project_root(cfg)
    metadata = cfgmod.resolve_path(cfg, "metadata")
    asset_root = cfgmod.resolve_path(cfg, "external_assets")
    storage = require_storage_gate(
        asset_root,
        operation="OCR benchmark construction",
        anticipated_c_gib=0.05,
        anticipated_asset_gib=0.05,
    )
    manifest_path = _output_path(root, args.output)
    summary_path = _output_path(root, args.summary)
    selection_path = _output_path(root, args.selection_report)
    existing = [
        path for path in (manifest_path, summary_path, selection_path) if path.exists()
    ]
    if existing and not args.force:
        raise SystemExit(
            "benchmark outputs already exist; rerun with --force after reviewing: "
            + ", ".join(str(path) for path in existing)
        )

    split_manifest = metadata / "information_extraction_split_manifest.csv"
    rows = [
        row
        for row in read_csv_rows(split_manifest)
        if row.get("project_split") == "dev_select"
        and row.get("dataset", "").casefold() in DATASET_LICENSES
        and row.get("is_private", "").casefold() == "false"
        and row.get("is_usable", "").casefold() == "true"
    ]
    available_counts = Counter(row["dataset"].casefold() for row in rows)
    analyzed = [
        analyze_benchmark_page(
            row,
            project_root=root,
            license_id=DATASET_LICENSES[row["dataset"].casefold()]["id"],
        )
        for row in sorted(rows, key=lambda value: value["page_id"])
    ]
    selected = select_benchmark_rows(
        analyzed,
        fatura_count=args.fatura_count,
        seed=args.seed,
    )
    source_commit = _git_revision(root)
    summary = write_benchmark_outputs(
        selected,
        manifest_path=manifest_path,
        summary_path=summary_path,
        selection_path=selection_path,
        seed=args.seed,
        source_commit=source_commit,
        available_counts=available_counts,
        storage_gate=storage,
        minimum_total=args.minimum_total,
        minimum_fatura=args.fatura_count,
        duration_seconds=time.perf_counter() - started,
    )
    print(json.dumps(summary, indent=2))
    return 0


def _counts(records: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(
        sorted(
            Counter(str(record.get(key, "")) for record in records).items()
        )
    )


def _duplicate_group_count(
    records: Sequence[Mapping[str, Any]], key: str
) -> int:
    counts = Counter(str(record.get(key, "")) for record in records)
    return sum(value > 1 for value in counts.values())


def _selection_markdown(summary: Mapping[str, Any]) -> str:
    counts = summary["counts_by_dataset"]
    lines = [
        "# OCR Benchmark Selection",
        "",
        f"Build: `{summary['build_id']}`",
        "",
        f"Manifest SHA-256: `{summary['manifest_sha256']}`",
        "",
        "## Selection",
        "",
        f"- Total DEV_SELECT pages: {summary['sample_count']}",
        f"- FUNSD: {counts.get('funsd', 0)} (all available)",
        f"- SROIE: {counts.get('sroie', 0)} (all available)",
        f"- FATURA: {counts.get('fatura', 0)} (deterministic stratified sample)",
        f"- Private rows: {summary['private_row_count']}",
        "",
        "FATURA selection gives exact source hashes first priority and then uses "
        "seeded round-robin coverage across template family, document type, "
        "quality, density, character size, resolution, table, amount, date, "
        "and identifier strata.",
        "",
        "A deterministic subset receives a 37-degree diagnostic transform flag; "
        "the raw source image is never changed or copied into a raw directory.",
        "",
        "## Governance",
        "",
        "Only DEV_SELECT pages are present. This benchmark may select OCR models "
        "and inference settings but may not fit weights or calibration. "
        "DEV_CALIBRATION, TEST_IN_DOMAIN, CORU, and Gmail are absent.",
        "",
        "## License scope",
        "",
    ]
    for dataset, entry in sorted(summary["license_sources"].items()):
        lines.append(
            f"- {dataset.upper()}: `{entry['id']}` — {entry['note']} "
            f"Source: {entry['source']}"
        )
    lines.extend(
        [
            "",
            "The manifest contains paths and hashes only. Raw images, normalized "
            "annotations, benchmark renderings, and predictions are not committed "
            "by this selection step.",
            "",
        ]
    )
    return "\n".join(lines)


def _output_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _git_revision(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
