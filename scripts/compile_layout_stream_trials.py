#!/usr/bin/env python3
"""Compile public OCR-v2 LayoutXLM stream manifests into one auditable ledger."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config as cfgmod  # noqa: E402
from src.ocr.model_registry import ModelRegistry  # noqa: E402
from src.ocr.stack_binding import build_ocr_stack_binding  # noqa: E402
from src.rotation_common import (  # noqa: E402
    atomic_write_text,
    configuration_hash,
    read_csv_rows,
    sha256_file,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument(
        "--trial",
        action="append",
        required=True,
        metavar="LABEL|OCR_PROFILE|MANIFEST",
        help="repeat for A, B, and the existing C baseline",
    )
    parser.add_argument("--selected-profile")
    parser.add_argument(
        "--model-setup",
        default=str(PROJECT_ROOT / "reports" / "ocr" / "model_setup.json"),
    )
    parser.add_argument(
        "--model-registry",
        default=str(PROJECT_ROOT / "reports" / "ocr_upgrade" / "model_registry.json"),
    )
    parser.add_argument(
        "--checkpoint",
        help=(
            "layout checkpoint override; defaults to "
            "layout_model.inference_checkpoint from --config"
        ),
    )
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument(
        "--output",
        default=str(
            PROJECT_ROOT / "reports" / "ocr_upgrade" / "layout_stream_trials.csv"
        ),
    )
    args = parser.parse_args()

    trials = [_parse_trial(value) for value in args.trial]
    labels = [label for label, _, _ in trials]
    if len(labels) != len(set(labels)):
        parser.error("trial labels must be unique")
    if args.selected_profile and args.selected_profile not in set(labels):
        parser.error("--selected-profile must match a supplied trial label")

    cfg = cfgmod.load_config(args.config)
    try:
        checkpoint = resolve_layout_checkpoint(cfg, args.checkpoint)
    except ValueError as exc:
        parser.error(str(exc))
    checkpoint_model = checkpoint / "model.safetensors"
    if not checkpoint_model.is_file():
        raise SystemExit(f"checkpoint model is missing: {checkpoint_model}")
    source_commit = _git_commit()
    rows = []
    for label, ocr_profile, manifest_path in trials:
        started = time.perf_counter()
        registry = _registry_for_profile(
            args.model_setup,
            args.model_registry,
            ocr_profile=ocr_profile,
        )
        stack = build_ocr_stack_binding(
            cfg,
            registry,
            ocr_profile=ocr_profile,
        )
        rows.append(
            summarize_stream_manifest(
                manifest_path,
                trial_id=label,
                ocr_profile=ocr_profile,
                selected=label == args.selected_profile,
                ocr_stack=stack,
                checkpoint_sha256=sha256_file(checkpoint_model),
                source_commit=source_commit,
                device=args.device,
                duration_seconds=time.perf_counter() - started,
            )
        )
    _write_csv(Path(args.output), rows)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "trial_count": len(rows),
                "selected_profile": args.selected_profile,
            },
            indent=2,
        )
    )
    return 0


def resolve_layout_checkpoint(
    cfg: Mapping[str, Any],
    override: str | Path | None,
) -> Path:
    if override is not None:
        return Path(override).expanduser().resolve()
    configured = cfg.get("layout_model", {}).get("inference_checkpoint")
    if not configured:
        raise ValueError(
            "layout_model.inference_checkpoint must be configured when "
            "--checkpoint is omitted"
        )
    checkpoint = Path(str(configured)).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = cfgmod.project_root(cfg) / checkpoint
    return checkpoint.resolve()


def summarize_stream_manifest(
    manifest_path: str | Path,
    *,
    trial_id: str,
    ocr_profile: str,
    selected: bool,
    ocr_stack: Mapping[str, Any],
    checkpoint_sha256: str,
    source_commit: str,
    device: str,
    duration_seconds: float,
) -> dict[str, Any]:
    path = Path(manifest_path).resolve()
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"model stream manifest is empty: {path}")
    build_ids = {row.get("build_id", "") for row in rows}
    if len(build_ids) != 1 or "" in build_ids:
        raise ValueError(f"manifest build IDs are missing or mixed: {path}")
    private_rows = sum(
        str(row.get("is_private", "")).casefold() != "false" for row in rows
    )
    if private_rows:
        raise ValueError(f"private or unmarked row found in stream trial: {path}")

    usable = [row for row in rows if row.get("is_usable") == "true"]
    counts = Counter(
        (row.get("project_split", ""), row.get("token_source", ""))
        for row in usable
    )
    train_total = sum(
        count for (split, _), count in counts.items() if split == "train"
    )
    dev_total = sum(
        count for (split, _), count in counts.items() if split == "dev_select"
    )
    forbidden_variant_count = sum(
        count
        for (split, stream), count in counts.items()
        if split not in {"train", "dev_select"}
        and stream in {"paddleocr", "hybrid", "ocr_noise"}
    )
    if forbidden_variant_count:
        raise ValueError(
            "OCR-derived variants are allowed only on TRAIN and DEV_SELECT: "
            f"{path} contains {forbidden_variant_count} forbidden rows"
        )
    ocr_rows = [
        row
        for row in usable
        if row.get("token_source") in {"paddleocr", "hybrid"}
    ]
    configuration_sha256 = configuration_hash(
        {
            "schema_version": "1.0",
            "trial_id": trial_id,
            "ocr_profile": ocr_profile,
            "manifest_sha256": sha256_file(path),
            "ocr_stack_binding": dict(ocr_stack),
        }
    )
    return {
        "trial_id": trial_id,
        "status": "passed",
        "selected": bool(selected),
        "ocr_profile": ocr_profile,
        "build_id": next(iter(build_ids)),
        "split": "train_and_dev_select",
        "manifest_path": str(path),
        "manifest_sha256": sha256_file(path),
        "detector_sha256": ocr_stack["detector_sha256"],
        "recognizer_sha256": ocr_stack["recognizer_sha256"],
        "checkpoint_sha256": checkpoint_sha256,
        "calibration_sha256": "",
        "configuration_sha256": configuration_sha256,
        "source_commit": source_commit,
        "device": device,
        "sample_count": len(rows),
        "failure_count": len(rows) - len(usable),
        "duration_seconds": duration_seconds,
        "private_row_count": 0,
        "gmail_fit_rows": 0,
        "usable_example_count": len(usable),
        "train_example_count": train_total,
        "train_ground_truth_count": counts[("train", "ground_truth")],
        "train_paddleocr_count": counts[("train", "paddleocr")],
        "train_hybrid_count": counts[("train", "hybrid")],
        "train_ocr_noise_count": counts[("train", "ocr_noise")],
        "train_ground_truth_ratio": _ratio(
            counts[("train", "ground_truth")], train_total
        ),
        "train_paddleocr_ratio": _ratio(
            counts[("train", "paddleocr")], train_total
        ),
        "train_hybrid_ratio": _ratio(
            counts[("train", "hybrid")], train_total
        ),
        "dev_select_example_count": dev_total,
        "dev_select_ground_truth_count": counts[
            ("dev_select", "ground_truth")
        ],
        "dev_select_paddleocr_count": counts[("dev_select", "paddleocr")],
        "dev_select_hybrid_count": counts[("dev_select", "hybrid")],
        "mean_alignment_coverage": _mean(
            ocr_rows, "alignment_coverage"
        ),
        "mean_entity_retention": _mean(
            ocr_rows, "entity_retention_rate"
        ),
        "mean_relation_retention": _mean(
            ocr_rows, "relation_retention_rate"
        ),
        "mean_canonical_retention": _mean(
            ocr_rows, "canonical_retention_rate"
        ),
        "nonselection_ocr_variant_count": forbidden_variant_count,
        "test_coru_gmail_used_for_selection": False,
    }


def _parse_trial(value: str) -> tuple[str, str, Path]:
    parts = value.split("|", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError(
            "--trial must use LABEL|OCR_PROFILE|MANIFEST"
        )
    label, profile, path = parts
    profile = profile.casefold()
    if profile not in {"original", "custom", "adaptive"}:
        raise ValueError(f"unsupported trial OCR profile: {profile}")
    manifest = Path(path).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    return label, profile, manifest


def _registry_for_profile(
    model_setup: str | Path,
    model_registry: str | Path,
    *,
    ocr_profile: str,
) -> ModelRegistry:
    choice = "original" if ocr_profile == "original" else "auto"
    return ModelRegistry.from_setup(
        model_setup,
        upgrade_registry=model_registry,
        detector_choice=choice,
        general_choice=choice,
        thai_choice=choice,
    )


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _mean(rows: list[dict[str, str]], field: str) -> float | None:
    values = [
        float(row[field])
        for row in rows
        if str(row.get(field, "")).strip()
    ]
    return sum(values) / len(values) if values else None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("no layout stream trials to write")
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=list(rows[0]),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, stream.getvalue())


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()


if __name__ == "__main__":
    raise SystemExit(main())
