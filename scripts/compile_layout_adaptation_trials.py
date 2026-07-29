#!/usr/bin/env python3
"""Compile LayoutXLM adaptation evidence and enforce the downstream gate."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rotation_common import atomic_write_text, sha256_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trial",
        action="append",
        required=True,
        metavar=(
            "ID|STRATEGY|TRAINING_JSON|REFERENCE_JSON|REAL_OCR_JSON|"
            "ROTATED_OCR_JSON|END_TO_END_CSV|OCR_CONFIGURATION"
        ),
    )
    parser.add_argument("--baseline-trial", required=True)
    parser.add_argument("--selected-trial", required=True)
    parser.add_argument(
        "--comparison-only-trial",
        action="append",
        default=[],
        help=(
            "Trial retained as a regression baseline but ineligible for final "
            "selection, such as the old checkpoint evaluated cross-build."
        ),
    )
    parser.add_argument(
        "--output",
        default=str(
            PROJECT_ROOT
            / "reports"
            / "ocr_upgrade"
            / "layout_adaptation_trials.csv"
        ),
    )
    args = parser.parse_args()

    definitions = [_parse_trial(value) for value in args.trial]
    trial_ids = [definition[0] for definition in definitions]
    if len(trial_ids) != len(set(trial_ids)):
        parser.error("trial IDs must be unique")
    known_trial_ids = set(trial_ids)
    for selected in (
        args.baseline_trial,
        args.selected_trial,
        *args.comparison_only_trial,
    ):
        if selected not in known_trial_ids:
            parser.error(f"unknown trial ID: {selected}")

    rows = [compile_trial(*definition) for definition in definitions]
    selected = select_downstream_trial(
        rows,
        baseline_trial=args.baseline_trial,
        selected_trial=args.selected_trial,
        comparison_only_trials=set(args.comparison_only_trial),
    )
    _write_csv(Path(args.output), rows)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "trial_count": len(rows),
                "baseline_trial": args.baseline_trial,
                "selected_trial": args.selected_trial,
                "selected_score": selected["selection_score"],
                "comparison_only_trials": sorted(
                    args.comparison_only_trial
                ),
            },
            indent=2,
        )
    )
    return 0


def compile_trial(
    trial_id: str,
    strategy: str,
    training_path: Path,
    reference_path: Path,
    real_ocr_path: Path,
    rotated_ocr_path: Path,
    end_to_end_path: Path,
    ocr_configuration: str,
) -> dict[str, Any]:
    training = _json(training_path)
    reference = _evaluation(
        reference_path,
        expected_rotation=0.0,
        expected_token_sources={"ground_truth"},
    )
    real_ocr = _evaluation(
        real_ocr_path,
        expected_rotation=0.0,
        expected_token_sources={"paddleocr"},
    )
    rotated = _evaluation(
        rotated_ocr_path,
        expected_rotation=37.0,
        expected_token_sources={"paddleocr"},
    )
    end_to_end = _csv_row(
        end_to_end_path,
        "configuration",
        ocr_configuration,
    )
    checkpoint = Path(str(training.get("checkpoint", "")))
    checkpoint_model = checkpoint / "model.safetensors"
    if not checkpoint_model.is_file():
        raise FileNotFoundError(checkpoint_model)
    checkpoint_sha = sha256_file(checkpoint_model)
    evaluation_hashes = {
        str(value.get("checkpoint_sha256") or value.get("checkpoint_model_sha256"))
        for value in (reference, real_ocr, rotated)
    }
    if evaluation_hashes != {checkpoint_sha}:
        raise ValueError(f"evaluation checkpoint mismatch for {trial_id}")
    if str(end_to_end.get("checkpoint_sha256", "")) != checkpoint_sha:
        raise ValueError(f"end-to-end checkpoint mismatch for {trial_id}")
    cross_build_values = {
        bool(value.get("cross_build_comparison", False))
        for value in (reference, real_ocr, rotated)
    }
    if len(cross_build_values) != 1:
        raise ValueError(
            f"mixed cross-build evaluation status for {trial_id}"
        )
    for field in ("detector_sha256", "recognizer_sha256"):
        values = {
            str(value.get(field, ""))
            for value in (reference, real_ocr, rotated, end_to_end)
        }
        if len(values) != 1 or "" in values:
            raise ValueError(
                f"mixed or missing {field} in downstream trial {trial_id}"
            )
    preprocessing_hashes = {
        str(value.get("preprocessing_sha256", ""))
        for value in (reference, real_ocr, rotated)
    }
    if len(preprocessing_hashes) != 1 or "" in preprocessing_hashes:
        raise ValueError(
            f"mixed or missing preprocessing hash for {trial_id}"
        )
    manifest_hashes = {
        str(value.get("manifest_sha256", ""))
        for value in (reference, real_ocr, rotated)
    }
    if len(manifest_hashes) != 1 or "" in manifest_hashes:
        raise ValueError(f"mixed evaluation manifests for {trial_id}")
    for value in (training, reference, real_ocr, rotated):
        if int(
            value.get(
                "private_row_count",
                value.get("private_example_count", 0),
            )
        ):
            raise ValueError(f"private row found in downstream trial {trial_id}")

    reference_metrics = dict(reference.get("metrics") or {})
    real_metrics = dict(real_ocr.get("metrics") or {})
    rotated_metrics = dict(rotated.get("metrics") or {})
    metrics = {
        "reference_entity_f1": _float(
            reference_metrics, "entity_token_f1"
        ),
        "real_ocr_entity_f1": _float(real_metrics, "entity_token_f1"),
        "canonical_evidence_f1": _float(
            reference_metrics, "canonical_evidence_token_f1"
        ),
        "critical_field_exact_match": _float(
            end_to_end, "critical_field_exact_match"
        ),
        "relation_f1": _float(real_metrics, "relation_f1"),
        "rotated_real_ocr_entity_f1": _float(
            rotated_metrics, "entity_token_f1"
        ),
        "document_macro_f1": _float(
            reference_metrics, "document_macro_f1"
        ),
        "document_accuracy": _float(
            reference_metrics, "document_accuracy"
        ),
        "end_to_end_entity_f1": _float(
            end_to_end, "end_to_end_entity_f1"
        ),
        "end_to_end_canonical_accuracy": _float(
            end_to_end, "end_to_end_canonical_accuracy"
        ),
        "end_to_end_relation_f1": _float(
            end_to_end, "end_to_end_relation_f1"
        ),
    }
    selection_score = downstream_selection_score(**{
        key: metrics[key]
        for key in (
            "reference_entity_f1",
            "real_ocr_entity_f1",
            "canonical_evidence_f1",
            "critical_field_exact_match",
            "relation_f1",
            "rotated_real_ocr_entity_f1",
            "document_macro_f1",
        )
    })
    return {
        "trial_id": trial_id,
        "strategy": strategy,
        "status": "passed",
        "selected": False,
        "selection_candidate": False,
        "eligible": False,
        "cross_build_comparison": next(iter(cross_build_values)),
        "build_id": reference.get("build_id")
        or reference.get("evaluation_build_id"),
        "split": "dev_select",
        "manifest_sha256": next(iter(manifest_hashes)),
        "detector_sha256": reference.get("detector_sha256"),
        "recognizer_sha256": reference.get("recognizer_sha256"),
        "checkpoint_sha256": checkpoint_sha,
        "calibration_sha256": reference.get("calibration_sha256") or "",
        "configuration_sha256": reference.get("configuration_sha256"),
        "source_commit": reference.get("source_commit"),
        "device": reference.get("device"),
        "sample_count": sum(
            int(value.get("sample_count", value.get("example_count", 0)))
            for value in (reference, real_ocr, rotated)
        )
        + int(end_to_end.get("sample_count", 0)),
        "failure_count": sum(
            int(value.get("failure_count", 0))
            for value in (reference, real_ocr, rotated)
        )
        + int(end_to_end.get("failure_count", 0)),
        "duration_seconds": sum(
            float(value.get("duration_seconds", 0.0))
            for value in (training, reference, real_ocr, rotated)
        )
        + float(end_to_end.get("duration_seconds", 0.0)),
        "private_row_count": 0,
        "gmail_fit_rows": int(training.get("gmail_fit_rows", 0)),
        "initial_checkpoint": training.get("initial_checkpoint")
        or training.get("source_checkpoint"),
        "requested_epochs": training.get("requested_epochs"),
        "completed_epochs": training.get("completed_epochs"),
        "optimizer_steps": training.get("optimizer_steps"),
        "learning_rates": json.dumps(
            training.get("learning_rates") or {},
            sort_keys=True,
            separators=(",", ":"),
        ),
        "checkpoint_reload_passed": training.get(
            "checkpoint_reload_passed", False
        ),
        **metrics,
        "selection_score": selection_score,
        "reference_regression_ok": False,
        "canonical_regression_ok": False,
        "document_regression_ok": False,
        "relation_regression_ok_or_benefit": False,
        "regression_reason": "",
        "training_report": str(training_path),
        "reference_evaluation": str(reference_path),
        "real_ocr_evaluation": str(real_ocr_path),
        "rotated_ocr_evaluation": str(rotated_ocr_path),
        "end_to_end_evaluation": str(end_to_end_path),
        "ocr_configuration": ocr_configuration,
    }


def select_downstream_trial(
    rows: list[dict[str, Any]],
    *,
    baseline_trial: str,
    selected_trial: str,
    comparison_only_trials: set[str] | None = None,
) -> dict[str, Any]:
    """Apply regression gates and select only build-bound final candidates."""
    comparison_only = set(comparison_only_trials or ())
    trial_ids = {str(row.get("trial_id", "")) for row in rows}
    for trial_id in (baseline_trial, selected_trial, *comparison_only):
        if trial_id not in trial_ids:
            raise ValueError(f"unknown downstream trial ID: {trial_id}")
    if selected_trial in comparison_only:
        raise ValueError("selected downstream trial is comparison-only")

    baseline = next(
        row for row in rows if row["trial_id"] == baseline_trial
    )
    for row in rows:
        trial_id = str(row["trial_id"])
        row["selection_candidate"] = trial_id not in comparison_only
        row.update(regression_gate(row, baseline))
        if (
            bool(row.get("cross_build_comparison", False))
            and row["selection_candidate"]
        ):
            raise ValueError(
                f"cross-build trial must be comparison-only: {trial_id}"
            )

    eligible = [
        row
        for row in rows
        if row["eligible"] and row["selection_candidate"]
    ]
    if not eligible:
        raise ValueError(
            "no build-bound downstream adaptation trial passes regression gates"
        )
    best_score = max(float(row["selection_score"]) for row in eligible)
    selected = next(
        row for row in rows if row["trial_id"] == selected_trial
    )
    if not selected["eligible"]:
        raise ValueError("selected downstream trial fails regression gates")
    if float(selected["selection_score"]) + 1e-12 < best_score:
        raise ValueError(
            "selected downstream trial is not the best eligible "
            "build-bound score"
        )
    for row in rows:
        row["selected"] = row is selected
    return selected


def downstream_selection_score(
    *,
    reference_entity_f1: float,
    real_ocr_entity_f1: float,
    canonical_evidence_f1: float,
    critical_field_exact_match: float,
    relation_f1: float,
    rotated_real_ocr_entity_f1: float,
    document_macro_f1: float,
) -> float:
    values = (
        reference_entity_f1,
        real_ocr_entity_f1,
        canonical_evidence_f1,
        critical_field_exact_match,
        relation_f1,
        rotated_real_ocr_entity_f1,
        document_macro_f1,
    )
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        raise ValueError("downstream selection metrics must be finite in [0, 1]")
    return (
        0.25 * reference_entity_f1
        + 0.20 * real_ocr_entity_f1
        + 0.15 * canonical_evidence_f1
        + 0.15 * critical_field_exact_match
        + 0.10 * relation_f1
        + 0.10 * rotated_real_ocr_entity_f1
        + 0.05 * document_macro_f1
    )


def regression_gate(
    row: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    reference_ok = (
        float(row["reference_entity_f1"])
        >= float(baseline["reference_entity_f1"]) - 0.02
    )
    canonical_ok = (
        float(row["canonical_evidence_f1"])
        >= float(baseline["canonical_evidence_f1"]) - 0.02
    )
    document_ok = (
        float(row["document_accuracy"])
        >= float(baseline["document_accuracy"]) - 0.02
    )
    relation_ok = (
        float(row["relation_f1"]) >= float(baseline["relation_f1"]) - 0.02
        or float(row["end_to_end_entity_f1"])
        >= float(baseline["end_to_end_entity_f1"]) + 0.02
    )
    reload_ok = bool(row.get("checkpoint_reload_passed", False))
    reasons = []
    for passed, reason in (
        (reference_ok, "reference_entity_drop_gt_0.02"),
        (canonical_ok, "canonical_drop_gt_0.02"),
        (document_ok, "document_accuracy_drop_gt_0.02"),
        (relation_ok, "relation_regression_without_e2e_gain"),
        (reload_ok, "checkpoint_reload_failed"),
    ):
        if not passed:
            reasons.append(reason)
    return {
        "eligible": not reasons,
        "reference_regression_ok": reference_ok,
        "canonical_regression_ok": canonical_ok,
        "document_regression_ok": document_ok,
        "relation_regression_ok_or_benefit": relation_ok,
        "regression_reason": ";".join(reasons),
    }


def _parse_trial(value: str) -> tuple[str, str, Path, Path, Path, Path, Path, str]:
    parts = value.split("|", 7)
    if len(parts) != 8 or not all(parts):
        raise ValueError("invalid --trial downstream evidence tuple")
    trial_id, strategy, *paths, configuration = parts
    resolved = tuple(Path(path).resolve() for path in paths)
    for path in resolved:
        if not path.is_file():
            raise FileNotFoundError(path)
    return trial_id, strategy, *resolved, configuration


def _evaluation(
    path: Path,
    *,
    expected_rotation: float,
    expected_token_sources: set[str],
) -> dict[str, Any]:
    value = _json(path)
    if value.get("split") != "dev_select":
        raise ValueError(f"downstream selection must use DEV_SELECT: {path}")
    if float(value.get("rotation_angle", 0.0)) != expected_rotation:
        raise ValueError(f"unexpected rotation angle in {path}")
    actual_sources = {
        str(source) for source in (value.get("token_sources") or [])
    }
    if actual_sources != expected_token_sources:
        raise ValueError(
            f"unexpected token sources in {path}: "
            f"{sorted(actual_sources)!r} != {sorted(expected_token_sources)!r}"
        )
    return value


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _csv_row(path: Path, field: str, expected: str) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle) if row.get(field) == expected
        ]
    if len(rows) != 1:
        raise ValueError(f"expected one {field}={expected} row in {path}")
    if rows[0].get("status") != "passed":
        raise ValueError(f"end-to-end configuration did not pass: {path}")
    return rows[0]


def _float(value: Mapping[str, Any], key: str) -> float:
    result = float(value[key])
    if not math.isfinite(result):
        raise ValueError(f"non-finite metric {key}")
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


if __name__ == "__main__":
    raise SystemExit(main())
