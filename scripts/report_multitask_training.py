#!/usr/bin/env python3
"""Materialize compact final training history and bounded trial evidence."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config as cfgmod  # noqa: E402
from src.rotation_common import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    sha256_file,
)


TRIALS = (
    {
        "trial": "trial_1_ground_truth",
        "streams": "ground_truth",
        "upright_probability": 1.0,
        "checkpoint_selection": "upright dev_select",
        "notes": "ground-truth baseline",
    },
    {
        "trial": "trial_2_rotation_variants",
        "streams": "ground_truth,paddleocr,hybrid",
        "upright_probability": 1.0,
        "checkpoint_selection": "upright dev_select",
        "notes": "real OCR-noise and hybrid target streams",
    },
    {
        "trial": "trial_3_dynamic_rotation",
        "streams": "ground_truth,paddleocr,hybrid",
        "upright_probability": 0.2,
        "checkpoint_selection": "upright dev_select",
        "notes": "80 percent arbitrary-angle augmentation",
    },
    {
        "trial": "trial_4_balanced_rotation",
        "streams": "ground_truth,paddleocr,hybrid",
        "upright_probability": 0.6,
        "checkpoint_selection": "upright dev_select",
        "notes": "selected 60/40 upright/arbitrary robustness compromise",
    },
)

TRIAL_COLUMNS = (
    "trial", "streams", "upright_probability", "rotation_probability",
    "clean_entity_f1", "clean_canonical_f1", "clean_relation_f1", "clean_composite",
    "ocr_variant_composite", "rotated_37_composite", "three_gate_mean_composite",
    "checkpoint_selection", "selected_for_final", "notes",
)

HISTORY_COLUMNS = (
    "epoch", "loss", "entity_token_f1", "canonical_evidence_token_f1",
    "relation_f1", "document_macro_f1", "upright_composite_score",
    "rotated_37_entity_token_f1", "rotated_37_canonical_evidence_token_f1",
    "rotated_37_relation_f1", "rotated_37_document_macro_f1",
    "rotated_37_composite_score", "selection_composite_score",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument(
        "--training-report",
        help=(
            "Selected OCR-upgrade training report to promote. When omitted "
            "with --adaptation-trials, the selected ledger row supplies it."
        ),
    )
    parser.add_argument(
        "--adaptation-trials",
        help="Completed layout_adaptation_trials.csv selection ledger.",
    )
    parser.add_argument("--selected-trial")
    args = parser.parse_args()
    cfg = cfgmod.load_config(args.config)
    report_root = cfgmod.resolve_path(cfg, "reports") / "final_model"
    evaluation_root = report_root / "evaluations"
    trial_rows = [_trial_row(definition, evaluation_root) for definition in TRIALS]
    atomic_write_csv(report_root / "hyperparameter_trials.csv", trial_rows, TRIAL_COLUMNS)

    adaptation_rows: list[dict[str, str]] = []
    selected_adaptation: dict[str, str] | None = None
    adaptation_path: Path | None = None
    if args.adaptation_trials:
        adaptation_path = _resolve(args.adaptation_trials)
        adaptation_rows = _csv(adaptation_path)
        selected_adaptation = selected_adaptation_trial(
            adaptation_rows,
            selected_trial=args.selected_trial,
        )
    elif args.selected_trial:
        parser.error("--selected-trial requires --adaptation-trials")

    if args.training_report:
        training_path = _resolve(args.training_report)
    elif selected_adaptation is not None:
        training_path = _resolve(selected_adaptation["training_report"])
    else:
        training_path = report_root / "multitask_training_final.json"
    if not training_path.is_file():
        raise FileNotFoundError(
            f"final training report is not available: {training_path}"
        )
    training = json.loads(training_path.read_text(encoding="utf-8"))
    if selected_adaptation is not None:
        validate_adaptation_promotion(
            selected_adaptation,
            training,
            training_path=training_path,
        )
        atomic_write_json(
            report_root / "multitask_training_final.json",
            training,
        )
        atomic_write_json(
            cfgmod.resolve_path(cfg, "reports")
            / "information_extraction"
            / "layout_model_training.json",
            training,
        )
    history_rows = [_history_row(item) for item in training.get("validation_history", [])]
    atomic_write_csv(report_root / "training_history.csv", history_rows, HISTORY_COLUMNS)
    summary = {
        key: training.get(key)
        for key in (
            "schema_version", "profile", "build_id", "manifest_path", "manifest_sha256",
            "public_only", "gmail_fit_rows", "source_checkpoint", "architecture",
            "visual_backbone", "license", "device",
            "mixed_precision", "token_sources", "train_examples", "validation_examples",
            "train_windows", "validation_windows", "rotated_validation_windows",
            "training_target_counts", "dynamic_rotation", "mean_task_losses",
            "optimizer_steps", "micro_steps", "requested_epochs", "completed_epochs",
            "early_stopping_patience", "stopped_early", "stop_reason", "best_epoch",
            "best_composite_score", "checkpoint_selection", "validation", "checkpoint",
            "checkpoint_reload_passed", "checkpoint_reload_max_difference", "duration_seconds",
            "limitations",
        )
    }
    summary["bounded_development_trial_count"] = len(trial_rows)
    summary["selected_development_trial"] = "trial_4_balanced_rotation"
    if selected_adaptation is not None and adaptation_path is not None:
        summary.update(
            {
                "bounded_adaptation_trial_count": len(adaptation_rows),
                "selected_adaptation_trial": selected_adaptation["trial_id"],
                "adaptation_selection_score": float(
                    selected_adaptation["selection_score"]
                ),
                "adaptation_ledger": str(adaptation_path),
                "adaptation_ledger_sha256": sha256_file(adaptation_path),
                "canonical_promotion_source": str(training_path),
            }
        )
    atomic_write_json(report_root / "training_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


def _trial_row(definition: dict[str, Any], evaluation_root: Path) -> dict[str, Any]:
    number = definition["trial"].split("_", 2)[1]
    prefix = f"trial_{number}_dev_select"
    clean = _metrics(evaluation_root / f"{prefix}_ground_truth.json")
    variants = _metrics(evaluation_root / f"{prefix}_variants.json")
    rotated = _metrics(evaluation_root / f"{prefix}_ground_truth_rotated_37.json")
    composites = [
        item["composite_score"] for item in (clean, variants, rotated) if item
    ]
    return {
        **definition,
        "rotation_probability": 1.0 - float(definition["upright_probability"]),
        "clean_entity_f1": clean.get("entity_token_f1"),
        "clean_canonical_f1": clean.get("canonical_evidence_token_f1"),
        "clean_relation_f1": clean.get("relation_f1"),
        "clean_composite": clean.get("composite_score"),
        "ocr_variant_composite": variants.get("composite_score"),
        "rotated_37_composite": rotated.get("composite_score"),
        "three_gate_mean_composite": statistics.fmean(composites) if composites else None,
        "selected_for_final": definition["trial"] == "trial_4_balanced_rotation",
    }


def _metrics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8")).get("metrics") or {})


def _history_row(item: dict[str, Any]) -> dict[str, Any]:
    rotated = dict(item.get("rotated_37") or {})
    return {
        "epoch": item.get("epoch"),
        "loss": item.get("loss"),
        "entity_token_f1": item.get("entity_token_f1"),
        "canonical_evidence_token_f1": item.get("canonical_evidence_token_f1"),
        "relation_f1": item.get("relation_f1"),
        "document_macro_f1": item.get("document_macro_f1"),
        "upright_composite_score": item.get("composite_score"),
        "rotated_37_entity_token_f1": rotated.get("entity_token_f1"),
        "rotated_37_canonical_evidence_token_f1": rotated.get("canonical_evidence_token_f1"),
        "rotated_37_relation_f1": rotated.get("relation_f1"),
        "rotated_37_document_macro_f1": rotated.get("document_macro_f1"),
        "rotated_37_composite_score": rotated.get("composite_score"),
        "selection_composite_score": item.get("selection_composite_score"),
    }


def selected_adaptation_trial(
    rows: list[dict[str, str]],
    *,
    selected_trial: str | None = None,
) -> dict[str, str]:
    selected = [
        row
        for row in rows
        if _truthy(row.get("selected"))
        and (
            selected_trial is None
            or row.get("trial_id") == selected_trial
        )
    ]
    if len(selected) != 1:
        raise ValueError(
            "adaptation ledger must contain exactly one matching selected trial"
        )
    row = selected[0]
    if (
        not _truthy(row.get("selection_candidate"))
        or not _truthy(row.get("eligible"))
        or _truthy(row.get("cross_build_comparison"))
        or not _truthy(row.get("checkpoint_reload_passed"))
    ):
        raise ValueError(
            "selected adaptation trial is not a reload-verified build-bound "
            "candidate"
        )
    return row


def validate_adaptation_promotion(
    row: Mapping[str, Any],
    training: Mapping[str, Any],
    *,
    training_path: Path,
) -> None:
    declared_training = _resolve(str(row.get("training_report", "")))
    if declared_training != training_path.resolve():
        raise ValueError("selected training report path does not match ledger")
    checkpoint = Path(str(training.get("checkpoint", ""))).resolve()
    model_path = checkpoint / "model.safetensors"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    checks = {
        "profile": training.get("profile") == "final",
        "public_only": training.get("public_only") is True,
        "gmail_fit_rows": int(training.get("gmail_fit_rows", -1)) == 0,
        "checkpoint_reload_passed": (
            training.get("checkpoint_reload_passed") is True
        ),
        "build_id": str(training.get("build_id", ""))
        == str(row.get("build_id", "")),
        "manifest_sha256": str(training.get("manifest_sha256", ""))
        == str(row.get("manifest_sha256", "")),
        "checkpoint_sha256": sha256_file(model_path)
        == str(row.get("checkpoint_sha256", "")),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "selected adaptation promotion binding failed: "
            + ", ".join(failed)
        )


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"adaptation ledger is empty: {path}")
    return rows


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _truthy(value: Any) -> bool:
    return str(value).casefold() in {"1", "true", "yes"}


if __name__ == "__main__":
    raise SystemExit(main())
