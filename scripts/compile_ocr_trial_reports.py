#!/usr/bin/env python3
"""Compile bounded detector/general/Thai trial ledgers and acceptance decisions."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ocr.trials import recognizer_acceptance  # noqa: E402
from src.rotation_common import atomic_write_json, atomic_write_text  # noqa: E402


DETECTOR_EXTERNAL_ROOT = Path(
    "D:/CSX4201/vision-info-extraction-assets/checkpoints/ocr_upgrade/detector"
)
METRIC_PROVENANCE_FIELDS = (
    "build_id",
    "split",
    "manifest_sha256",
    "detector_sha256",
    "recognizer_sha256",
    "checkpoint_sha256",
    "calibration_sha256",
    "configuration_sha256",
    "source_commit",
    "device",
    "sample_count",
    "failure_count",
    "duration_seconds",
    "private_row_count",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--general-custom-report")
    parser.add_argument("--general-custom-metadata")
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "reports/ocr_upgrade"),
    )
    args = parser.parse_args()
    if bool(args.general_custom_report) != bool(args.general_custom_metadata):
        parser.error(
            "--general-custom-report and --general-custom-metadata are required together"
        )
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    detector_rows = detector_trial_rows()
    _validate_metric_rows(detector_rows)
    _write_csv(output_root / "detector_trials.csv", detector_rows)
    general_rows, acceptance = recognition_trial_rows(
        custom_report=Path(args.general_custom_report)
        if args.general_custom_report
        else None,
        custom_metadata=Path(args.general_custom_metadata)
        if args.general_custom_metadata
        else None,
    )
    _validate_metric_rows(general_rows)
    _write_csv(output_root / "recognizer_trials.csv", general_rows)
    atomic_write_json(output_root / "recognizer_acceptance.json", acceptance)
    thai_rows = thai_trial_rows()
    _validate_metric_rows(thai_rows)
    _write_csv(output_root / "thai_trials.csv", thai_rows)
    print(
        json.dumps(
            {
                "detector_trial_count": len(detector_rows),
                "recognizer_trial_count": len(general_rows),
                "thai_trial_count": len(thai_rows),
                "general_custom_accepted": acceptance.get("accepted", False),
            },
            indent=2,
        )
    )
    return 0


def detector_trial_rows() -> list[dict[str, Any]]:
    training_data = _json(
        PROJECT_ROOT / "reports/ocr_upgrade/training_data_summary.json"
    )
    detector_manifest_sha = str(
        (training_data.get("detector") or {}).get("manifest_sha256", "")
    )
    baseline_path = (
        PROJECT_ROOT
        / "reports/ocr_upgrade/detector_trials/det_baseline_official.json"
    )
    baseline = _json(baseline_path)
    metrics = dict(baseline.get("metrics") or {})
    rows = [
        {
            "trial_id": metrics.get("trial_id", "det_baseline_official"),
            "build_id": baseline.get("build_id"),
            "status": "passed",
            "selected": True,
            "accepted": True,
            "model_name": baseline.get("model_name"),
            "training_mode": "official_inference_baseline",
            "split": baseline.get("split"),
            "manifest_sha256": baseline.get("manifest_sha256"),
            "sample_count": baseline.get("sample_count"),
            "failure_count": baseline.get("failure_count"),
            "duration_seconds": baseline.get("duration_seconds"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
            "small_text_recall": metrics.get("small_text_recall"),
            "critical_region_recall": metrics.get("critical_region_recall"),
            "time_per_page_seconds": metrics.get("time_per_page_seconds"),
            "detector_sha256": baseline.get("detector_sha256"),
            "recognizer_sha256": baseline.get("recognizer_sha256"),
            "checkpoint_sha256": baseline.get("checkpoint_sha256"),
            "calibration_sha256": baseline.get("calibration_sha256"),
            "configuration_sha256": baseline.get("configuration_sha256"),
            "source_commit": baseline.get("source_commit"),
            "device": baseline.get("device"),
            "private_row_count": baseline.get("private_row_count"),
            "observed_steps": 0,
            "peak_allocated_mib": None,
            "peak_reserved_mib": None,
            "rejection_reason": "",
            "evidence_path": baseline_path.as_posix(),
        }
    ]
    reasons = {
        "det_ft_official_aug": (
            "bounded_stop: FP32 batch 4 reached 9,777 MiB allocated and "
            "10,129 MiB reserved on an 8,151 MiB device; 10:22:14 ETA after step 20"
        ),
        "det_ft_official_amp": (
            "failed_before_valid_window: CUDA 13 NVRTC DLL was unavailable; "
            "no checkpoint or development metric was produced"
        ),
        "det_ft_official_amp_nvrtc": (
            "bounded_stop: O2 AMP repeatedly reduced the loss scale after inf/NaN; "
            "12,108 MiB allocated, 13,639 MiB reserved, and 6:06:01 ETA at step 20"
        ),
    }
    for trial_id in (
        "det_ft_official_aug",
        "det_ft_official_amp",
        "det_ft_official_amp_nvrtc",
    ):
        root = DETECTOR_EXTERNAL_ROOT / trial_id
        metadata = _json(root / "run_metadata.json")
        log_path = root / "logs/train.log"
        log = log_path.read_text(encoding="utf-8", errors="replace")
        rows.append(
            {
                "trial_id": trial_id,
                "build_id": (
                    "ocr-detector-trial-"
                    + str(metadata["resolved_config_sha256"])[:16]
                ),
                "status": "failed_or_bounded_stopped",
                "selected": False,
                "accepted": False,
                "model_name": "PP-OCRv6_medium_det",
                "training_mode": "public_train_finetune",
                "split": "train_with_dev_select_gate",
                "manifest_sha256": detector_manifest_sha,
                "sample_count": 0,
                "failure_count": 1,
                "duration_seconds": metadata.get("duration_seconds"),
                "precision": None,
                "recall": None,
                "f1": None,
                "small_text_recall": None,
                "critical_region_recall": None,
                "time_per_page_seconds": None,
                "detector_sha256": metadata.get(
                    "source_checkpoint_sha256"
                ),
                "recognizer_sha256": None,
                "checkpoint_sha256": metadata.get(
                    "source_checkpoint_sha256"
                ),
                "calibration_sha256": None,
                "configuration_sha256": metadata.get("resolved_config_sha256"),
                "source_commit": metadata.get("source_commit"),
                "device": "gpu:0",
                "private_row_count": metadata.get("private_row_count", 0),
                "observed_steps": _last_integer(log, r"global_step:\s*(\d+)"),
                "peak_allocated_mib": _last_integer(
                    log, r"max_mem_allocated:\s*(\d+)\s*MB"
                ),
                "peak_reserved_mib": _last_integer(
                    log, r"max_mem_reserved:\s*(\d+)\s*MB"
                ),
                "rejection_reason": reasons[trial_id],
                "evidence_path": log_path.as_posix(),
            }
        )
    return rows


def recognition_trial_rows(
    *,
    custom_report: Path | None,
    custom_metadata: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline_path = (
        PROJECT_ROOT
        / "reports/ocr_upgrade/recognizer_trials/rec_general_baseline_official.json"
    )
    baseline = _json(baseline_path)
    baseline_metrics = dict(baseline.get("metrics") or {})
    acceptance: dict[str, Any] = {
        "accepted": False,
        "reason": "no completed custom general recognizer report was supplied",
        "selected_trial_id": "rec_general_baseline_official",
    }
    rows = [_recognition_row(baseline_path, baseline, selected=True, accepted=True)]
    if custom_report is not None and custom_metadata is not None:
        candidate = _json(custom_report)
        metadata = _json(custom_metadata)
        candidate_metrics = dict(candidate.get("metrics") or {})
        acceptance = recognizer_acceptance(
            baseline_metrics,
            candidate_metrics,
            export_reload_passed=(
                metadata.get("status") == "passed"
                and metadata.get("export_sha256") == candidate.get("recognizer_sha256")
            ),
            dictionary_match_passed=(
                int(
                    candidate_metrics.get("dictionary", {}).get(
                        "compatible_evaluated_samples", 0
                    )
                )
                == int(candidate.get("sample_count", 0))
            ),
            private_row_count=int(candidate.get("private_row_count", -1)),
            maximum_time_per_sample_seconds=max(
                0.05,
                2.0
                * float(
                    baseline_metrics.get("time_per_sample_seconds", 0.0) or 0.0
                ),
            ),
        )
        acceptance.update(
            {
                "baseline_trial_id": baseline_metrics.get("trial_id"),
                "candidate_trial_id": candidate_metrics.get("trial_id"),
                "selected_trial_id": (
                    candidate_metrics.get("trial_id")
                    if acceptance["accepted"]
                    else baseline_metrics.get("trial_id")
                ),
                "baseline_report": baseline_path.as_posix(),
                "candidate_report": custom_report.as_posix(),
            }
        )
        rows[0]["selected"] = not acceptance["accepted"]
        rows.append(
            _recognition_row(
                custom_report,
                candidate,
                selected=bool(acceptance["accepted"]),
                accepted=bool(acceptance["accepted"]),
                rejection_reason=(
                    ""
                    if acceptance["accepted"]
                    else "; ".join(
                        key
                        for key, passed in acceptance["criteria"].items()
                        if not passed
                    )
                ),
            )
        )
    return rows, acceptance


def thai_trial_rows() -> list[dict[str, Any]]:
    baseline_path = (
        PROJECT_ROOT
        / "reports/ocr_upgrade/recognizer_trials/rec_thai_baseline_official.json"
    )
    custom_path = (
        PROJECT_ROOT
        / "reports/ocr_upgrade/recognizer_trials/rec_thai_synthetic.json"
    )
    baseline = _json(baseline_path)
    custom = _json(custom_path)
    return [
        _recognition_row(
            baseline_path,
            baseline,
            selected=False,
            accepted=True,
            rejection_reason="custom Thai synthetic DEV_SELECT result was materially better",
        ),
        _recognition_row(
            custom_path,
            custom,
            selected=True,
            accepted=True,
            claim_boundary=(
                "Synthetic OFL-font DEV_SELECT and integration evidence only; "
                "no real public Thai benchmark claim."
            ),
        ),
    ]


def _recognition_row(
    path: Path,
    report: Mapping[str, Any],
    *,
    selected: bool,
    accepted: bool,
    rejection_reason: str = "",
    claim_boundary: str = "",
) -> dict[str, Any]:
    metrics = dict(report.get("metrics") or {})
    return {
        "trial_id": metrics.get("trial_id"),
        "build_id": report.get("build_id"),
        "track": metrics.get("track"),
        "status": report.get("status", "passed"),
        "selected": selected,
        "accepted": accepted,
        "model_name": report.get("model_name"),
        "split": report.get("split"),
        "manifest_sha256": report.get("manifest_sha256"),
        "sample_count": report.get("sample_count"),
        "failure_count": report.get("failure_count"),
        "duration_seconds": report.get("duration_seconds"),
        "cer": metrics.get("cer"),
        "wer": metrics.get("wer"),
        "exact_line_accuracy": metrics.get("exact_line_accuracy"),
        "numeric_exact_match": metrics.get("numeric_exact_match"),
        "amount_exact_match": metrics.get("amount_exact_match"),
        "date_exact_match": metrics.get("date_exact_match"),
        "identifier_exact_match": metrics.get("identifier_exact_match"),
        "currency_exact_match": metrics.get("currency_exact_match"),
        "email_exact_match": metrics.get("email_exact_match"),
        "turkish_character_accuracy": metrics.get("turkish_character_accuracy"),
        "english_exact_match": metrics.get("english_exact_match"),
        "confidence_ece": metrics.get("confidence_ece"),
        "confidence_brier": metrics.get("confidence_brier"),
        "time_per_sample_seconds": metrics.get("time_per_sample_seconds"),
        "selection_score": metrics.get("selection_score"),
        "detector_sha256": report.get("detector_sha256"),
        "recognizer_sha256": report.get("recognizer_sha256"),
        "checkpoint_sha256": report.get("checkpoint_sha256"),
        "calibration_sha256": report.get("calibration_sha256"),
        "configuration_sha256": report.get("configuration_sha256"),
        "source_commit": report.get("source_commit"),
        "device": report.get("device"),
        "private_row_count": report.get("private_row_count"),
        "rejection_reason": rejection_reason,
        "claim_boundary": claim_boundary,
        "evidence_path": path.as_posix(),
    }


def _validate_metric_rows(rows: list[Mapping[str, Any]]) -> None:
    for index, row in enumerate(rows):
        missing = [
            field for field in METRIC_PROVENANCE_FIELDS if field not in row
        ]
        if missing:
            raise ValueError(
                f"metric ledger row {index} is missing provenance: "
                + ", ".join(missing)
            )


def _last_integer(text: str, pattern: str) -> int | None:
    values = re.findall(pattern, text)
    return int(values[-1]) if values else None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty trial ledger: {path}")
    fields = list(rows[0])
    for row in rows:
        if list(row) != fields:
            raise ValueError(f"inconsistent trial ledger fields: {path}")
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, stream.getvalue())


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
