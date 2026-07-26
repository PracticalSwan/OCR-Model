#!/usr/bin/env python3
"""Compile final OCR-upgrade narratives from executed public evidence."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rotation_common import atomic_write_text  # noqa: E402


JSON_INPUTS = {
    "baseline": "baseline_snapshot.json",
    "benchmark": "benchmark_summary.json",
    "training_data": "training_data_summary.json",
    "training_environment": "training_environment.json",
    "pretrained_models": "pretrained_training_models.json",
    "registry": "model_registry.json",
    "ocr_selection": "ocr_model_selection.json",
    "preprocessing": "preprocessing_selection.json",
    "adaptive_rendering": "adaptive_rendering_metrics.json",
    "tiling": "tiling_metrics.json",
    "crop_padding": "crop_padding_metrics.json",
    "recognition_retry": "recognition_retry_metrics.json",
    "orientation": "orientation_metrics.json",
    "critical_fields": "critical_field_metrics.json",
    "calibration": "calibration_metrics.json",
    "locked_ocr": "locked_test_ocr_metrics.json",
    "locked_end_to_end": "locked_test_end_to_end_metrics.json",
    "angles": "angle_metrics.json",
    "unseen_coru": "unseen_coru_metrics.json",
    "private": "private_aggregate.json",
}

CSV_INPUTS = {
    "detector_trials": "detector_trials.csv",
    "recognizer_trials": "recognizer_trials.csv",
    "thai_trials": "thai_trials.csv",
    "ocr_comparison": "ocr_model_comparison.csv",
    "layout_stream_trials": "layout_stream_trials.csv",
    "layout_adaptation_trials": "layout_adaptation_trials.csv",
}


@dataclass(frozen=True)
class ErrorCategory:
    group: str
    label: str
    key: str
    source: str
    impact: int
    feasibility: int
    unit: str


ERROR_CATEGORIES = (
    ErrorCategory(
        "Detection",
        "Missed small text",
        "detection_missed_small_text",
        "locked_ocr",
        5,
        4,
        "missed-region signals",
    ),
    ErrorCategory(
        "Detection",
        "Merged regions",
        "detection_merged_region_signals",
        "locked_ocr",
        4,
        3,
        "region-count signals",
    ),
    ErrorCategory(
        "Detection",
        "Split regions",
        "detection_split_region_signals",
        "locked_ocr",
        3,
        3,
        "region-count signals",
    ),
    ErrorCategory(
        "Detection",
        "False positives",
        "detection_false_positives",
        "locked_ocr",
        3,
        4,
        "region signals",
    ),
    ErrorCategory(
        "Detection",
        "Table misses",
        "detection_table_miss_pages",
        "locked_ocr",
        5,
        3,
        "pages",
    ),
    ErrorCategory(
        "Detection",
        "Rotated misses",
        "detection_rotated_miss_pages",
        "locked_ocr",
        5,
        3,
        "pages",
    ),
    ErrorCategory(
        "Detection",
        "Low-contrast misses",
        "detection_low_contrast_miss_pages",
        "locked_ocr",
        4,
        4,
        "pages",
    ),
    ErrorCategory(
        "Recognition",
        "Number confusion",
        "recognition_number_confusions",
        "locked_ocr",
        5,
        4,
        "token signals",
    ),
    ErrorCategory(
        "Recognition",
        "Decimal confusion",
        "recognition_decimal_confusions",
        "locked_ocr",
        5,
        4,
        "token signals",
    ),
    ErrorCategory(
        "Recognition",
        "Identifier corruption",
        "recognition_identifier_corruptions",
        "locked_ocr",
        5,
        3,
        "token signals",
    ),
    ErrorCategory(
        "Recognition",
        "Turkish-character errors",
        "recognition_turkish_character_errors",
        "locked_ocr",
        4,
        4,
        "character signals",
    ),
    ErrorCategory(
        "Recognition",
        "Spacing errors",
        "recognition_spacing_error_pages",
        "locked_ocr",
        2,
        4,
        "pages",
    ),
    ErrorCategory(
        "Recognition",
        "Punctuation loss",
        "recognition_punctuation_losses",
        "locked_ocr",
        3,
        4,
        "character signals",
    ),
    ErrorCategory(
        "Recognition",
        "Crop too tight",
        "recognition_crop_tight_retry_signals",
        "locked_ocr",
        4,
        5,
        "retry signals",
    ),
    ErrorCategory(
        "Recognition",
        "Low-resolution errors",
        "recognition_low_resolution_error_pages",
        "locked_ocr",
        4,
        4,
        "pages",
    ),
    ErrorCategory(
        "Recognition",
        "Language-route errors",
        "recognition_language_route_errors",
        "locked_ocr",
        5,
        4,
        "pages",
    ),
    ErrorCategory(
        "Orientation",
        "Wrong cardinal orientation",
        "wrong_cardinal_orientation",
        "orientation",
        5,
        4,
        "page-angle cases",
    ),
    ErrorCategory(
        "Orientation",
        "Wrong deskew",
        "wrong_deskew",
        "orientation",
        4,
        3,
        "page-angle cases",
    ),
    ErrorCategory(
        "Orientation",
        "Close candidate ambiguity",
        "close_candidate_ambiguity",
        "orientation",
        3,
        3,
        "page-angle cases",
    ),
    ErrorCategory(
        "Orientation",
        "Multi-orientation page",
        "multi_orientation_page",
        "orientation",
        5,
        2,
        "page-angle cases",
    ),
    ErrorCategory(
        "Downstream",
        "OCR text correct but entity wrong",
        "downstream_ocr_correct_entity_wrong_pages",
        "locked_ocr",
        5,
        3,
        "pages",
    ),
    ErrorCategory(
        "Downstream",
        "OCR text wrong and entity wrong",
        "downstream_ocr_wrong_entity_wrong_pages",
        "locked_ocr",
        5,
        3,
        "pages",
    ),
    ErrorCategory(
        "Downstream",
        "Field evidence found but abstained",
        "downstream_field_evidence_found_but_abstained",
        "locked_ocr",
        4,
        4,
        "field signals",
    ),
    ErrorCategory(
        "Downstream",
        "Relation missed",
        "downstream_relation_misses",
        "locked_ocr",
        4,
        2,
        "relation signals",
    ),
    ErrorCategory(
        "Downstream",
        "Arithmetic inconsistency",
        "downstream_arithmetic_inconsistency_pages",
        "locked_ocr",
        5,
        4,
        "pages",
    ),
    ErrorCategory(
        "Downstream",
        "Table grouping failure",
        "downstream_table_grouping_failure_pages",
        "locked_ocr",
        5,
        3,
        "pages",
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-root",
        default=str(PROJECT_ROOT / "reports" / "ocr_upgrade"),
    )
    args = parser.parse_args()
    root = Path(args.report_root).resolve()
    evidence = load_evidence(root)
    outputs = {
        "error_analysis.md": render_error_analysis(evidence),
        "final_ocr_model_card.md": render_model_card(evidence),
        "final_upgrade_summary.md": render_upgrade_summary(evidence),
    }
    for name, body in outputs.items():
        atomic_write_text(root / name, body)
    print(
        json.dumps(
            {
                "status": "passed",
                "report_root": str(root),
                "outputs": sorted(outputs),
            },
            indent=2,
        )
    )
    return 0


def load_evidence(root: Path) -> dict[str, Any]:
    missing = [
        str(root / filename)
        for filename in (*JSON_INPUTS.values(), *CSV_INPUTS.values())
        if not (root / filename).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "final OCR-upgrade report inputs are missing: " + ", ".join(missing)
        )
    evidence: dict[str, Any] = {
        name: _json(root / filename) for name, filename in JSON_INPUTS.items()
    }
    evidence.update(
        {
            name: _csv(root / filename)
            for name, filename in CSV_INPUTS.items()
        }
    )
    return evidence


def error_category_rows(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    source_counts = {
        "locked_ocr": dict(evidence["locked_ocr"].get("error_counts") or {}),
        "orientation": dict(
            (evidence["orientation"].get("metrics") or {}).get("error_counts")
            or {}
        ),
    }
    source_samples = {
        "locked_ocr": _integer(evidence["locked_ocr"].get("sample_count")),
        "orientation": _integer(evidence["orientation"].get("sample_count")),
    }
    rows = []
    for category in ERROR_CATEGORIES:
        counts = source_counts[category.source]
        if category.key not in counts:
            raise KeyError(
                "required error category is missing: "
                f"{category.source}.{category.key}"
            )
        count = _integer(counts[category.key])
        sample_count = source_samples[category.source]
        frequency = count / sample_count if sample_count else 0.0
        rows.append(
            {
                "group": category.group,
                "category": category.label,
                "key": category.key,
                "source": category.source,
                "count": count,
                "unit": category.unit,
                "sample_count": sample_count,
                "frequency_per_page": frequency,
                "impact": category.impact,
                "feasibility": category.feasibility,
                "priority_score": (
                    frequency * category.impact * category.feasibility
                ),
            }
        )
    return rows


def render_error_analysis(evidence: Mapping[str, Any]) -> str:
    rows = error_category_rows(evidence)
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row["priority_score"]),
            -int(row["count"]),
            str(row["category"]),
        ),
    )
    lines = [
        "# OCR upgrade public error analysis",
        "",
        "This report uses the one-time public TEST_IN_DOMAIN OCR evaluation "
        "for detection, recognition, and downstream signals. Orientation "
        "signals come from the public DEV_SELECT rotation diagnostic. It "
        "contains no private filenames, OCR text, images, or per-document "
        "predictions.",
        "",
        "The priority score is `frequency per evaluated page × downstream "
        "impact × implementation feasibility`. Impact and feasibility use "
        "a declared 1–5 engineering rubric. The score ranks bounded next "
        "steps; it is not a model-quality metric.",
        "",
        "## Ranked priorities",
        "",
        "| Rank | Group | Category | Count | Unit | Per page | Impact | "
        "Feasibility | Priority |",
        "|---:|---|---|---:|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        (
            f"| {index} | {row['group']} | {row['category']} | "
            f"{row['count']} | {row['unit']} | "
            f"{row['frequency_per_page']:.4f} | {row['impact']} | "
            f"{row['feasibility']} | {row['priority_score']:.4f} |"
        )
        for index, row in enumerate(ranked, start=1)
    )
    lines.extend(
        [
            "",
            "## Measurement boundaries",
            "",
            "- Missed-small-text, false-positive, number, punctuation, "
            "identifier, field, and relation values are occurrence signals "
            "and can exceed the evaluated page count.",
            "- Merged and split region values are deterministic region-count "
            "imbalance signals, not manually labeled merge/split events.",
            "- Low-contrast, low-resolution, rotation, arithmetic, and table "
            "values are reproducible page-level diagnostics from the executed "
            "pipeline.",
            "- A zero means the bounded diagnostic did not observe that "
            "signal. It does not prove the failure mode is impossible.",
            "- TEST_IN_DOMAIN was evaluated only after the OCR and LayoutXLM "
            "stack was frozen. CORU and private outputs were not used for "
            "selection or subsequent tuning.",
            "",
        ]
    )
    return "\n".join(lines)


def render_model_card(evidence: Mapping[str, Any]) -> str:
    detector = _selected(evidence["detector_trials"])
    general = _selected(
        evidence["recognizer_trials"],
        predicate=lambda row: row.get("track") == "general",
    )
    thai = _selected(evidence["thai_trials"])
    layout = _selected(evidence["layout_adaptation_trials"])
    selection = evidence["ocr_selection"]
    locked_ocr = evidence["locked_ocr"]
    locked_e2e = evidence["locked_end_to_end"]
    calibration = evidence["calibration"]
    benchmark = evidence["benchmark"]
    registry_defaults = evidence["registry"].get("defaults") or {}
    lines = [
        "# Final OCR and information-extraction model card",
        "",
        "## Scope",
        "",
        "This artifact is a document information-extraction pre-model for "
        "scanned forms, receipts, and invoices. It combines PaddleOCR "
        "detection and recognition with a LayoutXLM multi-task checkpoint. "
        "The four-zone K-Means output is display-only and never controls OCR "
        "or orientation.",
        "",
        "The measured system is not production-ready and is not claimed to "
        "work accurately on every document.",
        "",
        "## Selected stack",
        "",
        "| Component | Selected artifact | SHA-256 | Decision |",
        "|---|---|---|---|",
        f"| Detector | {_cell(detector.get('model_name'))} | "
        f"`{_cell(detector.get('detector_sha256'))}` | Original retained "
        "after bounded fine-tuning attempts did not produce an eligible "
        "checkpoint. |",
        f"| General recognizer | {_cell(general.get('model_name'))} | "
        f"`{_cell(general.get('recognizer_sha256'))}` | Original retained by "
        "the declared development gate. |",
        f"| Thai recognizer | {_cell(thai.get('model_name'))} | "
        f"`{_cell(thai.get('recognizer_sha256'))}` | Custom synthetic-data "
        "model selected within its stated evidence boundary. |",
        f"| OCR profile | {_cell(selection.get('selected_configuration'))} "
        f"({_cell(selection.get('selected_profile'))}) | "
        f"`{_cell(selection.get('configuration_hash') or selection.get('configuration_sha256'))}` | "
        "Selected on public DEV_SELECT only. |",
        f"| LayoutXLM | {_cell(layout.get('trial_id'))} | "
        f"`{_cell(layout.get('checkpoint_sha256'))}` | Best eligible "
        "downstream trial under the declared regression gates. |",
        "",
        f"Registry defaults: detector `{_cell(registry_defaults.get('detector'))}`, "
        f"general recognizer `{_cell(registry_defaults.get('general_recognizer'))}`, "
        f"Thai recognizer `{_cell(registry_defaults.get('thai_recognizer'))}`.",
        "",
        "## Data and selection boundaries",
        "",
        f"- OCR model comparison used {benchmark.get('sample_count')} public "
        f"DEV_SELECT pages: {_mapping(benchmark.get('counts_by_dataset'))}.",
        "- Detector and recognizer fitting used public TRAIN only. "
        "DEV_SELECT selected models and adaptive components.",
        "- DEV_CALIBRATION was reserved for confidence calibration.",
        "- TEST_IN_DOMAIN, CORU, and Gmail/private documents contributed zero "
        "fit or selection rows.",
        "- The Thai custom recognizer was trained and selected using synthetic "
        "OFL-font evidence. No real labeled public Thai accuracy benchmark was "
        "available, so its claim boundary remains synthetic-data performance "
        "plus integration behavior.",
        "- Dataset and font licenses are recorded in the benchmark report, "
        "training-data report, third-party notices, and model registry.",
        "",
        "## Adaptive OCR behavior",
        "",
        f"- Preprocessing: "
        f"`{_cell(evidence['preprocessing'].get('chosen_profile'))}`.",
        f"- Adaptive PDF rerender rate: "
        f"{_percent(evidence['adaptive_rendering'].get('rerender_trigger_rate'))}.",
        f"- Tiling metric deltas: "
        f"{_mapping(evidence['tiling'].get('metric_deltas'))}.",
        f"- Crop-padding profile: "
        f"`{_cell(evidence['crop_padding'].get('selected_default_profile'))}`.",
        f"- Recognition retry metric deltas: "
        f"{_mapping(evidence['recognition_retry'].get('metric_deltas'))}.",
        f"- Orientation selection accuracy: "
        f"{_percent(_metric(evidence['orientation'], 'orientation_selection_accuracy'))}; "
        "K-Means instantiated: no.",
        "",
        "## Locked public evaluation",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| TEST pages | {locked_ocr.get('sample_count')} |",
        f"| OCR polygon F1 | {_number(_metric(locked_ocr, 'polygon_f1'))} |",
        f"| OCR recognized-text coverage | "
        f"{_number(_metric(locked_ocr, 'recognized_text_coverage'))} |",
        f"| OCR WER | {_number(_metric(locked_ocr, 'wer'))} |",
        f"| Critical-field exact match | "
        f"{_number(_metric(locked_ocr, 'critical_field_exact_match'))} |",
        f"| End-to-end entity F1 | "
        f"{_number(_metric(locked_e2e, 'entity_f1'))} |",
        f"| End-to-end relation F1 | "
        f"{_number(_metric(locked_e2e, 'relation_f1'))} |",
        f"| End-to-end canonical-field accuracy | "
        f"{_number(_metric(locked_e2e, 'canonical_field_accuracy'))} |",
        f"| Calibration ECE | {_number(_metric(calibration, 'ece'))} |",
        "",
        "## Fallback, cache, and portability",
        "",
        "- The original detector and recognizers remain registered as explicit "
        "fallbacks.",
        "- OCR caches are bound to model, preprocessing, and configuration "
        "hashes; mismatched entries are rejected.",
        "- Portable packages contain selected inference artifacts and allowed "
        "fallbacks only. Training data, crops, caches, environments, logs, and "
        "private material are excluded.",
        "",
        "## Known limitations",
        "",
        "- OCR remains the main end-to-end bottleneck, especially for small, "
        "low-contrast, rotated, and tightly cropped text.",
        "- Relation supervision is limited to FUNSD, so relation results do "
        "not generalize across every document family.",
        "- CORU lacks compatible token polygons and is reported as an unseen "
        "operational QA/text-recall evaluation, not a fitting source.",
        "- Thai accuracy is not established on a real labeled public benchmark.",
        "- K-Means zones are diagnostic display output. The failed exact-angle "
        "estimator remains disabled for inference.",
        "",
    ]
    return "\n".join(lines)


def render_upgrade_summary(evidence: Mapping[str, Any]) -> str:
    baseline = evidence["baseline"]
    selected_ocr = evidence["ocr_selection"]
    selected_layout = _selected(evidence["layout_adaptation_trials"])
    locked_ocr = evidence["locked_ocr"]
    locked_e2e = evidence["locked_end_to_end"]
    coru = evidence["unseen_coru"]
    private = evidence["private"]
    stream = _selected(evidence["layout_stream_trials"])
    lines = [
        "# Final OCR upgrade summary",
        "",
        "The upgrade completed a public-only OCR training and selection "
        "lifecycle, rebuilt OCR-realistic LayoutXLM streams, evaluated bounded "
        "downstream adaptation, recalibrated the frozen stack, and reserved "
        "locked TEST_IN_DOMAIN, unseen CORU, and private Gmail operation for "
        "their declared final stages.",
        "",
        "## Decisions",
        "",
        f"- Selected OCR configuration: "
        f"`{_cell(selected_ocr.get('selected_configuration'))}` "
        f"(`{_cell(selected_ocr.get('selected_profile'))}`).",
        f"- Selected preprocessing: "
        f"`{_cell(evidence['preprocessing'].get('chosen_profile'))}`.",
        f"- Selected LayoutXLM trial: `{_cell(selected_layout.get('trial_id'))}` "
        f"with checkpoint `{_cell(selected_layout.get('checkpoint_sha256'))}`.",
        f"- Selected stream manifest: `{_cell(stream.get('manifest_path'))}` "
        f"with {stream.get('train_example_count')} TRAIN and "
        f"{stream.get('dev_select_example_count')} DEV_SELECT examples.",
        "- Original OCR remains available as a fallback. The custom Thai "
        "recognizer remains bounded to synthetic Thai selection evidence.",
        "",
        "## Measured results",
        "",
        "| Evidence | Baseline | Final |",
        "|---|---:|---:|",
        f"| Upright OCR polygon F1 | "
        f"{_number((baseline.get('baseline_ocr') or {}).get('polygon_f1'))} | "
        f"{_number(_metric(locked_ocr, 'polygon_f1'))} |",
        f"| Upright OCR recognized-text coverage | "
        f"{_number((baseline.get('baseline_ocr') or {}).get('recognized_text_coverage'))} | "
        f"{_number(_metric(locked_ocr, 'recognized_text_coverage'))} |",
        f"| Upright OCR WER | "
        f"{_number((baseline.get('baseline_ocr') or {}).get('wer'))} | "
        f"{_number(_metric(locked_ocr, 'wer'))} |",
        f"| End-to-end entity F1 | "
        f"{_number((baseline.get('baseline_end_to_end') or {}).get('entity_f1'))} | "
        f"{_number(_metric(locked_e2e, 'entity_f1'))} |",
        f"| End-to-end relation F1 | "
        f"{_number((baseline.get('baseline_end_to_end') or {}).get('relation_f1'))} | "
        f"{_number(_metric(locked_e2e, 'relation_f1'))} |",
        f"| End-to-end canonical-field accuracy | "
        f"{_number((baseline.get('baseline_end_to_end') or {}).get('canonical_field_accuracy'))} | "
        f"{_number(_metric(locked_e2e, 'canonical_field_accuracy'))} |",
        f"| Unseen CORU QA answer-text recall | "
        f"{_number((baseline.get('unseen_coru') or {}).get('qa_answer_text_recall'))} | "
        f"{_number(coru.get('qa_answer_text_recall'))} |",
        "",
        "The baseline upright OCR and end-to-end values used three pages, while "
        "the final locked result uses the full frozen TEST_IN_DOMAIN sample. "
        "The table preserves both executed results but does not treat that "
        "difference as a controlled paired experiment.",
        "",
        "## Privacy and evaluation boundaries",
        "",
        f"- Locked public failures: {locked_ocr.get('failure_count')} of "
        f"{locked_ocr.get('sample_count')}.",
        f"- CORU: {coru.get('successful_pages')} successful and "
        f"{coru.get('failed_pages')} failed pages from a fixed "
        f"{coru.get('sample_pages')}-page sample.",
        f"- Private operation: {private.get('successful_documents')} successful "
        f"and {private.get('failed_documents')} failed documents; published "
        "output is aggregate-only.",
        f"- Private/Gmail fit rows: {private.get('gmail_fit_rows')}.",
        "- No decision or configuration change was made from TEST_IN_DOMAIN, "
        "CORU, or private output.",
        "",
        "## Remaining limitations",
        "",
        "- End-to-end extraction remains bounded by OCR quality.",
        "- Detector fine-tuning was stopped for documented memory, numerical, "
        "and runtime constraints; the stronger eligible original detector was "
        "retained.",
        "- The general custom recognizer improved several development metrics "
        "but missed the declared WER gate, so it was not made the default.",
        "- Thai evidence is synthetic and does not establish real-world Thai "
        "accuracy.",
        "- Physical Apple hardware remains untested. Docker Linux/AMD64 is the "
        "supported macOS route.",
        "",
        "## Reproduction entry points",
        "",
        "```powershell",
        "python scripts/compile_ocr_upgrade_reports.py",
        "python -m pytest -q",
        "python -m compileall -q src scripts tests",
        "D:\\CSX4201\\vision-info-extraction-assets\\environments\\ie-ocr\\Scripts\\python.exe scripts\\verify_information_extraction.py --complete",
        "```",
        "",
        "The exact training, evaluation, inference, portable-build, and privacy "
        "commands are recorded in the repository documentation and executed "
        "reports. Reported hashes are authoritative for artifact identity.",
        "",
    ]
    return "\n".join(lines)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"expected non-empty CSV: {path}")
    return rows


def _selected(
    rows: list[dict[str, Any]],
    *,
    predicate: Callable[[Mapping[str, Any]], bool] | None = None,
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if _truthy(row.get("selected"))
        and (predicate is None or predicate(row))
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one selected row, found {len(matches)}"
        )
    return matches[0]


def _truthy(value: Any) -> bool:
    return str(value).casefold() in {"1", "true", "yes"}


def _integer(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _metric(report: Mapping[str, Any], name: str) -> Any:
    metrics = report.get("metrics")
    if isinstance(metrics, Mapping) and name in metrics:
        return metrics[name]
    return report.get(name)


def _number(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "not available"


def _percent(value: Any) -> str:
    try:
        return f"{100.0 * float(value):.2f}%"
    except (TypeError, ValueError):
        return "not available"


def _cell(value: Any) -> str:
    if value is None or str(value).strip() == "":
        return "not available"
    return str(value).replace("|", "\\|")


def _mapping(value: Any) -> str:
    if not isinstance(value, Mapping) or not value:
        return "not available"
    return ", ".join(
        f"{key}={_number(item) if isinstance(item, float) else item}"
        for key, item in sorted(value.items())
    )


if __name__ == "__main__":
    raise SystemExit(main())
