from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.run_integration_smoke import _command_record, _resolve_checkpoint
from scripts.verify_information_extraction import (
    _integration_semantic_errors,
    _expected_portable_provenance,
    _load_execution_evidence,
    _metric_report_provenance_errors,
    _private_name_scan,
    _secret_scan,
    _split_leakage_summary,
    _upgrade_report_inventory,
    _valid_locked_unseen_evaluation,
)
from scripts.record_ocr_upgrade_verification import (
    _load_ledger,
    _parse_additional_hashes,
    _portable_path,
)


def test_integration_command_records_effective_checkpoint_and_paths(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "selected-checkpoint"
    command = _command_record(
        config_path=tmp_path / "config.yaml",
        device="gpu:0",
        model_setup_path=tmp_path / "model_setup.json",
        checkpoint=checkpoint,
        artifact_root=tmp_path / "artifacts",
        output_path=tmp_path / "integration.json",
    )

    assert command[command.index("--model-checkpoint") + 1] == str(checkpoint)
    assert "--model-setup" in command
    assert "--artifact-root" in command
    assert "--output" in command


def test_integration_smoke_checkpoint_default_comes_from_config(
    tmp_path: Path,
) -> None:
    configured = tmp_path / "configured-checkpoint"
    cfg = {
        "paths": {"project_root": str(tmp_path)},
        "layout_model": {"inference_checkpoint": str(configured)},
    }

    assert _resolve_checkpoint(cfg, None) == configured.resolve()


def test_integration_provenance_covers_the_learned_worker_call_path() -> None:
    root = Path(__file__).resolve().parents[1]
    sections = []
    for relative, marker in (
        ("scripts/run_integration_smoke.py", "required_sources = {"),
        ("scripts/verify_information_extraction.py", "expected_sources = {"),
    ):
        text = (root / relative).read_text(encoding="utf-8")
        sections.append(text.split(marker, 1)[1].split("\n    }", 1)[0])

    required = {
        "entity_worker_client": "entity_worker_client.py",
        "layout_entity_worker": "layout_entity_worker.py",
        "multitask_inference": "multitask_inference.py",
        "layoutxlm_model": "layoutxlm_model.py",
    }
    for section in sections:
        for key, filename in required.items():
            assert f'"{key}"' in section
            assert f'"{filename}"' in section


def test_integration_verifier_accepts_learned_document_type_for_generic_fixture() -> None:
    payload = {
        "source_type": "image",
        "document_type": {"label": "invoice", "confidence": 0.75},
        "rotation_display": {"purpose": "display_only"},
        "pages": [{
            "ocr": {"words": [{"text": "TOTAL"}]},
            "entities": [{"id": "entity-1"}],
            "key_value_pairs": [{"id": "relation-1"}],
        }],
    }

    assert _integration_semantic_errors("unknown_upright_image", payload) == []


def test_split_leakage_summary_catches_document_and_hash_crossovers() -> None:
    rows = [
        {
            "dataset": "funsd", "document_id": "doc-1", "page_id": "p1",
            "split_group_id": "group-1", "duplicate_group_id": "dup-1",
            "sha256": "abc", "project_split": "train",
        },
        {
            "dataset": "funsd", "document_id": "doc-1", "page_id": "p2",
            "split_group_id": "group-2", "duplicate_group_id": "dup-2",
            "sha256": "abc", "project_split": "test_in_domain",
        },
    ]

    summary = _split_leakage_summary(rows)

    assert summary["violation_count"] == 2
    assert any(value.startswith("document:") for value in summary["violation_sample"])
    assert any(value.startswith("sha256:") for value in summary["violation_sample"])


def test_streaming_publication_scans_find_private_name_and_secret(tmp_path) -> None:
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "private_file_inventory.csv").write_text(
        "relative_path\nprivate/source-sensitive.pdf\n", encoding="utf-8"
    )
    candidate = tmp_path / "candidate.txt"
    candidate.write_text(
        "source-sensitive.pdf\n"
        + "api_" + "key='" + "abcdefghijklmnop" + "1234'\n",
        encoding="utf-8",
    )

    assert _private_name_scan([candidate], metadata) == 1
    assert _secret_scan([candidate])


def test_private_name_scan_supports_live_inventory_columns(tmp_path) -> None:
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "private_file_inventory.csv").write_text(
        "file_id,original_filename,current_relative_path\n"
        "private_1,private-source.pdf,data/raw/private/gmail/private-source.pdf\n",
        encoding="utf-8",
    )
    candidate = tmp_path / "candidate.txt"
    candidate.write_text("private-source.pdf\n", encoding="utf-8")

    assert _private_name_scan([candidate], metadata) == 1


def test_private_name_scan_fails_closed_without_inventory(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="inventory"):
        _private_name_scan([], tmp_path)

    (tmp_path / "private_file_inventory.csv").write_text(
        "file_id,original_filename\nprivate_1,\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="no usable filename"):
        _private_name_scan([], tmp_path)


def test_unseen_evaluation_requires_locked_100_page_zero_failure_run() -> None:
    checkpoint_hash = "a" * 64
    report = {
        "dataset": "coru",
        "split": "unseen_domain_test",
        "public_only": True,
        "private_page_count": 0,
        "sample_pages": 100,
        "successful_pages": 100,
        "failed_pages": 0,
        "checkpoint_model_sha256": checkpoint_hash,
    }

    assert _valid_locked_unseen_evaluation(
        report, checkpoint_model_sha256=checkpoint_hash
    )
    report["successful_pages"] = 99
    report["failed_pages"] = 1
    assert not _valid_locked_unseen_evaluation(
        report, checkpoint_model_sha256=checkpoint_hash
    )


def test_metric_report_provenance_accepts_json_and_csv_alias(
    tmp_path: Path,
) -> None:
    values = {
        "build_id": "build",
        "split": "dev_select",
        "manifest_sha256": "a" * 64,
        "detector_sha256": "b" * 64,
        "recognizer_sha256": "c" * 64,
        "checkpoint_sha256": "d" * 64,
        "calibration_sha256": "",
        "configuration_hash": "e" * 64,
        "source_commit": "1" * 40,
        "device": "gpu:0",
        "sample_count": 1,
        "failure_count": 0,
        "duration_seconds": 1.0,
        "private_row_count": 0,
    }
    json_path = tmp_path / "metrics.json"
    json_path.write_text(json.dumps(values), encoding="utf-8")
    csv_path = tmp_path / "metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values))
        writer.writeheader()
        writer.writerow(values)

    assert _metric_report_provenance_errors(json_path) == []
    assert _metric_report_provenance_errors(csv_path) == []


def test_upgrade_inventory_reports_missing_and_bad_provenance(
    tmp_path: Path,
) -> None:
    (tmp_path / "training_environment.json").write_text(
        json.dumps({"build_id": "incomplete"}),
        encoding="utf-8",
    )

    inventory = _upgrade_report_inventory(
        tmp_path,
        required_files=(
            "training_environment.json",
            "final_upgrade_summary.md",
        ),
        metric_files=("training_environment.json",),
    )

    assert inventory["missing"] == ["final_upgrade_summary.md"]
    assert "training_environment.json" in inventory["provenance_errors"]
    assert any(
        "duration_seconds" in error
        for error in inventory["provenance_errors"][
            "training_environment.json"
        ]
    )


def test_execution_evidence_requires_exact_passing_artifact_backed_matrix(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence.json"
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps({"source_commit": "a" * 40}),
        encoding="utf-8",
    )
    artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    evidence.write_text(
        json.dumps(
            {
                "checks": [
                    {
                        "name": "host_tests",
                        "command": "python -m pytest -q",
                        "status": "passed",
                        "evidence_path": str(artifact),
                        "timestamp": "2026-07-26T12:00:00+00:00",
                        "relevant_hashes": {
                            "source_commit": "a" * 40,
                            "evidence_sha256": artifact_sha256,
                        },
                        "detail": {"passed": 365, "skipped": 2},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    checks, errors = _load_execution_evidence(
        evidence,
        required_names=("host_tests",),
    )

    assert errors == []
    assert checks[0]["passed"] is True
    assert checks[0]["evidence_path"] == str(artifact)

    payload = json.loads(evidence.read_text(encoding="utf-8"))
    payload["checks"][0]["status"] = "failed"
    payload["checks"][0]["evidence_path"] = str(tmp_path / "missing.json")
    payload["checks"].append(dict(payload["checks"][0]))
    evidence.write_text(json.dumps(payload), encoding="utf-8")

    checks, errors = _load_execution_evidence(
        evidence,
        required_names=("host_tests", "compileall"),
    )

    assert checks[0]["passed"] is False
    assert any("status_not_passed" in error for error in errors)
    assert any("evidence_path_missing" in error for error in errors)
    assert any("duplicate_name" in error for error in errors)
    assert any("missing_checks:compileall" in error for error in errors)


def test_execution_evidence_rejects_hash_or_source_commit_drift(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence.json"
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps({"source_commit": "b" * 40}),
        encoding="utf-8",
    )
    evidence.write_text(
        json.dumps(
            {
                "checks": [
                    {
                        "name": "host_tests",
                        "command": "python -m pytest -q",
                        "status": "passed",
                        "evidence_path": str(artifact),
                        "timestamp": "2026-07-30T12:00:00+00:00",
                        "relevant_hashes": {
                            "source_commit": "a" * 40,
                            "evidence_sha256": "0" * 64,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    checks, errors = _load_execution_evidence(
        evidence,
        required_names=("host_tests",),
    )

    assert checks[0]["passed"] is False
    assert any("evidence_sha256_mismatch" in error for error in errors)
    assert any("evidence_source_commit_mismatch" in error for error in errors)


def test_portable_execution_evidence_rejects_generation_mismatch(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "portable_verification.json"
    artifact.write_text(
        json.dumps(
            {
                "source_commit": "a" * 40,
                "source_tree_sha256": "b" * 64,
                "source_candidate_file_count": 10,
                "source_tree_dirty_at_build": False,
            }
        ),
        encoding="utf-8",
    )
    artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    row_hashes = {
        "source_commit": "a" * 40,
        "source_tree_sha256": "c" * 64,
        "source_candidate_file_count": 11,
        "source_tree_dirty_at_build": "false",
        "evidence_sha256": artifact_sha256,
    }
    evidence = tmp_path / "ledger.json"
    evidence.write_text(
        json.dumps(
            {
                "portable_generation": dict(row_hashes),
                "checks": [
                    {
                        "name": "portable_package_verification",
                        "command": "verify portable package",
                        "status": "passed",
                        "evidence_path": str(artifact),
                        "timestamp": "2026-07-30T12:00:00+00:00",
                        "relevant_hashes": row_hashes,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    checks, errors = _load_execution_evidence(
        evidence,
        required_names=("portable_package_verification",),
        expected_portable_provenance={
            "source_commit": "a" * 40,
            "source_tree_sha256": "b" * 64,
            "source_candidate_file_count": 10,
            "source_tree_dirty_at_build": False,
        },
    )

    assert checks[0]["passed"] is False
    assert any("portable_source_tree_sha256_mismatch" in error for error in errors)
    assert any(
        "portable_source_candidate_file_count_mismatch" in error
        for error in errors
    )


def test_portable_expected_provenance_rejects_report_selected_old_package(
    tmp_path: Path,
) -> None:
    old_package = tmp_path / "old" / "OCR_Model"
    current_package = tmp_path / "current" / "OCR_Model"
    old_package.mkdir(parents=True)
    current_package.mkdir(parents=True)
    old_generation = {
        "source_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "source_candidate_file_count": 10,
        "source_tree_dirty_at_build": False,
    }
    current_generation = {
        "source_commit": "c" * 40,
        "source_tree_sha256": "d" * 64,
        "source_candidate_file_count": 20,
        "source_tree_dirty_at_build": False,
    }
    (old_package / "BUILD_INFO.json").write_text(
        json.dumps(old_generation),
        encoding="utf-8",
    )
    (current_package / "BUILD_INFO.json").write_text(
        json.dumps(current_generation),
        encoding="utf-8",
    )
    report = tmp_path / "portable_verification.json"
    report.write_text(
        json.dumps(
            {
                **old_generation,
                "package": {"directory": str(old_package)},
            }
        ),
        encoding="utf-8",
    )

    provenance, errors = _expected_portable_provenance(
        report,
        package_directory=current_package,
        expected_source_commit=current_generation["source_commit"],
    )

    assert provenance is None
    assert errors == ["portable_verification_directory_not_designated"]


def test_verification_recorder_helpers_reject_duplicate_or_unknown_checks(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps(
            {
                "checks": [
                    {"name": "all_original_tests"},
                    {"name": "all_original_tests"},
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        _load_ledger(ledger)

    ledger.write_text(
        json.dumps({"checks": [{"name": "invented_check"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown"):
        _load_ledger(ledger)


def test_verification_recorder_rejects_reserved_hash_overrides() -> None:
    with pytest.raises(ValueError, match="recorder-controlled"):
        _parse_additional_hashes(
            ["source_commit=" + "b" * 40],
            reserved_keys={"source_commit", "evidence_sha256"},
        )
    with pytest.raises(ValueError, match="more than once"):
        _parse_additional_hashes(
            ["model_sha256=" + "a" * 64, "model_sha256=" + "b" * 64],
            reserved_keys=set(),
        )

    assert _parse_additional_hashes(
        ["model_sha256=" + "a" * 64],
        reserved_keys={"source_commit"},
    ) == {"model_sha256": "a" * 64}

    project_file = Path(__file__).resolve()
    assert _portable_path(project_file).endswith(
        "tests/test_information_verifier.py"
    )
