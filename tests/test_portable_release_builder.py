import json
import zipfile
from pathlib import Path

import pytest

import scripts.build_portable_release as builder
from scripts.build_portable_release import copy_application


def _write_private_inventory(
    source_root: Path,
    *,
    filename: str = "private_statement_9284.pdf",
) -> None:
    metadata = source_root / "data" / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / "private_file_inventory.csv").write_text(
        "original_filename,current_relative_path\n"
        f"{filename},private/{filename}\n",
        encoding="utf-8",
    )


def test_portable_builder_includes_license_and_contribution_policy(
    tmp_path: Path,
) -> None:
    target = tmp_path / "OCR_Model"

    copy_application(target)

    license_text = (target / "LICENSE").read_text(encoding="utf-8")
    contributing_text = (target / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "MIT License" in license_text
    assert "Copyright (c) 2026 Sithu Win San" in license_text
    assert "solo academic project" in contributing_text
    assert "pull requests" in contributing_text
    assert "welcome" in contributing_text
    assert not (target / "docs" / "devpost").exists()


def test_portable_registry_keeps_originals_and_selected_custom_only(
    tmp_path: Path, monkeypatch
) -> None:
    source_root = tmp_path / "source"
    registry_root = source_root / "reports" / "ocr_upgrade"
    registry_root.mkdir(parents=True)
    custom = tmp_path / "custom"
    custom.mkdir()
    (custom / "model.json").write_text("{}", encoding="utf-8")
    registry = {
        "defaults": {
            "detector": "original",
            "general_recognizer": "original",
            "thai_recognizer": "custom",
        },
        "models": {
            "det_original": {
                "variant": "original",
                "upstream_base": "det",
                "available": True,
                "accepted": True,
                "selected": True,
            },
            "thai_custom": {
                "variant": "custom",
                "upstream_base": "thai",
                "available": True,
                "accepted": True,
                "selected": True,
                "local_path": str(custom),
            },
            "general_rejected": {
                "variant": "custom",
                "upstream_base": "general",
                "available": True,
                "accepted": False,
                "selected": False,
                "local_path": str(tmp_path / "must-not-copy"),
            },
        },
    }
    (registry_root / "model_registry.json").write_text(
        json.dumps(registry), encoding="utf-8"
    )
    monkeypatch.setattr(builder, "PROJECT_ROOT", source_root)
    target = tmp_path / "target"

    builder._copy_upgrade_registry(target)

    portable = json.loads(
        (
            target / "reports" / "ocr_upgrade" / "model_registry.json"
        ).read_text(encoding="utf-8")
    )
    assert portable["models"]["det_original"]["portable_included"] is True
    assert portable["models"]["thai_custom"]["portable_included"] is True
    assert portable["models"]["general_rejected"]["available"] is False
    assert (
        target
        / "assets"
        / "ocr_models"
        / "thai_custom"
        / "model.json"
    ).is_file()


def test_selected_layout_checkpoint_uses_config_and_calibration_binding(
    tmp_path: Path, monkeypatch
) -> None:
    source_root = tmp_path / "source"
    checkpoint = tmp_path / "selected" / "checkpoint"
    checkpoint.mkdir(parents=True)
    model = checkpoint / "model.safetensors"
    model.write_bytes(b"selected-layout-model")
    expected_hash = builder.sha256_file(model)
    (source_root / "models").mkdir(parents=True)
    (source_root / "config.yaml").write_text(
        "layout_model:\n"
        f"  inference_checkpoint: {checkpoint.as_posix()}\n",
        encoding="utf-8",
    )
    (source_root / "models" / "multitask_calibration.json").write_text(
        json.dumps({"checkpoint_model_sha256": expected_hash}),
        encoding="utf-8",
    )
    monkeypatch.setattr(builder, "PROJECT_ROOT", source_root)

    resolved, actual_hash, calibration = builder.selected_layout_checkpoint(
        tmp_path / "legacy-assets"
    )

    assert resolved == checkpoint.resolve()
    assert actual_hash == expected_hash
    assert calibration["checkpoint_model_sha256"] == expected_hash


def test_selected_layout_checkpoint_rejects_stale_calibration(
    tmp_path: Path, monkeypatch
) -> None:
    source_root = tmp_path / "source"
    checkpoint = tmp_path / "selected" / "checkpoint"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"new-layout-model")
    (source_root / "models").mkdir(parents=True)
    (source_root / "config.yaml").write_text(
        "layout_model:\n"
        f"  inference_checkpoint: {checkpoint.as_posix()}\n",
        encoding="utf-8",
    )
    (source_root / "models" / "multitask_calibration.json").write_text(
        json.dumps({"checkpoint_model_sha256": "0" * 64}),
        encoding="utf-8",
    )
    monkeypatch.setattr(builder, "PROJECT_ROOT", source_root)

    with pytest.raises(
        ValueError,
        match="selected checkpoint is not bound to the current calibration",
    ):
        builder.selected_layout_checkpoint(tmp_path / "legacy-assets")


def test_privacy_audit_scans_final_payload_and_passes_safe_files(
    tmp_path: Path, monkeypatch
) -> None:
    source_root = tmp_path / "source"
    _write_private_inventory(source_root)
    target = tmp_path / "OCR_Model"
    target.mkdir()
    (target / "BUILD_INFO.json").write_text(
        '{"source_tree_dirty_at_build": false}\n',
        encoding="utf-8",
    )
    (target / "README.md").write_text("Safe synthetic package.\n", encoding="utf-8")
    monkeypatch.setattr(builder, "PROJECT_ROOT", source_root)

    audit = builder.privacy_audit(target)

    assert audit["status"] == "pass"
    assert audit["credentials_included"] is False
    assert audit["private_filename_inventory_count"] == 1
    assert (target / "PRIVACY_AUDIT.json").is_file()


def test_privacy_audit_rejects_private_filename_content(
    tmp_path: Path, monkeypatch
) -> None:
    source_root = tmp_path / "source"
    private_name = "private_statement_9284.pdf"
    _write_private_inventory(source_root, filename=private_name)
    target = tmp_path / "OCR_Model"
    target.mkdir()
    (target / "leak.txt").write_text(
        f"source={private_name}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(builder, "PROJECT_ROOT", source_root)

    with pytest.raises(ValueError, match="privacy audit failed"):
        builder.privacy_audit(target)

    audit = json.loads(
        (target / "PRIVACY_AUDIT.json").read_text(encoding="utf-8")
    )
    assert audit["private_filename_hit_files"] == ["leak.txt"]


def test_privacy_audit_rejects_secret_pattern(
    tmp_path: Path, monkeypatch
) -> None:
    source_root = tmp_path / "source"
    _write_private_inventory(source_root)
    target = tmp_path / "OCR_Model"
    target.mkdir()
    (target / "credentials.txt").write_text(
        'api_key = "' + ("a" * 32) + '"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(builder, "PROJECT_ROOT", source_root)

    with pytest.raises(ValueError, match="privacy audit failed"):
        builder.privacy_audit(target)

    audit = json.loads(
        (target / "PRIVACY_AUDIT.json").read_text(encoding="utf-8")
    )
    assert audit["credentials_included"] is True
    assert audit["secret_hit_files"] == ["credentials.txt"]


def test_privacy_audit_scans_large_binary_files_without_skipping(
    tmp_path: Path, monkeypatch
) -> None:
    source_root = tmp_path / "source"
    private_name = "private_statement_9284.pdf"
    _write_private_inventory(source_root, filename=private_name)
    target = tmp_path / "OCR_Model"
    target.mkdir()
    (target / "large.bin").write_bytes(
        b"\0" * (5 * 1024 * 1024)
        + private_name.encode("utf-8")
        + b'\napi_key="' + (b"a" * 32) + b'"\n'
    )
    monkeypatch.setattr(builder, "PROJECT_ROOT", source_root)

    with pytest.raises(ValueError, match="privacy audit failed"):
        builder.privacy_audit(target)

    audit = json.loads(
        (target / "PRIVACY_AUDIT.json").read_text(encoding="utf-8")
    )
    assert audit["private_filename_hit_files"] == ["large.bin"]
    assert audit["secret_hit_files"] == ["large.bin"]


def test_privacy_audit_fails_closed_on_empty_private_inventory(
    tmp_path: Path, monkeypatch
) -> None:
    source_root = tmp_path / "source"
    metadata = source_root / "data" / "metadata"
    metadata.mkdir(parents=True)
    (metadata / "private_file_inventory.csv").write_text(
        "original_filename,current_relative_path\n",
        encoding="utf-8",
    )
    target = tmp_path / "OCR_Model"
    target.mkdir()
    monkeypatch.setattr(builder, "PROJECT_ROOT", source_root)

    with pytest.raises(ValueError, match="no usable filename"):
        builder.privacy_audit(target)


def test_prepare_target_refuses_every_existing_target(tmp_path: Path) -> None:
    target = tmp_path / "OCR_Model"
    (target / ".runtime").mkdir(parents=True)

    with pytest.raises(FileExistsError, match="choose a new isolated"):
        builder.prepare_target(target)

    assert target.is_dir()


def test_build_location_rejects_installed_working_copy() -> None:
    with pytest.raises(ValueError, match="must never target"):
        builder.validate_build_location(
            builder.INSTALLED_TARGET,
            builder.DEFAULT_ASSET_ROOT,
        )


def test_build_location_rejects_caller_redefined_asset_root() -> None:
    with pytest.raises(ValueError, match="asset root must be exactly"):
        builder.validate_build_location(
            Path("C:/review-staging/OCR_Model"),
            Path("C:/review-staging"),
        )


def test_build_location_accepts_authorized_staging_root() -> None:
    builder.validate_build_location(
        builder.DEFAULT_ASSET_ROOT
        / "release-staging"
        / "test"
        / "OCR_Model",
        builder.DEFAULT_ASSET_ROOT,
    )


def test_release_provenance_must_remain_unchanged() -> None:
    initial = {
        "source_commit": "a" * 40,
        "source_tree_dirty": False,
        "source_tree_sha256": "b" * 64,
        "source_candidate_file_count": 10,
        "source_missing_candidate_paths": [],
    }
    builder.require_unchanged_provenance(initial, dict(initial))

    changed = dict(initial)
    changed["source_tree_sha256"] = "c" * 64
    with pytest.raises(RuntimeError, match="source_tree_sha256"):
        builder.require_unchanged_provenance(initial, changed)


def test_copy_tree_rejects_source_symlink_or_reparse_point(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    link = source / "linked.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable on this Windows host")

    with pytest.raises(ValueError, match="reparse point"):
        builder.copy_tree(source, tmp_path / "target")


def test_payload_manifest_hashes_every_preexisting_file(
    tmp_path: Path,
) -> None:
    target = tmp_path / "OCR_Model"
    (target / "nested").mkdir(parents=True)
    (target / "README.md").write_text("portable\n", encoding="utf-8")
    (target / "nested" / "model.bin").write_bytes(b"model")

    manifest = builder.write_payload_manifest(target)

    assert manifest["file_count"] == 2
    assert {record["path"] for record in manifest["files"]} == {
        "README.md",
        "nested/model.bin",
    }
    assert (target / "PAYLOAD_MANIFEST.json").is_file()


def test_zip_creation_is_crc_checked_and_single_root(
    tmp_path: Path,
) -> None:
    target = tmp_path / "OCR_Model"
    target.mkdir()
    (target / "README.md").write_text("portable\n", encoding="utf-8")
    builder.write_payload_manifest(target)

    archive, digest, verification = builder.make_zip(target)

    assert archive.is_file()
    assert len(digest) == 64
    assert verification == {
        "entry_count": 2,
        "root": "OCR_Model",
        "crc_passed": True,
        "duplicate_count": 0,
        "unsafe_path_count": 0,
        "manifest_file_count": 2,
        "manifest_match": True,
    }


def test_zip_validation_rejects_traversal_path(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../escape.txt", "unsafe")

    with pytest.raises(ValueError, match="ZIP validation failed"):
        builder.validate_zip_archive(archive, "OCR_Model")


def test_zip_creation_rejects_payload_changed_after_manifest(
    tmp_path: Path,
) -> None:
    target = tmp_path / "OCR_Model"
    target.mkdir()
    readme = target / "README.md"
    readme.write_text("portable\n", encoding="utf-8")
    builder.write_payload_manifest(target)
    readme.write_text("mutated\n", encoding="utf-8")

    with pytest.raises(ValueError, match="changed after manifest"):
        builder.make_zip(target)
