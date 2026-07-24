import json
from pathlib import Path

import scripts.build_portable_release as builder
from scripts.build_portable_release import copy_application


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
