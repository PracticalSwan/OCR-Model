from __future__ import annotations

import json

import pytest

from scripts.report_multitask_training import (
    selected_adaptation_trial,
    validate_adaptation_promotion,
)
from scripts.train_multitask_model import _training_report_targets
from src.rotation_common import sha256_file


def test_trial_training_report_does_not_overwrite_canonical_reports(
    tmp_path,
) -> None:
    targets = _training_report_targets(
        tmp_path,
        profile="final",
        trial_id="adapt_a",
        publish_canonical=False,
    )

    assert targets == [
        tmp_path / "ocr_upgrade" / "layout_training" / "adapt_a.json"
    ]


def test_selected_trial_can_be_explicitly_published_canonically(
    tmp_path,
) -> None:
    targets = _training_report_targets(
        tmp_path,
        profile="final",
        trial_id="fresh_selected",
        publish_canonical=True,
    )

    assert targets == [
        tmp_path
        / "ocr_upgrade"
        / "layout_training"
        / "fresh_selected.json",
        tmp_path / "final_model" / "multitask_training_final.json",
        tmp_path
        / "information_extraction"
        / "layout_model_training.json",
    ]


def _adaptation_row(tmp_path) -> dict[str, str]:
    return {
        "trial_id": "continued_a",
        "selected": "True",
        "selection_candidate": "True",
        "eligible": "True",
        "cross_build_comparison": "False",
        "checkpoint_reload_passed": "True",
        "training_report": str(tmp_path / "training.json"),
        "build_id": "build-v2",
        "manifest_sha256": "a" * 64,
        "selection_score": "0.8",
    }


def test_selected_adaptation_must_be_build_bound(tmp_path) -> None:
    row = _adaptation_row(tmp_path)

    assert selected_adaptation_trial([row])["trial_id"] == "continued_a"
    row["cross_build_comparison"] = "True"
    with pytest.raises(ValueError, match="build-bound"):
        selected_adaptation_trial([row])


def test_adaptation_promotion_validates_checkpoint_and_manifest(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"model")
    training_path = tmp_path / "training.json"
    row = _adaptation_row(tmp_path)
    row["checkpoint_sha256"] = sha256_file(
        checkpoint / "model.safetensors"
    )
    training = {
        "profile": "final",
        "public_only": True,
        "gmail_fit_rows": 0,
        "checkpoint_reload_passed": True,
        "build_id": row["build_id"],
        "manifest_sha256": row["manifest_sha256"],
        "checkpoint": str(checkpoint),
    }
    training_path.write_text(json.dumps(training), encoding="utf-8")

    validate_adaptation_promotion(
        row,
        training,
        training_path=training_path,
    )

    training["build_id"] = "wrong"
    with pytest.raises(ValueError, match="build_id"):
        validate_adaptation_promotion(
            row,
            training,
            training_path=training_path,
        )
