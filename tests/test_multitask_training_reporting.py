from __future__ import annotations

from scripts.train_multitask_model import _training_report_targets


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
