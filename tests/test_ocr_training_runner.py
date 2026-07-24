from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.ocr.training_runner import (
    deep_merge,
    load_trial_definition,
    resolve_paddle_config,
)


def test_deep_merge_replaces_lists_and_preserves_unmodified_mapping() -> None:
    base = {
        "Global": {"epoch_num": 100, "use_gpu": True},
        "Train": {"loader": {"batch_size_per_card": 8}, "items": [1, 2]},
    }
    patch = {
        "Global": {"epoch_num": 2},
        "Train": {"items": [3], "loader": {"num_workers": 2}},
    }

    merged = deep_merge(base, patch)

    assert merged["Global"] == {"epoch_num": 2, "use_gpu": True}
    assert merged["Train"]["items"] == [3]
    assert merged["Train"]["loader"] == {
        "batch_size_per_card": 8,
        "num_workers": 2,
    }
    assert base["Global"]["epoch_num"] == 100


def test_resolved_trial_config_injects_external_output_and_checkpoint(
    tmp_path: Path,
) -> None:
    vendor = tmp_path / "vendor"
    config_dir = vendor / "configs"
    config_dir.mkdir(parents=True)
    base_path = config_dir / "base.yml"
    base_path.write_text(
        yaml.safe_dump(
            {
                "Global": {
                    "epoch_num": 100,
                    "pretrained_model": None,
                    "save_model_dir": "./output",
                },
                "Train": {
                    "dataset": {
                        "data_dir": "D:/assets/data/train",
                        "label_file_list": ["D:/assets/data/train_list.txt"],
                    }
                },
                "Eval": {
                    "dataset": {
                        "data_dir": "D:/assets/data/dev",
                        "label_file_list": ["D:/assets/data/dev_list.txt"],
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "checkpoint.pdparams"
    checkpoint.write_bytes(b"weights")
    definition_path = tmp_path / "trial.yml"
    definition_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "trial_id": "det_trial_01",
                "kind": "detector",
                "base_config": "configs/base.yml",
                "source_checkpoint": checkpoint.as_posix(),
                "overrides": {
                    "Global": {"epoch_num": 2},
                    "Train": {"loader": {"batch_size_per_card": 4}},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    definition = load_trial_definition(definition_path, expected_kind="detector")
    output = tmp_path / "artifacts" / "det_trial_01"

    resolved = resolve_paddle_config(
        definition,
        vendor_root=vendor,
        output_root=output,
    )

    assert resolved["Global"]["epoch_num"] == 2
    assert resolved["Global"]["pretrained_model"] == checkpoint.as_posix()
    assert resolved["Global"]["save_model_dir"] == (
        output / "training"
    ).as_posix()
    assert resolved["Train"]["loader"]["batch_size_per_card"] == 4


@pytest.mark.parametrize("forbidden", ["test_in_domain", "gmail", "private_test"])
def test_trial_definition_refuses_forbidden_training_paths(
    tmp_path: Path,
    forbidden: str,
) -> None:
    checkpoint = tmp_path / "checkpoint.pdparams"
    checkpoint.write_bytes(b"weights")
    definition_path = tmp_path / "trial.yml"
    definition_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "trial_id": "rec_trial_01",
                "kind": "general_recognizer",
                "base_config": "configs/base.yml",
                "source_checkpoint": checkpoint.as_posix(),
                "overrides": {
                    "Train": {
                        "dataset": {
                            "label_file_list": [f"D:/data/{forbidden}/train.txt"]
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="forbidden training data"):
        load_trial_definition(
            definition_path,
            expected_kind="general_recognizer",
        )
