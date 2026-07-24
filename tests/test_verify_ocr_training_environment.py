from __future__ import annotations

from scripts.verify_ocr_training_environment import (
    prepare_architecture_for_training,
)


class _PostProcessor:
    character = tuple("abc ")


def test_recognition_architecture_receives_dictionary_output_channels() -> None:
    config = {
        "Global": {},
        "Architecture": {
            "model_type": "rec",
            "Head": {"name": "MultiHead"},
        },
        "PostProcess": {"name": "CTCLabelDecode"},
        "Loss": {
            "loss_config_list": [
                {"CTCLoss": None},
                {"NRTRLoss": None},
            ]
        },
    }

    prepare_architecture_for_training(
        config,
        lambda _postprocess, _global: _PostProcessor(),
    )

    assert config["Architecture"]["Head"]["out_channels_list"] == {
        "CTCLabelDecode": 4,
        "NRTRLabelDecode": 7,
    }


def test_detection_architecture_needs_no_character_channels() -> None:
    config = {
        "Architecture": {
            "model_type": "det",
            "Head": {"name": "DBHead"},
        }
    }

    prepare_architecture_for_training(
        config,
        lambda *_: (_ for _ in ()).throw(AssertionError("must not be called")),
    )

    assert "out_channels" not in config["Architecture"]["Head"]
