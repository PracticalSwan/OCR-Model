from __future__ import annotations

from scripts.verify_ocr_training_environment import (
    prepare_architecture_for_training,
    resolve_vendor_config_paths,
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


def test_vendor_dictionary_path_is_resolved(tmp_path) -> None:
    config = {
        "Global": {
            "character_dict_path": "ppocr/utils/dict/ppocrv6_dict.txt",
        }
    }

    resolve_vendor_config_paths(config, tmp_path)

    assert config["Global"]["character_dict_path"] == str(
        (tmp_path / "ppocr/utils/dict/ppocrv6_dict.txt").resolve()
    )
