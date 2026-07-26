"""Deterministic, bounded OCR-noise variants for public LayoutXLM training."""
from __future__ import annotations

import copy
import hashlib
import random
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping


OCR_NOISE_AUGMENTATION_VERSION = "2.0-rotation-safe-jitter"


@dataclass(frozen=True)
class OCRNoiseConfig:
    seed: int = 42
    example_probability: float = 0.20
    maximum_token_fraction: float = 0.15
    maximum_transformations: int = 3
    maximum_box_jitter_ratio: float = 0.015

    def __post_init__(self) -> None:
        for name in ("example_probability", "maximum_token_fraction"):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 1 <= int(self.maximum_transformations) <= 5:
            raise ValueError("maximum transformations must be within [1, 5]")
        if not 0.0 <= float(self.maximum_box_jitter_ratio) <= 0.05:
            raise ValueError("maximum box jitter ratio must be in [0, 0.05]")


def build_noisy_example(
    source: Mapping[str, Any],
    *,
    config: OCRNoiseConfig = OCRNoiseConfig(),
) -> dict[str, Any] | None:
    """Return a separate noisy training example or None, never mutate source."""
    if source.get("is_private") is not False:
        raise ValueError("private or unmarked examples are refused for OCR noise")
    if str(source.get("project_split", "")).casefold() != "train":
        return None
    identity = str(source.get("example_id") or source.get("page_id") or "")
    if not identity:
        raise ValueError("OCR-noise source example lacks a stable identity")
    rng = random.Random(_seed(identity, config.seed))
    if rng.random() >= float(config.example_probability):
        return None
    output = copy.deepcopy(dict(source))
    tokens = [dict(token) for token in output.get("tokens") or []]
    if not tokens:
        return None
    eligible = [
        index for index, token in enumerate(tokens) if str(token.get("text", "")).strip()
    ]
    if not eligible:
        return None
    maximum_tokens = max(
        1,
        min(
            int(config.maximum_transformations),
            round(len(eligible) * float(config.maximum_token_fraction)),
        ),
    )
    count = rng.randint(1, maximum_tokens)
    chosen = rng.sample(eligible, k=count)
    transformations: list[dict[str, Any]] = []
    methods = (
        _character_confusion,
        _punctuation_deletion,
        _spacing_change,
        _decimal_separator_confusion,
        _box_jitter,
    )
    page = output.get("page") if isinstance(output.get("page"), Mapping) else {}
    width = int(output.get("width") or page.get("width") or 1000)
    height = int(output.get("height") or page.get("height") or 1000)
    for token_index in chosen:
        token = tokens[token_index]
        order = list(methods)
        rng.shuffle(order)
        applied = None
        for method in order:
            applied = method(
                token,
                rng=rng,
                width=width,
                height=height,
                config=config,
            )
            if applied is not None:
                break
        if applied is not None:
            transformations.append(
                {
                    "token_index": token_index,
                    "token_id": str(token.get("id", "")),
                    **applied,
                }
            )
    if not transformations:
        return None
    output["tokens"] = tokens
    output["example_id"] = f"{identity}__ocr_noise"
    output["token_source"] = "ocr_noise"
    output["inference_realistic"] = False
    output["training_only"] = True
    output["ocr_noise"] = {
        "schema_version": "1.0",
        "implementation_version": OCR_NOISE_AUGMENTATION_VERSION,
        "seed": config.seed,
        "example_seed": _seed(identity, config.seed),
        "configuration": asdict(config),
        "transformations": transformations,
        "transformed_token_count": len(transformations),
        "token_count": len(tokens),
        "corruption_rate": len(transformations) / max(1, len(tokens)),
        "unmodified_source_preserved": True,
        "token_ids_preserved": True,
        "relations_preserved": True,
        "canonical_evidence_preserved": True,
        "private_row_count": 0,
    }
    return output


def _character_confusion(
    token: dict[str, Any], **_: Any
) -> dict[str, Any] | None:
    text = str(token.get("text", ""))
    pairs = (
        ("0", "O"),
        ("O", "0"),
        ("1", "I"),
        ("I", "1"),
        ("l", "1"),
        ("5", "S"),
        ("S", "5"),
        ("8", "B"),
        ("B", "8"),
    )
    for source, replacement in pairs:
        if source in text:
            changed = text.replace(source, replacement, 1)
            token["text"] = changed
            return {
                "type": "character_substitution",
                "before": text,
                "after": changed,
                "alignment_safe": True,
            }
    return None


def _punctuation_deletion(
    token: dict[str, Any], **_: Any
) -> dict[str, Any] | None:
    text = str(token.get("text", ""))
    match = re.search(r"[-/:]", text)
    if match is None:
        return None
    changed = text[: match.start()] + text[match.end() :]
    if not changed.strip():
        return None
    token["text"] = changed
    return {
        "type": "punctuation_deletion",
        "before": text,
        "after": changed,
        "alignment_safe": True,
    }


def _spacing_change(
    token: dict[str, Any], **_: Any
) -> dict[str, Any] | None:
    text = str(token.get("text", ""))
    if " " in text:
        changed = text.replace(" ", "", 1)
        change_type = "missing_space"
    elif len(text) >= 6 and text.isalnum():
        center = len(text) // 2
        changed = text[:center] + " " + text[center:]
        change_type = "word_split"
    else:
        return None
    token["text"] = changed
    return {
        "type": change_type,
        "before": text,
        "after": changed,
        "alignment_safe": True,
    }


def _decimal_separator_confusion(
    token: dict[str, Any], **_: Any
) -> dict[str, Any] | None:
    text = str(token.get("text", ""))
    match = re.search(r"\d([.,])\d{1,2}(?!\d)", text)
    if match is None:
        return None
    source = match.group(1)
    replacement = "," if source == "." else "."
    changed = text[: match.start(1)] + replacement + text[match.end(1) :]
    token["text"] = changed
    return {
        "type": "decimal_separator_confusion",
        "before": text,
        "after": changed,
        "alignment_safe": True,
    }


def _box_jitter(
    token: dict[str, Any],
    *,
    rng: random.Random,
    width: int,
    height: int,
    config: OCRNoiseConfig,
    **_: Any,
) -> dict[str, Any] | None:
    try:
        polygon = [
            [float(point[0]), float(point[1])] for point in token.get("polygon") or []
        ]
    except (TypeError, ValueError, IndexError):
        return None
    if len(polygon) < 4:
        return None
    if width <= 0 or height <= 0:
        return None
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    minimum_x, maximum_polygon_x = min(xs), max(xs)
    minimum_y, maximum_polygon_y = min(ys), max(ys)
    if (
        minimum_x < 0.0
        or minimum_y < 0.0
        or maximum_polygon_x > float(width)
        or maximum_polygon_y > float(height)
        or maximum_polygon_x <= minimum_x
        or maximum_polygon_y <= minimum_y
    ):
        return None
    maximum_x = max(1.0, width * config.maximum_box_jitter_ratio)
    maximum_y = max(1.0, height * config.maximum_box_jitter_ratio)
    minimum_delta_x = max(-maximum_x, -minimum_x)
    maximum_delta_x = min(maximum_x, float(width) - maximum_polygon_x)
    minimum_delta_y = max(-maximum_y, -minimum_y)
    maximum_delta_y = min(maximum_y, float(height) - maximum_polygon_y)
    if minimum_delta_x > maximum_delta_x or minimum_delta_y > maximum_delta_y:
        return None
    delta_x = rng.uniform(minimum_delta_x, maximum_delta_x)
    delta_y = rng.uniform(minimum_delta_y, maximum_delta_y)
    changed = [[x + delta_x, y + delta_y] for x, y in polygon]
    token["polygon"] = changed
    xs = [point[0] for point in changed]
    ys = [point[1] for point in changed]
    token["bbox"] = [min(xs), min(ys), max(xs), max(ys)]
    return {
        "type": "box_jitter",
        "delta_x": delta_x,
        "delta_y": delta_y,
        "alignment_safe": True,
    }


def _seed(identity: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{identity}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")
