"""Strict multilingual PaddleOCR integration and candidate selection."""

from typing import TYPE_CHECKING, Any

from .model_registry import REQUIRED_MODEL_NAMES, ModelRegistry

if TYPE_CHECKING:
    from .pipeline import MultilingualOCR

__all__ = ["REQUIRED_MODEL_NAMES", "ModelRegistry", "MultilingualOCR"]


def __getattr__(name: str) -> Any:
    """Keep training/data utilities independent of the full inference stack."""
    if name == "MultilingualOCR":
        from .pipeline import MultilingualOCR

        return MultilingualOCR
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
