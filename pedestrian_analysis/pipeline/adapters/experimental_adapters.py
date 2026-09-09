"""Experimental tracker adapter placeholders."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pipeline.adapters.base import BaseTrackerAdapter


class DeepOCSORTAdapter(BaseTrackerAdapter):
    """Experimental Deep OC-SORT adapter placeholder."""

    name = "deep_ocsort"

    def __init__(self, **_: Any) -> None:
        raise NotImplementedError(
            "Deep OC-SORT is currently not implemented in this repository. "
            "Use bot_sort, byte_track or ocsort for now."
        )

    def update_with_detections(self, detections: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


class PBEVFormerTrackerAdapter(BaseTrackerAdapter):
    """Experimental PBEVFormer adapter stub."""

    name = "pb_evformer"

    def __init__(self, config_path: str, weights_path: str, **_: Any) -> None:
        if not Path(config_path).is_file():
            raise RuntimeError(f"PBEVFormer config not found: '{config_path}'")
        if not Path(weights_path).is_file():
            raise RuntimeError(f"PBEVFormer weights not found: '{weights_path}'")
        raise NotImplementedError(
            "PBEVFormer integration is experimental and not implemented in this repository yet."
        )

    def update_with_detections(self, detections: Any) -> Any:  # pragma: no cover
        raise NotImplementedError
