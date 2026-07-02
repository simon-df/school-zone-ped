"""Tracker adapter layer for the pedestrian tracking pipeline.

This module provides a small adapter abstraction so the UI and pipeline can
select tracking backends without coupling to a single implementation.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class BaseTrackerAdapter:
    """Interface for tracker implementations used by the tracking pipeline."""

    name: str = "base"

    def update_with_detections(self, detections: Any) -> Any:
        """Run tracker state update for the current detections."""
        raise NotImplementedError("Subclasses must implement update_with_detections().")


class UltralyticsBoTSORTAdapter(BaseTrackerAdapter):
    """BoT-SORT tracker adapter backed by supervision."""

    name = "bot_sort"

    def __init__(self, **_: Any) -> None:
        try:
            import supervision as sv
        except ImportError as exc:  # pragma: no cover - runtime dependency guard
            raise RuntimeError(
                "supervision is required for the default BoT-SORT tracker. "
                "Install the project dependencies first."
            ) from exc

        if not hasattr(sv, "BoTSORT"):
            raise RuntimeError(
                "The installed supervision package does not expose BoTSORT. "
                "Install supervision 0.18+ (or newer) or choose ByteTrack instead."
            )
        self._tracker = sv.BoTSORT()

    def update_with_detections(self, detections: Any) -> Any:
        return self._tracker.update_with_detections(detections)


class UltralyticsByteTrackAdapter(BaseTrackerAdapter):
    """ByteTrack tracker adapter backed by supervision."""

    name = "byte_track"

    def __init__(self, **_: Any) -> None:
        try:
            import supervision as sv
        except ImportError as exc:  # pragma: no cover - runtime dependency guard
            raise RuntimeError(
                "supervision is required for ByteTrack. Install the project dependencies first."
            ) from exc

        if not hasattr(sv, "ByteTrack"):
            raise RuntimeError(
                "The installed supervision package does not expose ByteTrack. "
                "Install supervision 0.18+ (or newer) to use ByteTrack."
            )
        self._tracker = sv.ByteTrack()

    def update_with_detections(self, detections: Any) -> Any:
        return self._tracker.update_with_detections(detections)


class OCSORTAdapter(BaseTrackerAdapter):
    """Experimental OC-SORT adapter placeholder."""

    name = "ocsort"

    def __init__(self, **_: Any) -> None:
        raise NotImplementedError(
            "OC-SORT is not implemented in this repository yet. "
            "Install the experimental tracker dependencies and wire the adapter before use."
        )

    def update_with_detections(self, detections: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


class DeepOCSORTAdapter(BaseTrackerAdapter):
    """Experimental Deep OC-SORT adapter placeholder."""

    name = "deep_ocsort"

    def __init__(self, **_: Any) -> None:
        raise NotImplementedError(
            "Deep OC-SORT is experimental and not available in this repository yet."
        )

    def update_with_detections(self, detections: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


class PBEVFormerTrackerAdapter(BaseTrackerAdapter):
    """Experimental PBEVFormer adapter stub."""

    name = "pb_evformer"

    def __init__(self, config_path: str, weights_path: str, **_: Any) -> None:
        from pathlib import Path

        if not Path(config_path).is_file():
            raise RuntimeError(f"PBEVFormer config not found: '{config_path}'")
        if not Path(weights_path).is_file():
            raise RuntimeError(f"PBEVFormer weights not found: '{weights_path}'")
        raise NotImplementedError(
            "PBEVFormer integration is experimental and not implemented in this repository yet."
        )

    def update_with_detections(self, detections: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


def create_tracker_adapter(tracker_type: str, **kwargs: Any) -> BaseTrackerAdapter:
    """Create a tracker adapter from a user-facing tracker name."""
    if not tracker_type or not str(tracker_type).strip():
        raise ValueError("tracker_type must be a non-empty tracker name")
    normalized = str(tracker_type).strip().lower().replace("-", "_")
    registry = {
        "bot_sort": UltralyticsBoTSORTAdapter,
        "byte_track": UltralyticsByteTrackAdapter,
        "ocsort": OCSORTAdapter,
        "deep_ocsort": DeepOCSORTAdapter,
        "pb_evformer": PBEVFormerTrackerAdapter,
    }
    adapter_cls = registry.get(normalized)
    if adapter_cls is None:
        supported = ", ".join(sorted(registry))
        raise ValueError(f"Unsupported tracker type '{tracker_type}'. Supported values: {supported}")
    return adapter_cls(**kwargs)
