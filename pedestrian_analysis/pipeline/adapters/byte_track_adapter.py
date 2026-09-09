"""ByteTrack adapter implementation."""
from __future__ import annotations

from typing import Any

from pipeline.adapters.base import BaseTrackerAdapter


class UltralyticsByteTrackAdapter(BaseTrackerAdapter):
    """ByteTrack tracker adapter backed by supervision."""

    name = "byte_track"

    def __init__(self, **_: Any) -> None:
        try:
            import supervision as sv
        except ImportError as exc:  # pragma: no cover - runtime dependency guard
            raise RuntimeError(
                "supervision is required for ByteTrack. Install project dependencies first."
            ) from exc

        if not hasattr(sv, "ByteTrack"):
            raise RuntimeError(
                "The installed supervision package does not expose ByteTrack. "
                "Install supervision 0.18+ or choose another tracker."
            )
        self._tracker = sv.ByteTrack()

    def update_with_detections(self, detections: Any) -> Any:
        return self._tracker.update_with_detections(detections)
