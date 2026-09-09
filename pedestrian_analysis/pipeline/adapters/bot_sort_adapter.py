"""BoT-SORT adapter implementation."""
from __future__ import annotations

from typing import Any

from pipeline.adapters.base import BaseTrackerAdapter


class UltralyticsBoTSORTAdapter(BaseTrackerAdapter):
    """BoT-SORT tracker adapter backed by supervision."""

    name = "bot_sort"

    def __init__(self, **_: Any) -> None:
        try:
            import supervision as sv
        except ImportError as exc:  # pragma: no cover - runtime dependency guard
            raise RuntimeError(
                "supervision is required for BoT-SORT. Install project dependencies first."
            ) from exc

        if not hasattr(sv, "BoTSORT"):
            raise RuntimeError(
                "The installed supervision package does not expose BoTSORT. "
                "Install supervision 0.18+ or choose another tracker."
            )
        self._tracker = sv.BoTSORT()

    def update_with_detections(self, detections: Any) -> Any:
        return self._tracker.update_with_detections(detections)
