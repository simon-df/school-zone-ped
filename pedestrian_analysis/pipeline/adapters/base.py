"""Base tracker adapter interface."""
from __future__ import annotations

from typing import Any


class BaseTrackerAdapter:
    """Interface for tracker implementations used by the tracking pipeline."""

    name: str = "base"

    def update_with_detections(self, detections: Any) -> Any:
        """Run tracker state update for the current detections."""
        raise NotImplementedError("Subclasses must implement update_with_detections().")
