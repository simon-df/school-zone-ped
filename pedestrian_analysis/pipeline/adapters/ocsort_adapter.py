"""OC-SORT adapter implementation."""
from __future__ import annotations

from typing import Any

import numpy as np

from pipeline.adapters.base import BaseTrackerAdapter


class OCSORTAdapter(BaseTrackerAdapter):
    """OC-SORT tracker adapter backed by boxmot."""

    name = "ocsort"

    def __init__(self, **kwargs: Any) -> None:
        try:
            import boxmot
        except ImportError as exc:  # pragma: no cover - runtime dependency guard
            raise RuntimeError(
                "boxmot is required for OC-SORT. Install with: pip install boxmot"
            ) from exc

        candidate_classes = [
            getattr(boxmot, "OCSORT", None),
            getattr(boxmot, "OcSort", None),
            getattr(boxmot, "OCSort", None),
        ]
        tracker_cls = next((cls for cls in candidate_classes if cls is not None), None)
        if tracker_cls is None:
            raise RuntimeError(
                "Installed boxmot package does not expose an OC-SORT tracker class "
                "(expected one of OCSORT/OcSort/OCSort)."
            )

        try:
            self._tracker = tracker_cls(**kwargs)
        except TypeError as exc:
            raise RuntimeError(
                f"Failed to initialize OC-SORT with the provided kwargs: {exc}"
            ) from exc

    def _detections_to_boxmot(self, detections: Any) -> np.ndarray:
        xyxy = np.asarray(getattr(detections, "xyxy", np.empty((0, 4))), dtype=np.float32)
        if xyxy.size == 0:
            return np.empty((0, 6), dtype=np.float32)

        confidence = getattr(detections, "confidence", None)
        if confidence is None:
            confidence = np.ones((xyxy.shape[0],), dtype=np.float32)
        else:
            confidence = np.asarray(confidence, dtype=np.float32)

        class_id = getattr(detections, "class_id", None)
        if class_id is None:
            class_id = np.zeros((xyxy.shape[0],), dtype=np.float32)
        else:
            class_id = np.asarray(class_id, dtype=np.float32)

        return np.column_stack((xyxy, confidence, class_id)).astype(np.float32)

    def update_with_detections(self, detections: Any) -> Any:
        try:
            import supervision as sv
        except ImportError as exc:  # pragma: no cover - runtime dependency guard
            raise RuntimeError(
                "supervision is required to convert detections for OC-SORT output."
            ) from exc

        dets = self._detections_to_boxmot(detections)

        update_fn = getattr(self._tracker, "update", None)
        if update_fn is None:
            raise RuntimeError("OC-SORT tracker instance does not provide an update(...) method.")

        tracks = np.asarray(update_fn(dets), dtype=np.float32)
        if tracks.size == 0:
            return sv.Detections(
                xyxy=np.empty((0, 4), dtype=np.float32),
                confidence=np.empty((0,), dtype=np.float32),
                class_id=np.empty((0,), dtype=np.int32),
                tracker_id=np.empty((0,), dtype=np.int32),
            )

        if tracks.ndim != 2 or tracks.shape[1] < 5:
            raise RuntimeError(
                "Unexpected OC-SORT output format. Expected at least 5 columns: x1,y1,x2,y2,track_id."
            )

        xyxy = tracks[:, :4]
        tracker_id = tracks[:, 4].astype(np.int32)
        confidence = tracks[:, 5] if tracks.shape[1] > 5 else np.ones((tracks.shape[0],), dtype=np.float32)
        class_id = tracks[:, 6] if tracks.shape[1] > 6 else np.zeros((tracks.shape[0],), dtype=np.float32)

        return sv.Detections(
            xyxy=xyxy.astype(np.float32),
            confidence=np.asarray(confidence, dtype=np.float32),
            class_id=np.asarray(class_id, dtype=np.int32),
            tracker_id=tracker_id,
        )
