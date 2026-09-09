"""DETR-style detector wrapper backed by Ultralytics RT-DETR."""

from __future__ import annotations

from typing import Any

from pipeline.detectors.ultralytics_yolo_detector import UltralyticsYOLODetector


class RTDETRDetector(UltralyticsYOLODetector):
    """RT-DETR wrapper that keeps output compatible with ``supervision.Detections``."""

    name = "rtdetr"

    @staticmethod
    def _load_model(model_name: str) -> Any:
        try:
            from ultralytics import RTDETR
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RuntimeError(
                "Ultralytics RT-DETR is unavailable in this environment. "
                "Install a compatible ultralytics build or provide a YOLO-compatible fallback."
            ) from exc
        return RTDETR(model_name)
