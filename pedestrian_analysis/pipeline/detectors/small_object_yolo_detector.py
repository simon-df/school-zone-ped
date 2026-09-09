"""Ultralytics small-object detector wrapper with tiny-target-friendly defaults."""

from __future__ import annotations

from typing import Any

from config import DEFAULT_IMAGE_SIZE
from pipeline.detectors.ultralytics_yolo_detector import UltralyticsYOLODetector

_DEFAULT_SMALL_OBJECT_IMAGE_SIZE = max(DEFAULT_IMAGE_SIZE, 1280)


class SmallObjectYOLODetector(UltralyticsYOLODetector):
    """YOLO wrapper intended for user-supplied small-object checkpoints."""

    name = "small_object_yolo"

    def __init__(
        self,
        model_name: str,
        confidence: float,
        nms_iou: float | None = 0.6,
        imgsz: int | None = _DEFAULT_SMALL_OBJECT_IMAGE_SIZE,
        agnostic_nms: bool = True,
        max_det: int = 300,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            confidence=confidence,
            nms_iou=nms_iou,
            imgsz=imgsz,
            agnostic_nms=agnostic_nms,
            max_det=max_det,
            **kwargs,
        )
