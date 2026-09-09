"""Ultralytics YOLO detector wrappers."""

from __future__ import annotations

from typing import Any

import numpy as np

from pipeline.detectors.base import BaseDetector, filter_detections_by_classes


class UltralyticsYOLODetector(BaseDetector):
    """YOLOv8 detector that returns ``supervision.Detections``."""

    name = "yolov8"

    def __init__(self, model_name: str, confidence: float, **kwargs: Any) -> None:
        super().__init__(model_name=model_name, confidence=confidence, **kwargs)
        self._model = self._load_model(model_name)

    @staticmethod
    def _load_model(model_name: str) -> Any:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RuntimeError(
                "Ultralytics is required for YOLO detectors. Install project dependencies first."
            ) from exc
        return YOLO(model_name)

    @property
    def class_names(self) -> dict[int, str]:
        names = getattr(self._model, "names", {})
        if isinstance(names, dict):
            return {int(idx): str(name) for idx, name in names.items()}
        return {idx: str(name) for idx, name in enumerate(names)}

    def _predict(self, frame: np.ndarray) -> Any:
        predict_kwargs = dict(self.extra_predict_kwargs)
        if self.nms_iou is not None:
            predict_kwargs["iou"] = self.nms_iou
        if self.imgsz is not None:
            predict_kwargs["imgsz"] = self.imgsz
        return self._model.predict(frame, conf=self.confidence, verbose=False, **predict_kwargs)

    def detect(self, frame: np.ndarray) -> Any:
        try:
            import supervision as sv
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RuntimeError(
                "supervision is required to convert detections. Install project dependencies first."
            ) from exc

        results = self._predict(frame)
        detections = sv.Detections.from_ultralytics(results[0])
        return filter_detections_by_classes(detections, self.class_names, self.class_filters)


class UltralyticsYOLOLargeDetector(UltralyticsYOLODetector):
    """Larger YOLOv8 preset for higher-recall pedestrian detection."""

    name = "yolov8_large"
