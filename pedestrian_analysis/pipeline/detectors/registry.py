"""Detector registry and factory helpers."""

from __future__ import annotations

from typing import Any

from config import DEFAULT_CONFIDENCE, DEFAULT_DETECTOR_CLASSES, DEFAULT_DETECTOR_TYPE, DEFAULT_MODEL_NAME
from pipeline.detectors.base import BaseDetector
from pipeline.detectors.rtdetr_detector import RTDETRDetector
from pipeline.detectors.small_object_yolo_detector import SmallObjectYOLODetector
from pipeline.detectors.ultralytics_yolo_detector import UltralyticsYOLODetector, UltralyticsYOLOLargeDetector

DETECTOR_REGISTRY: dict[str, type[BaseDetector]] = {
    "yolov8": UltralyticsYOLODetector,
    "yolov8_large": UltralyticsYOLOLargeDetector,
    "small_object_yolo": SmallObjectYOLODetector,
    "rtdetr": RTDETRDetector,
}

DETECTOR_DEFAULT_MODELS: dict[str, str] = {
    "yolov8": DEFAULT_MODEL_NAME,
    "yolov8_large": "yolov8l.pt",
    "small_object_yolo": "yolov8l.pt",
    "rtdetr": "rtdetr-l.pt",
}


def normalize_detector_key(detector_type: str | None) -> str:
    selected = detector_type or DEFAULT_DETECTOR_TYPE
    return str(selected).strip().lower().replace("-", "_")


def resolve_detector_model(detector_type: str | None, model_name: str | None = None) -> str:
    """Resolve the checkpoint path/name for a detector type."""
    normalized = normalize_detector_key(detector_type)
    if normalized not in DETECTOR_DEFAULT_MODELS:
        supported = ", ".join(sorted(DETECTOR_DEFAULT_MODELS))
        raise ValueError(
            f"Unsupported detector type '{detector_type}'. Supported detectors: {supported}."
        )
    return model_name or DETECTOR_DEFAULT_MODELS[normalized]


def create_detector(
    detector_type: str | None = None,
    model_name: str | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
    class_filters: tuple[str | int, ...] | list[str | int] | None = None,
    **kwargs: Any,
) -> BaseDetector:
    """Create a detector adapter from detector name and optional checkpoint path."""
    normalized = normalize_detector_key(detector_type)
    if normalized not in DETECTOR_REGISTRY:
        supported = ", ".join(sorted(DETECTOR_REGISTRY))
        raise ValueError(
            f"Unsupported detector type '{detector_type}'. Supported detectors: {supported}."
        )

    detector_cls = DETECTOR_REGISTRY[normalized]
    resolved_model = resolve_detector_model(normalized, model_name)
    resolved_classes = class_filters if class_filters is not None else DEFAULT_DETECTOR_CLASSES
    return detector_cls(
        model_name=resolved_model,
        confidence=confidence,
        class_filters=resolved_classes,
        **kwargs,
    )
