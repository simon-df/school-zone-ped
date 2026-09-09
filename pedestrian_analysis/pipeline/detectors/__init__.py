"""Public detector API."""

from pipeline.detectors.base import BaseDetector, filter_detections_by_classes, normalize_class_filters
from pipeline.detectors.registry import DETECTOR_DEFAULT_MODELS, DETECTOR_REGISTRY, create_detector, resolve_detector_model
from pipeline.detectors.rtdetr_detector import RTDETRDetector
from pipeline.detectors.small_object_yolo_detector import SmallObjectYOLODetector
from pipeline.detectors.ultralytics_yolo_detector import UltralyticsYOLODetector, UltralyticsYOLOLargeDetector

__all__ = [
    "BaseDetector",
    "DETECTOR_DEFAULT_MODELS",
    "DETECTOR_REGISTRY",
    "RTDETRDetector",
    "SmallObjectYOLODetector",
    "UltralyticsYOLODetector",
    "UltralyticsYOLOLargeDetector",
    "create_detector",
    "filter_detections_by_classes",
    "normalize_class_filters",
    "resolve_detector_model",
]
