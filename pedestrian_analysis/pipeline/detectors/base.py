"""Base detector interface and shared filtering helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

import numpy as np


def normalize_class_filters(class_filters: Iterable[str | int] | None) -> tuple[set[int], set[str]]:
    """Normalize detector class filters into ID and lowercase-name sets."""
    ids: set[int] = set()
    names: set[str] = set()
    if class_filters is None:
        return ids, names

    for item in class_filters:
        if item is None:
            continue
        text = str(item).strip()
        if not text:
            continue
        if text.lstrip("-").isdigit():
            ids.add(int(text))
        else:
            names.add(text.lower())
    return ids, names


def filter_detections_by_classes(
    detections: Any,
    class_names: dict[int, str],
    class_filters: Iterable[str | int] | None,
) -> Any:
    """Filter a ``supervision.Detections`` object by class IDs or names."""
    class_ids, class_name_filters = normalize_class_filters(class_filters)
    if not class_ids and not class_name_filters:
        return detections
    if getattr(detections, "class_id", None) is None or len(detections) == 0:
        return detections

    resolved_names = {
        int(idx): str(name).strip().lower()
        for idx, name in class_names.items()
        if str(name).strip()
    }
    mask = np.array(
        [
            int(class_id) in class_ids or resolved_names.get(int(class_id), "") in class_name_filters
            for class_id in detections.class_id
        ],
        dtype=bool,
    )
    return detections[mask]


class BaseDetector(ABC):
    """Interface for detector implementations used by the tracking pipeline."""

    name = "base_detector"

    def __init__(
        self,
        model_name: str,
        confidence: float,
        class_filters: Iterable[str | int] | None = None,
        nms_iou: float | None = None,
        imgsz: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.confidence = float(confidence)
        self.class_filters = tuple(class_filters) if class_filters is not None else ()
        self.nms_iou = nms_iou
        self.imgsz = imgsz
        self.extra_predict_kwargs = dict(kwargs)

    @property
    @abstractmethod
    def class_names(self) -> dict[int, str]:
        """Return the loaded model's class-name mapping."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> Any:
        """Run inference on a BGR frame and return ``supervision.Detections``."""
