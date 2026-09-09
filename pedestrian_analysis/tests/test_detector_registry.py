"""Tests for detector registry and class filtering helpers."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import supervision as sv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.detectors import base, registry


class _DummyDetector(base.BaseDetector):
    name = "dummy"

    @property
    def class_names(self) -> dict[int, str]:
        return {0: "person", 1: "child", 2: "car"}

    def detect(self, frame):
        return frame


def test_create_detector_uses_default_model_and_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "DETECTOR_REGISTRY", {"dummy": _DummyDetector})
    monkeypatch.setattr(registry, "DETECTOR_DEFAULT_MODELS", {"dummy": "dummy.pt"})

    detector = registry.create_detector("dummy", confidence=0.25, class_filters=("person", "child"))

    assert isinstance(detector, _DummyDetector)
    assert detector.model_name == "dummy.pt"
    assert detector.confidence == pytest.approx(0.25)
    assert detector.class_filters == ("person", "child")


def test_create_detector_rejects_unknown_name() -> None:
    with pytest.raises(ValueError):
        registry.create_detector("missing")


def test_filter_detections_by_class_names_keeps_only_requested_classes() -> None:
    detections = sv.Detections(
        xyxy=np.array([[0, 0, 10, 10], [10, 10, 20, 20], [20, 20, 30, 30]], dtype=float),
        confidence=np.array([0.9, 0.8, 0.7], dtype=float),
        class_id=np.array([0, 1, 2], dtype=int),
    )

    filtered = base.filter_detections_by_classes(
        detections,
        {0: "person", 1: "child", 2: "car"},
        ("person", "1"),
    )

    assert len(filtered) == 2
    assert filtered.class_id.tolist() == [0, 1]
