"""Tests for detector integration inside the tracking pipeline."""

from __future__ import annotations

import os
import sys

import numpy as np
import supervision as sv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline import tracker


class _DetectorStub:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return sv.Detections(
            xyxy=np.array([[10, 20, 30, 40]], dtype=float),
            confidence=np.array([0.85], dtype=float),
            class_id=np.array([0], dtype=int),
        )


class _TrackerAdapterStub:
    def update_with_detections(self, detections):
        detections.tracker_id = np.array([7], dtype=int)
        return detections


def test_extract_trajectories_uses_detector_wrapper(monkeypatch) -> None:
    detector = _DetectorStub()
    tracker_adapter = _TrackerAdapterStub()

    monkeypatch.setattr(tracker, "get_video_metadata", lambda path: {"frame_count": 1, "width": 64, "height": 64, "fps": 25.0})
    monkeypatch.setattr(tracker, "iter_frames", lambda path, frame_skip=1: iter([(0, np.zeros((64, 64, 3), dtype=np.uint8))]))
    monkeypatch.setattr(tracker, "create_detector", lambda **kwargs: detector)
    monkeypatch.setattr(tracker, "create_tracker_adapter", lambda name: tracker_adapter)
    monkeypatch.setattr(tracker, "pixel_to_meter", lambda x, y, H: (x / 10.0, y / 10.0))
    monkeypatch.setattr(tracker, "label_behaviors", lambda df, fps: df)

    df = tracker.extract_trajectories_from_video(
        video_path="unused.mp4",
        H=np.eye(3),
        detector_type="rtdetr",
        detector_classes="person,child",
        tracker_type="byte_track",
    )

    assert detector.calls == 1
    assert len(df) == 1
    assert df.iloc[0]["id"] == 7
    assert df.iloc[0]["bbox_x1"] == 10.0
    assert df.iloc[0]["confidence"] == 0.85
