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
    monkeypatch.setattr(tracker, "label_behaviors", lambda df, *args, **kwargs: df)

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
    assert df.iloc[0]["px"] == 20.0
    assert df.iloc[0]["py"] == 30.0


def test_extract_trajectories_smooths_bbox_center_per_track(monkeypatch) -> None:
    detector = _DetectorStub()
    tracker_adapter = _TrackerAdapterStub()
    boxes = [
        np.array([[10, 20, 30, 40]], dtype=float),
        np.array([[20, 30, 40, 50]], dtype=float),
        np.array([[90, 190, 110, 210]], dtype=float),
    ]
    frames = [(idx, np.zeros((64, 64, 3), dtype=np.uint8)) for idx in range(len(boxes))]

    def detect(frame):
        idx = detector.calls
        detector.calls += 1
        return sv.Detections(
            xyxy=boxes[idx],
            confidence=np.array([0.85], dtype=float),
            class_id=np.array([0], dtype=int),
        )

    detector.detect = detect

    monkeypatch.setattr(tracker, "get_video_metadata", lambda path: {"frame_count": 3, "width": 64, "height": 64, "fps": 25.0})
    monkeypatch.setattr(tracker, "iter_frames", lambda path, frame_skip=1: iter(frames))
    monkeypatch.setattr(tracker, "create_detector", lambda **kwargs: detector)
    monkeypatch.setattr(tracker, "create_tracker_adapter", lambda name: tracker_adapter)
    monkeypatch.setattr(tracker, "pixel_to_meter", lambda x, y, H: (x, y))
    monkeypatch.setattr(tracker, "label_behaviors", lambda df, *args, **kwargs: df)

    df = tracker.extract_trajectories_from_video(video_path="unused.mp4", H=np.eye(3))

    assert list(df["px"]) == [20.0, 25.0, 30.0]
    assert list(df["py"]) == [30.0, 35.0, 40.0]
    assert list(df["x"]) == [20.0, 25.0, 30.0]
    assert list(df["y"]) == [30.0, 35.0, 40.0]


def test_draw_tracking_annotations_uses_behavior_colors_and_labels(monkeypatch) -> None:
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    labels: list[tuple[str, tuple[int, int, int]]] = []
    original_put_text = tracker.cv2.putText

    def capture_put_text(*args, **kwargs):
        labels.append((args[1], tuple(args[5])))
        return original_put_text(*args, **kwargs)

    monkeypatch.setattr(tracker.cv2, "putText", capture_put_text)

    annotated = tracker.draw_tracking_annotations(
        frame,
        track_ids=[7, 8],
        bboxes=[(4, 4, 12, 12), (16, 4, 24, 12)],
        confidences=[0.91, 0.77],
        behaviors=["crossing", "waiting"],
    )

    assert tuple(annotated[4, 4]) == tracker._BEHAVIOR_BOX_COLORS["crossing"]
    assert tuple(annotated[4, 16]) == tracker._BEHAVIOR_BOX_COLORS["waiting"]
    assert labels[0][0] == "ID:7 0.91 [crossing]"
    assert labels[0][1] == tracker._BEHAVIOR_BOX_COLORS["crossing"]
    assert labels[1][0] == "ID:8 0.77 [waiting]"
    assert labels[1][1] == tracker._BEHAVIOR_BOX_COLORS["waiting"]
