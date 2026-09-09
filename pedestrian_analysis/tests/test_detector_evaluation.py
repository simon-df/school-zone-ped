"""Tests for local detector evaluation helpers."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.detector_evaluation import compute_iou, estimate_frame_recall, summarize_scene_counts


class _DetectionsStub:
    def __init__(self, xyxy: list[list[float]]) -> None:
        self.xyxy = np.asarray(xyxy, dtype=float)

    def __len__(self) -> int:
        return len(self.xyxy)


def test_summarize_scene_counts_aggregates_frames() -> None:
    summary = summarize_scene_counts(
        [
            {"scene": "a", "detector": "yolov8", "frame": 0, "detection_count": 2},
            {"scene": "a", "detector": "yolov8", "frame": 5, "detection_count": 4},
            {"scene": "b", "detector": "rtdetr", "frame": 0, "detection_count": 1},
        ]
    )

    row = summary[(summary["scene"] == "a") & (summary["detector"] == "yolov8")].iloc[0]
    assert row["frames_evaluated"] == 2
    assert row["total_detections"] == 6
    assert row["mean_detections_per_frame"] == 3


def test_estimate_frame_recall_supports_gt_count_annotations() -> None:
    recall = estimate_frame_recall(
        _DetectionsStub([[0, 0, 10, 10], [10, 10, 20, 20]]),
        pd.DataFrame([{"scene": "sample", "frame": 0, "gt_count": 4}]),
    )

    assert recall == 0.5


def test_estimate_frame_recall_matches_boxes_with_iou() -> None:
    recall = estimate_frame_recall(
        _DetectionsStub([[0, 0, 10, 10], [50, 50, 70, 70]]),
        pd.DataFrame(
            [
                {"bbox_x1": 1, "bbox_y1": 1, "bbox_x2": 11, "bbox_y2": 11},
                {"bbox_x1": 80, "bbox_y1": 80, "bbox_x2": 90, "bbox_y2": 90},
            ]
        ),
        iou_threshold=0.3,
    )

    assert recall == 0.5
    assert compute_iou(np.array([0, 0, 10, 10]), np.array([1, 1, 11, 11])) > 0.6
