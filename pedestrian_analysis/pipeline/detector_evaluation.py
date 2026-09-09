"""Helpers for local detector comparison and lightweight recall estimation."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


def annotate_detections(frame: np.ndarray, detections: Any, class_names: dict[int, str] | None = None) -> np.ndarray:
    """Draw bounding boxes, class labels and scores onto a frame copy."""
    annotated = frame.copy()
    class_names = class_names or {}
    total = len(detections)
    for idx in range(total):
        x1, y1, x2, y2 = (int(round(v)) for v in detections.xyxy[idx])
        score = (
            float(detections.confidence[idx])
            if getattr(detections, "confidence", None) is not None and idx < len(detections.confidence)
            else 0.0
        )
        class_id = (
            int(detections.class_id[idx])
            if getattr(detections, "class_id", None) is not None and idx < len(detections.class_id)
            else -1
        )
        label = class_names.get(class_id, str(class_id if class_id >= 0 else "det"))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{label} {score:.2f}",
            (x1, max(y1 - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return annotated


def summarize_scene_counts(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Summarize per-scene/per-detector detection counts."""
    df = pd.DataFrame(list(rows))
    if df.empty:
        return pd.DataFrame(
            columns=["scene", "detector", "frames_evaluated", "total_detections", "mean_detections_per_frame"]
        )

    grouped = (
        df.groupby(["scene", "detector"], as_index=False)
        .agg(frames_evaluated=("frame", "nunique"), total_detections=("detection_count", "sum"))
    )
    grouped["mean_detections_per_frame"] = grouped["total_detections"] / grouped["frames_evaluated"].clip(lower=1)
    return grouped


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU for two ``xyxy`` boxes."""
    inter_x1 = max(float(box_a[0]), float(box_b[0]))
    inter_y1 = max(float(box_a[1]), float(box_b[1]))
    inter_x2 = min(float(box_a[2]), float(box_b[2]))
    inter_y2 = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    if intersection <= 0:
        return 0.0

    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(0.0, float(box_a[3]) - float(box_a[1]))
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(0.0, float(box_b[3]) - float(box_b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def estimate_frame_recall(
    detections: Any,
    annotations: pd.DataFrame,
    *,
    iou_threshold: float = 0.3,
) -> float | None:
    """Estimate recall from either GT counts or GT boxes for a single frame."""
    if annotations.empty:
        return None
    if "gt_count" in annotations.columns:
        gt_count = int(annotations["gt_count"].iloc[0])
        if gt_count <= 0:
            return None
        return min(len(detections), gt_count) / gt_count

    required = {"bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"}
    if not required.issubset(annotations.columns):
        return None

    gt_boxes = annotations.loc[:, ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy(dtype=float)
    if len(gt_boxes) == 0:
        return None
    if len(detections) == 0:
        return 0.0

    pred_boxes = np.asarray(detections.xyxy, dtype=float)
    matched_gt: set[int] = set()
    true_positives = 0
    for pred_box in pred_boxes:
        best_idx = -1
        best_iou = 0.0
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            iou = compute_iou(pred_box, gt_box)
            if iou >= iou_threshold and iou > best_iou:
                best_idx = gt_idx
                best_iou = iou
        if best_idx >= 0:
            matched_gt.add(best_idx)
            true_positives += 1
    return true_positives / len(gt_boxes)


def load_annotations(annotation_path: str | Path | None) -> pd.DataFrame:
    """Load optional annotation CSV used for lightweight recall estimation."""
    if not annotation_path:
        return pd.DataFrame()
    path = Path(annotation_path)
    if not path.is_file():
        raise FileNotFoundError(f"Annotation CSV not found: '{path}'")
    return pd.read_csv(path)
