"""Tracking pipeline: YOLOv8 + ByteTrack (default) and PBEVFormer adapter (optional).

The central entry points are:
* :func:`extract_trajectories_from_video` – batch processing returning a DataFrame.
* :func:`run_tracking_with_preview` – same processing but sending preview frames
  and progress updates via a :class:`queue.Queue` for live UI feedback.
"""

from __future__ import annotations

import logging
import queue
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd

from utils.video_utils import get_video_metadata, iter_frames, create_video_writer
from utils.threading_utils import send_status, send_progress, send_preview
from pipeline.calibration import pixel_to_meter

logger = logging.getLogger(__name__)

# Person class index in COCO (used by YOLOv8)
_PERSON_CLASS_ID = 0

# Columns for an empty trajectory DataFrame
_TRAJECTORY_COLUMNS = ["id", "frame", "x", "y", "px", "py",
                       "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "confidence"]


# ---------------------------------------------------------------------------
# Annotation helper
# ---------------------------------------------------------------------------

def draw_tracking_annotations(
    frame: np.ndarray,
    track_ids: list[int],
    bboxes: list[tuple[float, float, float, float]],
    confidences: list[float] | None = None,
    foot_points_px: list[tuple[float, float]] | None = None,
    foot_points_m: list[tuple[float, float]] | None = None,
    behaviors: list[str] | None = None,
) -> np.ndarray:
    """Draw bounding boxes, IDs and optional annotations on a copy of *frame*.

    Args:
        frame: BGR NumPy image.
        track_ids: List of integer track IDs.
        bboxes: List of ``(x1, y1, x2, y2)`` bounding boxes.
        confidences: Optional list of confidence scores per detection.
        foot_points_px: Optional list of ``(px, py)`` foot-point pixel coords.
        foot_points_m: Optional list of ``(xm, ym)`` metre coordinates.
        behaviors: Optional list of behavior label strings.

    Returns:
        Annotated BGR image (copy of *frame*).
    """
    annotated = frame.copy()
    n = len(track_ids)
    for i in range(n):
        tid = track_ids[i]
        x1, y1, x2, y2 = (int(round(v)) for v in bboxes[i])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"ID:{tid}"
        if confidences is not None and i < len(confidences):
            label += f" {confidences[i]:.2f}"
        if behaviors is not None and i < len(behaviors):
            label += f" [{behaviors[i]}]"
        cv2.putText(annotated, label, (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        if foot_points_px is not None and i < len(foot_points_px):
            fpx, fpy = (int(round(v)) for v in foot_points_px[i])
            cv2.circle(annotated, (fpx, fpy), 4, (0, 0, 255), -1)

        if foot_points_m is not None and i < len(foot_points_m):
            xm, ym = foot_points_m[i]
            txt = f"({xm:.1f},{ym:.1f})m"
            cv2.putText(annotated, txt, (x1, min(y2 + 15, annotated.shape[0] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1, cv2.LINE_AA)
    return annotated


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_trajectories_from_video(
    video_path: str | Path,
    H: np.ndarray,
    model_name: str = "yolov8n.pt",
    confidence: float = 0.4,
    frame_skip: int = 1,
    output_video_path: str | Path | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> pd.DataFrame:
    """Extract pedestrian trajectories from *video_path* using YOLOv8 + ByteTrack.

    Args:
        video_path: Path to the source video.
        H: 3 × 3 homography matrix (pixel → metre).
        model_name: YOLOv8 model file name or path.
        confidence: Detection confidence threshold.
        frame_skip: Process every *frame_skip*-th frame.
        output_video_path: Optional path to write an annotated output video.
        progress_callback: Optional ``(fraction, message)`` callback.

    Returns:
        DataFrame with columns ``id``, ``frame``, ``x``, ``y``, ``px``, ``py``,
        ``bbox_x1``, ``bbox_y1``, ``bbox_x2``, ``bbox_y2``, ``confidence``.
    """
    from ultralytics import YOLO
    import supervision as sv

    video_path = Path(video_path)
    meta = get_video_metadata(video_path)
    total_frames = meta["frame_count"]
    w, h = meta["width"], meta["height"]

    model = YOLO(model_name)
    tracker = sv.ByteTrack()

    writer: cv2.VideoWriter | None = None
    if output_video_path is not None:
        writer = create_video_writer(output_video_path, w, h, meta["fps"])

    rows: list[dict] = []

    try:
        for frame_idx, frame in iter_frames(video_path, frame_skip=frame_skip):
            results = model.predict(frame, conf=confidence, classes=[_PERSON_CLASS_ID], verbose=False)
            detections = sv.Detections.from_ultralytics(results[0])
            detections = tracker.update_with_detections(detections)

            track_ids, bboxes, confs, feet_px, feet_m = [], [], [], [], []

            for j in range(len(detections)):
                tid = detections.tracker_id[j] if detections.tracker_id is not None else -1
                bbox = detections.xyxy[j]  # x1,y1,x2,y2
                conf = float(detections.confidence[j]) if detections.confidence is not None else 0.0

                # Foot point = centre-bottom of bbox
                fpx = float((bbox[0] + bbox[2]) / 2)
                fpy = float(bbox[3])
                xm, ym = pixel_to_meter(fpx, fpy, H)

                rows.append({
                    "id": int(tid),
                    "frame": frame_idx,
                    "x": xm,
                    "y": ym,
                    "px": fpx,
                    "py": fpy,
                    "bbox_x1": float(bbox[0]),
                    "bbox_y1": float(bbox[1]),
                    "bbox_x2": float(bbox[2]),
                    "bbox_y2": float(bbox[3]),
                    "confidence": conf,
                })
                track_ids.append(int(tid))
                bboxes.append((float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])))
                confs.append(conf)
                feet_px.append((fpx, fpy))
                feet_m.append((xm, ym))

            if writer is not None:
                annotated = draw_tracking_annotations(frame, track_ids, bboxes, confs, feet_px, feet_m)
                writer.write(annotated)

            if progress_callback is not None and total_frames > 0:
                progress_callback(frame_idx / total_frames, f"Frame {frame_idx}/{total_frames}")
    finally:
        if writer is not None:
            writer.release()

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=_TRAJECTORY_COLUMNS)


# ---------------------------------------------------------------------------
# Queue-based preview version
# ---------------------------------------------------------------------------

def run_tracking_with_preview(
    video_path: str | Path,
    H: np.ndarray,
    result_queue: queue.Queue,
    model_name: str = "yolov8n.pt",
    confidence: float = 0.4,
    frame_skip: int = 1,
    preview_every_n: int = 5,
    output_video_path: str | Path | None = None,
    cancelled_fn: Callable[[], bool] | None = None,
) -> pd.DataFrame:
    """Like :func:`extract_trajectories_from_video` but streams updates to *result_queue*.

    Sends queue messages of types ``"status"``, ``"progress"``, ``"preview"``.
    The caller is responsible for draining the queue in the main thread.

    Args:
        video_path: Path to the source video.
        H: 3 × 3 homography matrix (pixel → metre).
        result_queue: Queue to receive status/progress/preview messages.
        model_name: YOLOv8 model file name or path.
        confidence: Detection confidence threshold.
        frame_skip: Process every *frame_skip*-th frame.
        preview_every_n: Send a preview frame every *preview_every_n* processed frames.
        output_video_path: Optional path to write an annotated output video.
        cancelled_fn: Optional callable returning ``True`` when the user requests
            cancellation.

    Returns:
        Completed trajectory DataFrame.
    """
    from ultralytics import YOLO
    import supervision as sv

    video_path = Path(video_path)
    send_status(result_queue, f"Loading video: {video_path.name}")
    meta = get_video_metadata(video_path)
    total_frames = meta["frame_count"]
    w, h = meta["width"], meta["height"]

    send_status(result_queue, f"Loading model: {model_name}")
    model = YOLO(model_name)
    tracker = sv.ByteTrack()

    writer: cv2.VideoWriter | None = None
    if output_video_path is not None:
        writer = create_video_writer(output_video_path, w, h, meta["fps"])

    rows: list[dict] = []
    processed = 0

    try:
        for frame_idx, frame in iter_frames(video_path, frame_skip=frame_skip):
            if cancelled_fn is not None and cancelled_fn():
                send_status(result_queue, "Cancelled by user.")
                break

            results = model.predict(frame, conf=confidence, classes=[_PERSON_CLASS_ID], verbose=False)
            detections = sv.Detections.from_ultralytics(results[0])
            detections = tracker.update_with_detections(detections)

            track_ids, bboxes, confs, feet_px, feet_m = [], [], [], [], []

            for j in range(len(detections)):
                tid = detections.tracker_id[j] if detections.tracker_id is not None else -1
                bbox = detections.xyxy[j]
                conf = float(detections.confidence[j]) if detections.confidence is not None else 0.0

                fpx = float((bbox[0] + bbox[2]) / 2)
                fpy = float(bbox[3])
                xm, ym = pixel_to_meter(fpx, fpy, H)

                rows.append({
                    "id": int(tid),
                    "frame": frame_idx,
                    "x": xm,
                    "y": ym,
                    "px": fpx,
                    "py": fpy,
                    "bbox_x1": float(bbox[0]),
                    "bbox_y1": float(bbox[1]),
                    "bbox_x2": float(bbox[2]),
                    "bbox_y2": float(bbox[3]),
                    "confidence": conf,
                })
                track_ids.append(int(tid))
                bboxes.append((float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])))
                confs.append(conf)
                feet_px.append((fpx, fpy))
                feet_m.append((xm, ym))

            annotated = draw_tracking_annotations(frame, track_ids, bboxes, confs, feet_px, feet_m)

            if writer is not None:
                writer.write(annotated)

            processed += 1
            if processed % preview_every_n == 0:
                send_preview(result_queue, annotated.copy())

            if total_frames > 0:
                send_progress(result_queue, frame_idx / total_frames)
                if frame_idx % 50 == 0:
                    send_status(result_queue, f"Processing frame {frame_idx}/{total_frames} …")
    finally:
        if writer is not None:
            writer.release()

    send_status(result_queue, "Tracking complete.")
    send_progress(result_queue, 1.0)

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=_TRAJECTORY_COLUMNS)


# ---------------------------------------------------------------------------
# Optional PBEVFormer adapter skeleton
# ---------------------------------------------------------------------------

class PBEVFormerTrackerAdapter:
    """Adapter interface for PBEVFormer-based pedestrian detection.

    This is a documented **stub**.  A real implementation must:

    1. Load the PBEVFormer model from *config_path* and *weights_path*.
    2. Accept a BGR frame and return a list of detections with track IDs.
    3. Map BEV output coordinates to the same metric space as :func:`pixel_to_meter`.

    Expected input per call:  ``frame`` – BGR NumPy array (H × W × 3).
    Expected output per call: list of dicts with keys ``id``, ``x``, ``y``,
                              ``bbox_x1``, ``bbox_y1``, ``bbox_x2``, ``bbox_y2``.

    Raises:
        NotImplementedError: Always – replace with a real implementation.
        RuntimeError: When the model fails to load.
    """

    def __init__(self, config_path: str, weights_path: str) -> None:
        """Load the PBEVFormer model.

        Args:
            config_path: Path to the PBEVFormer YAML config file.
            weights_path: Path to the PBEVFormer pre-trained weights file.

        Raises:
            RuntimeError: When config or weights files cannot be found.
            NotImplementedError: Always – replace with a real implementation.
        """
        if not Path(config_path).is_file():
            raise RuntimeError(f"PBEVFormer config not found: '{config_path}'")
        if not Path(weights_path).is_file():
            raise RuntimeError(f"PBEVFormer weights not found: '{weights_path}'")
        raise NotImplementedError(
            "PBEVFormer integration is not yet implemented. "
            "Implement PBEVFormerTrackerAdapter to enable this mode."
        )

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Run inference on *frame* and return detections.

        Args:
            frame: BGR NumPy array.

        Returns:
            List of detection dicts (see class docstring for schema).

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("PBEVFormer detect() is not implemented.")
