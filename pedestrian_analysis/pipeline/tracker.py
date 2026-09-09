"""Tracking pipeline for pedestrian trajectory extraction.

The pipeline is intentionally split into:
* detection and tracking via an adapter layer
* optional preview/video export
* tracking metrics and debug helpers
"""

from __future__ import annotations

from collections import deque
import queue
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd

from config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_DETECTOR_CLASSES,
    DEFAULT_DETECTOR_TYPE,
    DEFAULT_FRAME_SKIP,
    DEFAULT_TRACKER_TYPE,
)
from pedestrian_analysis.pipeline.behavior_labeling import (
    STATE_APPROACHING,
    STATE_CROSSED,
    STATE_CROSSING,
    STATE_WAITING,
    label_behaviors,
)
from pedestrian_analysis.pipeline.group_analysis import detect_groups_per_frame
from pipeline.calibration import pixel_to_meter
from pipeline.detectors import create_detector, resolve_detector_model
from pipeline.tracker_adapters import create_tracker_adapter
from utils.threading_utils import send_preview, send_progress, send_status
from utils.video_utils import create_video_writer, get_video_metadata, iter_frames

import logging
logger = logging.getLogger(__name__)

# Maximum gap between track fragments before a heuristic ID-switch candidate is ignored.
_MAX_GAP_FRAMES_FOR_ID_SWITCH = 2
# Maximum spatial distance between track endpoints that still looks like a plausible switch.
_ID_SWITCH_DISTANCE_THRESHOLD_M = 1.0
_TRAJECTORY_COLUMNS = [
    "id",
    "frame",
    "x",
    "y",
    "px",
    "py",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "confidence",
]
_REFERENCE_POINT_WINDOW = 3
_DEFAULT_BEHAVIOR_LABELING_KWARGS = {
    "street_start_m": 2.0,
    "street_end_m": 6.0,
    "speed_threshold_ms": 0.5,
    "waiting_min_frames": 10,
    "smooth_window": 10,
}
_DEFAULT_BOX_COLOR = (0, 255, 0)
_BEHAVIOR_BOX_COLORS = {
    STATE_WAITING: (180, 130, 70),
    STATE_APPROACHING: (0, 165, 255),
    STATE_CROSSING: (0, 255, 0),
    STATE_CROSSED: (128, 128, 128),
}


def _compute_bbox_center(bbox: np.ndarray | tuple[float, float, float, float]) -> tuple[float, float]:
    return (float((bbox[0] + bbox[2]) / 2.0), float((bbox[1] + bbox[3]) / 2.0))


def _get_smoothed_reference_point(
    track_id: int,
    bbox: np.ndarray | tuple[float, float, float, float],
    track_points: dict[int, deque[tuple[float, float]]],
) -> tuple[float, float]:
    history = track_points.setdefault(track_id, deque(maxlen=_REFERENCE_POINT_WINDOW))
    history.append(_compute_bbox_center(bbox))
    points = np.asarray(history, dtype=float)
    return float(np.median(points[:, 0])), float(np.median(points[:, 1]))


def _label_trajectory_behaviors(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    return label_behaviors(df, fps=fps, **_DEFAULT_BEHAVIOR_LABELING_KWARGS)


def _get_current_behaviors(
    rows: list[dict[str, Any]],
    frame_idx: int,
    track_ids: list[int],
    fps: float,
) -> list[str] | None:
    if not rows or not track_ids:
        return None

    labeled = _label_trajectory_behaviors(pd.DataFrame(rows), fps=fps)
    current = labeled[labeled["frame"] == frame_idx]
    if current.empty or "behavior" not in current.columns:
        return None

    current = current.drop_duplicates(subset=["id"], keep="last")
    behavior_by_id = current.set_index("id")["behavior"].to_dict()
    return [str(behavior_by_id.get(track_id, "")) for track_id in track_ids]


def _get_behavior_box_color(behavior: str | None) -> tuple[int, int, int]:
    if behavior is None:
        return _DEFAULT_BOX_COLOR
    return _BEHAVIOR_BOX_COLORS.get(str(behavior).lower(), _DEFAULT_BOX_COLOR)


def draw_tracking_annotations(
    frame: np.ndarray,
    track_ids: list[int],
    bboxes: list[tuple[float, float, float, float]],
    confidences: list[float] | None = None,
    foot_points_px: list[tuple[float, float]] | None = None,
    foot_points_m: list[tuple[float, float]] | None = None,
    behaviors: list[str] | None = None,
) -> np.ndarray:
    """Draw tracking overlays on a frame copy.

    Args:
        frame: Source BGR image.
        track_ids: Integer track identifiers.
        bboxes: Bounding boxes as ``(x1, y1, x2, y2)`` tuples.
        confidences: Optional per-track confidences.
        foot_points_px: Optional pixel foot points.
        foot_points_m: Optional metre foot points.
        behaviors: Optional behavior labels.

    Returns:
        An annotated copy of the input frame.
    """
    annotated = frame.copy()
    for i, tid in enumerate(track_ids):
        x1, y1, x2, y2 = (int(round(v)) for v in bboxes[i])
        behavior = behaviors[i] if behaviors is not None and i < len(behaviors) else None
        box_color = _get_behavior_box_color(behavior)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

        label = f"ID:{tid}"
        if confidences is not None and i < len(confidences):
            label += f" {confidences[i]:.2f}"
        if behavior:
            label += f" [{behavior}]"
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            box_color,
            1,
            cv2.LINE_AA,
        )

        if foot_points_px is not None and i < len(foot_points_px):
            fpx, fpy = (int(round(v)) for v in foot_points_px[i])
            cv2.circle(annotated, (fpx, fpy), 4, (0, 0, 255), -1)

        if foot_points_m is not None and i < len(foot_points_m):
            xm, ym = foot_points_m[i]
            txt = f"({xm:.1f},{ym:.1f})m"
            cv2.putText(
                annotated,
                txt,
                (x1, min(y2 + 15, annotated.shape[0] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 165, 0),
                1,
                cv2.LINE_AA,
            )
    return annotated


def _build_tracker_adapter(tracker_type: str | None) -> Any:
    selected = tracker_type or DEFAULT_TRACKER_TYPE
    logger.info("Using tracker adapter: %s", selected)
    return create_tracker_adapter(selected)


def _parse_detector_classes(detector_classes: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if detector_classes is None:
        return DEFAULT_DETECTOR_CLASSES
    if isinstance(detector_classes, str):
        return tuple(part.strip() for part in detector_classes.split(",") if part.strip())
    return tuple(str(part).strip() for part in detector_classes if str(part).strip())


def _build_detector(
    model_name: str | None,
    confidence: float,
    detector_type: str | None,
    detector_classes: str | list[str] | tuple[str, ...] | None,
    detector_kwargs: dict[str, Any] | None = None,
) -> Any:
    selected_detector = detector_type or DEFAULT_DETECTOR_TYPE
    resolved_model_name = resolve_detector_model(selected_detector, model_name)
    selected_classes = _parse_detector_classes(detector_classes)
    logger.info("Using detector: %s (%s)", selected_detector, resolved_model_name)
    return create_detector(
        detector_type=selected_detector,
        model_name=resolved_model_name,
        confidence=confidence,
        class_filters=selected_classes,
        **(detector_kwargs or {}),
    )


def extract_trajectories_from_video(
    video_path: str | Path,
    H: np.ndarray,
    model_name: str | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
    frame_skip: int = DEFAULT_FRAME_SKIP,
    output_video_path: str | Path | None = None,
    output_csv_path: str | Path | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
    tracker_type: str | None = None,
    detector_type: str | None = None,
    detector_classes: str | list[str] | tuple[str, ...] | None = None,
    detector_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Extract pedestrian trajectories from *video_path* using a tracker adapter.

    Args:
        video_path: Source video path.
        H: Homography mapping pixel coordinates to metres.
        model_name: Detector checkpoint name or path.
        confidence: Detection confidence threshold.
        frame_skip: Skip every ``frame_skip``-th frame.
        output_video_path: Optional annotated output video path.
        output_csv_path: Optional CSV output path. When provided, the trajectories are saved automatically.
        progress_callback: Optional callback receiving ``(fraction, message)``.
        tracker_type: Tracker adapter name (e.g. ``bot_sort``) or research group name
            (e.g. ``research_top_down_occlusion``).
        detector_type: Detector adapter name (e.g. ``yolov8_large`` or ``rtdetr``).
        detector_classes: Optional comma-separated class names/IDs to keep before tracking.
        detector_kwargs: Optional detector-specific inference kwargs (e.g. ``imgsz`` or ``iou``).

    Returns:
        A trajectory DataFrame with the required columns ``id``, ``frame``, ``x``, ``y``,
        ``px``, ``py``, ``bbox_x1``, ``bbox_y1``, ``bbox_x2``, ``bbox_y2`` and ``confidence``.
    """
    video_path = Path(video_path)
    meta = get_video_metadata(video_path)
    total_frames = meta["frame_count"]
    width, height = meta["width"], meta["height"]

    detector = _build_detector(model_name, confidence, detector_type, detector_classes, detector_kwargs)
    tracker = _build_tracker_adapter(tracker_type)

    writer: cv2.VideoWriter | None = None
    if output_video_path is not None:
        writer = create_video_writer(output_video_path, width, height, meta["fps"])

    rows: list[dict[str, Any]] = []
    track_points: dict[int, deque[tuple[float, float]]] = {}
    try:
        for frame_idx, frame in iter_frames(video_path, frame_skip=frame_skip):
            detections = detector.detect(frame)
            detections = tracker.update_with_detections(detections)

            track_ids, bboxes, confs, feet_px, feet_m = [], [], [], [], []
            for j in range(len(detections)):
                tid = detections.tracker_id[j] if detections.tracker_id is not None else -1
                bbox = detections.xyxy[j]
                conf = float(detections.confidence[j]) if detections.confidence is not None else 0.0

                fpx, fpy = _get_smoothed_reference_point(int(tid), bbox, track_points)
                xm, ym = pixel_to_meter(fpx, fpy, H)

                rows.append(
                    {
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
                    }
                )
                track_ids.append(int(tid))
                bboxes.append((float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])))
                confs.append(conf)
                feet_px.append((fpx, fpy))
                feet_m.append((xm, ym))

            if writer is not None:
                behaviors = _get_current_behaviors(rows, frame_idx, track_ids, meta["fps"])
                annotated = draw_tracking_annotations(frame, track_ids, bboxes, confs, feet_px, feet_m, behaviors=behaviors)
                writer.write(annotated)

            if progress_callback is not None and total_frames > 0:
                progress_callback(frame_idx / total_frames, f"Frame {frame_idx}/{total_frames}")
    finally:
        if writer is not None:
            writer.release()

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=_TRAJECTORY_COLUMNS)
    df = _label_trajectory_behaviors(df, fps=meta["fps"])
    if output_csv_path is not None:
        from pipeline.trajectory_io import save_trajectories

        save_trajectories(df, output_csv_path)
    return df


def run_tracking_with_preview(
    video_path: str | Path,
    H: np.ndarray,
    result_queue: queue.Queue,
    model_name: str | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
    frame_skip: int = DEFAULT_FRAME_SKIP,
    preview_every_n: int = 5,
    output_video_path: str | Path | None = None,
    output_csv_path: str | Path | None = None,
    cancelled_fn: Callable[[], bool] | None = None,
    tracker_type: str | None = None,
    detector_type: str | None = None,
    detector_classes: str | list[str] | tuple[str, ...] | None = None,
    detector_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Track pedestrians while streaming preview frames and progress to *result_queue*.

    Args:
        video_path: Source video path.
        H: Homography mapping pixel coordinates to metres.
        result_queue: Queue receiving ``status``, ``progress``, ``preview`` and ``result`` messages.
        model_name: Detector checkpoint name or path.
        confidence: Detection confidence threshold.
        frame_skip: Skip every ``frame_skip``-th frame.
        preview_every_n: Send preview frames every ``preview_every_n`` processed frames.
        output_video_path: Optional annotated output video path.
        output_csv_path: Optional CSV output path. When provided, the trajectories are saved automatically.
        cancelled_fn: Optional callback returning ``True`` when the user aborts.
        tracker_type: Tracker adapter name (e.g. ``bot_sort``) or research group name
            (e.g. ``research_top_down_occlusion``).
        detector_type: Detector adapter name (e.g. ``yolov8_large`` or ``rtdetr``).
        detector_classes: Optional comma-separated class names/IDs to keep before tracking.
        detector_kwargs: Optional detector-specific inference kwargs (e.g. ``imgsz`` or ``iou``).

    Returns:
        A trajectory DataFrame with the same required columns as :func:`extract_trajectories_from_video`.
    """
    video_path = Path(video_path)
    send_status(result_queue, f"Loading video: {video_path.name}")
    meta = get_video_metadata(video_path)
    total_frames = meta["frame_count"]
    width, height = meta["width"], meta["height"]

    resolved_model_name = resolve_detector_model(detector_type or DEFAULT_DETECTOR_TYPE, model_name)
    send_status(
        result_queue,
        f"Loading detector: {detector_type or DEFAULT_DETECTOR_TYPE} ({resolved_model_name})",
    )
    detector = _build_detector(model_name, confidence, detector_type, detector_classes, detector_kwargs)
    tracker = _build_tracker_adapter(tracker_type)

    writer: cv2.VideoWriter | None = None
    if output_video_path is not None:
        writer = create_video_writer(output_video_path, width, height, meta["fps"])

    rows: list[dict[str, Any]] = []
    track_points: dict[int, deque[tuple[float, float]]] = {}
    processed = 0
    try:
        for frame_idx, frame in iter_frames(video_path, frame_skip=frame_skip):
            if cancelled_fn is not None and cancelled_fn():
                send_status(result_queue, "Cancelled by user.")
                break

            detections = detector.detect(frame)
            detections = tracker.update_with_detections(detections)

            track_ids, bboxes, confs, feet_px, feet_m = [], [], [], [], []
            for j in range(len(detections)):
                tid = detections.tracker_id[j] if detections.tracker_id is not None else -1
                bbox = detections.xyxy[j]
                conf = float(detections.confidence[j]) if detections.confidence is not None else 0.0

                fpx, fpy = _get_smoothed_reference_point(int(tid), bbox, track_points)
                xm, ym = pixel_to_meter(fpx, fpy, H)

                rows.append(
                    {
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
                    }
                )
                track_ids.append(int(tid))
                bboxes.append((float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])))
                confs.append(conf)
                feet_px.append((fpx, fpy))
                feet_m.append((xm, ym))

            behaviors = _get_current_behaviors(rows, frame_idx, track_ids, meta["fps"])
            annotated = draw_tracking_annotations(frame, track_ids, bboxes, confs, feet_px, feet_m, behaviors=behaviors)
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
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=_TRAJECTORY_COLUMNS)
    df = _label_trajectory_behaviors(df, fps=meta["fps"])
    df = detect_groups_per_frame(df, proximity_m=1.5, min_group_frames=10, smooth_window=10)

    # df = df.append(df_behavior, ignore_index=False).append(df_groups, ignore_index=False)

    
    if output_csv_path is not None:
        from pipeline.trajectory_io import save_trajectories

        save_trajectories(df, output_csv_path)
    return df


def compute_tracking_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Compute simple internal tracking-quality diagnostics.

    The implementation is intentionally heuristic-based and does not require ground-truth labels.
    It reports proxy metrics such as ID-switch estimates, fragmentation counts, and per-frame
    track activity, but MOTA/IDF1/HOTA remain ``None`` until ground-truth annotations are available.

    Args:
        df: Trajectory DataFrame with at least columns ``id``, ``frame``, ``x`` and ``y``.

    Returns:
        A dictionary containing ``id_switches``, ``fragmentations``,
        ``track_length_stats``, ``active_track_count_per_frame``,
        ``new_tracks_per_frame`` and ``lost_tracks_per_frame``.
    """
    if df.empty:
        return {
            "id_switches": 0,
            "fragmentations": 0,
            "track_length_stats": {"count": 0, "min": 0, "max": 0, "mean": 0.0},
            "active_track_count_per_frame": {},
            "new_tracks_per_frame": {},
            "lost_tracks_per_frame": {},
            "mota": None,
            "idf1": None,
            "hota": None,
        }

    track_lengths = df.groupby("id").size()
    active_tracks_per_frame = df.groupby("frame")["id"].nunique().to_dict()

    new_tracks_per_frame: dict[int, int] = {}
    lost_tracks_per_frame: dict[int, int] = {}
    previous_ids: set[int] = set()
    for frame in sorted(df["frame"].unique()):
        frame_ids = set(int(v) for v in df.loc[df["frame"] == frame, "id"].tolist())
        new_tracks_per_frame[int(frame)] = len(frame_ids - previous_ids)
        lost_tracks_per_frame[int(frame)] = len(previous_ids - frame_ids)
        previous_ids = frame_ids

    fragmentations = 0
    for _, track_df in df.groupby("id"):
        frames = sorted(int(v) for v in track_df["frame"].tolist())
        if len(frames) < 2:
            continue
        fragmentations += sum(1 for before, after in zip(frames, frames[1:]) if after - before > 1)

    id_switches = 0
    if {"x", "y"}.issubset(df.columns):
        track_segments: list[dict[str, Any]] = []
        for _, track_df in df.groupby("id"):
            ordered = track_df.sort_values("frame")
            if ordered.empty:
                continue
            track_segments.append(
                {
                    "id": int(ordered.iloc[0]["id"]),
                    "start_frame": int(ordered["frame"].min()),
                    "end_frame": int(ordered["frame"].max()),
                    "start_x": float(ordered.iloc[0]["x"]),
                    "start_y": float(ordered.iloc[0]["y"]),
                    "end_x": float(ordered.iloc[-1]["x"]),
                    "end_y": float(ordered.iloc[-1]["y"]),
                }
            )

        track_segments.sort(key=lambda item: (item["start_frame"], item["end_frame"]))
        for idx, current in enumerate(track_segments):
            for other in track_segments[idx + 1 :]:
                gap_frames = other["start_frame"] - current["end_frame"]
                if gap_frames > _MAX_GAP_FRAMES_FOR_ID_SWITCH:
                    continue
                distance = np.hypot(
                    current["end_x"] - other["start_x"],
                    current["end_y"] - other["start_y"],
                )
                if distance <= _ID_SWITCH_DISTANCE_THRESHOLD_M:
                    id_switches += 1
                    break

    return {
        "id_switches": id_switches,
        "fragmentations": fragmentations,
        "track_length_stats": {
            "count": int(track_lengths.count()),
            "min": int(track_lengths.min()),
            "max": int(track_lengths.max()),
            "mean": float(track_lengths.mean()),
        },
        "active_track_count_per_frame": active_tracks_per_frame,
        "new_tracks_per_frame": new_tracks_per_frame,
        "lost_tracks_per_frame": lost_tracks_per_frame,
        "mota": None,
        "idf1": None,
        "hota": None,
    }
