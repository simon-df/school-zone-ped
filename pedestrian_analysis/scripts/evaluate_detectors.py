"""Run local detector comparisons on BEV frames or videos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from config import DEFAULT_CONFIDENCE, DEFAULT_DETECTOR_CLASSES, DEFAULT_MODEL_NAME, OUTPUTS_DIR
from pipeline.detector_evaluation import annotate_detections, estimate_frame_recall, load_annotations, summarize_scene_counts
from pipeline.detectors import DETECTOR_DEFAULT_MODELS, DETECTOR_REGISTRY, create_detector
from utils.paths import ensure_directories
from utils.video_utils import iter_frames


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Video/image paths to evaluate locally.")
    parser.add_argument(
        "--detector",
        action="append",
        dest="detectors",
        help="Detector type to run. Repeat the flag or use 'all'. Defaults to yolov8 baseline.",
    )
    parser.add_argument("--model", action="append", default=[], help="Optional detector_type=checkpoint override.")
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--class-filter", default=",".join(DEFAULT_DETECTOR_CLASSES))
    parser.add_argument("--frame-step", type=int, default=30, help="Evaluate every Nth frame for videos.")
    parser.add_argument("--max-frames", type=int, default=20, help="Max sampled frames per scene.")
    parser.add_argument("--imgsz", type=int, default=None, help="Optional inference image size override.")
    parser.add_argument("--nms-iou", type=float, default=None, help="Optional NMS IoU override.")
    parser.add_argument("--output-dir", default=str(OUTPUTS_DIR / "detector_eval"))
    parser.add_argument("--annotations", default="", help="Optional CSV with scene/frame gt_count or GT boxes.")
    return parser.parse_args()


def _resolve_detector_models(model_overrides: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in model_overrides:
        if "=" not in item:
            raise ValueError(f"Model override must use detector_type=checkpoint format, got: '{item}'")
        detector_name, model_name = item.split("=", 1)
        overrides[detector_name.strip().lower().replace("-", "_")] = model_name.strip()
    return overrides


def _resolve_detectors(requested: list[str] | None) -> list[str]:
    if not requested:
        return ["yolov8"]
    normalized = [item.strip().lower().replace("-", "_") for item in requested if item.strip()]
    if "all" in normalized:
        return list(DETECTOR_REGISTRY)
    return normalized


def _iter_scene_frames(path: Path, frame_step: int, max_frames: int) -> list[tuple[int, Any]]:
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Failed to read image: '{path}'")
        return [(0, frame)]

    sampled: list[tuple[int, Any]] = []
    for frame_idx, frame in iter_frames(path, frame_skip=max(1, frame_step)):
        sampled.append((frame_idx, frame))
        if len(sampled) >= max_frames:
            break
    return sampled


def _scene_annotations(df: pd.DataFrame, scene: str, frame_idx: int) -> pd.DataFrame:
    if df.empty:
        return df
    subset = df[df["scene"].astype(str) == scene]
    if "frame" in subset.columns:
        subset = subset[subset["frame"] == frame_idx]
    return subset.reset_index(drop=True)


def main() -> None:
    args = _parse_args()
    detectors = _resolve_detectors(args.detectors)
    model_overrides = _resolve_detector_models(args.model)
    class_filters = tuple(part.strip() for part in args.class_filter.split(",") if part.strip())
    detector_kwargs: dict[str, Any] = {}
    if args.imgsz is not None:
        detector_kwargs["imgsz"] = args.imgsz
    if args.nms_iou is not None:
        detector_kwargs["nms_iou"] = args.nms_iou

    output_dir = Path(args.output_dir)
    overlay_dir = output_dir / "overlays"
    ensure_directories(output_dir, overlay_dir)

    annotations = load_annotations(args.annotations or None)
    frame_rows: list[dict[str, Any]] = []

    for detector_name in detectors:
        model_name = model_overrides.get(detector_name, DETECTOR_DEFAULT_MODELS.get(detector_name, DEFAULT_MODEL_NAME))
        detector = create_detector(
            detector_type=detector_name,
            model_name=model_name,
            confidence=args.confidence,
            class_filters=class_filters,
            **detector_kwargs,
        )
        for input_path in args.inputs:
            scene_path = Path(input_path)
            scene = scene_path.stem
            scene_output_dir = overlay_dir / detector_name / scene
            ensure_directories(scene_output_dir)
            for frame_idx, frame in _iter_scene_frames(scene_path, args.frame_step, args.max_frames):
                detections = detector.detect(frame)
                annotated = annotate_detections(frame, detections, detector.class_names)
                cv2.imwrite(str(scene_output_dir / f"frame_{frame_idx:06d}.jpg"), annotated)
                recall_estimate = estimate_frame_recall(
                    detections,
                    _scene_annotations(annotations, scene, frame_idx),
                )
                frame_rows.append(
                    {
                        "scene": scene,
                        "frame": frame_idx,
                        "detector": detector_name,
                        "model_name": model_name,
                        "detection_count": len(detections),
                        "recall_estimate": recall_estimate,
                    }
                )

    frame_df = pd.DataFrame(frame_rows)
    frame_df.to_csv(output_dir / "frame_summary.csv", index=False)
    summarize_scene_counts(frame_rows).to_csv(output_dir / "scene_summary.csv", index=False)
    if not frame_df.empty and "recall_estimate" in frame_df.columns:
        recall_df = (
            frame_df.dropna(subset=["recall_estimate"])
            .groupby(["scene", "detector"], as_index=False)
            .agg(mean_recall_estimate=("recall_estimate", "mean"))
        )
        recall_df.to_csv(output_dir / "recall_summary.csv", index=False)


if __name__ == "__main__":
    main()
