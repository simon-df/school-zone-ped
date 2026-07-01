"""Homography calibration helpers.

Provides functions to compute, validate, save and load a homography matrix
that maps image pixel coordinates to real-world metre coordinates.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core calibration
# ---------------------------------------------------------------------------


def compute_homography(
    src_pixel_points: np.ndarray,
    dst_meter_points: np.ndarray,
) -> np.ndarray:
    """Compute the homography matrix from four pixel–metre point correspondences.

    Args:
        src_pixel_points: Array of shape (4, 2) with pixel coordinates.
        dst_meter_points: Array of shape (4, 2) with world/metre coordinates.

    Returns:
        3 × 3 homography matrix *H* such that metre ≈ H @ pixel (homogeneous).

    Raises:
        ValueError: When the input arrays do not have shape (4, 2).
        RuntimeError: When OpenCV cannot compute the homography (degenerate config).
    """
    src = np.asarray(src_pixel_points, dtype=np.float64)
    dst = np.asarray(dst_meter_points, dtype=np.float64)

    if src.shape != (4, 2):
        raise ValueError(f"src_pixel_points must have shape (4, 2), got {src.shape}")
    if dst.shape != (4, 2):
        raise ValueError(f"dst_meter_points must have shape (4, 2), got {dst.shape}")

    H, mask = cv2.findHomography(src, dst, method=0)
    if H is None:
        raise RuntimeError("OpenCV could not compute a valid homography. Check your point correspondences.")
    logger.debug("Computed homography:\n%s", H)
    return H


def pixel_to_meter(px: float, py: float, H: np.ndarray) -> tuple[float, float]:
    """Transform a single pixel coordinate to metre coordinates using *H*.

    Args:
        px: X coordinate in pixels.
        py: Y coordinate in pixels.
        H: 3 × 3 homography matrix returned by :func:`compute_homography`.

    Returns:
        ``(x_m, y_m)`` in the real-world metre coordinate system.
    """
    src_pt = np.array([[[px, py]]], dtype=np.float64)
    dst_pt = cv2.perspectiveTransform(src_pt, H)
    x_m, y_m = float(dst_pt[0, 0, 0]), float(dst_pt[0, 0, 1])
    return x_m, y_m


def validate_calibration(
    frame: np.ndarray,
    H: np.ndarray,
    grid_spacing_m: float = 1.0,
) -> np.ndarray:
    """Apply *H* to *frame* and draw a metric grid on the bird's-eye view.

    Args:
        frame: BGR source image.
        H: 3 × 3 homography matrix.
        grid_spacing_m: Distance between grid lines in metres.

    Returns:
        BGR bird's-eye-view image with metric grid overlay.
    """
    h, w = frame.shape[:2]
    bev = cv2.warpPerspective(frame, H, (w, h))

    # Determine visible metre range by back-projecting image corners
    corners_px = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64).reshape(-1, 1, 2)
    corners_m = cv2.perspectiveTransform(corners_px, H).reshape(-1, 2)
    min_x, min_y = corners_m.min(axis=0)
    max_x, max_y = corners_m.max(axis=0)

    H_inv = np.linalg.inv(H)

    # Grid lines parallel to Y-axis
    x = np.arange(np.floor(min_x / grid_spacing_m) * grid_spacing_m, max_x + grid_spacing_m, grid_spacing_m)
    for xv in x:
        pt_top = cv2.perspectiveTransform(np.array([[[xv, min_y]]], dtype=np.float64), H_inv)[0, 0]
        pt_bot = cv2.perspectiveTransform(np.array([[[xv, max_y]]], dtype=np.float64), H_inv)[0, 0]
        p1 = (int(round(pt_top[0])), int(round(pt_top[1])))
        p2 = (int(round(pt_bot[0])), int(round(pt_bot[1])))
        cv2.line(bev, p1, p2, (0, 200, 0), 1, cv2.LINE_AA)

    # Grid lines parallel to X-axis
    y = np.arange(np.floor(min_y / grid_spacing_m) * grid_spacing_m, max_y + grid_spacing_m, grid_spacing_m)
    for yv in y:
        pt_left = cv2.perspectiveTransform(np.array([[[min_x, yv]]], dtype=np.float64), H_inv)[0, 0]
        pt_right = cv2.perspectiveTransform(np.array([[[max_x, yv]]], dtype=np.float64), H_inv)[0, 0]
        p1 = (int(round(pt_left[0])), int(round(pt_left[1])))
        p2 = (int(round(pt_right[0])), int(round(pt_right[1])))
        cv2.line(bev, p1, p2, (0, 200, 0), 1, cv2.LINE_AA)

    return bev


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_calibration(H: np.ndarray, path: str | Path) -> None:
    """Save homography matrix *H* as a ``.npy`` file.

    Args:
        H: 3 × 3 homography matrix.
        path: Destination file path (should end in ``.npy``).

    Raises:
        ValueError: When *path* is a directory.
    """
    from utils.validation import assert_is_not_directory

    path = Path(path)
    assert_is_not_directory(path, label="calibration path")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), H)
    logger.info("Saved calibration to '%s'", path)


def load_calibration(path: str | Path) -> np.ndarray:
    """Load a homography matrix from a ``.npy`` file.

    Args:
        path: Path to the ``.npy`` file.

    Returns:
        3 × 3 homography matrix.

    Raises:
        FileNotFoundError: When *path* does not exist.
    """
    from utils.validation import assert_is_file

    path = Path(path)
    assert_is_file(path, label="calibration file")
    H = np.load(str(path))
    logger.info("Loaded calibration from '%s'", path)
    return H


def save_calibration_metadata(
    json_path: str | Path,
    original_image_path: str,
    pixel_points: list[list[float]],
    meter_points: list[list[float]],
    grid_spacing_m: float,
    homography_file: str,
) -> None:
    """Save calibration metadata as a JSON file.

    Args:
        json_path: Destination ``.json`` file path.
        original_image_path: Path to the source calibration image.
        pixel_points: List of four ``[px, py]`` pixel points.
        meter_points: List of four ``[x_m, y_m]`` world points.
        grid_spacing_m: Grid spacing in metres.
        homography_file: Path to the saved ``.npy`` homography file.
    """
    from utils.io_utils import save_json

    metadata = {
        "original_image_path": original_image_path,
        "timestamp": datetime.now().isoformat(),
        "pixel_points": pixel_points,
        "meter_points": meter_points,
        "grid_spacing_m": grid_spacing_m,
        "homography_file": homography_file,
    }
    save_json(metadata, json_path)
    logger.info("Saved calibration metadata to '%s'", json_path)
