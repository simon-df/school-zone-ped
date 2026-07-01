"""Image utility functions: conversion, scaling, annotation."""

from __future__ import annotations

import logging
from typing import Sequence

import cv2
import numpy as np
from PIL import Image

try:
    from PIL import ImageTk as _ImageTk
except ImportError:  # Tkinter not available in this environment
    _ImageTk = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def bgr_to_pil(frame: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR frame to a Pillow :class:`~PIL.Image.Image`.

    Args:
        frame: NumPy array in BGR format (H × W × 3).

    Returns:
        RGB Pillow image.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def pil_to_tkimage(image: Image.Image):
    """Convert a Pillow image to a Tkinter-compatible :class:`~PIL.ImageTk.PhotoImage`.

    Args:
        image: Pillow image to convert.

    Returns:
        :class:`~PIL.ImageTk.PhotoImage` ready for use on a Tkinter canvas.

    Raises:
        RuntimeError: When Tkinter is not available in the current environment.
    """
    if _ImageTk is None:
        raise RuntimeError("Tkinter is not available. Install python3-tk to use this function.")
    return _ImageTk.PhotoImage(image)


def bgr_to_tkimage(frame: np.ndarray):
    """Convert an OpenCV BGR frame directly to a Tkinter PhotoImage.

    Args:
        frame: NumPy array in BGR format (H × W × 3).

    Returns:
        :class:`~PIL.ImageTk.PhotoImage` ready for use on a Tkinter canvas.
    """
    return pil_to_tkimage(bgr_to_pil(frame))


def scale_image_for_canvas(
    image: Image.Image,
    canvas_width: int,
    canvas_height: int,
) -> tuple[Image.Image, float]:
    """Scale *image* to fit inside *canvas_width* × *canvas_height* while preserving aspect ratio.

    Args:
        image: Original Pillow image.
        canvas_width: Target canvas width in pixels.
        canvas_height: Target canvas height in pixels.

    Returns:
        A tuple of ``(scaled_image, scale_factor)`` where *scale_factor* < 1 means
        the image was shrunk.  The same factor applies to both axes.
    """
    orig_w, orig_h = image.size
    scale = min(canvas_width / orig_w, canvas_height / orig_h, 1.0)
    new_w = max(1, int(orig_w * scale))
    new_h = max(1, int(orig_h * scale))
    scaled = image.resize((new_w, new_h), Image.LANCZOS)
    return scaled, scale


def canvas_coords_to_image_coords(
    canvas_x: float,
    canvas_y: float,
    scale: float,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> tuple[float, float]:
    """Map canvas pixel coordinates back to original image coordinates.

    Args:
        canvas_x: X coordinate on the Tkinter canvas.
        canvas_y: Y coordinate on the Tkinter canvas.
        scale: Scale factor returned by :func:`scale_image_for_canvas`.
        offset_x: Horizontal offset (in pixels) applied when drawing on the canvas.
        offset_y: Vertical offset (in pixels) applied when drawing on the canvas.

    Returns:
        ``(img_x, img_y)`` in the coordinate space of the original image.
    """
    img_x = (canvas_x - offset_x) / scale
    img_y = (canvas_y - offset_y) / scale
    return img_x, img_y


def draw_points_on_image(
    image: np.ndarray,
    points: Sequence[tuple[float, float]],
    color: tuple[int, int, int] = (0, 0, 255),
    radius: int = 8,
    thickness: int = 2,
) -> np.ndarray:
    """Draw numbered circles at *points* on a copy of *image*.

    Args:
        image: BGR NumPy image to annotate.
        points: Sequence of ``(x, y)`` coordinates in image space.
        color: BGR color of the circles and text.
        radius: Circle radius in pixels.
        thickness: Circle border thickness in pixels.

    Returns:
        Annotated copy of *image*.
    """
    annotated = image.copy()
    for i, (px, py) in enumerate(points, start=1):
        cx, cy = int(round(px)), int(round(py))
        cv2.circle(annotated, (cx, cy), radius, color, thickness)
        cv2.putText(
            annotated,
            str(i),
            (cx + radius + 2, cy - radius),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated
