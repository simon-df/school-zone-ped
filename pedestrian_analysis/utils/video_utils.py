"""Video utility functions: metadata reading, writer creation and frame iteration."""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import cv2
import numpy as np

import logging
logger = logging.getLogger(__name__)


def get_video_metadata(path: str | Path) -> dict:
    """Read basic metadata from a video file.

    Args:
        path: Path to the video file.

    Returns:
        Dictionary with keys ``width``, ``height``, ``fps``, ``frame_count``.

    Raises:
        FileNotFoundError: When *path* does not exist.
        RuntimeError: When OpenCV cannot open the file.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: '{path}'")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: '{path}'")
    try:
        return {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
    finally:
        cap.release()


def create_video_writer(
    output_path: str | Path,
    width: int,
    height: int,
    fps: float,
    fourcc: str = "mp4v",
) -> cv2.VideoWriter:
    """Create a :class:`cv2.VideoWriter` for writing annotated frames.

    Args:
        output_path: Destination file path (e.g. ``"output.mp4"``).
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Frames per second.
        fourcc: FourCC code string (default ``"mp4v"``).

    Returns:
        Opened :class:`cv2.VideoWriter` instance.

    Raises:
        ValueError: When the output path points to a directory.
        RuntimeError: When the writer cannot be opened.
    """
    from utils.validation import assert_is_not_directory

    output_path = Path(output_path)
    assert_is_not_directory(output_path, label="output_path")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
    writer = cv2.VideoWriter(str(output_path), fourcc_code, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for path: '{output_path}'")
    return writer


def iter_frames(
    path: str | Path,
    frame_skip: int = 1,
) -> Generator[tuple[int, np.ndarray], None, None]:
    """Iterate over frames of a video file.

    Args:
        path: Path to the video file.
        frame_skip: Yield every *frame_skip*-th frame (default 1 = every frame).

    Yields:
        Tuples of ``(frame_index, frame)`` where *frame* is a BGR NumPy array.

    Raises:
        FileNotFoundError: When *path* does not exist.
        RuntimeError: When OpenCV cannot open the file.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: '{path}'")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: '{path}'")
    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip == 0:
                yield frame_idx, frame
            frame_idx += 1
    finally:
        cap.release()
