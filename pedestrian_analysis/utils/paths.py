"""Path utilities: directory creation and output path generation."""

import os
from datetime import datetime
from pathlib import Path

import logging
logger = logging.getLogger(__name__)


def ensure_directories(*dirs: Path | str) -> None:
    """Create all given directories (and parents) if they do not exist.

    Args:
        *dirs: One or more directory paths to create.
    """
    for d in dirs:
        path = Path(d)
        path.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured directory: %s", path)


def get_timestamp() -> str:
    """Return a filesystem-safe timestamp string (YYYYMMDD_HHMMSS).

    Returns:
        Timestamp string suitable for use in file names.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_output_path(base_dir: Path | str, stem: str, suffix: str, *, timestamped: bool = True) -> Path:
    """Build a unique output file path inside *base_dir*.

    Args:
        base_dir: Directory in which the file will be created.
        stem: Base name without extension.
        suffix: File extension including the leading dot (e.g. ``".csv"``).
        timestamped: When *True*, append a timestamp to *stem* to avoid collisions.

    Returns:
        Full :class:`pathlib.Path` for the output file.
    """
    base_dir = Path(base_dir)
    ensure_directories(base_dir)
    name = f"{stem}_{get_timestamp()}{suffix}" if timestamped else f"{stem}{suffix}"
    return base_dir / name


def setup_project_directories() -> None:
    """Create all required project directories defined in :mod:`config`."""
    from config import (
        DATA_DIR,
        VIDEOS_DIR,
        TRAJECTORIES_DIR,
        CALIBRATION_DIR,
        PREVIEWS_DIR,
        OUTPUTS_DIR,
        FIGURES_DIR,
        REPORTS_DIR,
        OUTPUT_VIDEOS_DIR,
    )
    ensure_directories(
        DATA_DIR,
        VIDEOS_DIR,
        TRAJECTORIES_DIR,
        CALIBRATION_DIR,
        PREVIEWS_DIR,
        OUTPUTS_DIR,
        FIGURES_DIR,
        REPORTS_DIR,
        OUTPUT_VIDEOS_DIR,
    )
    logger.info("Project directories ready.")
