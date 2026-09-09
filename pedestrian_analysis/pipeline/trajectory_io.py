"""Trajectory I/O: save/load CSV files and validate their structure."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.validation import validate_trajectory_dataframe, assert_is_not_directory, assert_is_file

import logging
logger = logging.getLogger(__name__)

# Columns that are always written when present
_OPTIONAL_COLUMNS: tuple[str, ...] = (
    "px",
    "py",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "confidence",
    "speed_ms",
    "heading_deg",
    "behavior",
    "group_id",
    "cohesion_m",
    "alignment_deg",
    "min_separation_m",
)


def save_trajectories(df: pd.DataFrame, path: str | Path) -> None:
    """Validate and save *df* to a CSV file.

    Args:
        df: Trajectory DataFrame with at least columns ``id``, ``frame``,
            ``x``, ``y``.
        path: Destination file path (must end in ``.csv``).

    Raises:
        ValueError: When *path* points to a directory or *df* lacks required columns.
    """
    path = Path(path)
    assert_is_not_directory(path, label="trajectory output path")
    validate_trajectory_dataframe(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(path), index=False)
    logger.info("Saved %d trajectory rows to '%s'", len(df), path)


def load_trajectories(path: str | Path) -> pd.DataFrame:
    """Load a trajectory CSV file and validate its columns.

    Args:
        path: Path to the ``.csv`` file.

    Returns:
        Validated :class:`pandas.DataFrame`.

    Raises:
        FileNotFoundError: When *path* does not exist.
        ValueError: When required columns are missing.
    """
    path = Path(path)
    assert_is_file(path, label="trajectory CSV")
    df = pd.read_csv(str(path))
    validate_trajectory_dataframe(df)
    logger.info("Loaded %d trajectory rows from '%s'", len(df), path)
    return df
