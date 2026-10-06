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

# Default frame rate assumed when a trajectory CSV has no ``timestamp`` column.
DEFAULT_TRAJECTORY_FPS = 10.0


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


def load_trajectory_csv(csv_path: str | Path, fps: float = DEFAULT_TRAJECTORY_FPS) -> pd.DataFrame:
    """Load and normalize a trajectory CSV (TP Analysis tab + TP training pipeline).

    Accepts both this app's native schema (``id``, ``frame``, ``x``, ``y``)
    and the ``track_id``/``x_m``/``y_m`` schema described in earlier planning
    docs; both are normalized to ``id``/``frame``/``timestamp``/``x``/``y``.
    When no ``timestamp`` column is present it is derived as ``frame / fps``.

    Raises:
        ValueError: When required columns are missing.
    """
    df = pd.read_csv(csv_path)

    rename_map = {}
    if "track_id" in df.columns and "id" not in df.columns:
        rename_map["track_id"] = "id"
    if "x_m" in df.columns and "x" not in df.columns:
        rename_map["x_m"] = "x"
    if "y_m" in df.columns and "y" not in df.columns:
        rename_map["y_m"] = "y"
    if rename_map:
        df = df.rename(columns=rename_map)

    required = {"id", "frame", "x", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if "timestamp" not in df.columns:
        df = df.assign(timestamp=df["frame"] / fps)

    df = df[["id", "frame", "timestamp", "x", "y"]].dropna(subset=["id", "frame", "x", "y"])
    df["id"] = df["id"].astype(int)
    df["frame"] = df["frame"].astype(int)
    return df.sort_values(["frame", "id"]).reset_index(drop=True)
