"""Validation helpers for file paths and DataFrames."""

import os
from pathlib import Path
from typing import Sequence

import pandas as pd


# Required columns for a trajectory DataFrame
REQUIRED_TRAJECTORY_COLUMNS: tuple[str, ...] = ("id", "frame", "x", "y")


def assert_is_file(path: str | Path, label: str = "path") -> None:
    """Raise :class:`FileNotFoundError` if *path* is not an existing file.

    Args:
        path: Path to validate.
        label: Human-readable label used in the error message.

    Raises:
        FileNotFoundError: When *path* does not point to a regular file.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"{label} is not a valid file: '{p}'")


def assert_file_extension(path: str | Path, allowed: Sequence[str], label: str = "file") -> None:
    """Raise :class:`ValueError` if the file extension is not in *allowed*.

    Args:
        path: Path whose extension should be checked.
        allowed: Collection of permitted lower-case extensions (e.g. ``[".mp4", ".avi"]``).
        label: Human-readable label for the error message.

    Raises:
        ValueError: When the extension is not in *allowed*.
    """
    ext = Path(path).suffix.lower()
    if ext not in [a.lower() for a in allowed]:
        raise ValueError(f"{label} has unexpected extension '{ext}'. Allowed: {list(allowed)}")


def assert_is_not_directory(path: str | Path, label: str = "output path") -> None:
    """Raise :class:`ValueError` when *path* points to a directory.

    This prevents accidentally treating a directory as a file output.

    Args:
        path: Path to check.
        label: Human-readable label for the error message.

    Raises:
        ValueError: When *path* is an existing directory.
    """
    p = Path(path)
    if p.is_dir():
        raise ValueError(f"{label} must not be a directory: '{p}'")


def assert_dataframe_columns(df: pd.DataFrame, required: Sequence[str], label: str = "DataFrame") -> None:
    """Raise :class:`ValueError` when *df* is missing any column in *required*.

    Args:
        df: DataFrame to inspect.
        required: Column names that must be present.
        label: Human-readable label for the error message.

    Raises:
        ValueError: When one or more required columns are absent.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def validate_trajectory_dataframe(df: pd.DataFrame) -> None:
    """Validate that *df* contains all mandatory trajectory columns.

    Args:
        df: Trajectory DataFrame to check.

    Raises:
        ValueError: When mandatory columns are absent.
    """
    assert_dataframe_columns(df, REQUIRED_TRAJECTORY_COLUMNS, label="Trajectory DataFrame")
