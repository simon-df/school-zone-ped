"""Kinematics computation using PedPy (with pure-NumPy fallback).

Computes per-pedestrian speed and heading angle from trajectory data.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _compute_kinematics_numpy(df: pd.DataFrame, fps: float, frame_step: int = 5) -> pd.DataFrame:
    """Pure-NumPy fallback for speed and heading computation.

    Computes speed and heading by finite differences over *frame_step* frames.

    Args:
        df: Trajectory DataFrame with columns ``id``, ``frame``, ``x``, ``y``.
        fps: Frames per second used to convert frame differences to seconds.
        frame_step: Number of frames used for the finite difference.

    Returns:
        *df* augmented with columns ``speed_ms`` and ``heading_deg``.
    """
    result_rows = []
    for ped_id, group in df.groupby("id"):
        g = group.sort_values("frame").copy()
        x = g["x"].to_numpy()
        y = g["y"].to_numpy()
        frames = g["frame"].to_numpy()

        n = len(g)
        speed = np.full(n, np.nan)
        heading = np.full(n, np.nan)

        for i in range(n):
            j = min(i + frame_step, n - 1)
            dt_frames = frames[j] - frames[i]
            if dt_frames == 0:
                continue
            dt_s = dt_frames / fps
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            dist = np.hypot(dx, dy)
            speed[i] = dist / dt_s
            heading[i] = np.degrees(np.arctan2(dy, dx)) % 360.0

        g["speed_ms"] = speed
        g["heading_deg"] = heading
        result_rows.append(g)

    if not result_rows:
        df["speed_ms"] = np.nan
        df["heading_deg"] = np.nan
        return df

    return pd.concat(result_rows, ignore_index=True)


def compute_kinematics(df: pd.DataFrame, fps: float, frame_step: int = 5) -> pd.DataFrame:
    """Compute speed (m/s) and heading (°) for each pedestrian trajectory.

    Attempts to use PedPy if it is installed and compatible; otherwise falls
    back to a pure-NumPy finite-difference implementation.

    Args:
        df: Trajectory DataFrame with at minimum columns ``id``, ``frame``,
            ``x``, ``y``.
        fps: Frames per second of the source video.
        frame_step: Number of frames used for the finite difference.

    Returns:
        *df* augmented with columns ``speed_ms`` and ``heading_deg``.
    """
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")

    try:
        import pedpy  # noqa: F401

        logger.info("PedPy available – using PedPy kinematics.")
        return _compute_kinematics_pedpy(df, fps, frame_step)
    except Exception as exc:
        logger.warning("PedPy not available or incompatible (%s); using NumPy fallback.", exc)
        return _compute_kinematics_numpy(df, fps, frame_step)


def _compute_kinematics_pedpy(df: pd.DataFrame, fps: float, frame_step: int) -> pd.DataFrame:
    """Attempt to compute kinematics via PedPy.

    PedPy's public API may differ between versions. This wrapper tries the
    known API and raises on failure so the caller can fall back gracefully.

    Args:
        df: Trajectory DataFrame.
        fps: Frames per second.
        frame_step: Frame step for finite differences.

    Returns:
        DataFrame with speed and heading columns added.

    Raises:
        Exception: On any PedPy API incompatibility.
    """
    import pedpy

    # PedPy >= 1.1 uses TrajectoryData + compute_individual_speed
    traj_data = pedpy.TrajectoryData(
        data=df[["id", "frame", "x", "y"]].rename(columns={"id": "id", "frame": "frame"}),
        frame_rate=fps,
    )
    speed_df = pedpy.compute_individual_speed(
        traj_data=traj_data,
        frame_step=frame_step,
        speed_calculation=pedpy.SpeedCalculation.BORDER_SINGLE_SIDED,
    )
    merged = df.merge(speed_df[["id", "frame", "speed"]], on=["id", "frame"], how="left")
    merged.rename(columns={"speed": "speed_ms"}, inplace=True)

    # Compute heading via NumPy since PedPy doesn't expose it
    merged = _add_heading(merged, fps, frame_step)
    return merged


def _add_heading(df: pd.DataFrame, fps: float, frame_step: int) -> pd.DataFrame:
    """Add ``heading_deg`` column computed from position differences.

    Args:
        df: Trajectory DataFrame with ``id``, ``frame``, ``x``, ``y``.
        fps: Frames per second (unused here, kept for API symmetry).
        frame_step: Frame step for finite differences.

    Returns:
        *df* with ``heading_deg`` column added.
    """
    result_rows = []
    for _, group in df.groupby("id"):
        g = group.sort_values("frame").copy()
        x = g["x"].to_numpy()
        y = g["y"].to_numpy()
        n = len(g)
        heading = np.full(n, np.nan)
        for i in range(n):
            j = min(i + frame_step, n - 1)
            if j == i:
                continue
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            heading[i] = np.degrees(np.arctan2(dy, dx)) % 360.0
        g["heading_deg"] = heading
        result_rows.append(g)
    if not result_rows:
        df["heading_deg"] = np.nan
        return df
    return pd.concat(result_rows, ignore_index=True)
