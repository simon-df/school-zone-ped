"""Frame-rate resampling of trajectory DataFrames.

ETH/UCY is annotated at 2.5 Hz while this app's drone exports and the TP
Analysis tab defaults use 10 Hz. :func:`resample_trajectories` linearly
interpolates (upsampling) or picks interpolated samples on a coarser grid
(decimation) so every track lies on a uniform ``1 / target_fps`` time grid.

The output ``frame`` column is the integer index on that grid
(``round(timestamp * target_fps)``), so consecutive samples of a track have
consecutive frame numbers -- which is what the sliding-window extraction in
:mod:`training.tp.datasets` relies on.

CLI (resample an exported CSV, e.g. to match the UI's 10 Hz assumption)::

    python -m training.tp.resample input.csv output.csv --fps 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

RESAMPLED_COLUMNS: tuple[str, ...] = ("id", "frame", "timestamp", "x", "y")
_EPS = 1e-6


def infer_fps(df: pd.DataFrame, time_col: str = "timestamp", id_col: str = "id") -> float:
    """Estimate the sampling rate as ``1 / median`` per-track time step."""
    steps: list[np.ndarray] = []
    for _, group in df.groupby(id_col):
        t = np.sort(group[time_col].to_numpy(dtype=np.float64))
        dt = np.diff(t)
        steps.append(dt[dt > _EPS])
    all_steps = np.concatenate(steps) if steps else np.empty(0)
    if all_steps.size == 0:
        raise ValueError("Cannot infer fps: no track has two samples with distinct timestamps.")
    return float(1.0 / np.median(all_steps))


def _resample_segment(t: np.ndarray, xy: np.ndarray, target_fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate one gap-free segment onto the integer grid ``k / target_fps``."""
    k_start = int(np.ceil(t[0] * target_fps - _EPS))
    k_end = int(np.floor(t[-1] * target_fps + _EPS))
    if k_end < k_start:
        return np.empty(0, dtype=np.int64), np.empty((0, 2))
    grid_k = np.arange(k_start, k_end + 1, dtype=np.int64)
    grid_t = grid_k / target_fps
    if t.size == 1:
        return grid_k, np.repeat(xy, grid_k.size, axis=0)
    x = np.interp(grid_t, t, xy[:, 0])
    y = np.interp(grid_t, t, xy[:, 1])
    return grid_k, np.stack([x, y], axis=1)


def resample_trajectories(
    df: pd.DataFrame,
    target_fps: float,
    *,
    time_col: str = "timestamp",
    id_col: str = "id",
    max_gap_s: Optional[float] = None,
) -> pd.DataFrame:
    """Resample every track of *df* onto a uniform ``target_fps`` time grid.

    Args:
        df: Trajectory DataFrame with columns ``id``, ``x``, ``y`` and a time
            column in seconds (``timestamp`` by default).
        target_fps: Output sampling rate in Hz.
        time_col: Name of the time column (seconds).
        id_col: Name of the track id column.
        max_gap_s: Tracks are split at time gaps larger than this; no samples
            are interpolated across such gaps. ``None`` uses 2.5x the median
            sampling interval of *df* (bridges single missed samples only).

    Returns:
        DataFrame with columns ``id``, ``frame``, ``timestamp``, ``x``, ``y``
        where ``frame = round(timestamp * target_fps)`` and
        ``timestamp = frame / target_fps``; sorted by ``frame``, ``id``.
    """
    if target_fps <= 0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")
    missing = {id_col, time_col, "x", "y"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for resampling: {sorted(missing)}")
    if df.empty:
        return pd.DataFrame(columns=list(RESAMPLED_COLUMNS))

    if max_gap_s is None:
        try:
            max_gap_s = 2.5 / infer_fps(df, time_col=time_col, id_col=id_col)
        except ValueError:
            max_gap_s = float("inf")

    ids: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    for track_id, group in df.groupby(id_col, sort=True):
        g = group.sort_values(time_col).drop_duplicates(subset=time_col, keep="first")
        t = g[time_col].to_numpy(dtype=np.float64)
        xy = g[["x", "y"]].to_numpy(dtype=np.float64)
        split_at = np.nonzero(np.diff(t) > max_gap_s + _EPS)[0] + 1
        for seg_t, seg_xy in zip(np.split(t, split_at), np.split(xy, split_at)):
            grid_k, grid_xy = _resample_segment(seg_t, seg_xy, target_fps)
            if grid_k.size == 0:
                continue
            ids.append(np.full(grid_k.size, track_id))
            frames.append(grid_k)
            positions.append(grid_xy)

    if not frames:
        return pd.DataFrame(columns=list(RESAMPLED_COLUMNS))

    frame_arr = np.concatenate(frames)
    xy_arr = np.concatenate(positions)
    out = pd.DataFrame(
        {
            "id": np.concatenate(ids),
            "frame": frame_arr,
            "timestamp": frame_arr / target_fps,
            "x": xy_arr[:, 0],
            "y": xy_arr[:, 1],
        }
    )
    # Segments of one track never overlap in time, but guard against duplicates anyway.
    out = out.drop_duplicates(subset=["id", "frame"], keep="first")
    return out.sort_values(["frame", "id"]).reset_index(drop=True)


def main(argv: Optional[list[str]] = None) -> None:
    from pipeline.trajectory_io import load_trajectory_csv

    parser = argparse.ArgumentParser(description="Resample a trajectory CSV to a target frame rate.")
    parser.add_argument("input", help="Input trajectory CSV (id/frame/timestamp/x/y or track_id/x_m/y_m).")
    parser.add_argument("output", help="Output CSV path.")
    parser.add_argument("--fps", type=float, default=10.0, help="Target frame rate in Hz (default: 10).")
    parser.add_argument(
        "--source-fps",
        type=float,
        default=10.0,
        help="Frame rate used to derive timestamps when the input has no timestamp column (default: 10).",
    )
    parser.add_argument("--max-gap-s", type=float, default=None, help="Split tracks at time gaps larger than this.")
    args = parser.parse_args(argv)

    df = load_trajectory_csv(args.input, fps=args.source_fps)
    out = resample_trajectories(df, args.fps, max_gap_s=args.max_gap_s)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Resampled {len(df)} rows -> {len(out)} rows at {args.fps:g} Hz: {args.output}")


if __name__ == "__main__":
    main()
