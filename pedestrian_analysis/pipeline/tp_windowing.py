"""Observation-window extraction utilities for the TP prediction preview."""
from __future__ import annotations

import numpy as np
import pandas as pd


def get_current_positions(df: pd.DataFrame, frame_idx: int) -> pd.DataFrame:
    """Return the rows of *df* at exactly *frame_idx*."""
    return df[df["frame"] == frame_idx]


def extract_observation_windows(
    df: pd.DataFrame,
    frame_idx: int,
    obs_window: int,
    min_obs_frames: int = 2,
) -> dict[int, np.ndarray]:
    """Build per-track observation windows ending at *frame_idx* (inclusive).

    Args:
        df: Trajectory DataFrame with columns ``id``, ``frame``, ``x``, ``y``.
        frame_idx: Current frame index (inclusive upper bound of the window).
        obs_window: Number of trailing frames to include.
        min_obs_frames: Minimum number of points required to keep a track.

    Returns:
        Mapping of track id -> array of shape ``(obs_len, 2)``, sorted by frame.
    """
    window = df[(df["frame"] <= frame_idx) & (df["frame"] > frame_idx - obs_window)]
    history: dict[int, np.ndarray] = {}
    for track_id, group in window.groupby("id"):
        g = group.sort_values("frame")
        if len(g) < min_obs_frames:
            continue
        history[int(track_id)] = g[["x", "y"]].to_numpy(dtype=np.float64)
    return history


def stack_observations(history: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Stack per-track histories into a single batch array for TP adapters.

    Tracks may have observation windows of differing length (e.g. a
    pedestrian just entered the frame); each history is truncated to the
    shortest available length so they can be stacked into one array.

    Returns:
        ``(track_ids, positions)`` where ``positions`` has shape
        ``(num_pedestrians, obs_len, 2)`` and ``track_ids`` has shape
        ``(num_pedestrians,)``, both empty when *history* is empty.
    """
    if not history:
        return np.empty((0,), dtype=np.int64), np.empty((0, 0, 2), dtype=np.float64)

    track_ids = np.array(sorted(history.keys()), dtype=np.int64)
    min_len = min(arr.shape[0] for arr in history.values())
    positions = np.stack([history[tid][-min_len:] for tid in track_ids], axis=0)
    return track_ids, positions
