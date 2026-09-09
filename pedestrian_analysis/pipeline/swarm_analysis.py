"""Swarm behaviour metrics: cohesion, alignment, separation, leader-follower."""

from __future__ import annotations


import numpy as np
import pandas as pd
from scipy.stats import circstd

import logging
logger = logging.getLogger(__name__)


def compute_cohesion(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-pedestrian distance to the group centroid for each frame.

    Adds column ``cohesion_m`` to *df*.  Pedestrians not in any group
    (``group_id == -1``) receive ``NaN``.

    Args:
        df: Trajectory DataFrame with columns ``id``, ``frame``, ``x``, ``y``,
            ``group_id``.

    Returns:
        *df* augmented with column ``cohesion_m``.
    """
    if "group_id" not in df.columns:
        raise ValueError("DataFrame must have 'group_id' column.")

    df = df.copy()
    df["cohesion_m"] = np.nan

    for (frame_val, group_id), group in df.groupby(["frame", "group_id"]):
        if group_id < 0 or len(group) < 2:
            continue
        cx = group["x"].mean()
        cy = group["y"].mean()
        distances = np.sqrt((group["x"] - cx) ** 2 + (group["y"] - cy) ** 2)
        df.loc[group.index, "cohesion_m"] = distances.values

    return df


def compute_alignment(df: pd.DataFrame) -> pd.DataFrame:
    """Compute circular standard deviation of heading angles per group per frame.

    Adds column ``alignment_deg`` (lower = more aligned).  Pedestrians not in
    a group receive ``NaN``.

    Args:
        df: Trajectory DataFrame with columns ``id``, ``frame``, ``heading_deg``,
            ``group_id``.

    Returns:
        *df* augmented with column ``alignment_deg``.
    """
    if "group_id" not in df.columns:
        raise ValueError("DataFrame must have 'group_id' column.")
    if "heading_deg" not in df.columns:
        raise ValueError("DataFrame must have 'heading_deg' column. Run compute_kinematics first.")

    df = df.copy()
    df["alignment_deg"] = np.nan

    for (frame_val, group_id), group in df.groupby(["frame", "group_id"]):
        if group_id < 0 or len(group) < 2:
            continue
        headings_rad = np.radians(group["heading_deg"].dropna().values)
        if len(headings_rad) < 2:
            continue
        circ_std_deg = float(np.degrees(circstd(headings_rad)))
        df.loc[group.index, "alignment_deg"] = circ_std_deg

    return df


def compute_separation(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the minimum pairwise inter-pedestrian distance per group per frame.

    Adds column ``min_separation_m``.  Pedestrians not in a group receive ``NaN``.

    Args:
        df: Trajectory DataFrame with columns ``id``, ``frame``, ``x``, ``y``,
            ``group_id``.

    Returns:
        *df* augmented with column ``min_separation_m``.
    """
    if "group_id" not in df.columns:
        raise ValueError("DataFrame must have 'group_id' column.")

    df = df.copy()
    df["min_separation_m"] = np.nan

    for (frame_val, group_id), group in df.groupby(["frame", "group_id"]):
        if group_id < 0 or len(group) < 2:
            continue
        coords = group[["x", "y"]].to_numpy()
        # Pairwise distances
        min_dist = np.inf
        n = len(coords)
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(coords[i] - coords[j])
                if d < min_dist:
                    min_dist = d
        df.loc[group.index, "min_separation_m"] = float(min_dist) if np.isfinite(min_dist) else np.nan

    return df


def compute_leader_follower(
    df: pd.DataFrame,
    fps: float,
    max_delay_s: float = 5.0,
) -> pd.DataFrame:
    """Identify the leader of each group as the pedestrian who starts crossing first.

    Adds column ``is_leader`` (boolean).

    Args:
        df: Trajectory DataFrame with columns ``id``, ``frame``, ``group_id``,
            ``behavior`` (requires ``crossing`` state).
        fps: Frames per second used to convert frame counts to seconds.
        max_delay_s: Maximum delay (seconds) within which a follower can still
            be associated with the same leader event.

    Returns:
        *df* augmented with column ``is_leader``.
    """
    if "behavior" not in df.columns:
        raise ValueError("DataFrame must have 'behavior' column. Run label_behaviors first.")
    if "group_id" not in df.columns:
        raise ValueError("DataFrame must have 'group_id' column.")

    df = df.copy()
    df["is_leader"] = False

    for group_id, group_df in df[df["group_id"] >= 0].groupby("group_id"):
        crossing = group_df[group_df["behavior"] == "crossing"]
        if crossing.empty:
            continue
        # First crossing frame per pedestrian
        first_cross = crossing.groupby("id")["frame"].min()
        leader_id = first_cross.idxmin()
        leader_frame = first_cross.min()

        # Mark all rows for the leader
        df.loc[(df["id"] == leader_id) & (df["group_id"] == group_id), "is_leader"] = True

        # Optionally verify followers are within max_delay_s
        max_delay_frames = max_delay_s * fps
        for pid, cross_frame in first_cross.items():
            if pid == leader_id:
                continue
            if cross_frame - leader_frame > max_delay_frames:
                logger.debug(
                    "Pedestrian %s crossed more than %.1f s after group leader (%s). Not counted as follower.",
                    pid,
                    max_delay_s,
                    leader_id,
                )

    return df
