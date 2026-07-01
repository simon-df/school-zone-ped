"""Group detection and analysis using DBSCAN clustering with temporal smoothing."""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)


def detect_groups_per_frame(
    df: pd.DataFrame,
    proximity_m: float = 1.5,
    min_group_frames: int = 10,
    smooth_window: int = 10,
) -> pd.DataFrame:
    """Assign group IDs to each (id, frame) pair using DBSCAN per frame.

    Steps:
    1. For every frame, run DBSCAN on (x, y) positions.
    2. Temporally smooth group assignments across consecutive frames.
    3. Discard groups that persist fewer than *min_group_frames* frames.

    Args:
        df: Trajectory DataFrame with columns ``id``, ``frame``, ``x``, ``y``.
        proximity_m: DBSCAN *epsilon* (metres).
        min_group_frames: Minimum number of frames a group must be visible.
        smooth_window: Number of frames used for temporal smoothing.

    Returns:
        *df* augmented with column ``group_id``
        (``-1`` means noise / no group).
    """
    df = df.copy()
    df["group_id"] = -1

    frame_labels: dict[int, dict[int, int]] = {}  # frame -> {ped_id -> group_label}

    for frame_val, frame_df in df.groupby("frame"):
        if len(frame_df) < 2:
            frame_labels[int(frame_val)] = {int(r["id"]): -1 for _, r in frame_df.iterrows()}
            continue

        coords = frame_df[["x", "y"]].to_numpy()
        labels = DBSCAN(eps=proximity_m, min_samples=2).fit_predict(coords)
        frame_labels[int(frame_val)] = {
            int(row["id"]): int(labels[i])
            for i, (_, row) in enumerate(frame_df.iterrows())
        }

    # Temporal smoothing: majority vote over smooth_window frames per pedestrian
    ped_ids = df["id"].unique()
    all_frames = sorted(df["frame"].unique())

    # Build a mapping (ped_id, frame) -> raw_label
    raw: dict[tuple[int, int], int] = {}
    for frame_val, pid_map in frame_labels.items():
        for pid, label in pid_map.items():
            raw[(pid, frame_val)] = label

    smoothed: dict[tuple[int, int], int] = {}
    for pid in ped_ids:
        pid_frames = sorted(f for (p, f) in raw if p == pid)
        for i, frame_val in enumerate(pid_frames):
            window_start = max(0, i - smooth_window // 2)
            window_end = min(len(pid_frames), i + smooth_window // 2 + 1)
            window_labels = [raw.get((pid, pid_frames[j]), -1) for j in range(window_start, window_end)]
            # Majority vote (excluding noise)
            non_noise = [lb for lb in window_labels if lb >= 0]
            if non_noise:
                majority = max(set(non_noise), key=non_noise.count)
            else:
                majority = -1
            smoothed[(pid, frame_val)] = majority

    # Count group persistence
    group_frame_counts: dict[tuple[int, int], int] = defaultdict(int)  # (pid, group_label) -> frame_count
    for (pid, _), label in smoothed.items():
        if label >= 0:
            group_frame_counts[(pid, label)] += 1

    # Assign back to DataFrame
    for idx, row in df.iterrows():
        pid = int(row["id"])
        frame_val = int(row["frame"])
        label = smoothed.get((pid, frame_val), -1)
        if label >= 0 and group_frame_counts.get((pid, label), 0) < min_group_frames:
            label = -1
        df.at[idx, "group_id"] = label

    return df


def compute_group_statistics(df: pd.DataFrame) -> dict:
    """Compute aggregate group statistics for paper-ready reporting.

    Args:
        df: Trajectory DataFrame with columns ``id``, ``frame``, ``group_id``,
            and optionally ``behavior``, ``waiting_duration_s``.

    Returns:
        Dictionary with keys:
        ``group_count``, ``size_histogram``, ``split_events``,
        ``median_waiting_time_s``, ``partial_crossing_rate``.
    """
    if "group_id" not in df.columns:
        raise ValueError("DataFrame must have 'group_id' column. Run detect_groups_per_frame first.")

    groups = df[df["group_id"] >= 0]
    if groups.empty:
        return {
            "group_count": 0,
            "size_histogram": {},
            "split_events": 0,
            "median_waiting_time_s": float("nan"),
            "partial_crossing_rate": float("nan"),
        }

    # Number of distinct groups
    unique_groups = groups["group_id"].unique()
    group_count = len(unique_groups)

    # Group size histogram
    sizes = groups.groupby("group_id")["id"].nunique()
    size_histogram = sizes.value_counts().sort_index().to_dict()

    # Heuristic split-event detection: group seen in two sub-groups later
    split_events = _count_split_events(df)

    # Median waiting time
    median_waiting_time_s = float("nan")
    if "waiting_duration_s" in df.columns:
        wt = df.groupby("id")["waiting_duration_s"].max().dropna()
        if not wt.empty:
            median_waiting_time_s = float(wt.median())

    # Partial crossing rate
    partial_crossing_rate = float("nan")
    if "behavior" in df.columns:
        ped_ids = df["id"].unique()
        partial_count = 0
        crossed_count = 0
        for pid in ped_ids:
            g = df[df["id"] == pid]
            behaviors = set(g["behavior"].dropna())
            if "crossing" in behaviors and "crossed" not in behaviors:
                partial_count += 1
            if "crossing" in behaviors:
                crossed_count += 1
        if crossed_count > 0:
            partial_crossing_rate = partial_count / crossed_count

    return {
        "group_count": group_count,
        "size_histogram": size_histogram,
        "split_events": split_events,
        "median_waiting_time_s": median_waiting_time_s,
        "partial_crossing_rate": partial_crossing_rate,
    }


def _count_split_events(df: pd.DataFrame) -> int:
    """Heuristically count the number of group split events.

    A split is counted when a group ID that was together in a frame is later
    seen with different sub-group IDs.

    Args:
        df: Trajectory DataFrame with ``group_id``, ``id``, ``frame``.

    Returns:
        Number of detected split events.
    """
    splits = 0
    if "group_id" not in df.columns:
        return splits

    df_g = df[df["group_id"] >= 0].copy()
    if df_g.empty:
        return splits

    # For each pedestrian, track changes in group membership over time
    for pid, pdata in df_g.groupby("id"):
        pdata = pdata.sort_values("frame")
        prev_label = None
        for _, row in pdata.iterrows():
            label = row["group_id"]
            if prev_label is not None and prev_label != label and prev_label >= 0 and label >= 0:
                splits += 1
            prev_label = label
    return splits
