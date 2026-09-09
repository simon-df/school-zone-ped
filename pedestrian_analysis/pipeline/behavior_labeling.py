"""Behavior labeling for pedestrian trajectories.

Assigns one of four states to each (id, frame) pair:
* ``waiting``    – standing near the kerb
* ``approaching`` – moving toward the crossing
* ``crossing``   – inside the street zone
* ``crossed``    – has left the far side of the crossing
"""

from __future__ import annotations


import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# State constants
STATE_WAITING = "waiting"
STATE_APPROACHING = "approaching"
STATE_CROSSING = "crossing"
STATE_CROSSED = "crossed"


def label_behaviors(
    df: pd.DataFrame,
    street_start_m: float,
    street_end_m: float,
    fps: float,
    speed_threshold_ms: float = 0.3,
    waiting_min_frames: int = 5,
    smooth_window: int = 10,
) -> pd.DataFrame:
    """Assign behavior labels to every row in *df*.

    The labelling is performed per pedestrian and uses hysteresis to avoid
    rapid state flickering.

    Args:
        df: Trajectory DataFrame with columns ``id``, ``frame``, ``x``,
            ``speed_ms`` (optional; computed if absent).
        street_start_m: X-coordinate (m) where the crossing begins.
        street_end_m: X-coordinate (m) where the crossing ends.
        fps: Frames per second (used to convert frame counts to seconds).
        speed_threshold_ms: Speed below which a pedestrian is considered
            stationary.
        waiting_min_frames: Minimum consecutive stationary frames to enter
            the *waiting* state.
        smooth_window: Rolling window for speed smoothing (frames).

    Returns:
        *df* augmented with columns:
        ``behavior``, ``waiting_start_frame``, ``crossing_start_frame``,
        ``waiting_duration_s``.
    """
    if "speed_ms" not in df.columns:
        from pipeline.pedpy_analysis import compute_kinematics

        df = compute_kinematics(df, fps)

    result_rows: list[pd.DataFrame] = []

    for ped_id, group in df.groupby("id"):
        g = group.sort_values("frame").copy()
        n = len(g)

        # yi = g["y"].to_numpy()
        speed = g["speed_ms"].fillna(0.0).to_numpy()

        # Smooth speed to reduce noise
        if smooth_window > 1:
            kernel = np.ones(smooth_window) / smooth_window
            speed_s = np.convolve(speed, kernel, mode="same")
        else:
            speed_s = speed

        behaviors = np.full(n, STATE_APPROACHING, dtype=object)
        waiting_start = np.full(n, np.nan)
        crossing_start = np.full(n, np.nan)
        waiting_duration = np.full(n, np.nan)

        state = STATE_APPROACHING
        w_start: int | None = None
        c_start: int | None = None
        stationary_count = 0
        frames_arr = g["frame"].to_numpy()

        for i in range(n):
            yi = g["y"].to_numpy()[i]
            vi = speed_s[i]
            inside_street = street_start_m <= yi <= street_end_m
            past_street = yi > street_end_m

            # Hysteresis for waiting detection
            if vi < speed_threshold_ms:
                stationary_count += 1
            else:
                stationary_count = 0

            if past_street:
                state = STATE_CROSSED
            elif inside_street:
                if state != STATE_CROSSING:
                    state = STATE_CROSSING
                    c_start = i
            elif stationary_count >= waiting_min_frames and not inside_street and not past_street:
                state = STATE_WAITING
                if w_start is None:
                    w_start = i
            else:
                if state == STATE_WAITING and stationary_count < waiting_min_frames:
                    state = STATE_APPROACHING
                    w_start = None

            behaviors[i] = state
            if w_start is not None:
                waiting_start[i] = float(frames_arr[w_start])
                waiting_duration[i] = float(frames_arr[i] - frames_arr[w_start]) / fps
            if c_start is not None:
                crossing_start[i] = float(frames_arr[c_start])

        g["behavior"] = behaviors
        g["waiting_start_frame"] = waiting_start
        g["crossing_start_frame"] = crossing_start
        g["waiting_duration_s"] = waiting_duration
        result_rows.append(g)

    if not result_rows:
        for col in ("behavior", "waiting_start_frame", "crossing_start_frame", "waiting_duration_s"):
            df[col] = np.nan
        return df

    return pd.concat(result_rows, ignore_index=True)


def compute_crossing_events(
    df: pd.DataFrame,
    fps: float,
) -> pd.DataFrame:
    """Summarise crossing events: one row per pedestrian.

    Args:
        df: Labeled trajectory DataFrame (output of :func:`label_behaviors`).
        fps: Frames per second.

    Returns:
        Summary DataFrame with columns ``id``, ``crossing_start_frame``,
        ``crossing_end_frame``, ``crossing_duration_s``, ``waiting_duration_s``.
    """
    if "behavior" not in df.columns:
        raise ValueError("DataFrame must have a 'behavior' column. Run label_behaviors first.")

    records = []
    for ped_id, g in df.groupby("id"):
        g = g.sort_values("frame")
        crossing = g[g["behavior"] == STATE_CROSSING]
        if crossing.empty:
            continue
        c_start = int(crossing["frame"].min())
        c_end = int(crossing["frame"].max())
        c_dur = (c_end - c_start) / fps
        w_dur = g["waiting_duration_s"].max()
        if pd.isna(w_dur):
            w_dur = 0.0
        records.append(
            {
                "id": ped_id,
                "crossing_start_frame": c_start,
                "crossing_end_frame": c_end,
                "crossing_duration_s": c_dur,
                "waiting_duration_s": w_dur,
            }
        )
    if not records:
        return pd.DataFrame(
            columns=["id", "crossing_start_frame", "crossing_end_frame", "crossing_duration_s", "waiting_duration_s"]
        )
    return pd.DataFrame(records)
