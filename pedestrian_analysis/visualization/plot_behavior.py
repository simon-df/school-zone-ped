"""Behavior timeline plot: state bars per pedestrian."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

import logging
logger = logging.getLogger(__name__)

_STATE_COLORS = {
    "waiting": "steelblue",
    "approaching": "orange",
    "crossing": "green",
    "crossed": "gray",
}


def plot_behavior_timeline(
    df: pd.DataFrame,
    fps: float = 25.0,
    output_path: str | Path | None = None,
    title: str = "Behavior Timeline",
) -> Path | None:
    """Plot a horizontal bar timeline of behavior states per pedestrian.

    Args:
        df: Trajectory DataFrame with columns ``id``, ``frame``, ``behavior``.
        fps: Frames per second (used to convert frame numbers to seconds on X-axis).
        output_path: Optional save path for the PNG.
        title: Plot title.

    Returns:
        :class:`pathlib.Path` of the saved file, or ``None``.
    """
    if "behavior" not in df.columns:
        logger.warning("No 'behavior' column found – skipping timeline plot.")
        return None

    ped_ids = sorted(df["id"].unique())
    fig, ax = plt.subplots(figsize=(12, max(4, len(ped_ids) * 0.4 + 2)), dpi=150)

    for y_pos, pid in enumerate(ped_ids):
        g = df[df["id"] == pid].sort_values("frame")
        frames = g["frame"].to_numpy()
        states = g["behavior"].to_numpy()

        if len(frames) == 0:
            continue

        # Draw segments
        start_f = frames[0]
        cur_state = states[0]
        for i in range(n):
            if states[i] != cur_state:
                end_f = frames[i]
                t_start = start_f / fps
                t_end = end_f / fps
                color = _STATE_COLORS.get(str(cur_state), "purple")
                ax.barh(y_pos, t_end - t_start, left=t_start, height=0.7, color=color, alpha=0.85)
                start_f = frames[i]
                cur_state = states[i]

        # Draw the final segment
        end_f = frames[-1]
        t_start = start_f / fps
        t_end = end_f / fps
        if t_end > t_start:
            color = _STATE_COLORS.get(str(cur_state), "purple")
            ax.barh(y_pos, t_end - t_start, left=t_start, height=0.7, color=color, alpha=0.85)

    ax.set_yticks(range(len(ped_ids)))
    ax.set_yticklabels([str(p) for p in ped_ids], fontsize=8)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Pedestrian ID", fontsize=12)
    ax.set_title(title, fontsize=14)

    patches = [mpatches.Patch(color=c, label=s) for s, c in _STATE_COLORS.items()]
    ax.legend(handles=patches, loc="upper right", fontsize=9)
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        logger.info("Saved behavior timeline to '%s'", output_path)
        plt.close(fig)
        return output_path

    plt.close(fig)
    return None
