"""Trajectory plot: lines, start/end markers, street zone and group centroids."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


def plot_trajectories(
    df: pd.DataFrame,
    street_start_m: float = 2.0,
    street_end_m: float = 6.0,
    output_path: str | Path | None = None,
    title: str = "Pedestrian Trajectories",
    show_group_centroids: bool = False,
) -> Path | None:
    """Plot pedestrian trajectories with start/end markers and a street zone.

    Uses Matplotlib for a paper-ready static export.

    Args:
        df: Trajectory DataFrame with columns ``id``, ``x``, ``y``.
        street_start_m: X-coordinate (m) where the crossing begins.
        street_end_m: X-coordinate (m) where the crossing ends.
        output_path: If given, save PNG to this path.
        title: Plot title.
        show_group_centroids: When True, draw dashed group centroid paths.

    Returns:
        :class:`pathlib.Path` of the saved file, or ``None`` when no path given.
    """
    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)

    # Street zone
    ax.axvspan(street_start_m, street_end_m, alpha=0.15, color="gray", label="Street zone")

    colors = plt.cm.tab20.colors
    ped_ids = df["id"].unique()

    for idx, pid in enumerate(ped_ids):
        g = df[df["id"] == pid].sort_values("frame")
        color = colors[int(idx) % len(colors)]
        ax.plot(g["x"], g["y"], color=color, linewidth=1.2, alpha=0.8)
        if len(g) > 0:
            # Start marker: triangle
            ax.plot(g["x"].iloc[0], g["y"].iloc[0], "^", color=color, markersize=7, zorder=5)
            # End marker: square
            ax.plot(g["x"].iloc[-1], g["y"].iloc[-1], "s", color=color, markersize=6, zorder=5)

    if show_group_centroids and "group_id" in df.columns:
        for gid, gdf in df[df["group_id"] >= 0].groupby("group_id"):
            centroid = gdf.groupby("frame")[["x", "y"]].mean().reset_index().sort_values("frame")
            ax.plot(centroid["x"], centroid["y"], "--k", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("X (m)", fontsize=12, fontfamily="DejaVu Sans")
    ax.set_ylabel("Y (m)", fontsize=12, fontfamily="DejaVu Sans")
    ax.set_title(title, fontsize=14, fontfamily="DejaVu Sans")
    ax.grid(True, linestyle="--", alpha=0.4)

    start_patch = mpatches.Patch(color="gray", alpha=0.3, label="Street zone")
    ax.legend(handles=[start_patch], loc="upper right", fontsize=10)

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        logger.info("Saved trajectory plot to '%s'", output_path)
        plt.close(fig)
        return output_path

    plt.close(fig)
    return None
