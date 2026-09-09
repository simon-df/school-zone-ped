"""Heatmap plot of pedestrian position density."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import logging
logger = logging.getLogger(__name__)


def plot_heatmap(
    df: pd.DataFrame,
    bins: int = 50,
    output_path: str | Path | None = None,
    title: str = "Pedestrian Position Heatmap",
) -> Path | None:
    """Plot a 2-D histogram heatmap of pedestrian positions.

    Args:
        df: Trajectory DataFrame with columns ``x`` and ``y``.
        bins: Number of bins in each dimension.
        output_path: Optional save path for the PNG.
        title: Plot title.

    Returns:
        :class:`pathlib.Path` of the saved file, or ``None``.
    """
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

    x = df["x"].dropna().to_numpy()
    y = df["y"].dropna().to_numpy()

    if len(x) < 2:
        logger.warning("Not enough data points for heatmap.")
        plt.close(fig)
        return None

    h, xedges, yedges, img = ax.hist2d(x, y, bins=bins, cmap="hot_r")
    plt.colorbar(img, ax=ax, label="Count")

    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        logger.info("Saved heatmap to '%s'", output_path)
        plt.close(fig)
        return output_path

    plt.close(fig)
    return None
