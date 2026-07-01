"""Group statistics plots: size histogram and optional split-event overview."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def plot_group_statistics(
    stats: dict,
    output_path: str | Path | None = None,
    title: str = "Group Size Distribution",
) -> Path | None:
    """Plot a bar chart of group size frequencies.

    Args:
        stats: Dictionary from :func:`pipeline.group_analysis.compute_group_statistics`.
        output_path: Optional save path for the PNG.
        title: Plot title.

    Returns:
        :class:`pathlib.Path` of the saved file, or ``None``.
    """
    hist = stats.get("size_histogram", {})
    if not hist:
        logger.warning("Empty size histogram – skipping group plot.")
        return None

    sizes = sorted(hist.keys())
    counts = [hist[s] for s in sizes]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.bar([str(s) for s in sizes], counts, color="steelblue", edgecolor="white")
    ax.set_xlabel("Group size (# pedestrians)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    split_events = stats.get("split_events", 0)
    ax.annotate(
        f"Split events: {split_events}",
        xy=(0.98, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=10,
    )

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        logger.info("Saved group statistics plot to '%s'", output_path)
        plt.close(fig)
        return output_path

    plt.close(fig)
    return None
