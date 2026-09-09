"""Swarm metrics plot: cohesion, alignment and separation over time."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import logging
logger = logging.getLogger(__name__)


def plot_swarm_metrics(
    df: pd.DataFrame,
    fps: float = 25.0,
    output_path: str | Path | None = None,
    title: str = "Swarm Metrics",
) -> Path | None:
    """Plot cohesion, alignment and separation as time series.

    Expects columns ``cohesion_m``, ``alignment_deg``, ``min_separation_m``
    (all optional; missing columns are skipped).

    Args:
        df: Trajectory DataFrame with swarm metric columns and ``frame``.
        fps: Frames per second for the time axis.
        output_path: Optional save path for the PNG.
        title: Overall figure title.

    Returns:
        :class:`pathlib.Path` of the saved file, or ``None``.
    """
    metrics = [
        ("cohesion_m", "Cohesion (m)", "steelblue"),
        ("alignment_deg", "Alignment (°)", "darkorange"),
        ("min_separation_m", "Min Separation (m)", "seagreen"),
    ]
    available = [(col, lbl, col_c) for col, lbl, col_c in metrics if col in df.columns]

    if not available:
        logger.warning("No swarm metric columns found – skipping swarm plot.")
        return None

    n_plots = len(available)
    fig, axes = plt.subplots(n_plots, 1, figsize=(11, 3 * n_plots), dpi=150, sharex=True)
    if n_plots == 1:
        axes = [axes]

    for ax, (col, label, color) in zip(axes, available):
        grouped = df.groupby("frame")[col].mean()
        time_s = grouped.index / fps
        ax.plot(time_s, grouped.values, color=color, linewidth=1.5)
        ax.set_ylabel(label, fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Time (s)", fontsize=12)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        logger.info("Saved swarm metrics plot to '%s'", output_path)
        plt.close(fig)
        return output_path

    plt.close(fig)
    return None
