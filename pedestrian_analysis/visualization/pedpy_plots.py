"""PedPy-based static trajectory overview plot (with a matplotlib fallback)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

import logging
logger = logging.getLogger(__name__)


def create_pedpy_overview_figure(df: pd.DataFrame, frame_rate: float = 10.0):
    """Create a static figure showing all observed trajectories.

    Uses PedPy's ``plot_trajectories`` when available; falls back to a plain
    per-track matplotlib line plot otherwise (mirrors the fallback pattern
    used in :func:`pipeline.pedpy_analysis.compute_kinematics`).

    Args:
        df: Trajectory DataFrame with columns ``id``, ``frame``, ``x``, ``y``.
        frame_rate: Frames per second (required by PedPy's ``TrajectoryData``).

    Returns:
        ``(fig, ax)``.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    if df.empty:
        ax.set_title("All Observed Trajectories (no data loaded)")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        return fig, ax

    try:
        import pedpy

        traj_data = pedpy.TrajectoryData(data=df[["id", "frame", "x", "y"]], frame_rate=frame_rate)
        pedpy.plot_trajectories(traj=traj_data, axes=ax, traj_alpha=0.4, traj_width=1.2)
        ax.set_title("All Observed Trajectories (PedPy)")
    except Exception as exc:
        logger.warning("PedPy plotting unavailable (%s); using matplotlib fallback.", exc)
        for _track_id, group in df.groupby("id"):
            g = group.sort_values("frame")
            ax.plot(g["x"], g["y"], alpha=0.5, linewidth=1.2)
        ax.set_title("All Observed Trajectories")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="datalim")
    return fig, ax
