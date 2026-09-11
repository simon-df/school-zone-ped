"""Drawing helpers for the interactive TP prediction preview."""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def draw_current_frame(ax, current_df: pd.DataFrame, history_dict: dict[int, np.ndarray]) -> None:
    """Draw current waypoints (red dots + id labels) and observed history lines."""
    for _, row in current_df.iterrows():
        track_id = int(row["id"])
        x, y = float(row["x"]), float(row["y"])

        history = history_dict.get(track_id)
        if history is not None and len(history) >= 2:
            ax.plot(history[:, 0], history[:, 1], color="black", linewidth=1.0, alpha=0.6)

        ax.scatter(x, y, s=80, color="red", edgecolors="white", zorder=5)
        ax.text(x, y, f" {track_id}", fontsize=8, color="black")


def draw_predictions(
    ax,
    predictions: dict[int, np.ndarray],
    num_modes: int,
    show_all_modes: bool = True,
    top_k: int = 3,
) -> None:
    """Draw predicted trajectories as dashed, per-mode colored lines.

    Args:
        ax: Matplotlib axes.
        predictions: Mapping of track id -> array of shape ``(num_modes, pred_len, 2)``.
        num_modes: Number of modes actually present in *predictions*.
        show_all_modes: When ``False``, only the first *top_k* modes are drawn.
        top_k: Number of modes to draw when *show_all_modes* is ``False``.
    """
    modes_to_show = range(num_modes) if show_all_modes else range(min(top_k, num_modes))
    for track_preds in predictions.values():
        for mode_idx in modes_to_show:
            if mode_idx >= track_preds.shape[0]:
                continue
            future = track_preds[mode_idx]
            alpha = 0.8 if mode_idx == 0 else 0.3
            linewidth = 2.0 if mode_idx == 0 else 1.0
            color = plt.cm.tab10(mode_idx % 10)
            ax.plot(future[:, 0], future[:, 1], linestyle="--", linewidth=linewidth, color=color, alpha=alpha)
