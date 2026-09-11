"""Tests for TP Analysis tab support modules: windowing, pedpy/prediction plots."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.tp_windowing import extract_observation_windows, get_current_positions, stack_observations
from visualization.pedpy_plots import create_pedpy_overview_figure
from visualization.prediction_plots import draw_current_frame, draw_predictions


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 2],
            "frame": [0, 1, 2, 1, 2],
            "x": [0.0, 1.0, 2.0, 5.0, 5.2],
            "y": [0.0, 0.0, 0.0, 0.0, 0.1],
        }
    )


def test_get_current_positions_filters_by_frame() -> None:
    df = _sample_df()
    current = get_current_positions(df, 1)
    assert set(current["id"]) == {1, 2}
    assert len(current) == 2


def test_extract_observation_windows_respects_min_obs_frames() -> None:
    df = _sample_df()
    history = extract_observation_windows(df, frame_idx=2, obs_window=5, min_obs_frames=2)

    assert set(history.keys()) == {1, 2}
    np.testing.assert_allclose(history[1], [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    np.testing.assert_allclose(history[2], [[5.0, 0.0], [5.2, 0.1]])


def test_extract_observation_windows_excludes_short_tracks() -> None:
    df = pd.DataFrame({"id": [9], "frame": [2], "x": [1.0], "y": [1.0]})
    history = extract_observation_windows(df, frame_idx=2, obs_window=5, min_obs_frames=2)
    assert history == {}


def test_stack_observations_truncates_to_shortest_and_sorts_by_id() -> None:
    history = {
        2: np.array([[5.0, 0.0], [5.2, 0.1]]),
        1: np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
    }
    track_ids, positions = stack_observations(history)

    assert list(track_ids) == [1, 2]
    assert positions.shape == (2, 2, 2)
    np.testing.assert_allclose(positions[0], [[1.0, 0.0], [2.0, 0.0]])  # last 2 of track 1
    np.testing.assert_allclose(positions[1], [[5.0, 0.0], [5.2, 0.1]])


def test_stack_observations_empty_history() -> None:
    track_ids, positions = stack_observations({})
    assert track_ids.shape == (0,)
    assert positions.shape == (0, 0, 2)


def test_create_pedpy_overview_figure_handles_empty_df() -> None:
    fig, ax = create_pedpy_overview_figure(pd.DataFrame(columns=["id", "frame", "x", "y"]))
    assert fig is not None
    assert "no data loaded" in ax.get_title().lower()


def test_create_pedpy_overview_figure_falls_back_without_pedpy() -> None:
    fig, ax = create_pedpy_overview_figure(_sample_df(), frame_rate=10.0)
    assert fig is not None
    # pedpy is not installed in this environment, so the matplotlib fallback path runs.
    assert len(ax.get_lines()) == 2


def test_draw_current_frame_draws_history_and_points() -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    current_df = get_current_positions(_sample_df(), 2)
    history = extract_observation_windows(_sample_df(), 2, obs_window=5)

    draw_current_frame(ax, current_df, history)

    assert len(ax.collections) == 2  # one scatter point per track
    assert len(ax.lines) == 2  # history line per track with >=2 points
    plt.close(fig)


def test_draw_predictions_respects_show_all_modes_and_top_k() -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    predictions = {1: np.random.rand(5, 3, 2)}

    draw_predictions(ax, predictions, num_modes=5, show_all_modes=False, top_k=2)
    assert len(ax.lines) == 2

    fig2, ax2 = plt.subplots()
    draw_predictions(ax2, predictions, num_modes=5, show_all_modes=True)
    assert len(ax2.lines) == 5

    plt.close(fig)
    plt.close(fig2)
