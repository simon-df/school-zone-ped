"""Tests for pipeline/behavior_labeling.py"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from pipeline.behavior_labeling import label_behaviors, STATE_WAITING, STATE_CROSSING, STATE_CROSSED, STATE_APPROACHING


def _make_df(x_values, speed_values, ped_id: int = 1, fps: float = 25.0) -> pd.DataFrame:
    """Build a minimal trajectory DataFrame for one pedestrian."""
    n = len(x_values)
    return pd.DataFrame({
        "id": [ped_id] * n,
        "frame": list(range(n)),
        "x": x_values,
        "y": [0.0] * n,
        "speed_ms": speed_values,
    })


class TestLabelBehaviors:
    def test_approaching_state(self):
        """Fast-moving pedestrian not in street zone → approaching."""
        n = 20
        df = _make_df(
            x_values=[0.5] * n,       # before the street
            speed_values=[1.0] * n,   # moving fast
        )
        result = label_behaviors(df, street_start_m=2.0, street_end_m=6.0, fps=25.0)
        assert "behavior" in result.columns
        # Before the street and fast → approaching
        assert (result["behavior"] == STATE_APPROACHING).all(), result["behavior"].values

    def test_waiting_state(self):
        """Slow pedestrian near the kerb for enough frames → waiting."""
        n = 30
        df = _make_df(
            x_values=[1.0] * n,       # before street
            speed_values=[0.0] * n,   # stationary
        )
        result = label_behaviors(
            df, street_start_m=2.0, street_end_m=6.0, fps=25.0, waiting_min_frames=5
        )
        # After the min_frames window the state should be waiting
        waiting_rows = result[result["frame"] >= 5]
        assert (waiting_rows["behavior"] == STATE_WAITING).all(), waiting_rows["behavior"].values

    def test_crossing_state(self):
        """Pedestrian inside the street zone → crossing."""
        n = 20
        df = _make_df(
            x_values=[4.0] * n,       # inside street zone [2, 6]
            speed_values=[1.0] * n,
        )
        result = label_behaviors(df, street_start_m=2.0, street_end_m=6.0, fps=25.0)
        assert (result["behavior"] == STATE_CROSSING).all(), result["behavior"].values

    def test_crossed_state(self):
        """Pedestrian past the street end → crossed."""
        n = 20
        df = _make_df(
            x_values=[8.0] * n,       # past street end of 6.0
            speed_values=[1.0] * n,
        )
        result = label_behaviors(df, street_start_m=2.0, street_end_m=6.0, fps=25.0)
        assert (result["behavior"] == STATE_CROSSED).all(), result["behavior"].values

    def test_state_sequence(self):
        """A complete crossing sequence should include all four states."""
        fps = 25.0
        # 20 stationary frames → waiting, then 20 moving frames in street → crossing,
        # then 20 frames past the street → crossed
        n_wait = 20
        n_cross = 20
        n_past = 20
        x = [1.0] * n_wait + [4.0] * n_cross + [8.0] * n_past
        spd = [0.0] * n_wait + [1.5] * n_cross + [1.5] * n_past
        df = _make_df(x, spd)
        result = label_behaviors(df, street_start_m=2.0, street_end_m=6.0, fps=fps, waiting_min_frames=5)
        states = set(result["behavior"])
        assert STATE_WAITING in states, f"Missing 'waiting'. Got: {states}"
        assert STATE_CROSSING in states, f"Missing 'crossing'. Got: {states}"
        assert STATE_CROSSED in states, f"Missing 'crossed'. Got: {states}"

    def test_output_columns(self):
        """Result DataFrame must have all required output columns."""
        n = 10
        df = _make_df([1.0] * n, [0.5] * n)
        result = label_behaviors(df, 2.0, 6.0, fps=25.0)
        for col in ("behavior", "waiting_start_frame", "crossing_start_frame", "waiting_duration_s"):
            assert col in result.columns, f"Missing column: {col}"
