"""Tests for tracking metrics helpers."""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.tracker import compute_tracking_metrics


def test_tracking_metrics_report_basic_stats() -> None:
    df = pd.DataFrame(
        [
            {"id": 1, "frame": 0, "x": 0.0, "y": 0.0},
            {"id": 1, "frame": 1, "x": 0.1, "y": 0.0},
            {"id": 2, "frame": 0, "x": 1.0, "y": 0.0},
            {"id": 2, "frame": 2, "x": 1.2, "y": 0.0},
        ]
    )
    metrics = compute_tracking_metrics(df)

    assert metrics["track_length_stats"]["count"] == 2
    assert metrics["id_switches"] >= 0
    # Track 2 skips frame 1, so the fragmentation heuristic should report one gap.
    assert metrics["fragmentations"] == 1
    assert metrics["active_track_count_per_frame"][0] == 2
    assert metrics["new_tracks_per_frame"][0] == 2
    assert metrics["lost_tracks_per_frame"][2] == 1
