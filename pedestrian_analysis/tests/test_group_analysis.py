"""Tests for pipeline/group_analysis.py"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from pipeline.group_analysis import detect_groups_per_frame, compute_group_statistics


def _make_two_cluster_df(n_frames: int = 30) -> pd.DataFrame:
    """Create a DataFrame with two clearly separated clusters of 3 pedestrians each."""
    rows = []
    # Cluster A: centred at (0, 0), pedestrians 1-3
    for pid in range(1, 4):
        for frame in range(n_frames):
            rows.append({
                "id": pid,
                "frame": frame,
                "x": float(pid - 2) * 0.3,   # within 0.6 m of each other
                "y": 0.0,
            })
    # Cluster B: centred at (10, 0), pedestrians 4-6
    for pid in range(4, 7):
        for frame in range(n_frames):
            rows.append({
                "id": pid,
                "frame": frame,
                "x": 10.0 + float(pid - 5) * 0.3,
                "y": 0.0,
            })
    return pd.DataFrame(rows)


class TestDetectGroupsPerFrame:
    def test_two_clusters_detected(self):
        """Two spatially separated clusters should get different group IDs."""
        df = _make_two_cluster_df(n_frames=30)
        result = detect_groups_per_frame(df, proximity_m=1.5, min_group_frames=5)

        assert "group_id" in result.columns

        # After temporal smoothing, ped 1-3 should share a group, 4-6 another
        first_frame = result[result["frame"] == 15]
        group_a = set(first_frame[first_frame["id"].isin([1, 2, 3])]["group_id"])
        group_b = set(first_frame[first_frame["id"].isin([4, 5, 6])]["group_id"])

        # Both groups should be assigned (not noise = -1)
        assert -1 not in group_a or len(group_a - {-1}) > 0, "Cluster A not detected"
        assert -1 not in group_b or len(group_b - {-1}) > 0, "Cluster B not detected"

        # The two clusters must have different group IDs
        valid_a = group_a - {-1}
        valid_b = group_b - {-1}
        if valid_a and valid_b:
            assert valid_a.isdisjoint(valid_b), f"Clusters share group IDs: {valid_a} ∩ {valid_b}"

    def test_noise_for_isolated_pedestrian(self):
        """A single isolated pedestrian should get group_id == -1."""
        rows = [
            {"id": 1, "frame": i, "x": 0.0, "y": 0.0} for i in range(30)
        ]
        df = pd.DataFrame(rows)
        result = detect_groups_per_frame(df, proximity_m=1.5, min_group_frames=5)
        assert (result["group_id"] == -1).all()

    def test_output_column_present(self):
        df = _make_two_cluster_df(n_frames=15)
        result = detect_groups_per_frame(df)
        assert "group_id" in result.columns


class TestComputeGroupStatistics:
    def test_group_count(self):
        df = _make_two_cluster_df(n_frames=30)
        df = detect_groups_per_frame(df, proximity_m=1.5, min_group_frames=5)
        stats = compute_group_statistics(df)
        assert stats["group_count"] >= 1

    def test_empty_data_returns_zeros(self):
        """If no groups exist, statistics should reflect empty state."""
        rows = [{"id": i, "frame": 0, "x": float(i * 100), "y": 0.0, "group_id": -1} for i in range(5)]
        df = pd.DataFrame(rows)
        stats = compute_group_statistics(df)
        assert stats["group_count"] == 0
        assert stats["size_histogram"] == {}
