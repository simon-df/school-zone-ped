"""Tests for pipeline/swarm_analysis.py"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from pipeline.swarm_analysis import compute_cohesion, compute_alignment, compute_separation


def _make_group_df(positions: list[tuple[float, float]], frame: int = 0, group_id: int = 0) -> pd.DataFrame:
    """Create a single-frame DataFrame with given pedestrian positions."""
    rows = []
    for pid, (x, y) in enumerate(positions, start=1):
        rows.append({"id": pid, "frame": frame, "x": x, "y": y, "group_id": group_id})
    return pd.DataFrame(rows)


class TestComputeCohesion:
    def test_equidistant_square(self):
        """Four pedestrians at the corners of a 2×2 square.
        Each is at distance sqrt(2) ≈ 1.414 from the centroid (0, 0)."""
        positions = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
        df = _make_group_df(positions)
        result = compute_cohesion(df)
        expected_dist = np.sqrt(2.0)
        for v in result["cohesion_m"].dropna():
            assert abs(v - expected_dist) < 0.01, f"Expected {expected_dist:.3f}, got {v:.3f}"

    def test_collinear_points(self):
        """Three pedestrians at (-1,0), (0,0), (1,0); centroid is (0,0)."""
        positions = [(-1.0, 0.0), (0.0, 0.0), (1.0, 0.0)]
        df = _make_group_df(positions)
        result = compute_cohesion(df)
        expected = [1.0, 0.0, 1.0]
        for v, exp in zip(sorted(result["cohesion_m"].dropna()), sorted(expected)):
            assert abs(v - exp) < 0.01, f"Expected {exp}, got {v}"

    def test_no_group_is_nan(self):
        """Pedestrians with group_id == -1 should have NaN cohesion."""
        df = _make_group_df([(0, 0), (1, 0)], group_id=-1)
        result = compute_cohesion(df)
        assert result["cohesion_m"].isna().all()

    def test_raises_without_group_id(self):
        df = pd.DataFrame({"id": [1], "frame": [0], "x": [0.0], "y": [0.0]})
        with pytest.raises(ValueError):
            compute_cohesion(df)


class TestComputeAlignment:
    def test_identical_headings(self):
        """All pedestrians moving in the same direction → alignment_deg ≈ 0."""
        positions = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
        df = _make_group_df(positions)
        df["heading_deg"] = 90.0  # all pointing the same way
        result = compute_alignment(df)
        for v in result["alignment_deg"].dropna():
            assert v < 5.0, f"Expected low alignment std, got {v}"

    def test_raises_without_heading(self):
        df = _make_group_df([(0, 0), (1, 0)])
        with pytest.raises(ValueError):
            compute_alignment(df)


class TestComputeSeparation:
    def test_known_distance(self):
        """Two pedestrians 4 m apart; min_separation should be 4."""
        positions = [(0.0, 0.0), (4.0, 0.0)]
        df = _make_group_df(positions)
        result = compute_separation(df)
        for v in result["min_separation_m"].dropna():
            assert abs(v - 4.0) < 0.01, f"Expected 4.0, got {v}"

    def test_three_pedestrians(self):
        """Three pedestrians; min separation is the closest pair."""
        positions = [(0.0, 0.0), (1.0, 0.0), (10.0, 0.0)]
        df = _make_group_df(positions)
        result = compute_separation(df)
        for v in result["min_separation_m"].dropna():
            assert abs(v - 1.0) < 0.01, f"Expected 1.0 (closest pair), got {v}"
