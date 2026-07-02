"""Tests for pipeline/calibration.py"""

import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from pipeline.calibration import compute_homography, pixel_to_meter
from utils.validation import validate_trajectory_dataframe


# ---------------------------------------------------------------------------
# compute_homography
# ---------------------------------------------------------------------------

def _build_simple_correspondences():
    """
    Use a simple known projective mapping: a square in pixel space (100x100 at offset 50,50)
    mapped to a 4x4 metre square at the origin.
    """
    src = np.array([
        [50.0,  50.0],   # top-left
        [150.0, 50.0],   # top-right
        [150.0, 150.0],  # bottom-right
        [50.0,  150.0],  # bottom-left
    ], dtype=np.float64)

    dst = np.array([
        [0.0, 0.0],
        [4.0, 0.0],
        [4.0, 4.0],
        [0.0, 4.0],
    ], dtype=np.float64)
    return src, dst


class TestComputeHomography:
    def test_returns_3x3_matrix(self):
        src, dst = _build_simple_correspondences()
        H = compute_homography(src, dst)
        assert H.shape == (3, 3)

    def test_known_transform(self):
        """The centre of the square (100, 100) should map to (2, 2) m."""
        src, dst = _build_simple_correspondences()
        H = compute_homography(src, dst)
        x_m, y_m = pixel_to_meter(100.0, 100.0, H)
        assert abs(x_m - 2.0) < 0.05, f"Expected ~2.0 m, got {x_m}"
        assert abs(y_m - 2.0) < 0.05, f"Expected ~2.0 m, got {y_m}"

    def test_corner_transform(self):
        """Each corner pixel should map back to its metre coordinate."""
        src, dst = _build_simple_correspondences()
        H = compute_homography(src, dst)
        for (px, py), (mx, my) in zip(src, dst):
            rx, ry = pixel_to_meter(px, py, H)
            assert abs(rx - mx) < 0.05, f"X mismatch: {rx} vs {mx}"
            assert abs(ry - my) < 0.05, f"Y mismatch: {ry} vs {my}"

    def test_raises_on_wrong_shape(self):
        with pytest.raises(ValueError):
            compute_homography(np.zeros((3, 2)), np.zeros((4, 2)))

    def test_raises_on_wrong_columns(self):
        with pytest.raises(ValueError):
            compute_homography(np.zeros((4, 3)), np.zeros((4, 2)))


class TestValidateTrajectoryDataFrame:
    def test_requires_minimum_columns(self):
        df = pd.DataFrame({"id": [1], "frame": [0], "x": [0.0]})
        with pytest.raises(ValueError):
            validate_trajectory_dataframe(df)

    def test_accepts_minimal_columns(self):
        df = pd.DataFrame({"id": [1], "frame": [0], "x": [0.0], "y": [0.0]})
        validate_trajectory_dataframe(df)
