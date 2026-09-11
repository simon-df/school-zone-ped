"""Tests for the TP Analysis tab's CSV loading/normalization logic."""
from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ui.tabs_tp_analysis import load_trajectory_csv


def test_load_trajectory_csv_native_schema(tmp_path) -> None:
    csv_path = tmp_path / "native.csv"
    pd.DataFrame({"id": [1, 1], "frame": [0, 1], "x": [0.0, 1.0], "y": [0.0, 0.5]}).to_csv(csv_path, index=False)

    df = load_trajectory_csv(str(csv_path), fps=10.0)

    assert list(df.columns) == ["id", "frame", "timestamp", "x", "y"]
    assert df["timestamp"].tolist() == [0.0, 0.1]


def test_load_trajectory_csv_plan_schema_is_normalized(tmp_path) -> None:
    csv_path = tmp_path / "plan_schema.csv"
    pd.DataFrame(
        {"track_id": [2, 2], "frame": [0, 1], "timestamp": [0.0, 0.1], "x_m": [3.0, 3.5], "y_m": [1.0, 1.2]}
    ).to_csv(csv_path, index=False)

    df = load_trajectory_csv(str(csv_path))

    assert set(df.columns) == {"id", "frame", "timestamp", "x", "y"}
    assert df["id"].tolist() == [2, 2]
    assert df["x"].tolist() == [3.0, 3.5]


def test_load_trajectory_csv_missing_columns_raises(tmp_path) -> None:
    csv_path = tmp_path / "bad.csv"
    pd.DataFrame({"id": [1], "frame": [0]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        load_trajectory_csv(str(csv_path))


def test_load_trajectory_csv_sorts_by_frame_then_id(tmp_path) -> None:
    csv_path = tmp_path / "unsorted.csv"
    pd.DataFrame(
        {"id": [2, 1, 1, 2], "frame": [0, 1, 0, 1], "x": [0.0, 1.0, 2.0, 3.0], "y": [0.0, 0.0, 0.0, 0.0]}
    ).to_csv(csv_path, index=False)

    df = load_trajectory_csv(str(csv_path))

    assert df[["frame", "id"]].to_records(index=False).tolist() == [(0, 1), (0, 2), (1, 1), (1, 2)]
