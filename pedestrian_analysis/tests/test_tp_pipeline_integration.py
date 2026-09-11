"""Tests for TP-adapter integration into the tracking pipeline (Phase 2)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline import tracker
from pipeline.calibration import meter_to_pixel, pixel_to_meter


def test_meter_to_pixel_is_inverse_of_pixel_to_meter() -> None:
    H = np.array([[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 1.0]])

    x_m, y_m = pixel_to_meter(120.0, 340.0, H)
    px, py = meter_to_pixel(x_m, y_m, H)

    assert px == pytest.approx(120.0, abs=1e-6)
    assert py == pytest.approx(340.0, abs=1e-6)


def test_extract_trajectories_with_prediction_returns_both_dataframes(monkeypatch, tmp_path) -> None:
    observed = pd.DataFrame(
        {
            "id": [1, 1, 1],
            "frame": [0, 1, 2],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "px": [0.0, 10.0, 20.0],
            "py": [0.0, 0.0, 0.0],
        }
    )
    monkeypatch.setattr(tracker, "extract_trajectories_from_video", lambda video_path, H, **kw: observed)
    monkeypatch.setattr(tracker, "get_video_metadata", lambda path: {"fps": 10.0})

    df_observed, df_predictions = tracker.extract_trajectories_with_prediction(
        video_path="unused.mp4",
        H=np.eye(3),
        tp_model_type="dummy",
        tp_config={"pred_len": 2, "num_modes": 1},
    )

    assert df_observed is observed
    assert not df_predictions.empty
    assert set(df_predictions["id"]) == {1}
    assert df_predictions["frame"].max() == 4  # last observed frame (2) + pred_len (2)


def test_extract_trajectories_with_prediction_saves_csv(monkeypatch, tmp_path) -> None:
    observed = pd.DataFrame({"id": [1, 1], "frame": [0, 1], "x": [0.0, 1.0], "y": [0.0, 0.0]})
    monkeypatch.setattr(tracker, "extract_trajectories_from_video", lambda video_path, H, **kw: observed)
    monkeypatch.setattr(tracker, "get_video_metadata", lambda path: {"fps": 25.0})

    out_csv = tmp_path / "predictions.csv"
    _, df_predictions = tracker.extract_trajectories_with_prediction(
        video_path="unused.mp4",
        H=np.eye(3),
        tp_model_type="dummy",
        tp_config={"pred_len": 1},
        output_predictions_csv_path=out_csv,
    )

    assert out_csv.is_file()
    reloaded = pd.read_csv(out_csv)
    assert len(reloaded) == len(df_predictions)


def test_draw_prediction_overlay_draws_without_crashing() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    H = np.eye(3)
    predictions = pd.DataFrame(
        {
            "id": [1, 1],
            "mode": [0, 0],
            "frame": [1, 2],
            "frame_offset": [1, 2],
            "x_pred": [10.0, 20.0],
            "y_pred": [10.0, 20.0],
            "probability": [1.0, 1.0],
        }
    )

    annotated = tracker.draw_prediction_overlay(frame, predictions, {1: (0.0, 0.0)}, H)

    assert annotated.shape == frame.shape
    assert annotated.sum() > 0  # something was drawn


def test_draw_prediction_overlay_empty_predictions_returns_copy() -> None:
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    annotated = tracker.draw_prediction_overlay(frame, pd.DataFrame(), {}, np.eye(3))
    assert annotated.sum() == 0
    assert annotated is not frame


class _FakeWriter:
    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []
        self.released = False

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame)

    def release(self) -> None:
        self.released = True


def test_export_video_with_predictions_writes_all_frames(monkeypatch, tmp_path) -> None:
    frames = [(i, np.zeros((20, 20, 3), dtype=np.uint8)) for i in range(3)]
    fake_writer = _FakeWriter()

    monkeypatch.setattr(tracker, "get_video_metadata", lambda path: {"width": 20, "height": 20, "fps": 10.0})
    monkeypatch.setattr(tracker, "create_video_writer", lambda *a, **kw: fake_writer)
    monkeypatch.setattr(tracker, "iter_frames", lambda path, frame_skip=1: iter(frames))

    df_observed = pd.DataFrame({"id": [1], "frame": [0], "px": [5.0], "py": [5.0]})
    df_predictions = pd.DataFrame(
        {
            "id": [1],
            "mode": [0],
            "frame": [1],
            "frame_offset": [1],
            "x_pred": [1.0],
            "y_pred": [1.0],
            "probability": [1.0],
        }
    )

    result_path = tracker.export_video_with_predictions(
        video_path="unused.mp4",
        df_observed=df_observed,
        df_predictions=df_predictions,
        output_path=tmp_path / "out.mp4",
        H=np.eye(3),
    )

    assert len(fake_writer.frames) == 3
    assert fake_writer.released is True
    assert result_path == tmp_path / "out.mp4"
