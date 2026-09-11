"""Tests for the Phase 1 trajectory-prediction adapter architecture."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline import tp_adapters
from pipeline.tp_adapters import (
    BaseTPAdapter,
    DummyTPAdapter,
    create_tp_adapter,
    predict_trajectories_from_dataframe,
)


def test_dummy_adapter_constant_velocity_extrapolation() -> None:
    adapter = DummyTPAdapter(pred_len=3)
    # Single pedestrian moving with constant velocity (1, 0.5) per frame.
    observed = np.array([[[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]]])

    predictions = adapter.predict(observed, num_modes=2)

    assert predictions.shape == (1, 2, 3, 2)
    expected = np.array([[3.0, 1.5], [4.0, 2.0], [5.0, 2.5]])
    np.testing.assert_allclose(predictions[0, 0], expected)
    # Constant-velocity is deterministic: all modes must be identical.
    np.testing.assert_allclose(predictions[0, 0], predictions[0, 1])


def test_dummy_adapter_rejects_invalid_shape() -> None:
    adapter = DummyTPAdapter()
    with pytest.raises(ValueError):
        adapter.predict(np.zeros((2, 3)))


def test_create_tp_adapter_direct_name() -> None:
    adapter = create_tp_adapter("dummy", pred_len=5)

    assert isinstance(adapter, DummyTPAdapter)
    assert adapter.model_name == "dummy"
    assert adapter.pred_len == 5


def test_create_tp_adapter_unsupported_name_raises() -> None:
    with pytest.raises(ValueError):
        create_tp_adapter("not_a_real_model")


def test_create_tp_adapter_group_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FailingAdapter(BaseTPAdapter):
        name = "failing"

        def __init__(self, **kwargs):
            raise RuntimeError("missing optional dependency")

        def predict(self, observed_trajectories, num_modes=1, **kwargs):
            raise NotImplementedError

    monkeypatch.setattr(
        tp_adapters,
        "TP_ADAPTER_REGISTRY",
        {"failing": _FailingAdapter, "dummy": DummyTPAdapter},
    )
    monkeypatch.setattr(
        tp_adapters,
        "TP_MODEL_GROUPS",
        {"research_lightweight": ("failing", "dummy")},
    )

    adapter = create_tp_adapter("research_lightweight")

    assert isinstance(adapter, DummyTPAdapter)


def test_predict_trajectories_from_dataframe_shape_and_columns() -> None:
    df = pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 2],
            "frame": [0, 1, 2, 0, 1],
            "x": [0.0, 1.0, 2.0, 5.0, 5.0],
            "y": [0.0, 0.0, 0.0, 0.0, 1.0],
        }
    )

    predictions = predict_trajectories_from_dataframe(
        df, tp_type="dummy", pred_len=2, num_modes=1, fps=10.0
    )

    expected_columns = [
        "id",
        "mode",
        "frame",
        "frame_offset",
        "timestamp",
        "x_pred",
        "y_pred",
        "probability",
    ]
    assert list(predictions.columns) == expected_columns
    # Two pedestrians * 1 mode * 2 predicted frames each.
    assert len(predictions) == 4

    ped1 = predictions[predictions["id"] == 1].sort_values("frame_offset")
    np.testing.assert_allclose(ped1["x_pred"].to_numpy(), [3.0, 4.0])
    np.testing.assert_allclose(ped1["y_pred"].to_numpy(), [0.0, 0.0])
    assert ped1["frame"].tolist() == [3, 4]


def test_predict_trajectories_from_dataframe_skips_short_tracks() -> None:
    df = pd.DataFrame({"id": [1], "frame": [0], "x": [0.0], "y": [0.0]})

    predictions = predict_trajectories_from_dataframe(df, pred_len=2)

    assert predictions.empty


def test_predict_trajectories_from_dataframe_empty_input_returns_empty_df() -> None:
    df = pd.DataFrame(columns=["id", "frame", "x", "y"])

    predictions = predict_trajectories_from_dataframe(df)

    assert predictions.empty
