"""Tests for the Phase 2 torch-backed trajectory-prediction adapters."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pipeline.tp_adapters import (  # noqa: E402
    SocialGANAdapter,
    SocialLSTMAdapter,
    TransformerTPAdapter,
    create_tp_adapter,
)

_OBSERVED = np.array(
    [
        [[0.0, 0.0], [1.0, 0.2], [2.0, 0.4], [3.0, 0.6]],
        [[5.0, 5.0], [5.2, 5.1], [5.4, 5.2], [5.6, 5.3]],
    ]
)


@pytest.mark.parametrize(
    "adapter_cls,kwargs",
    [
        (SocialLSTMAdapter, {"hidden_dim": 8, "embedding_dim": 8, "max_modes": 4}),
        (SocialGANAdapter, {"hidden_dim": 8, "embedding_dim": 8, "noise_dim": 4}),
        (TransformerTPAdapter, {"d_model": 16, "nhead": 2, "num_layers": 1, "max_modes": 4}),
    ],
)
def test_torch_adapter_predict_shape(adapter_cls, kwargs) -> None:
    adapter = adapter_cls(pred_len=5, seed=0, **kwargs)

    predictions = adapter.predict(_OBSERVED, num_modes=3, pred_len=5)

    assert predictions.shape == (2, 3, 5, 2)
    assert np.isfinite(predictions).all()
    assert adapter.checkpoint_loaded is False


def test_social_lstm_deterministic_across_calls() -> None:
    adapter = SocialLSTMAdapter(pred_len=4, seed=1, hidden_dim=8, embedding_dim=8)

    first = adapter.predict(_OBSERVED, num_modes=2)
    second = adapter.predict(_OBSERVED, num_modes=2)

    # No dropout/sampling in Social-LSTM's decode path: repeat calls must match.
    np.testing.assert_allclose(first, second)


def test_transformer_adapter_clips_modes_and_pred_len_to_max() -> None:
    adapter = TransformerTPAdapter(
        pred_len=100, max_pred_len=10, max_modes=2, d_model=16, nhead=2, num_layers=1
    )

    predictions = adapter.predict(_OBSERVED, num_modes=5)

    assert predictions.shape == (2, 2, 10, 2)


def test_torch_adapter_rejects_invalid_shape() -> None:
    adapter = SocialLSTMAdapter(hidden_dim=8, embedding_dim=8)
    with pytest.raises(ValueError):
        adapter.predict(np.zeros((2, 3)))


def test_torch_adapter_missing_checkpoint_raises() -> None:
    with pytest.raises(FileNotFoundError):
        SocialLSTMAdapter(checkpoint="does_not_exist.pt", hidden_dim=8, embedding_dim=8)


def test_torch_adapter_checkpoint_round_trip(tmp_path) -> None:
    trained = SocialLSTMAdapter(hidden_dim=8, embedding_dim=8, seed=7)
    checkpoint_path = tmp_path / "social_lstm.pt"
    torch.save(trained.net.state_dict(), checkpoint_path)

    loaded = SocialLSTMAdapter(hidden_dim=8, embedding_dim=8, seed=123, checkpoint=str(checkpoint_path))

    assert loaded.checkpoint_loaded is True
    baseline = trained.predict(_OBSERVED, num_modes=2)
    reloaded = loaded.predict(_OBSERVED, num_modes=2)
    # Different seed but same checkpoint weights must produce identical output.
    np.testing.assert_allclose(baseline, reloaded)


def test_create_tp_adapter_social_lstm_by_name() -> None:
    adapter = create_tp_adapter("social_lstm", hidden_dim=8, embedding_dim=8)
    assert isinstance(adapter, SocialLSTMAdapter)


def test_create_tp_adapter_research_classical_group() -> None:
    adapter = create_tp_adapter("research_classical", hidden_dim=8, embedding_dim=8)
    assert isinstance(adapter, (SocialLSTMAdapter, SocialGANAdapter))
