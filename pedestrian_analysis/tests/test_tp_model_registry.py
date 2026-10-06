"""Tests for the TP model registry: checkpoint validation and adapter creation."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.tp_adapters import DummyTPAdapter, SocialLSTMAdapter
from pipeline.tp_model_registry import ModelCheckpointInfo, TPModelRegistry


def test_builtin_models_are_registered() -> None:
    available = TPModelRegistry.get_available_models()
    for name in ("dummy", "social_lstm", "social_gan", "transformer", "social_stgcnn", "trajectron_pp"):
        assert name in available


def test_get_model_class_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        TPModelRegistry.get_model_class("not_a_model")


def test_get_checkpoint_info_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        TPModelRegistry.get_checkpoint_info("not_a_model")


def test_create_adapter_dummy_no_checkpoint_required() -> None:
    adapter = TPModelRegistry.create_adapter("dummy", pred_len=5)
    assert isinstance(adapter, DummyTPAdapter)


def test_create_adapter_social_lstm_optional_checkpoint() -> None:
    adapter = TPModelRegistry.create_adapter("social_lstm", hidden_dim=8, embedding_dim=8)
    assert isinstance(adapter, SocialLSTMAdapter)
    assert adapter.checkpoint_loaded is False


def test_create_adapter_missing_required_checkpoint_raises() -> None:
    with pytest.raises(ValueError, match="requires a checkpoint file"):
        TPModelRegistry.create_adapter("social_stgcnn")


def test_create_adapter_nonexistent_checkpoint_path_raises() -> None:
    with pytest.raises(FileNotFoundError):
        TPModelRegistry.create_adapter("social_lstm", checkpoint_path="does_not_exist.pt")


def test_create_adapter_not_implemented_model_raises_with_repo_info() -> None:
    with pytest.raises(NotImplementedError, match="Social-STGCNN"):
        TPModelRegistry.create_adapter("social_stgcnn", checkpoint_path=__file__)


def test_checkpoint_info_flags_match_required_semantics() -> None:
    assert TPModelRegistry.get_checkpoint_info("dummy").required is False
    assert TPModelRegistry.get_checkpoint_info("social_lstm").required is False
    assert TPModelRegistry.get_checkpoint_info("social_stgcnn").required is True
    assert TPModelRegistry.get_checkpoint_info("trajectron_pp").required is True


def test_register_model_custom_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    class _StubAdapter(DummyTPAdapter):
        name = "stub"

    TPModelRegistry.register_model(
        "stub",
        _StubAdapter,
        ModelCheckpointInfo(required=False, official_repo="", download_instructions=""),
    )
    try:
        adapter = TPModelRegistry.create_adapter("stub")
        assert isinstance(adapter, _StubAdapter)
    finally:
        del TPModelRegistry._models["stub"]
        del TPModelRegistry._checkpoint_info["stub"]
