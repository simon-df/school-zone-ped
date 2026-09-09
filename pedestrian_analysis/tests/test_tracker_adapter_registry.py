"""Tests for tracker adapter registry and group fallback behavior."""
from __future__ import annotations

import os
import sys
import logging

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.adapters.base import BaseTrackerAdapter
from pipeline.adapters import registry


class _DummyAdapter(BaseTrackerAdapter):
    name = "dummy"

    def update_with_detections(self, detections):
        return detections


def test_create_tracker_adapter_direct_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "TRACKER_REGISTRY", {"dummy": _DummyAdapter})
    monkeypatch.setattr(registry, "TRACKER_GROUPS", {})

    adapter = registry.create_tracker_adapter("dummy")

    assert isinstance(adapter, _DummyAdapter)


def test_create_tracker_adapter_group_fallback(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    class _FailingAdapter(BaseTrackerAdapter):
        name = "failing"

        def __init__(self, **kwargs):
            raise RuntimeError("missing optional dependency")

        def update_with_detections(self, detections):
            return detections

    class _WorkingAdapter(_DummyAdapter):
        name = "working"

    monkeypatch.setattr(
        registry,
        "TRACKER_REGISTRY",
        {"failing": _FailingAdapter, "working": _WorkingAdapter, "third": _DummyAdapter},
    )
    monkeypatch.setattr(
        registry,
        "TRACKER_GROUPS",
        {"research_top_down_occlusion": ("failing", "working", "third")},
    )
    caplog.set_level(logging.WARNING)

    adapter = registry.create_tracker_adapter("research_top_down_occlusion")

    assert isinstance(adapter, _WorkingAdapter)
    assert "Failed to initialize tracker 'failing'" in caplog.text


def test_create_tracker_adapter_group_all_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FailingAdapter(BaseTrackerAdapter):
        name = "failing"

        def __init__(self, **kwargs):
            raise RuntimeError("dependency missing")

        def update_with_detections(self, detections):
            return detections

    monkeypatch.setattr(
        registry,
        "TRACKER_REGISTRY",
        {"a": _FailingAdapter, "b": _FailingAdapter, "c": _FailingAdapter},
    )
    monkeypatch.setattr(registry, "TRACKER_GROUPS", {"research_top_down_general": ("a", "b", "c")})
    monkeypatch.setattr(
        registry,
        "TRACKER_DEPENDENCY_HINTS",
        {"a": "pkg-a", "b": "pkg-b", "c": "pkg-c"},
    )

    with pytest.raises(RuntimeError) as exc:
        registry.create_tracker_adapter("research_top_down_general")

    message = str(exc.value)
    assert "No tracker from group 'research_top_down_general'" in message
    assert "a -> pkg-a" in message
    assert "b -> pkg-b" in message
    assert "c -> pkg-c" in message
