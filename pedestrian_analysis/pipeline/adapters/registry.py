"""Tracker adapter registry and group-based selection."""
from __future__ import annotations

import logging
from typing import Any

from pipeline.adapters.base import BaseTrackerAdapter
from pipeline.adapters.bot_sort_adapter import UltralyticsBoTSORTAdapter
from pipeline.adapters.byte_track_adapter import UltralyticsByteTrackAdapter
from pipeline.adapters.experimental_adapters import DeepOCSORTAdapter, PBEVFormerTrackerAdapter
from pipeline.adapters.ocsort_adapter import OCSORTAdapter

logger = logging.getLogger(__name__)

TRACKER_REGISTRY: dict[str, type[BaseTrackerAdapter]] = {
    "bot_sort": UltralyticsBoTSORTAdapter,
    "byte_track": UltralyticsByteTrackAdapter,
    "ocsort": OCSORTAdapter,
    "deep_ocsort": DeepOCSORTAdapter,
    "pb_evformer": PBEVFormerTrackerAdapter,
}

TRACKER_DEPENDENCY_HINTS: dict[str, str] = {
    "bot_sort": "supervision>=0.18",
    "byte_track": "supervision>=0.18",
    "ocsort": "boxmot (+ supervision for Detections conversion)",
    "deep_ocsort": "not implemented in this repository",
    "pb_evformer": "custom PBEVFormer implementation + weights",
}

TRACKER_GROUPS: dict[str, tuple[str, str, str]] = {
    "research_top_down_general": ("bot_sort", "ocsort", "byte_track"),
    "research_top_down_occlusion": ("bot_sort", "ocsort", "byte_track"),
    "research_top_down_small_targets": ("bot_sort", "byte_track", "ocsort"),
}


def _normalize_tracker_key(tracker_type: str) -> str:
    return str(tracker_type).strip().lower().replace("-", "_")


def _build_unsupported_type_error(tracker_type: str) -> ValueError:
    supported_trackers = ", ".join(sorted(TRACKER_REGISTRY))
    supported_groups = ", ".join(sorted(TRACKER_GROUPS))
    return ValueError(
        f"Unsupported tracker type '{tracker_type}'. "
        f"Supported trackers: {supported_trackers}. "
        f"Supported groups: {supported_groups}."
    )


def _create_single_tracker(tracker_key: str, **kwargs: Any) -> BaseTrackerAdapter:
    adapter_cls = TRACKER_REGISTRY[tracker_key]
    return adapter_cls(**kwargs)


def create_tracker_adapter(tracker_type: str, **kwargs: Any) -> BaseTrackerAdapter:
    """Create a tracker adapter from tracker name or research group name."""
    if not tracker_type or not str(tracker_type).strip():
        raise ValueError("tracker_type must be a non-empty tracker or group name")

    normalized = _normalize_tracker_key(tracker_type)
    if normalized in TRACKER_REGISTRY:
        return _create_single_tracker(normalized, **kwargs)

    if normalized not in TRACKER_GROUPS:
        raise _build_unsupported_type_error(tracker_type)

    failures: list[tuple[str, str]] = []
    for candidate in TRACKER_GROUPS[normalized]:
        try:
            adapter = _create_single_tracker(candidate, **kwargs)
            logger.info("Selected tracker '%s' from group '%s'.", candidate, normalized)
            return adapter
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            failures.append((candidate, reason))
            logger.warning(
                "Failed to initialize tracker '%s' for group '%s': %s. Falling back.",
                candidate,
                normalized,
                reason,
            )

    required_pkgs = ", ".join(
        f"{name} -> {TRACKER_DEPENDENCY_HINTS.get(name, 'see tracker docs')}"
        for name in TRACKER_GROUPS[normalized]
    )
    tried = "; ".join(f"{name}: {reason}" for name, reason in failures)
    raise RuntimeError(
        f"No tracker from group '{normalized}' could be initialized. "
        f"Attempted in order: {TRACKER_GROUPS[normalized]}. "
        f"Failure reasons: {tried}. "
        f"Required packages/setup: {required_pkgs}."
    )
