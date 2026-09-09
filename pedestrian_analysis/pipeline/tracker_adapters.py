"""Backward-compatible tracker adapter exports.

The concrete implementations live in :mod:`pipeline.adapters`.
"""
from pipeline.adapters import (
    BaseTrackerAdapter,
    DeepOCSORTAdapter,
    OCSORTAdapter,
    PBEVFormerTrackerAdapter,
    TRACKER_GROUPS,
    TRACKER_REGISTRY,
    UltralyticsBoTSORTAdapter,
    UltralyticsByteTrackAdapter,
    create_tracker_adapter,
)

__all__ = [
    "BaseTrackerAdapter",
    "UltralyticsBoTSORTAdapter",
    "UltralyticsByteTrackAdapter",
    "OCSORTAdapter",
    "DeepOCSORTAdapter",
    "PBEVFormerTrackerAdapter",
    "TRACKER_REGISTRY",
    "TRACKER_GROUPS",
    "create_tracker_adapter",
]
