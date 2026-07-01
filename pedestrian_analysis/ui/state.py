"""Shared application state container.

Holds objects that are shared across all UI tabs:
* Loaded homography matrix
* Loaded trajectory DataFrame
* Currently selected file paths
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class AppState:
    """Mutable, shared state object passed to every tab.

    Attributes:
        homography: Currently loaded 3 × 3 homography matrix, or ``None``.
        calibration_path: Path to the active ``.npy`` calibration file.
        trajectories: Currently loaded trajectory DataFrame, or ``None``.
        trajectory_path: Path to the active trajectory CSV file.
        video_path: Path to the currently selected source video.
        last_preview_frame: Last annotated frame displayed during extraction.
    """

    homography: Optional[np.ndarray] = None
    calibration_path: Optional[Path] = None
    trajectories: Optional[pd.DataFrame] = None
    trajectory_path: Optional[Path] = None
    video_path: Optional[Path] = None
    last_preview_frame: Optional[object] = None  # np.ndarray when set
