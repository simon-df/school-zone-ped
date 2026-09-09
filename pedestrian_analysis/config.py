"""Central configuration for the Pedestrian Crossing Trajectory Analysis project."""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
VIDEOS_DIR: Path = DATA_DIR / "videos"
TRAJECTORIES_DIR: Path = DATA_DIR / "trajectories"
CALIBRATION_DIR: Path = DATA_DIR / "calibration"
PREVIEWS_DIR: Path = DATA_DIR / "previews"
OUTPUTS_DIR: Path = BASE_DIR / "outputs"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
REPORTS_DIR: Path = OUTPUTS_DIR / "reports"
OUTPUT_VIDEOS_DIR: Path = OUTPUTS_DIR / "videos"

# ---------------------------------------------------------------------------
# Movement / analysis thresholds
# ---------------------------------------------------------------------------
SPEED_THRESHOLD_MS: float = 0.3        # m/s – below this → "waiting"
PROXIMITY_THRESHOLD_M: float = 1.5     # m   – max distance for group membership
MIN_GROUP_FRAMES: int = 10             # frames – minimum group persistence
TEMPORAL_SMOOTH_WINDOW: int = 10       # frames – smoothing window for labels
WAITING_MIN_FRAMES: int = 5            # frames – minimum to declare waiting state
LEADER_FOLLOWER_MAX_DELAY_S: float = 5.0  # s  – max delay for leader-follower

# ---------------------------------------------------------------------------
# Plot settings
# ---------------------------------------------------------------------------
PLOT_WIDTH_PX: int = 1200
PLOT_HEIGHT_PX: int = 800
PLOT_FONT_SIZE: int = 14
PLOT_FONT_FAMILY: str = "Arial"
PLOT_TEMPLATE: str = "plotly_white"
DPI: int = 300

# ---------------------------------------------------------------------------
# Tracker / model defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL_NAME: str = "yolov8n.pt"
DEFAULT_TRACKER_TYPE: str = "bot_sort"
TRACKER_CHOICES: tuple[str, ...] = (
    "bot_sort",
    "byte_track",
    "ocsort",
    "deep_ocsort",
    "pb_evformer",
    "research_top_down_general",
    "research_top_down_occlusion",
    "research_top_down_small_targets",
)
DEFAULT_FRAME_SKIP: int = 1
DEFAULT_CONFIDENCE: float = 0.4
DEFAULT_FPS: float = 25.0
DEFAULT_PREVIEW_EVERY_N: int = 5
DEFAULT_STREET_START_M: float = 2.0
DEFAULT_STREET_END_M: float = 6.0
DEFAULT_GRID_SPACING_M: float = 1.0

# ---------------------------------------------------------------------------
# Optional experimental: PBEVFormer integration
# ---------------------------------------------------------------------------
PBEVFORMER_ENABLED: bool = False
PBEVFORMER_CONFIG_PATH: str = ""
PBEVFORMER_WEIGHTS_PATH: str = ""
