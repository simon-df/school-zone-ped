"""Trajectory datasets for TP training: ETH/UCY, own CSV exports, PIE stub.

All datasets share :class:`BaseTrajectoryDataset`: a subclass only has to
return its *recordings* (one DataFrame per file/video with columns ``id``,
``timestamp`` [s], ``x``, ``y`` in metric top-down coordinates) from
:meth:`BaseTrajectoryDataset.load_recordings`. The base class then

1. resamples every recording to the training ``fps``
   (:func:`training.tp.resample.resample_trajectories`),
2. cuts sliding windows of ``obs_len + pred_len`` consecutive frames
   (:func:`extract_windows`) -- every agent present in all frames of a
   window is part of it, which provides the scene grouping used by the
   optional social pooling, and
3. optionally augments windows on access (:func:`augment_window`).

Windows are grouped by recording so that train/val/test splits can be made
*by recording* (:meth:`BaseTrajectoryDataset.subset`); windows of the same
recording are strongly correlated and must never be split across sets.

The datasets are plain map-style containers (``__len__``/``__getitem__``)
and work with :class:`torch.utils.data.DataLoader` without importing torch.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from pipeline.trajectory_io import load_trajectory_csv
from training.tp.resample import resample_trajectories

logger = logging.getLogger(__name__)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ETHUCY_ROOT = PACKAGE_ROOT / "data" / "ethucy"
ETHUCY_SCENES: tuple[str, ...] = ("eth", "hotel", "univ", "zara1", "zara2")
ETHUCY_NATIVE_FPS = 2.5


@dataclass
class TrajectoryWindow:
    """One training/evaluation sample: all agents of a scene over one time window.

    Attributes:
        obs: Observed absolute positions, shape ``(num_agents, obs_len, 2)``.
        fut: Future absolute positions, shape ``(num_agents, pred_len, 2)``.
        agent_ids: Track ids, shape ``(num_agents,)``.
        recording: Name of the source recording (file).
        start_frame: First frame of the window (on the resampled grid).
    """

    obs: np.ndarray
    fut: np.ndarray
    agent_ids: np.ndarray
    recording: str = ""
    start_frame: int = 0

    @property
    def num_agents(self) -> int:
        return int(self.obs.shape[0])


@dataclass
class AugmentationConfig:
    """Random geometric augmentation applied per window (same transform for all agents).

    Rotation, flips and scale jitter keep relative geometry between agents
    intact (so social pooling still sees a consistent scene). Strongly
    recommended for the small fine-tuning set.
    """

    rotate: bool = True
    flip: bool = True
    scale_range: tuple[float, float] = (0.9, 1.1)


def extract_windows(
    df: pd.DataFrame,
    obs_len: int,
    pred_len: int,
    stride: int = 1,
    min_agents: int = 1,
    recording: str = "",
) -> list[TrajectoryWindow]:
    """Cut sliding windows of ``obs_len + pred_len`` consecutive frames from *df*.

    Args:
        df: Resampled trajectories with columns ``id``, ``frame`` (consecutive
            integers per track), ``x``, ``y``.
        obs_len: Number of observed positions per window.
        pred_len: Number of future positions per window.
        stride: Step (in frames) between window starts.
        min_agents: Windows with fewer fully-present agents are skipped.
        recording: Recording name stored on each window.

    Tracks shorter than ``obs_len + pred_len`` frames are skipped; an agent is
    included in a window only if it is present in every frame of the window.
    """
    if obs_len < 2 or pred_len < 1:
        raise ValueError(f"Need obs_len >= 2 and pred_len >= 1, got obs_len={obs_len}, pred_len={pred_len}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    seq_len = obs_len + pred_len
    if df.empty:
        return []

    counts = df.groupby("id")["frame"].transform("size")
    df = df[counts >= seq_len]
    if df.empty:
        return []

    first_frame = int(df["frame"].min())
    last_frame = int(df["frame"].max())
    frame_index = np.arange(first_frame, last_frame + 1)
    xs = df.pivot_table(index="frame", columns="id", values="x", aggfunc="first").reindex(frame_index)
    ys = df.pivot_table(index="frame", columns="id", values="y", aggfunc="first").reindex(frame_index)
    agent_ids = xs.columns.to_numpy()
    x_arr = xs.to_numpy(dtype=np.float64)
    y_arr = ys.to_numpy(dtype=np.float64)
    present = ~np.isnan(x_arr)

    windows: list[TrajectoryWindow] = []
    for start in range(0, len(frame_index) - seq_len + 1, stride):
        cols = np.nonzero(present[start : start + seq_len].all(axis=0))[0]
        if cols.size < max(1, min_agents):
            continue
        traj = np.stack([x_arr[start : start + seq_len, cols], y_arr[start : start + seq_len, cols]], axis=-1)
        traj = np.transpose(traj, (1, 0, 2))  # (agents, seq_len, 2)
        windows.append(
            TrajectoryWindow(
                obs=traj[:, :obs_len].copy(),
                fut=traj[:, obs_len:].copy(),
                agent_ids=agent_ids[cols].copy(),
                recording=recording,
                start_frame=int(frame_index[start]),
            )
        )
    return windows


def normalize_window_translation(window: TrajectoryWindow) -> TrajectoryWindow:
    """Translate a window so the centroid of the agents' last observed points is the origin.

    For a single-agent window this is exactly "relative to the last observed
    point". A shared offset (instead of one per agent) keeps relative
    positions between agents intact for social pooling. The models work on
    displacements, so this does not change predictions -- it only keeps
    coordinates small and centres rotations/flips on the scene.
    """
    origin = window.obs[:, -1, :].mean(axis=0)
    return replace(window, obs=window.obs - origin, fut=window.fut - origin)


def augment_window(window: TrajectoryWindow, rng: np.random.Generator, cfg: AugmentationConfig) -> TrajectoryWindow:
    """Apply a random rotation / flip / scale (about the origin) to all agents of *window*."""
    transform = np.eye(2)
    if cfg.rotate:
        theta = rng.uniform(0.0, 2.0 * np.pi)
        c, s = np.cos(theta), np.sin(theta)
        transform = np.array([[c, -s], [s, c]]) @ transform
    if cfg.flip:
        if rng.random() < 0.5:
            transform = np.diag([-1.0, 1.0]) @ transform
        if rng.random() < 0.5:
            transform = np.diag([1.0, -1.0]) @ transform
    lo, hi = cfg.scale_range
    if hi > lo:
        transform = rng.uniform(lo, hi) * transform
    return replace(window, obs=window.obs @ transform.T, fut=window.fut @ transform.T)


class BaseTrajectoryDataset(ABC):
    """Generic dataset-loader interface yielding :class:`TrajectoryWindow` samples.

    Subclasses implement :meth:`load_recordings`. To add a new dataset,
    return one DataFrame per recording with columns ``id``, ``timestamp``
    (seconds), ``x``, ``y`` in *metric, top-down* coordinates.

    Args:
        obs_len: Observed positions per window.
        pred_len: Future positions per window.
        fps: Training frame rate; recordings are resampled to it.
        stride: Window stride in frames. ``None`` -> ``round(fps / 2.5)``,
            i.e. the same temporal density as the standard 2.5 Hz ETH/UCY
            protocol with stride 1.
        min_agents: Minimum number of fully-present agents per window.
        augment: Apply random :class:`AugmentationConfig` transforms on access.
        augmentation: Augmentation settings (defaults when ``None``).
        normalize_translation: Translate windows to the scene origin (see
            :func:`normalize_window_translation`).
        seed: Seed for the augmentation RNG (see :meth:`reseed`).
        max_gap_s: Forwarded to :func:`resample_trajectories`.
    """

    name: str = "base"

    def __init__(
        self,
        obs_len: int = 20,
        pred_len: int = 30,
        fps: float = 10.0,
        stride: Optional[int] = None,
        min_agents: int = 1,
        augment: bool = False,
        augmentation: Optional[AugmentationConfig] = None,
        normalize_translation: bool = True,
        seed: int = 0,
        max_gap_s: Optional[float] = None,
    ) -> None:
        self.obs_len = int(obs_len)
        self.pred_len = int(pred_len)
        self.fps = float(fps)
        self.stride = int(stride) if stride else max(1, int(round(self.fps / ETHUCY_NATIVE_FPS)))
        self.min_agents = int(min_agents)
        self.augment = bool(augment)
        self.augmentation = augmentation or AugmentationConfig()
        self.normalize_translation = bool(normalize_translation)
        self.max_gap_s = max_gap_s
        self._rng = np.random.default_rng(seed)
        self._windows_by_recording: Optional[dict[str, list[TrajectoryWindow]]] = None
        self._flat: Optional[list[TrajectoryWindow]] = None

    @abstractmethod
    def load_recordings(self) -> dict[str, pd.DataFrame]:
        """Return ``{recording_name: DataFrame(id, timestamp, x, y)}`` in metric top-down coordinates."""

    # ------------------------------------------------------------------
    def _build(self) -> dict[str, list[TrajectoryWindow]]:
        windows: dict[str, list[TrajectoryWindow]] = {}
        for name, df in self.load_recordings().items():
            resampled = resample_trajectories(df, self.fps, max_gap_s=self.max_gap_s)
            windows[name] = extract_windows(
                resampled,
                self.obs_len,
                self.pred_len,
                stride=self.stride,
                min_agents=self.min_agents,
                recording=name,
            )
            if not windows[name]:
                logger.warning(
                    "%s: recording '%s' yields no windows of %d frames at %.4g fps (tracks too short?).",
                    self.name,
                    name,
                    self.obs_len + self.pred_len,
                    self.fps,
                )
        return windows

    @property
    def windows_by_recording(self) -> dict[str, list[TrajectoryWindow]]:
        if self._windows_by_recording is None:
            self._windows_by_recording = self._build()
        return self._windows_by_recording

    @property
    def recordings(self) -> list[str]:
        return list(self.windows_by_recording.keys())

    @property
    def windows(self) -> list[TrajectoryWindow]:
        """All (un-augmented, un-normalized) windows, in recording order."""
        if self._flat is None:
            self._flat = [w for ws in self.windows_by_recording.values() for w in ws]
        return self._flat

    @property
    def num_agents(self) -> int:
        """Total number of agent-windows (training samples for the per-agent loss)."""
        return sum(w.num_agents for w in self.windows)

    def reseed(self, seed: int) -> None:
        """Reset the augmentation RNG (called per epoch for reproducible/resumable training)."""
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> TrajectoryWindow:
        window = self.windows[idx]
        if self.normalize_translation:
            window = normalize_window_translation(window)
        if self.augment:
            window = augment_window(window, self._rng, self.augmentation)
        return window

    def subset(self, recordings: Iterable[str], augment: Optional[bool] = None) -> "WindowSubset":
        """Return a dataset view restricted to *recordings* (split by recording, never by window)."""
        names = list(recordings)
        unknown = set(names) - set(self.windows_by_recording)
        if unknown:
            raise KeyError(f"Unknown recordings: {sorted(unknown)}")
        return WindowSubset(
            {name: self.windows_by_recording[name] for name in names},
            obs_len=self.obs_len,
            pred_len=self.pred_len,
            fps=self.fps,
            stride=self.stride,
            min_agents=self.min_agents,
            augment=self.augment if augment is None else augment,
            augmentation=self.augmentation,
            normalize_translation=self.normalize_translation,
        )


class WindowSubset(BaseTrajectoryDataset):
    """Dataset over precomputed windows (result of :meth:`BaseTrajectoryDataset.subset`)."""

    name = "subset"

    def __init__(self, windows_by_recording: dict[str, list[TrajectoryWindow]], **kwargs) -> None:
        super().__init__(**kwargs)
        self._windows_by_recording = dict(windows_by_recording)

    def load_recordings(self) -> dict[str, pd.DataFrame]:  # pragma: no cover - windows are preset
        raise RuntimeError("WindowSubset holds precomputed windows only.")


# ----------------------------------------------------------------------
# ETH/UCY
# ----------------------------------------------------------------------


def load_ethucy_file(path: str | Path, native_fps: float = ETHUCY_NATIVE_FPS) -> pd.DataFrame:
    """Parse one SGAN-format ETH/UCY file (``frame ped_id x y``, tab/space separated).

    Frame numbers in these files advance by a constant annotation step
    (10 for all ETH/UCY files); timestamps are derived as
    ``frame / step / native_fps`` seconds.
    """
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["frame", "id", "x", "y"], engine="python")
    df = df.dropna()
    if df.empty:
        return pd.DataFrame(columns=["id", "frame", "timestamp", "x", "y"])
    df["id"] = df["id"].astype(float).astype(int)
    df["frame"] = df["frame"].astype(float)
    steps = df.sort_values("frame").groupby("id")["frame"].diff().dropna()
    steps = steps[steps > 0]
    step = float(steps.mode().iloc[0]) if not steps.empty else 1.0
    df["timestamp"] = df["frame"] / step / native_fps
    df["frame"] = (df["frame"] / step).round().astype(int)
    return df[["id", "frame", "timestamp", "x", "y"]].reset_index(drop=True)


class ETHUCYDataset(BaseTrajectoryDataset):
    """ETH/UCY benchmark (eth, hotel, univ, zara1, zara2) with leave-one-scene-out splits.

    Expects the standard SGAN / Social-STGCNN layout produced by
    :mod:`training.tp.download_ethucy`::

        <root>/<test_scene>/{train,val,test}/*.txt

    where ``<root>/<scene>/train`` and ``val`` hold the *other* four scenes
    and ``test`` holds ``<scene>`` itself. Native rate is 2.5 Hz.
    """

    name = "ethucy"

    def __init__(
        self,
        root: str | Path = DEFAULT_ETHUCY_ROOT,
        test_scene: str = "eth",
        split: str = "train",
        **kwargs,
    ) -> None:
        if test_scene not in ETHUCY_SCENES:
            raise ValueError(f"Unknown ETH/UCY scene '{test_scene}'. Choose from {ETHUCY_SCENES}.")
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val' or 'test', got '{split}'")
        super().__init__(**kwargs)
        self.root = Path(root)
        self.test_scene = test_scene
        self.split = split

    def load_recordings(self) -> dict[str, pd.DataFrame]:
        split_dir = self.root / self.test_scene / self.split
        files = sorted(split_dir.glob("*.txt")) if split_dir.is_dir() else []
        if not files:
            raise FileNotFoundError(
                f"No ETH/UCY files found in '{split_dir}'. Download them first with:\n"
                f"  python -m training.tp.download_ethucy --output-dir {self.root}"
            )
        return {f.stem: load_ethucy_file(f) for f in files}


# ----------------------------------------------------------------------
# Own CSV exports
# ----------------------------------------------------------------------


def find_csv_files(paths: Sequence[str | Path]) -> list[Path]:
    """Expand files/directories into a sorted, de-duplicated list of CSV files."""
    found: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            found.extend(sorted(p.glob("*.csv")))
        elif p.is_file():
            found.append(p)
        else:
            raise FileNotFoundError(f"Trajectory CSV path not found: '{p}'")
    unique = list(dict.fromkeys(f.resolve() for f in found))
    if not unique:
        raise FileNotFoundError(f"No CSV files found in {list(map(str, paths))}")
    return unique


class OwnCSVDataset(BaseTrajectoryDataset):
    """The app's exported drone trajectory CSVs (one file = one recording).

    Columns ``id, frame, timestamp, x, y`` or ``track_id, frame, timestamp,
    x_m, y_m`` (normalized by :func:`pipeline.trajectory_io.load_trajectory_csv`).

    Args:
        paths: CSV files and/or directories containing CSV files.
        source_fps: Frame rate of the CSV ``frame`` column; used to derive
            timestamps when a CSV has none or when ``use_timestamps=False``.
        use_timestamps: Use the CSV ``timestamp`` column (seconds) for
            resampling; otherwise derive time as ``frame / source_fps``.
    """

    name = "own_csv"

    def __init__(
        self,
        paths: Sequence[str | Path],
        source_fps: float = 10.0,
        use_timestamps: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.paths = find_csv_files(paths)
        self.source_fps = float(source_fps)
        self.use_timestamps = bool(use_timestamps)

    def load_recordings(self) -> dict[str, pd.DataFrame]:
        recordings: dict[str, pd.DataFrame] = {}
        for path in self.paths:
            df = load_trajectory_csv(path, fps=self.source_fps)
            if not self.use_timestamps:
                df = df.assign(timestamp=df["frame"] / self.source_fps)
            name = path.stem
            if name in recordings:
                name = str(path)
            recordings[name] = df
        return recordings


# ----------------------------------------------------------------------
# PIE (not implemented)
# ----------------------------------------------------------------------


class PIEDataset(BaseTrajectoryDataset):
    """Placeholder for the PIE dataset (Pedestrian Intention Estimation).

    Not implemented: PIE (https://github.com/aras62/PIE) is recorded from an
    ego-vehicle dash camera and annotates pedestrians as *image-space*
    bounding boxes in a moving camera frame. The TP models here are trained
    on *top-down metric* trajectories (ETH/UCY, the app's drone exports).
    Using PIE would require either a different (image-space, ego-motion
    compensated) model formulation or a reliable image-to-ground projection
    with vehicle odometry. A future implementation only has to provide
    :meth:`load_recordings` returning metric top-down ``id/timestamp/x/y``
    DataFrames.
    """

    name = "pie"

    _MESSAGE = (
        "PIEDataset is not implemented. PIE is filmed from an ego-vehicle camera and provides "
        "image-space bounding boxes (pixels, moving camera), whereas this pipeline trains on "
        "top-down metric trajectories (metres, static frame) like ETH/UCY and the drone CSVs. "
        "Integrating PIE requires projecting boxes to the ground plane with ego-motion "
        "compensation (or an image-space model); see training/tp/datasets.py:PIEDataset."
    )

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(self._MESSAGE)

    def load_recordings(self) -> dict[str, pd.DataFrame]:  # pragma: no cover - never constructed
        raise NotImplementedError(self._MESSAGE)


DATASET_REGISTRY: dict[str, type[BaseTrajectoryDataset]] = {
    "ethucy": ETHUCYDataset,
    "own_csv": OwnCSVDataset,
    "pie": PIEDataset,
}


def split_recordings(recordings: Sequence[str], val_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    """Deterministically split recording names into ``(train, val)`` (by recording, never by window)."""
    names = sorted(recordings)
    if len(names) < 2 or val_fraction <= 0:
        return names, []
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(len(names)))
    n_val = min(len(names) - 1, max(1, int(round(val_fraction * len(names)))))
    val = sorted(names[i] for i in order[:n_val])
    train = sorted(names[i] for i in order[n_val:])
    return train, val


__all__ = [
    "AugmentationConfig",
    "BaseTrajectoryDataset",
    "DATASET_REGISTRY",
    "DEFAULT_ETHUCY_ROOT",
    "ETHUCYDataset",
    "ETHUCY_SCENES",
    "OwnCSVDataset",
    "PIEDataset",
    "TrajectoryWindow",
    "WindowSubset",
    "augment_window",
    "extract_windows",
    "find_csv_files",
    "load_ethucy_file",
    "normalize_window_translation",
    "split_recordings",
]
