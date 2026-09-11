"""Trajectory prediction (TP) adapter interface and implementations.

Mirrors the tracker-adapter pattern in :mod:`pipeline.adapters`: a small
abstract interface plus a registry of concrete adapters, selectable by name.

Phase 1 (MVP) shipped :class:`DummyTPAdapter` (constant-velocity
extrapolation) to validate the end-to-end architecture (config, adapter
selection, CSV output format).

Phase 2 adds real PyTorch-backed adapters (:class:`SocialLSTMAdapter`,
:class:`SocialGANAdapter`, :class:`TransformerTPAdapter`) with model
loading and multimodal (K-mode) inference. These are compact from-scratch
reimplementations of the architectural ideas behind the named papers, not
the original authors' code -- see :mod:`pipeline.tp_models`. Without a
checkpoint they run with randomly initialized weights (architecture
validation only); pass a real checkpoint path once one is trained/obtained.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

import logging

logger = logging.getLogger(__name__)


class BaseTPAdapter(ABC):
    """Base adapter for trajectory prediction models."""

    name: str = "base"

    @abstractmethod
    def predict(
        self,
        observed_trajectories: np.ndarray,
        num_modes: int = 1,
        **kwargs: Any,
    ) -> np.ndarray:
        """Predict future trajectories.

        Args:
            observed_trajectories: shape ``(num_pedestrians, obs_len, 2)``.
            num_modes: number of predicted modes per pedestrian.

        Returns:
            predictions: shape ``(num_pedestrians, num_modes, pred_len, 2)``.
        """
        raise NotImplementedError

    def load_model(self, checkpoint_path: Optional[str] = None) -> None:
        """Load model weights. No-op by default (e.g. for heuristic adapters)."""

    @property
    def model_name(self) -> str:
        return self.name


class DummyTPAdapter(BaseTPAdapter):
    """Constant-velocity extrapolation adapter, used to validate the architecture."""

    name = "dummy"

    def __init__(self, pred_len: int = 10, **_: Any) -> None:
        self.pred_len = int(pred_len)

    def load_model(self, checkpoint_path: Optional[str] = None) -> None:
        # No trainable weights; present for interface compatibility.
        return None

    def predict(
        self,
        observed_trajectories: np.ndarray,
        num_modes: int = 1,
        **kwargs: Any,
    ) -> np.ndarray:
        observed = np.asarray(observed_trajectories, dtype=np.float64)
        if observed.ndim != 3 or observed.shape[-1] != 2:
            raise ValueError(
                "observed_trajectories must have shape (num_pedestrians, obs_len, 2), "
                f"got {observed.shape}"
            )

        pred_len = int(kwargs.get("pred_len", self.pred_len))
        num_pedestrians, obs_len, _ = observed.shape

        if obs_len >= 2:
            velocity = observed[:, -1, :] - observed[:, -2, :]
        else:
            velocity = np.zeros((num_pedestrians, 2), dtype=np.float64)

        last_point = observed[:, -1, :]
        steps = np.arange(1, pred_len + 1, dtype=np.float64)
        # (num_pedestrians, pred_len, 2)
        offsets = steps[None, :, None] * velocity[:, None, :]
        single_mode = last_point[:, None, :] + offsets

        # Constant-velocity is deterministic: replicate across requested modes.
        predictions = np.repeat(single_mode[:, None, :, :], num_modes, axis=1)
        return predictions


class _TorchTPAdapter(BaseTPAdapter):
    """Shared plumbing for torch-backed adapters: seeding, eval mode, checkpoints."""

    def __init__(self, pred_len: int, seed: int, checkpoint: Optional[str]) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - runtime dependency guard
            raise RuntimeError(
                f"PyTorch is required for the '{self.name}' TP adapter. Install with: pip install torch"
            ) from exc

        self._torch = torch
        self.pred_len = int(pred_len)
        torch.manual_seed(int(seed))
        self.net = self._build_net()
        self.net.eval()
        self.checkpoint_loaded = False
        self.load_model(checkpoint)

    def _build_net(self) -> Any:
        raise NotImplementedError

    def load_model(self, checkpoint_path: Optional[str] = None) -> None:
        if not checkpoint_path:
            logger.warning(
                "%s: no checkpoint provided; using randomly initialized weights "
                "(architecture validation only, not a trained model).",
                self.name,
            )
            return

        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"{self.name} checkpoint not found: '{checkpoint_path}'")

        state_dict = self._torch.load(str(path), map_location="cpu")
        self.net.load_state_dict(state_dict)
        self.net.eval()
        self.checkpoint_loaded = True
        logger.info("%s: loaded checkpoint from '%s'.", self.name, checkpoint_path)

    def _validate_observed(self, observed_trajectories: np.ndarray) -> np.ndarray:
        observed = np.asarray(observed_trajectories, dtype=np.float32)
        if observed.ndim != 3 or observed.shape[-1] != 2:
            raise ValueError(
                "observed_trajectories must have shape (num_pedestrians, obs_len, 2), "
                f"got {observed.shape}"
            )
        return observed

    def _diffs_from_observed(self, observed: np.ndarray) -> np.ndarray:
        num_pedestrians, obs_len, _ = observed.shape
        if obs_len >= 2:
            return np.diff(observed, axis=1).astype(np.float32)
        return np.zeros((num_pedestrians, 1, 2), dtype=np.float32)

    def _reconstruct_absolute(self, last_point: np.ndarray, pred_diffs: np.ndarray) -> np.ndarray:
        cumulative = np.cumsum(pred_diffs, axis=2)
        return (cumulative + last_point[:, None, None, :]).astype(np.float64)


class SocialLSTMAdapter(_TorchTPAdapter):
    """Social-LSTM-inspired encoder/decoder with per-mode embedding conditioning."""

    name = "social_lstm"

    def __init__(
        self,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        max_modes: int = 20,
        pred_len: int = 30,
        checkpoint: Optional[str] = None,
        seed: int = 42,
        **_: Any,
    ) -> None:
        self._embedding_dim = embedding_dim
        self._hidden_dim = hidden_dim
        self._max_modes = max_modes
        super().__init__(pred_len=pred_len, seed=seed, checkpoint=checkpoint)

    def _build_net(self) -> Any:
        from pipeline.tp_models import SocialLSTMNet

        return SocialLSTMNet(
            embedding_dim=self._embedding_dim,
            hidden_dim=self._hidden_dim,
            max_modes=self._max_modes,
        )

    def predict(self, observed_trajectories: np.ndarray, num_modes: int = 5, **kwargs: Any) -> np.ndarray:
        observed = self._validate_observed(observed_trajectories)
        pred_len = int(kwargs.get("pred_len", self.pred_len))
        num_modes = min(int(num_modes), self._max_modes)

        diffs = self._diffs_from_observed(observed)
        last_point = observed[:, -1, :]

        with self._torch.no_grad():
            diffs_t = self._torch.from_numpy(diffs)
            h, c = self.net.encode(diffs_t)
            pred_diffs = self.net.decode(diffs_t[:, -1, :], h, c, num_modes=num_modes, pred_len=pred_len)

        return self._reconstruct_absolute(last_point, pred_diffs.numpy())


class SocialGANAdapter(_TorchTPAdapter):
    """Social-GAN-inspired encoder + noise-conditioned generator decoder."""

    name = "social_gan"

    def __init__(
        self,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        noise_dim: int = 8,
        pred_len: int = 30,
        checkpoint: Optional[str] = None,
        seed: int = 42,
        **_: Any,
    ) -> None:
        self._embedding_dim = embedding_dim
        self._hidden_dim = hidden_dim
        self._noise_dim = noise_dim
        super().__init__(pred_len=pred_len, seed=seed, checkpoint=checkpoint)
        self._sampler = self._torch.Generator().manual_seed(seed)

    def _build_net(self) -> Any:
        from pipeline.tp_models import SocialGANNet

        return SocialGANNet(
            embedding_dim=self._embedding_dim,
            hidden_dim=self._hidden_dim,
            noise_dim=self._noise_dim,
        )

    def predict(self, observed_trajectories: np.ndarray, num_modes: int = 20, **kwargs: Any) -> np.ndarray:
        observed = self._validate_observed(observed_trajectories)
        pred_len = int(kwargs.get("pred_len", self.pred_len))

        diffs = self._diffs_from_observed(observed)
        last_point = observed[:, -1, :]

        with self._torch.no_grad():
            diffs_t = self._torch.from_numpy(diffs)
            h, c = self.net.encode(diffs_t)
            pred_diffs = self.net.decode(
                diffs_t[:, -1, :], h, c, num_modes=num_modes, pred_len=pred_len, generator=self._sampler
            )

        return self._reconstruct_absolute(last_point, pred_diffs.numpy())


class TransformerTPAdapter(_TorchTPAdapter):
    """Transformer-encoder-based predictor with learned mode/step query decoding."""

    name = "transformer"

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        max_modes: int = 20,
        max_pred_len: int = 60,
        pred_len: int = 30,
        checkpoint: Optional[str] = None,
        seed: int = 42,
        **_: Any,
    ) -> None:
        self._d_model = d_model
        self._nhead = nhead
        self._num_layers = num_layers
        self._max_modes = max_modes
        self._max_pred_len = max_pred_len
        super().__init__(pred_len=pred_len, seed=seed, checkpoint=checkpoint)

    def _build_net(self) -> Any:
        from pipeline.tp_models import TransformerTPNet

        return TransformerTPNet(
            d_model=self._d_model,
            nhead=self._nhead,
            num_layers=self._num_layers,
            max_modes=self._max_modes,
            max_pred_len=self._max_pred_len,
        )

    def predict(self, observed_trajectories: np.ndarray, num_modes: int = 10, **kwargs: Any) -> np.ndarray:
        observed = self._validate_observed(observed_trajectories)
        pred_len = min(int(kwargs.get("pred_len", self.pred_len)), self._max_pred_len)
        num_modes = min(int(num_modes), self._max_modes)
        last_point = observed[:, -1, :]

        with self._torch.no_grad():
            observed_t = self._torch.from_numpy(observed)
            pred_diffs = self.net(observed_t, num_modes=num_modes, pred_len=pred_len)

        return self._reconstruct_absolute(last_point, pred_diffs.numpy())


TP_ADAPTER_REGISTRY: dict[str, type[BaseTPAdapter]] = {
    "dummy": DummyTPAdapter,
    "social_lstm": SocialLSTMAdapter,
    "social_gan": SocialGANAdapter,
    "transformer": TransformerTPAdapter,
}

TP_MODEL_GROUPS: dict[str, tuple[str, ...]] = {
    "research_classical": ("social_lstm", "social_gan", "dummy"),
    "research_transformer": ("transformer", "dummy"),
}


def _normalize_tp_key(tp_type: str) -> str:
    return str(tp_type).strip().lower().replace("-", "_")


def create_tp_adapter(tp_type: str, **kwargs: Any) -> BaseTPAdapter:
    """Create a TP adapter instance by name (or research group name)."""
    if not tp_type or not str(tp_type).strip():
        raise ValueError("tp_type must be a non-empty TP adapter or group name")

    normalized = _normalize_tp_key(tp_type)
    if normalized in TP_ADAPTER_REGISTRY:
        return TP_ADAPTER_REGISTRY[normalized](**kwargs)

    if normalized in TP_MODEL_GROUPS:
        failures: list[tuple[str, str]] = []
        for candidate in TP_MODEL_GROUPS[normalized]:
            try:
                adapter = TP_ADAPTER_REGISTRY[candidate](**kwargs)
                logger.info("Selected TP adapter '%s' from group '%s'.", candidate, normalized)
                return adapter
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                failures.append((candidate, reason))
                logger.warning(
                    "Failed to initialize TP adapter '%s' for group '%s': %s. Falling back.",
                    candidate,
                    normalized,
                    reason,
                )
        tried = "; ".join(f"{name}: {reason}" for name, reason in failures)
        raise RuntimeError(
            f"No TP adapter from group '{normalized}' could be initialized. Failures: {tried}."
        )

    supported = ", ".join(sorted(TP_ADAPTER_REGISTRY))
    groups = ", ".join(sorted(TP_MODEL_GROUPS)) or "(none)"
    raise ValueError(
        f"Unsupported TP model type '{tp_type}'. Supported TP models: {supported}. "
        f"Supported groups: {groups}."
    )


def predict_trajectories_from_dataframe(
    df: pd.DataFrame,
    tp_type: str = "dummy",
    obs_len: Optional[int] = None,
    pred_len: int = 10,
    num_modes: int = 1,
    fps: float = 25.0,
    **adapter_kwargs: Any,
) -> pd.DataFrame:
    """Run a TP adapter over an observed trajectory DataFrame.

    Args:
        df: Observed trajectory DataFrame with at least columns ``id``,
            ``frame``, ``x``, ``y`` (metric world coordinates).
        tp_type: Name of the TP adapter/group to use.
        obs_len: Number of trailing observed frames to feed the model per
            pedestrian. ``None`` uses all available frames for that id.
        pred_len: Number of future frames to predict.
        num_modes: Number of predicted modes per pedestrian.
        fps: Video frame rate, used to compute predicted frame offsets.

    Returns:
        DataFrame with columns ``id``, ``mode``, ``frame``, ``frame_offset``,
        ``timestamp``, ``x_pred``, ``y_pred``, ``probability``. Empty (but
        correctly columned) if *df* has no rows.
    """
    output_columns = [
        "id",
        "mode",
        "frame",
        "frame_offset",
        "timestamp",
        "x_pred",
        "y_pred",
        "probability",
    ]
    if df.empty:
        return pd.DataFrame(columns=output_columns)

    adapter = create_tp_adapter(tp_type, pred_len=pred_len, **adapter_kwargs)

    rows: list[dict[str, Any]] = []
    probability = 1.0 / max(num_modes, 1)

    for track_id, group in df.sort_values("frame").groupby("id"):
        points = group[["x", "y"]].to_numpy(dtype=np.float64)
        if obs_len is not None:
            points = points[-obs_len:]
        if points.shape[0] < 2:
            continue

        last_frame = int(group["frame"].iloc[-1])
        observed = points[np.newaxis, :, :]
        predictions = adapter.predict(observed, num_modes=num_modes, pred_len=pred_len)
        # predictions shape: (1, num_modes, pred_len, 2)

        for mode_idx in range(predictions.shape[1]):
            for step in range(predictions.shape[2]):
                frame_offset = step + 1
                x_pred, y_pred = predictions[0, mode_idx, step, :]
                rows.append(
                    {
                        "id": track_id,
                        "mode": mode_idx,
                        "frame": last_frame + frame_offset,
                        "frame_offset": frame_offset,
                        "timestamp": (last_frame + frame_offset) / fps,
                        "x_pred": float(x_pred),
                        "y_pred": float(y_pred),
                        "probability": probability,
                    }
                )

    return pd.DataFrame(rows, columns=output_columns)
