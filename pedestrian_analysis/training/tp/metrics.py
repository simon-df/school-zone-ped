"""Displacement metrics: ADE, FDE, minADE@K, minFDE@K (numpy, adapter-agnostic).

* ``ade`` / ``fde`` use the *first* predicted mode (for Social-GAN: one
  random sample; for constant velocity / Social-LSTM: deterministic).
* ``min_ade`` / ``min_fde`` take the best of the K modes per agent.

All values are in the units of the trajectories (metres for ETH/UCY and the
drone CSVs).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

METRIC_KEYS: tuple[str, ...] = ("ade", "fde", "min_ade", "min_fde")


def displacement_errors(pred_abs: np.ndarray, gt_abs: np.ndarray) -> dict[str, np.ndarray]:
    """Per-agent errors for predictions ``(N, K, T, 2)`` vs ground truth ``(N, T, 2)``."""
    pred_abs = np.asarray(pred_abs, dtype=np.float64)
    gt_abs = np.asarray(gt_abs, dtype=np.float64)
    if pred_abs.ndim != 4 or gt_abs.ndim != 3 or pred_abs.shape[2:] != gt_abs.shape[1:]:
        raise ValueError(f"Expected pred (N, K, T, 2) and gt (N, T, 2), got {pred_abs.shape} and {gt_abs.shape}")
    dist = np.linalg.norm(pred_abs - gt_abs[:, None], axis=-1)  # (N, K, T)
    ade_k = dist.mean(axis=-1)
    fde_k = dist[..., -1]
    return {"ade": ade_k[:, 0], "fde": fde_k[:, 0], "min_ade": ade_k.min(axis=1), "min_fde": fde_k.min(axis=1)}


class MetricAccumulator:
    """Accumulates per-agent errors over batches and reports means."""

    def __init__(self) -> None:
        self._parts: dict[str, list[np.ndarray]] = {key: [] for key in METRIC_KEYS}

    def update(self, pred_abs: np.ndarray, gt_abs: np.ndarray) -> None:
        for key, values in displacement_errors(pred_abs, gt_abs).items():
            self._parts[key].append(values)

    @property
    def num_agents(self) -> int:
        return int(sum(v.size for v in self._parts["ade"]))

    def compute(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for key in METRIC_KEYS:
            values = np.concatenate(self._parts[key]) if self._parts[key] else np.empty(0)
            out[key] = float(values.mean()) if values.size else float("nan")
        out["num_agents"] = self.num_agents
        return out


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    """Mean and (population) standard deviation, ignoring NaNs."""
    arr = np.asarray([v for v in values if v == v], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std())
