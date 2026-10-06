"""Losses for TP training: best-of-K variety loss and Social-GAN adversarial losses.

All models predict per-step *displacements* (matching the adapters, which
cumsum them from the last observed point). Losses compare positions relative
to the last observed point, i.e. ``cumsum(diffs)``.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


def relative_positions(diffs: torch.Tensor) -> torch.Tensor:
    """Cumulative sum over the time axis (second-to-last dim): displacements -> positions rel. to last obs."""
    return torch.cumsum(diffs, dim=-2)


def per_mode_displacement(pred_diffs: torch.Tensor, gt_diffs: torch.Tensor, squared: bool = True) -> torch.Tensor:
    """Per-step distance between every predicted mode and the ground truth.

    Args:
        pred_diffs: ``(N, K, T, 2)`` predicted displacements.
        gt_diffs: ``(N, T, 2)`` ground-truth displacements (first one is
            ``future[0] - obs[-1]``).
        squared: Return squared L2 distances (default) instead of L2 distances.

    Returns:
        ``(N, K, T)`` distances.
    """
    err = relative_positions(pred_diffs) - relative_positions(gt_diffs).unsqueeze(1)
    sq = (err**2).sum(dim=-1)
    return sq if squared else torch.sqrt(sq + 1e-12)


def variety_loss(pred_diffs: torch.Tensor, gt_diffs: torch.Tensor, squared: bool = True) -> torch.Tensor:
    """Best-of-K ("variety") loss: per agent, only the closest of the K modes is penalized.

    ``mean_N min_K mean_T ||p - g||^2`` (``squared=True``, the L2 loss used
    for training) or ``mean_N min_K mean_T ||p - g||`` (= minADE@K).
    Used for both Social-LSTM (K mode embeddings) and the Social-GAN
    generator (K noise samples). With ``K = 1`` this is a plain L2 loss.
    """
    per_mode = per_mode_displacement(pred_diffs, gt_diffs, squared=squared).mean(dim=-1)  # (N, K)
    return per_mode.min(dim=1).values.mean()


def discriminator_loss(real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy discriminator loss (real -> 1, fake -> 0) on raw logits."""
    real = F.binary_cross_entropy_with_logits(real_scores, torch.ones_like(real_scores))
    fake = F.binary_cross_entropy_with_logits(fake_scores, torch.zeros_like(fake_scores))
    return real + fake


def generator_adversarial_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    """Non-saturating generator loss: make the discriminator call fakes real."""
    return F.binary_cross_entropy_with_logits(fake_scores, torch.ones_like(fake_scores))
