"""Social-GAN trajectory discriminator (training only; never needed for inference).

Kept out of :mod:`pipeline.tp_models` on purpose: the inference checkpoint
for :class:`pipeline.tp_adapters.SocialGANAdapter` contains the generator
(:class:`pipeline.tp_models.SocialGANNet`) state_dict only.
"""
from __future__ import annotations

import torch
from torch import nn


class TrajectoryDiscriminator(nn.Module):
    """Small LSTM classifier scoring a full (observed + future) displacement sequence.

    Input: ``(N, obs_len - 1 + pred_len, 2)`` displacements. Output: ``(N,)``
    real/fake logits.
    """

    def __init__(self, embedding_dim: int = 64, hidden_dim: int = 64, mlp_dim: int = 64) -> None:
        super().__init__()
        self.input_embed = nn.Linear(2, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(mlp_dim, 1),
        )

    def forward(self, traj_diffs: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.encoder(torch.relu(self.input_embed(traj_diffs)))
        return self.classifier(h.squeeze(0)).squeeze(-1)
