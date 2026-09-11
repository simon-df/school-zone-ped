"""Lightweight PyTorch trajectory-prediction model backbones (Phase 2).

These are compact, from-scratch reimplementations of the *architectural
ideas* behind Social-LSTM, Social-GAN and transformer-based trajectory
predictors -- not the original authors' code. They ship with randomly
initialized weights unless a checkpoint is supplied through the adapters'
``load_model()``. Their purpose is to validate real model-loading +
multimodal inference plumbing so that genuine pretrained checkpoints can be
dropped in later without further architecture changes.
"""
from __future__ import annotations

import torch
from torch import nn


class SocialLSTMNet(nn.Module):
    """Encoder/decoder LSTM with a per-mode embedding for cheap multimodality."""

    def __init__(self, embedding_dim: int = 64, hidden_dim: int = 64, max_modes: int = 20) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_embed = nn.Linear(2, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.decoder_cell = nn.LSTMCell(embedding_dim, hidden_dim)
        self.output_head = nn.Linear(hidden_dim, 2)
        self.mode_embed = nn.Embedding(max_modes, hidden_dim)

    def encode(self, observed_diffs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.input_embed(observed_diffs)
        _, (h, c) = self.encoder(embedded)
        return h.squeeze(0), c.squeeze(0)

    def decode(
        self,
        last_diff: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
        num_modes: int,
        pred_len: int,
    ) -> torch.Tensor:
        batch = last_diff.shape[0]
        modes: list[torch.Tensor] = []
        for mode in range(num_modes):
            mode_bias = self.mode_embed.weight[mode].unsqueeze(0).expand(batch, -1)
            hm, cm = h + mode_bias, c
            step_input = last_diff
            steps: list[torch.Tensor] = []
            for _ in range(pred_len):
                embedded = self.input_embed(step_input)
                hm, cm = self.decoder_cell(embedded, (hm, cm))
                diff_pred = self.output_head(hm)
                steps.append(diff_pred)
                step_input = diff_pred
            modes.append(torch.stack(steps, dim=1))
        return torch.stack(modes, dim=1)


class SocialGANNet(nn.Module):
    """Encoder + noise-conditioned decoder, mirroring Social-GAN's generator."""

    def __init__(self, embedding_dim: int = 64, hidden_dim: int = 64, noise_dim: int = 8) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.input_embed = nn.Linear(2, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.noise_proj = nn.Linear(hidden_dim + noise_dim, hidden_dim)
        self.decoder_cell = nn.LSTMCell(embedding_dim, hidden_dim)
        self.output_head = nn.Linear(hidden_dim, 2)

    def encode(self, observed_diffs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.input_embed(observed_diffs)
        _, (h, c) = self.encoder(embedded)
        return h.squeeze(0), c.squeeze(0)

    def decode(
        self,
        last_diff: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
        num_modes: int,
        pred_len: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        batch = last_diff.shape[0]
        modes: list[torch.Tensor] = []
        for _ in range(num_modes):
            noise = torch.randn(batch, self.noise_dim, generator=generator)
            hm = torch.tanh(self.noise_proj(torch.cat([h, noise], dim=-1)))
            cm = c
            step_input = last_diff
            steps: list[torch.Tensor] = []
            for _ in range(pred_len):
                embedded = self.input_embed(step_input)
                hm, cm = self.decoder_cell(embedded, (hm, cm))
                diff_pred = self.output_head(hm)
                steps.append(diff_pred)
                step_input = diff_pred
            modes.append(torch.stack(steps, dim=1))
        return torch.stack(modes, dim=1)


class TransformerTPNet(nn.Module):
    """Transformer encoder with learned mode/step query embeddings (parallel decoding)."""

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        max_modes: int = 20,
        max_pred_len: int = 60,
        max_obs_len: int = 200,
    ) -> None:
        super().__init__()
        self.max_modes = max_modes
        self.max_pred_len = max_pred_len
        self.input_embed = nn.Linear(2, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_obs_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mode_embed = nn.Parameter(torch.randn(max_modes, d_model) * 0.02)
        self.step_embed = nn.Parameter(torch.randn(max_pred_len, d_model) * 0.02)
        self.output_head = nn.Linear(d_model, 2)

    def forward(self, observed: torch.Tensor, num_modes: int, pred_len: int) -> torch.Tensor:
        obs_len = observed.shape[1]
        x = self.input_embed(observed) + self.pos_embed[:, :obs_len, :]
        memory = self.encoder(x)
        context = memory.mean(dim=1)  # (batch, d_model)

        queries = self.mode_embed[:num_modes].unsqueeze(0) + context.unsqueeze(1)  # (batch, modes, d_model)
        queries = queries.unsqueeze(2) + self.step_embed[:pred_len].view(1, 1, pred_len, -1)
        return self.output_head(queries)  # (batch, modes, pred_len, 2)
