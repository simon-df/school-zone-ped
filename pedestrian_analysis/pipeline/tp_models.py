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


class PoolHiddenNet(nn.Module):
    """Social-GAN-style pooling of neighbour hidden states (optional interaction module).

    For every agent ``i`` and every agent ``j`` in the same scene (including
    ``i`` itself), the relative last observed position ``p_j - p_i`` is
    embedded, concatenated with ``h_j`` and passed through an MLP; the
    resulting messages are max-pooled over ``j``. Agents belong to the same
    scene when they share a ``scene_id`` (``None`` = one scene for the whole
    batch, which is what the adapters use at inference time: all agents
    present in the current frame).
    """

    def __init__(self, embedding_dim: int, hidden_dim: int, pool_dim: int) -> None:
        super().__init__()
        self.pool_dim = pool_dim
        self.rel_embed = nn.Linear(2, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pool_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        h: torch.Tensor,
        last_pos: torch.Tensor,
        scene_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_agents = h.shape[0]
        if scene_ids is None:
            scene_ids = torch.zeros(num_agents, dtype=torch.long, device=h.device)
        same_scene = scene_ids.unsqueeze(1) == scene_ids.unsqueeze(0)
        idx_i, idx_j = same_scene.nonzero(as_tuple=True)
        rel = last_pos[idx_j] - last_pos[idx_i]
        messages = self.mlp(torch.cat([self.rel_embed(rel), h[idx_j]], dim=-1))
        pooled = messages.new_zeros(num_agents, self.pool_dim)
        index = idx_i.unsqueeze(1).expand(-1, self.pool_dim)
        return pooled.scatter_reduce(0, index, messages, reduce="amax", include_self=False)


class _EncoderDecoderBase(nn.Module):
    """Shared encoder (+ optional social pooling) and autoregressive decoding loop."""

    def _init_encoder(self, embedding_dim: int, hidden_dim: int, use_social_pooling: bool) -> None:
        self.hidden_dim = hidden_dim
        self.use_social_pooling = bool(use_social_pooling)
        self.input_embed = nn.Linear(2, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        if self.use_social_pooling:
            # Only created when enabled so state_dict keys of non-pooling
            # checkpoints are unchanged (backward compatible).
            self.social_pool = PoolHiddenNet(embedding_dim, hidden_dim, pool_dim=hidden_dim)
            self.pool_proj = nn.Linear(hidden_dim * 2, hidden_dim)
            # Zero-init the residual projection: a pooling model initialised
            # from a non-pooling checkpoint starts out behaving identically.
            nn.init.zeros_(self.pool_proj.weight)
            nn.init.zeros_(self.pool_proj.bias)

    def encode(
        self,
        observed_diffs: torch.Tensor,
        last_pos: torch.Tensor | None = None,
        scene_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode ``(batch, obs_len - 1, 2)`` displacements into ``(h, c)``.

        ``last_pos`` (``(batch, 2)`` last observed absolute positions) and
        ``scene_ids`` (``(batch,)``) are only used when social pooling is
        enabled; ``last_pos`` is then required.
        """
        embedded = self.input_embed(observed_diffs)
        _, (h, c) = self.encoder(embedded)
        h, c = h.squeeze(0), c.squeeze(0)
        if self.use_social_pooling:
            if last_pos is None:
                raise ValueError("last_pos is required when use_social_pooling=True")
            pooled = self.social_pool(h, last_pos, scene_ids)
            h = h + self.pool_proj(torch.cat([h, pooled], dim=-1))
        return h, c

    def _rollout(self, step_input: torch.Tensor, hm: torch.Tensor, cm: torch.Tensor, pred_len: int) -> torch.Tensor:
        steps: list[torch.Tensor] = []
        for _ in range(pred_len):
            embedded = self.input_embed(step_input)
            hm, cm = self.decoder_cell(embedded, (hm, cm))
            diff_pred = self.output_head(hm)
            steps.append(diff_pred)
            step_input = diff_pred
        return torch.stack(steps, dim=1)


class SocialLSTMNet(_EncoderDecoderBase):
    """Encoder/decoder LSTM with a per-mode embedding for cheap multimodality.

    With ``use_social_pooling=True`` the encoder output is fused with a
    :class:`PoolHiddenNet` summary of co-present agents (interaction-aware).
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        max_modes: int = 20,
        use_social_pooling: bool = False,
    ) -> None:
        super().__init__()
        self.max_modes = max_modes
        self._init_encoder(embedding_dim, hidden_dim, use_social_pooling)
        self.decoder_cell = nn.LSTMCell(embedding_dim, hidden_dim)
        self.output_head = nn.Linear(hidden_dim, 2)
        self.mode_embed = nn.Embedding(max_modes, hidden_dim)

    def decode(
        self,
        last_diff: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
        num_modes: int,
        pred_len: int,
    ) -> torch.Tensor:
        """Return predicted displacements of shape ``(batch, num_modes, pred_len, 2)``."""
        if num_modes > self.max_modes:
            raise ValueError(f"num_modes={num_modes} exceeds max_modes={self.max_modes}")
        batch = last_diff.shape[0]
        # All modes are decoded in parallel as one (batch * num_modes) batch.
        mode_bias = self.mode_embed.weight[:num_modes]
        hm = (h.unsqueeze(1) + mode_bias.unsqueeze(0)).reshape(batch * num_modes, -1)
        cm = c.unsqueeze(1).expand(-1, num_modes, -1).reshape(batch * num_modes, -1)
        step_input = last_diff.unsqueeze(1).expand(-1, num_modes, -1).reshape(batch * num_modes, 2)
        return self._rollout(step_input, hm, cm, pred_len).reshape(batch, num_modes, pred_len, 2)


class SocialGANNet(_EncoderDecoderBase):
    """Encoder + noise-conditioned decoder, mirroring Social-GAN's generator.

    With ``use_social_pooling=True`` the encoder output is fused with a
    :class:`PoolHiddenNet` summary of co-present agents, as in Social-GAN.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        noise_dim: int = 8,
        use_social_pooling: bool = False,
    ) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self._init_encoder(embedding_dim, hidden_dim, use_social_pooling)
        self.noise_proj = nn.Linear(hidden_dim + noise_dim, hidden_dim)
        self.decoder_cell = nn.LSTMCell(embedding_dim, hidden_dim)
        self.output_head = nn.Linear(hidden_dim, 2)

    def decode(
        self,
        last_diff: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
        num_modes: int,
        pred_len: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Return predicted displacements of shape ``(batch, num_modes, pred_len, 2)``."""
        batch = last_diff.shape[0]
        if generator is not None:
            # Seeded sampler (CPU generator): draw per mode, then move to h's device.
            noise = torch.stack(
                [torch.randn(batch, self.noise_dim, generator=generator) for _ in range(num_modes)], dim=1
            ).to(h.device)
        else:
            noise = torch.randn(batch, num_modes, self.noise_dim, device=h.device)
        h_rep = h.unsqueeze(1).expand(-1, num_modes, -1)
        hm = torch.tanh(self.noise_proj(torch.cat([h_rep, noise], dim=-1))).reshape(batch * num_modes, -1)
        cm = c.unsqueeze(1).expand(-1, num_modes, -1).reshape(batch * num_modes, -1)
        step_input = last_diff.unsqueeze(1).expand(-1, num_modes, -1).reshape(batch * num_modes, 2)
        return self._rollout(step_input, hm, cm, pred_len).reshape(batch, num_modes, pred_len, 2)


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
