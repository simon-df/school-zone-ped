"""Train SocialLSTMNet / SocialGANNet trajectory predictors.

Produces checkpoints that load directly into the existing adapters
(``SocialLSTMAdapter(checkpoint=...)`` / ``SocialGANAdapter(checkpoint=...)``)
and the TP Analysis tab's *Checkpoint* field:

* ``best.pt`` / ``last.pt`` -- plain ``state_dict`` of the inference network
  (for Social-GAN: the generator only, keys matching ``SocialGANNet``).
* ``best.json`` / ``last.json`` -- sidecar metadata (model, architecture
  kwargs, fps, obs_len, pred_len, metrics, ...). The registry reads it to
  build a matching adapter and the UI warns when its settings differ.
* ``train_state.pt`` -- full training state (incl. discriminator and
  optimizers) for ``--resume``.

Training matches the adapters exactly: inputs are ``np.diff`` of the
observed positions, targets are the diffs of the future positions with the
first target ``future[0] - obs[-1]``; predictions are rebuilt by a cumsum
from the last observed point.

Examples (run from ``pedestrian_analysis/``)::

    python -m training.tp.train --model social_lstm --dataset ethucy --test-scene eth
    python -m training.tp.train --model social_gan --dataset ethucy --test-scene zara1 \\
        --use-social-pooling --epochs 100
    python -m training.tp.train --config my_run.yaml   # keys = CLI option names (dashes or underscores)
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from pipeline.tp_model_registry import load_checkpoint_metadata  # noqa: E402
from pipeline.tp_models import SocialGANNet, SocialLSTMNet  # noqa: E402
from training.tp.datasets import (  # noqa: E402
    DEFAULT_ETHUCY_ROOT,
    ETHUCY_SCENES,
    BaseTrajectoryDataset,
    ETHUCYDataset,
    OwnCSVDataset,
    TrajectoryWindow,
    split_recordings,
)
from training.tp.discriminator import TrajectoryDiscriminator  # noqa: E402
from training.tp.losses import discriminator_loss, generator_adversarial_loss, variety_loss  # noqa: E402
from training.tp.metrics import MetricAccumulator  # noqa: E402

logger = logging.getLogger(__name__)

MODEL_CHOICES: tuple[str, ...] = ("social_lstm", "social_gan")
DATASET_CHOICES: tuple[str, ...] = ("ethucy", "own_csv")
DEFAULT_OUTPUT_ROOT = PACKAGE_ROOT / "outputs" / "tp_training"
METADATA_FORMAT_VERSION = 1

# Fields that, when not given explicitly, are inherited from --init-checkpoint metadata.
_INHERITABLE = ("fps", "obs_len", "pred_len", "embedding_dim", "hidden_dim", "max_modes", "noise_dim", "use_social_pooling")
_HARD_DEFAULTS: dict[str, Any] = {
    "fps": 10.0,
    "obs_len": 20,
    "pred_len": 30,
    "embedding_dim": 64,
    "hidden_dim": 64,
    "max_modes": 20,
    "noise_dim": 8,
    "use_social_pooling": False,
}


@dataclass
class TrainConfig:
    """All training settings (CLI / YAML / programmatic)."""

    model: str = "social_lstm"
    dataset: str = "ethucy"
    data_root: str = str(DEFAULT_ETHUCY_ROOT)
    test_scene: str = "eth"
    csv: list[str] = field(default_factory=list)
    source_fps: float = 10.0
    val_fraction: float = 0.2
    fps: float = 10.0
    obs_len: int = 20
    pred_len: int = 30
    stride: Optional[int] = None
    min_agents: int = 1
    num_modes: int = 20
    epochs: int = 50
    lr: float = 1e-3
    d_lr: Optional[float] = None
    weight_decay: float = 0.0
    batch_size: int = 32
    grad_clip: float = 1.0
    adv_weight: float = 0.1
    patience: int = 10
    early_stop_metric: str = "min_ade"
    device: str = "auto"
    seed: int = 42
    output_dir: Optional[str] = None
    use_social_pooling: bool = False
    embedding_dim: int = 64
    hidden_dim: int = 64
    max_modes: int = 20
    noise_dim: int = 8
    augment: Optional[bool] = None
    init_checkpoint: Optional[str] = None
    freeze_encoder: bool = False
    resume: Optional[str] = None

    def validate(self) -> None:
        if self.model not in MODEL_CHOICES:
            raise ValueError(f"model must be one of {MODEL_CHOICES}, got '{self.model}'")
        if self.dataset not in DATASET_CHOICES:
            raise ValueError(f"dataset must be one of {DATASET_CHOICES}, got '{self.dataset}'")
        if self.obs_len < 2 or self.pred_len < 1:
            raise ValueError("obs_len must be >= 2 and pred_len >= 1")
        if self.num_modes < 1:
            raise ValueError("num_modes must be >= 1")
        if self.model == "social_lstm" and self.num_modes > self.max_modes:
            raise ValueError(f"num_modes ({self.num_modes}) must be <= max_modes ({self.max_modes}) for social_lstm")
        if self.early_stop_metric not in ("ade", "fde", "min_ade", "min_fde"):
            raise ValueError(f"Unknown early_stop_metric '{self.early_stop_metric}'")

    @property
    def model_kwargs(self) -> dict[str, Any]:
        """Constructor kwargs of the inference net (= adapter kwargs, stored in the sidecar)."""
        kwargs: dict[str, Any] = {
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "use_social_pooling": bool(self.use_social_pooling),
        }
        if self.model == "social_lstm":
            kwargs["max_modes"] = self.max_modes
        else:
            kwargs["noise_dim"] = self.noise_dim
        return kwargs


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def build_net(model: str, model_kwargs: dict[str, Any]) -> torch.nn.Module:
    if model == "social_lstm":
        return SocialLSTMNet(**model_kwargs)
    if model == "social_gan":
        return SocialGANNet(**model_kwargs)
    raise ValueError(f"Unknown model '{model}'")


def load_init_checkpoint(net: torch.nn.Module, path: str | Path) -> None:
    """Initialise *net* from a state_dict (or a ``train_state.pt``).

    Missing keys are tolerated only for the optional social pooling modules
    (so a non-pooling ETH/UCY checkpoint can seed a pooling model; the
    residual projection is zero-initialised so behaviour starts identical).
    """
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "net" in state and isinstance(state["net"], dict):
        state = state["net"]
    result = net.load_state_dict(state, strict=False)
    bad_missing = [k for k in result.missing_keys if not k.startswith(("social_pool.", "pool_proj."))]
    if result.unexpected_keys or bad_missing:
        raise RuntimeError(
            f"Init checkpoint '{path}' does not match the architecture: "
            f"missing={bad_missing}, unexpected={list(result.unexpected_keys)}"
        )
    if result.missing_keys:
        logger.info("Init checkpoint has no social pooling weights; they start from scratch: %s", result.missing_keys)


def freeze_encoder(net: torch.nn.Module) -> None:
    """Freeze the observation encoder (LSTM + shared input embedding)."""
    for module in (net.input_embed, net.encoder):
        for param in module.parameters():
            param.requires_grad_(False)


def collate_windows(windows: Sequence[TrajectoryWindow]) -> dict[str, torch.Tensor]:
    """Concatenate the agents of several windows; ``scene_ids`` keeps windows apart for pooling."""
    obs = np.concatenate([w.obs for w in windows], axis=0)
    fut = np.concatenate([w.fut for w in windows], axis=0)
    scene_ids = np.concatenate([np.full(w.num_agents, i) for i, w in enumerate(windows)])
    return {
        "obs": torch.as_tensor(obs, dtype=torch.float32),
        "fut": torch.as_tensor(fut, dtype=torch.float32),
        "scene_ids": torch.as_tensor(scene_ids, dtype=torch.long),
    }


def forward_net(
    net: torch.nn.Module,
    obs: torch.Tensor,
    scene_ids: Optional[torch.Tensor],
    num_modes: int,
    pred_len: int,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same computation as the adapters' ``predict``: returns ``(pred_diffs (N,K,T,2), obs_diffs)``."""
    obs_diffs = obs[:, 1:] - obs[:, :-1]
    h, c = net.encode(obs_diffs, last_pos=obs[:, -1], scene_ids=scene_ids)
    kwargs = {"generator": generator} if isinstance(net, SocialGANNet) else {}
    pred = net.decode(obs_diffs[:, -1], h, c, num_modes=num_modes, pred_len=pred_len, **kwargs)
    return pred, obs_diffs


def target_diffs(obs: torch.Tensor, fut: torch.Tensor) -> torch.Tensor:
    """Future displacements; the first one is ``fut[0] - obs[-1]`` (matches the adapters' cumsum)."""
    return torch.diff(torch.cat([obs[:, -1:], fut], dim=1), dim=1)


@torch.no_grad()
def evaluate_net(
    net: torch.nn.Module,
    loader: DataLoader,
    num_modes: int,
    pred_len: int,
    device: torch.device,
    seed: int = 0,
) -> dict[str, float]:
    net.eval()
    generator = torch.Generator().manual_seed(seed)
    acc = MetricAccumulator()
    for batch in loader:
        obs = batch["obs"].to(device)
        pred, _ = forward_net(net, obs, batch["scene_ids"].to(device), num_modes, pred_len, generator=generator)
        pred_abs = torch.cumsum(pred, dim=2) + obs[:, -1][:, None, None, :]
        acc.update(pred_abs.cpu().numpy(), batch["fut"].numpy())
    return acc.compute()


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def save_checkpoint(net: torch.nn.Module, path: Path, metadata: dict[str, Any]) -> None:
    """Save the inference ``state_dict`` + sidecar ``<stem>.json``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_cpu_state_dict(net), str(path))
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, default=str))


def build_metadata(cfg: TrainConfig, epoch: int, val_metrics: Optional[dict[str, float]], extra: Optional[dict[str, Any]]) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "format_version": METADATA_FORMAT_VERSION,
        "model": cfg.model,
        "model_kwargs": cfg.model_kwargs,
        "checkpoint_type": "state_dict",
        "generator_only": cfg.model == "social_gan",
        "fps": cfg.fps,
        "obs_len": cfg.obs_len,
        "pred_len": cfg.pred_len,
        "num_modes": cfg.num_modes,
        "dataset": cfg.dataset,
        "test_scene": cfg.test_scene if cfg.dataset == "ethucy" else None,
        "epoch": epoch,
        "val_metrics": val_metrics,
        "init_checkpoint": cfg.init_checkpoint,
        "train_config": asdict(cfg),
        "torch_version": torch.__version__,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if extra:
        meta.update(extra)
    return meta


# ----------------------------------------------------------------------
# training loop
# ----------------------------------------------------------------------


def fit(
    cfg: TrainConfig,
    train_ds: BaseTrajectoryDataset,
    val_ds: Optional[BaseTrajectoryDataset],
    output_dir: str | Path,
    extra_metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Train ``cfg.model`` on *train_ds*, early-stopping on *val_ds* (if given and non-empty).

    Without validation data every epoch overwrites ``best.pt`` (= ``last.pt``).

    Returns:
        ``{"best_checkpoint", "last_checkpoint", "best_epoch", "best_val", "history"}``.
    """
    cfg.validate()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)
    set_seed(cfg.seed)

    if len(train_ds) == 0:
        raise ValueError("Training dataset has no windows (tracks shorter than obs_len + pred_len?).")
    has_val = val_ds is not None and len(val_ds) > 0

    net = build_net(cfg.model, cfg.model_kwargs)
    if cfg.init_checkpoint:
        load_init_checkpoint(net, cfg.init_checkpoint)
    if cfg.freeze_encoder:
        freeze_encoder(net)
    net.to(device)
    opt_g = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay)

    disc: Optional[TrajectoryDiscriminator] = None
    opt_d: Optional[torch.optim.Optimizer] = None
    if cfg.model == "social_gan":
        disc = TrajectoryDiscriminator(cfg.embedding_dim, cfg.hidden_dim, input_scale=cfg.fps).to(device)
        opt_d = torch.optim.Adam(disc.parameters(), lr=cfg.d_lr or cfg.lr)

    start_epoch = 1
    best_val = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    state_path = output_dir / "train_state.pt"

    if cfg.resume:
        state = torch.load(str(cfg.resume), map_location="cpu", weights_only=True)
        net.load_state_dict(state["net"])
        opt_g.load_state_dict(state["opt_g"])
        if disc is not None and state.get("disc") is not None:
            disc.load_state_dict(state["disc"])
            opt_d.load_state_dict(state["opt_d"])
        start_epoch = int(state["epoch"]) + 1
        best_val = float(state["best_val"])
        best_epoch = int(state["best_epoch"])
        epochs_without_improvement = int(state["epochs_without_improvement"])
        history = list(state.get("history", []))
        logger.info("Resumed from '%s' at epoch %d.", cfg.resume, start_epoch)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_windows,
                              generator=torch.Generator().manual_seed(cfg.seed))
    val_loader = (
        DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_windows) if has_val else None
    )
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    logger.info(
        "Training %s on %d windows (%d agent samples), val=%s, device=%s, fps=%g obs=%d pred=%d K=%d pooling=%s",
        cfg.model, len(train_ds), train_ds.num_agents, len(val_ds) if has_val else 0, device,
        cfg.fps, cfg.obs_len, cfg.pred_len, cfg.num_modes, cfg.use_social_pooling,
    )

    for epoch in range(start_epoch, cfg.epochs + 1):
        # Per-epoch reseeding makes runs (and resumed runs) reproducible.
        torch.manual_seed(cfg.seed + epoch)
        train_ds.reseed(cfg.seed + epoch)
        train_loader.generator.manual_seed(cfg.seed + epoch)
        net.train()
        if disc is not None:
            disc.train()
        t0 = time.perf_counter()
        sums = {"loss": 0.0, "variety": 0.0, "adv": 0.0, "d_loss": 0.0}
        n_batches = 0
        for batch in train_loader:
            obs = batch["obs"].to(device)
            fut = batch["fut"].to(device)
            scene_ids = batch["scene_ids"].to(device)
            gt = target_diffs(obs, fut)
            pred, obs_diffs = forward_net(net, obs, scene_ids, cfg.num_modes, cfg.pred_len)
            loss_variety = variety_loss(pred, gt)
            loss = loss_variety

            if disc is not None:
                real_traj = torch.cat([obs_diffs, gt], dim=1)
                fake_traj = torch.cat([obs_diffs, pred[:, 0]], dim=1)
                d_loss = discriminator_loss(disc(real_traj), disc(fake_traj.detach()))
                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()
                adv = generator_adversarial_loss(disc(fake_traj))
                loss = loss + cfg.adv_weight * adv
                sums["adv"] += adv.item()
                sums["d_loss"] += d_loss.item()

            opt_g.zero_grad()
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            opt_g.step()
            sums["loss"] += loss.item()
            sums["variety"] += loss_variety.item()
            n_batches += 1

        record: dict[str, Any] = {"epoch": epoch, **{k: v / max(n_batches, 1) for k, v in sums.items()}}
        val_metrics = evaluate_net(net, val_loader, cfg.num_modes, cfg.pred_len, device, cfg.seed) if has_val else None
        if val_metrics is not None:
            record.update({f"val_{k}": v for k, v in val_metrics.items()})
        record["seconds"] = time.perf_counter() - t0
        history.append(record)

        meta = build_metadata(cfg, epoch, val_metrics, extra_metadata)
        save_checkpoint(net, last_path, meta)
        improved = False
        if val_metrics is None:
            improved = True
        else:
            score = val_metrics[cfg.early_stop_metric]
            if score < best_val - 1e-6:
                best_val, improved = score, True
        if improved:
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(net, best_path, meta)
        else:
            epochs_without_improvement += 1

        logger.info(
            "epoch %3d | loss %.4f (variety %.4f%s) | %s | %.1fs%s",
            epoch, record["loss"], record["variety"],
            f", adv {record['adv']:.3f}, D {record['d_loss']:.3f}" if disc is not None else "",
            (f"val ADE {val_metrics['ade']:.3f} FDE {val_metrics['fde']:.3f} "
             f"minADE@{cfg.num_modes} {val_metrics['min_ade']:.3f} minFDE {val_metrics['min_fde']:.3f}")
            if val_metrics else "no val",
            record["seconds"], " *" if improved else "",
        )

        torch.save(
            {
                "epoch": epoch,
                "net": _cpu_state_dict(net),
                "opt_g": opt_g.state_dict(),
                "disc": _cpu_state_dict(disc) if disc is not None else None,
                "opt_d": opt_d.state_dict() if opt_d is not None else None,
                "best_val": best_val,
                "best_epoch": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
                "history": history,
                "config": asdict(cfg),
            },
            str(state_path),
        )

        if has_val and cfg.patience > 0 and epochs_without_improvement >= cfg.patience:
            logger.info("Early stopping at epoch %d (best epoch %d).", epoch, best_epoch)
            break

    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    return {
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "best_epoch": best_epoch,
        "best_val": best_val if has_val else None,
        "history": history,
    }


# ----------------------------------------------------------------------
# datasets + end-to-end entry point
# ----------------------------------------------------------------------


def _dataset_kwargs(cfg: TrainConfig, augment: bool) -> dict[str, Any]:
    return dict(obs_len=cfg.obs_len, pred_len=cfg.pred_len, fps=cfg.fps, stride=cfg.stride,
                min_agents=cfg.min_agents, augment=augment, seed=cfg.seed)


def build_datasets(cfg: TrainConfig) -> tuple[BaseTrajectoryDataset, Optional[BaseTrajectoryDataset], Optional[BaseTrajectoryDataset]]:
    """Return ``(train, val, test)`` datasets. ETH/UCY: leave-one-scene-out; own CSV: split by recording."""
    augment = cfg.augment if cfg.augment is not None else cfg.dataset == "own_csv"
    if cfg.dataset == "ethucy":
        root = cfg.data_root
        train = ETHUCYDataset(root, cfg.test_scene, "train", **_dataset_kwargs(cfg, augment))
        val = ETHUCYDataset(root, cfg.test_scene, "val", **_dataset_kwargs(cfg, False))
        test = ETHUCYDataset(root, cfg.test_scene, "test", **_dataset_kwargs(cfg, False))
        return train, val, test
    if not cfg.csv:
        raise ValueError("--csv (files and/or directories) is required for --dataset own_csv")
    full = OwnCSVDataset(cfg.csv, source_fps=cfg.source_fps, **_dataset_kwargs(cfg, augment))
    train_recs, val_recs = split_recordings(full.recordings, cfg.val_fraction, cfg.seed)
    logger.info("Recording split -> train: %s | val: %s", train_recs, val_recs)
    return full.subset(train_recs, augment=augment), (full.subset(val_recs, augment=False) if val_recs else None), None


def default_output_dir(cfg: TrainConfig) -> Path:
    tag = f"{cfg.model}_{cfg.dataset}" + (f"_{cfg.test_scene}" if cfg.dataset == "ethucy" else "")
    if cfg.use_social_pooling:
        tag += "_pool"
    return DEFAULT_OUTPUT_ROOT / tag


def train(cfg: TrainConfig) -> dict[str, Any]:
    """Build datasets, train, and (ETH/UCY) evaluate ``best.pt`` on the held-out scene via the adapter."""
    cfg.validate()
    output_dir = Path(cfg.output_dir) if cfg.output_dir else default_output_dir(cfg)
    train_ds, val_ds, test_ds = build_datasets(cfg)
    result = fit(cfg, train_ds, val_ds, output_dir)

    if test_ds is not None and len(test_ds) > 0:
        from training.tp.evaluate import format_table, run_evaluation, write_report

        rows = run_evaluation(
            test_ds.windows,
            [(f"{cfg.model} (best)", cfg.model, result["best_checkpoint"])],
            num_modes=cfg.num_modes,
            pred_len=cfg.pred_len,
            seed=cfg.seed,
        )
        print(f"\nTest scene '{cfg.test_scene}' ({len(test_ds)} windows):")
        print(format_table(rows, cfg.num_modes))
        write_report(rows, output_dir / "test_results", {"test_scene": cfg.test_scene, **asdict(cfg)})
        result["test_results"] = rows

    print(f"\nBest checkpoint: {result['best_checkpoint']} (epoch {result['best_epoch']})")
    return result


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Options shared by ``train`` and ``finetune``."""
    parser.add_argument("--config", default=None, help="YAML file with option values (CLI flags override it).")
    parser.add_argument("--model", choices=MODEL_CHOICES, default=None)
    parser.add_argument("--fps", type=float, default=None, help="Training fps (default 10; ETH/UCY is resampled from 2.5 Hz).")
    parser.add_argument("--obs-len", type=int, default=None, help="Observed positions (default 20 = 2 s at 10 Hz).")
    parser.add_argument("--pred-len", type=int, default=None, help="Predicted positions (default 30 = 3 s at 10 Hz).")
    parser.add_argument("--stride", type=int, default=None, help="Window stride in frames (default: fps / 2.5).")
    parser.add_argument("--num-modes", type=int, default=20, help="K for the best-of-K loss and minADE@K.")
    parser.add_argument("--batch-size", type=int, default=32, help="Windows (scenes) per batch.")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--use-social-pooling", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable Social-GAN-style pooling over co-present agents.")
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--max-modes", type=int, default=None, help="Social-LSTM mode embeddings (default 20).")
    parser.add_argument("--noise-dim", type=int, default=None, help="Social-GAN noise size (default 8).")
    parser.add_argument("--adv-weight", type=float, default=0.1, help="Social-GAN adversarial loss weight.")
    parser.add_argument("--d-lr", type=float, default=None, help="Discriminator LR (default: --lr).")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stop-metric", choices=("ade", "fde", "min_ade", "min_fde"), default="min_ade",
                        help="Validation metric for early stopping / best.pt (default: minADE@K).")
    parser.add_argument("--min-agents", type=int, default=1)
    parser.add_argument("--source-fps", type=float, default=10.0, help="CSV frame rate if no timestamp column.")
    parser.add_argument("--init-checkpoint", default=None, help="Initialise from a (pretrained) checkpoint.")
    parser.add_argument("--freeze-encoder", action="store_true", help="Freeze input embedding + encoder LSTM.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Social-LSTM / Social-GAN TP models (checkpoints load into the adapters).")
    add_common_arguments(parser)
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="ethucy")
    parser.add_argument("--data-root", default=str(DEFAULT_ETHUCY_ROOT), help="ETH/UCY root (see download_ethucy).")
    parser.add_argument("--test-scene", choices=ETHUCY_SCENES, default="eth", help="Held-out ETH/UCY scene.")
    parser.add_argument("--csv", nargs="+", default=None, help="Own CSV files/dirs (--dataset own_csv).")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="own_csv: fraction of recordings for validation.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10, help="Early-stopping patience in epochs (0 = off).")
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=None,
                        help="Random rotation/flip/scale (default: on for own_csv, off for ethucy).")
    parser.add_argument("--resume", default=None, help="Resume from a train_state.pt.")
    return parser


def parse_args_with_config(parser: argparse.ArgumentParser, argv: Optional[list[str]]) -> argparse.Namespace:
    """Parse CLI args; values from ``--config`` YAML act as defaults that CLI flags override."""
    args = parser.parse_args(argv)
    if args.config:
        import yaml

        with open(args.config, encoding="utf-8") as fh:
            values = yaml.safe_load(fh) or {}
        if not isinstance(values, dict):
            raise SystemExit(f"Config '{args.config}' must contain a mapping of option names to values.")
        known = {action.dest for action in parser._actions}
        normalized = {str(k).replace("-", "_"): v for k, v in values.items()}
        unknown = set(normalized) - known
        if unknown:
            raise SystemExit(f"Unknown keys in config '{args.config}': {sorted(unknown)}")
        parser.set_defaults(**normalized)
        args = parser.parse_args(argv)
    return args


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    """Build a :class:`TrainConfig`; unset architecture/timing fields come from the init checkpoint's sidecar."""
    values = {k: v for k, v in vars(args).items() if k in TrainConfig.__dataclass_fields__ and v is not None}
    meta = load_checkpoint_metadata(args.init_checkpoint) if getattr(args, "init_checkpoint", None) else None
    if meta:
        values.setdefault("model", meta.get("model"))
        inherited = {**{k: meta.get(k) for k in ("fps", "obs_len", "pred_len")}, **(meta.get("model_kwargs") or {})}
        for key in _INHERITABLE:
            if key not in values and inherited.get(key) is not None:
                values[key] = inherited[key]
    for key, default in _HARD_DEFAULTS.items():
        values.setdefault(key, default)
    values.setdefault("model", "social_lstm")
    if "csv" in values and values["csv"] is None:
        values["csv"] = []
    cfg = TrainConfig(**values)
    if meta and meta.get("model") and meta["model"] != cfg.model:
        raise SystemExit(f"--init-checkpoint is a '{meta['model']}' checkpoint but --model is '{cfg.model}'.")
    cfg.validate()
    return cfg


def main(argv: Optional[list[str]] = None) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args_with_config(build_parser(), argv)
    return train(config_from_args(args))


if __name__ == "__main__":
    main()
