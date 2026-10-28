"""TP model registry: checkpoint metadata, validation, and adapter creation.

Wraps :mod:`pipeline.tp_adapters` (the actual, tested adapter interface used
by this app -- ``BaseTPAdapter.predict(observed_trajectories, num_modes,
**kwargs) -> np.ndarray``) with checkpoint-requirement metadata per model, so
the UI can show download instructions and validate checkpoint paths before
attempting to build an adapter.

This intentionally does NOT use a separate ``TrajectoryPredictorAdapter`` /
``PredictionResult`` dataclass API (as sketched in early planning docs);
that would duplicate/replace the adapter architecture that is already built,
tested and wired into the extraction pipeline and the TP Analysis tab.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from pipeline.tp_adapters import TP_ADAPTER_REGISTRY, BaseTPAdapter

import logging
logger = logging.getLogger(__name__)


def checkpoint_metadata_path(checkpoint_path: str | Path) -> Optional[Path]:
    """Return the sidecar metadata file for *checkpoint_path*, if one exists.

    The TP training pipeline (:mod:`training.tp.train`) writes ``<stem>.json``
    next to each checkpoint (e.g. ``best.pt`` -> ``best.json``);
    ``<checkpoint>.json`` (e.g. ``best.pt.json``) is accepted as well.
    """
    path = Path(checkpoint_path)
    for candidate in (path.with_suffix(".json"), Path(f"{path}.json")):
        if candidate.is_file():
            return candidate
    return None


def load_checkpoint_metadata(checkpoint_path: Optional[str | Path]) -> Optional[dict[str, Any]]:
    """Load the sidecar metadata of a checkpoint, or ``None`` if absent/unreadable."""
    if not checkpoint_path:
        return None
    meta_path = checkpoint_metadata_path(checkpoint_path)
    if meta_path is None:
        return None
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Ignoring unreadable checkpoint metadata '%s': %s", meta_path, exc)
        return None
    return metadata if isinstance(metadata, dict) else None


def checkpoint_settings_warnings(
    metadata: Optional[dict[str, Any]],
    *,
    obs_len: Optional[int] = None,
    pred_len: Optional[int] = None,
    fps: Optional[float] = None,
    model_name: Optional[str] = None,
) -> list[str]:
    """Compare current (UI) settings with the settings a checkpoint was trained with.

    Returns human-readable warnings for every mismatch of model name, frame
    rate, observation length or prediction length. Empty when *metadata* is
    ``None`` or everything matches.
    """
    if not metadata:
        return []
    warnings: list[str] = []
    trained_model = metadata.get("model")
    if model_name and trained_model and trained_model != model_name:
        warnings.append(f"checkpoint was trained for model '{trained_model}', not '{model_name}'")
    trained_fps = metadata.get("fps")
    if fps is not None and trained_fps is not None and abs(float(trained_fps) - float(fps)) > 1e-6:
        warnings.append(f"checkpoint was trained at {float(trained_fps):g} fps, current data/UI uses {float(fps):g} fps")
    for key, current in (("obs_len", obs_len), ("pred_len", pred_len)):
        trained = metadata.get(key)
        if current is not None and trained is not None and int(trained) != int(current):
            warnings.append(f"checkpoint was trained with {key}={int(trained)}, current setting is {int(current)}")
    return warnings


@dataclass
class ModelCheckpointInfo:
    """Metadata about a TP model's checkpoint requirements and provenance."""

    required: bool
    official_repo: str
    download_instructions: str
    default_config: dict[str, Any] = field(default_factory=dict)
    supported_configs: list[dict[str, Any]] = field(default_factory=list)
    checkpoint_format: str = "pytorch_state_dict"
    notes: str = ""


class TPModelRegistry:
    """Registry mapping TP model names to adapter classes + checkpoint metadata."""

    _models: dict[str, type[BaseTPAdapter]] = {}
    _checkpoint_info: dict[str, ModelCheckpointInfo] = {}

    @classmethod
    def register_model(
        cls,
        model_name: str,
        adapter_class: type[BaseTPAdapter],
        checkpoint_info: ModelCheckpointInfo,
    ) -> None:
        """Register a TP model with its adapter class and checkpoint metadata."""
        cls._models[model_name] = adapter_class
        cls._checkpoint_info[model_name] = checkpoint_info

    @classmethod
    def get_available_models(cls) -> list[str]:
        """Return the list of registered model names."""
        return list(cls._models.keys())

    @classmethod
    def get_model_class(cls, model_name: str) -> type[BaseTPAdapter]:
        """Return the adapter class registered for *model_name*."""
        if model_name not in cls._models:
            raise ValueError(f"Unknown model: {model_name}. Available models: {cls.get_available_models()}")
        return cls._models[model_name]

    @classmethod
    def get_checkpoint_info(cls, model_name: str) -> ModelCheckpointInfo:
        """Return checkpoint metadata registered for *model_name*."""
        if model_name not in cls._checkpoint_info:
            raise ValueError(f"Unknown model: {model_name}. Available models: {cls.get_available_models()}")
        return cls._checkpoint_info[model_name]

    @classmethod
    def create_adapter(
        cls,
        model_name: str,
        checkpoint_path: Optional[str] = None,
        **kwargs: Any,
    ) -> BaseTPAdapter:
        """Validate checkpoint requirements and instantiate the adapter for *model_name*.

        Args:
            model_name: Registered model name (e.g. ``"social_lstm"``).
            checkpoint_path: Optional path to a checkpoint file. Required for
                models whose :class:`ModelCheckpointInfo` has ``required=True``.
            **kwargs: Forwarded to the adapter constructor (e.g. ``pred_len``).

        Raises:
            ValueError: When a required checkpoint is missing.
            FileNotFoundError: When *checkpoint_path* does not exist.
        """
        adapter_class = cls.get_model_class(model_name)
        checkpoint_info = cls.get_checkpoint_info(model_name)

        if checkpoint_info.required and not checkpoint_path:
            raise ValueError(
                f"Model '{model_name}' requires a checkpoint file.\n\n"
                f"Download instructions:\n{checkpoint_info.download_instructions}\n\n"
                f"Official repository: {checkpoint_info.official_repo}"
            )

        if checkpoint_path:
            checkpoint_file = Path(checkpoint_path)
            if not checkpoint_file.exists():
                raise FileNotFoundError(
                    f"Checkpoint file not found: {checkpoint_path}\n"
                    f"Please verify the file path is correct."
                )
            metadata = load_checkpoint_metadata(checkpoint_path)
            if metadata is not None:
                trained_model = metadata.get("model")
                if trained_model and trained_model != model_name:
                    logger.warning(
                        "Checkpoint '%s' was trained for model '%s' but is loaded into '%s'.",
                        checkpoint_path,
                        trained_model,
                        model_name,
                    )
                else:
                    # Architecture kwargs recorded at training time (e.g.
                    # hidden_dim, use_social_pooling) are applied as defaults
                    # so the state_dict matches; explicit kwargs still win.
                    for key, value in (metadata.get("model_kwargs") or {}).items():
                        kwargs.setdefault(key, value)

        try:
            return adapter_class(checkpoint=checkpoint_path, **kwargs)
        except Exception as exc:
            raise type(exc)(
                f"Failed to create model '{model_name}': {exc}\n\n"
                f"Checkpoint: {checkpoint_path}\n"
                f"Official repo: {checkpoint_info.official_repo}"
            ) from exc

    @classmethod
    def initialize_builtin_models(cls) -> None:
        """Register all TP adapters known to :mod:`pipeline.tp_adapters`."""
        cls.register_model(
            model_name="dummy",
            adapter_class=TP_ADAPTER_REGISTRY["dummy"],
            checkpoint_info=ModelCheckpointInfo(
                required=False,
                official_repo="",
                download_instructions="",
                default_config={"obs_len": 20, "pred_len": 30, "num_modes": 1},
                notes="Constant-velocity baseline. No checkpoint required or supported.",
            ),
        )

        cls.register_model(
            model_name="social_lstm",
            adapter_class=TP_ADAPTER_REGISTRY["social_lstm"],
            checkpoint_info=ModelCheckpointInfo(
                required=False,
                official_repo="https://github.com/quancore/social-lstm (unofficial PyTorch reference)",
                download_instructions=(
                    "This adapter is a from-scratch reimplementation of the Social-LSTM "
                    "idea (see pipeline.tp_models.SocialLSTMNet), not the original "
                    "authors' code, so no official pretrained checkpoint is compatible. "
                    "Train your own checkpoint with the TP training pipeline "
                    "(python -m training.tp.train --model social_lstm ..., see "
                    "docs/TP_TRAINING.md) and pass its best.pt path. The sidecar "
                    "best.json (fps/obs_len/pred_len/architecture) is read automatically."
                ),
                default_config={
                    "obs_len": 20,
                    "pred_len": 30,
                    "frame_rate_hz": 10.0,
                    "embedding_dim": 64,
                    "hidden_dim": 64,
                    "max_modes": 20,
                    "num_modes": 5,
                    "use_social_pooling": False,
                },
                checkpoint_format="pytorch_state_dict",
                notes=(
                    "Runs with random weights (architecture validation only) when no checkpoint is given. "
                    "Optional Social-GAN-style social pooling (use_social_pooling=True, read from the "
                    "checkpoint's sidecar JSON) pools all pedestrians of the current frame together; "
                    "without it each pedestrian is predicted independently."
                ),
            ),
        )

        cls.register_model(
            model_name="social_gan",
            adapter_class=TP_ADAPTER_REGISTRY["social_gan"],
            checkpoint_info=ModelCheckpointInfo(
                required=False,
                official_repo="https://github.com/agrimgupta92/sgan",
                download_instructions=(
                    "The official Social-GAN checkpoints (eth_12.pt, hotel_12.pt, ...) are "
                    "trained with a different network (their sgan.models.SocialGAN) and "
                    "will NOT load into this repo's from-scratch reimplementation "
                    "(pipeline.tp_models.SocialGANNet) -- the state_dict keys won't match. "
                    "To use the official checkpoints you would need to vendor their model "
                    "code; to use this adapter's checkpoint loading, train your own "
                    "SocialGANNet with the TP training pipeline (python -m training.tp.train "
                    "--model social_gan ..., see docs/TP_TRAINING.md) and pass its best.pt "
                    "(generator-only state_dict) path."
                ),
                default_config={
                    "obs_len": 20,
                    "pred_len": 30,
                    "frame_rate_hz": 10.0,
                    "embedding_dim": 64,
                    "hidden_dim": 64,
                    "noise_dim": 8,
                    "num_modes": 20,
                    "use_social_pooling": False,
                },
                checkpoint_format="pytorch_state_dict",
                notes=(
                    "Runs with random weights (architecture validation only) when no checkpoint is given. "
                    "Optional Social-GAN-style social pooling (use_social_pooling=True, read from the "
                    "checkpoint's sidecar JSON) pools all pedestrians of the current frame together; "
                    "without it each pedestrian is predicted independently."
                ),
            ),
        )

        cls.register_model(
            model_name="transformer",
            adapter_class=TP_ADAPTER_REGISTRY["transformer"],
            checkpoint_info=ModelCheckpointInfo(
                required=False,
                official_repo="",
                download_instructions=(
                    "Custom transformer-encoder architecture (pipeline.tp_models."
                    "TransformerTPNet), not based on a specific published checkpoint. "
                    "Train your own and pass its state_dict path."
                ),
                default_config={"obs_len": 20, "pred_len": 30, "num_modes": 10, "d_model": 128, "nhead": 8, "num_layers": 2},
                checkpoint_format="pytorch_state_dict",
                notes="Runs with random weights (architecture validation only) when no checkpoint is given.",
            ),
        )

        cls.register_model(
            model_name="social_stgcnn",
            adapter_class=TP_ADAPTER_REGISTRY["social_stgcnn"],
            checkpoint_info=ModelCheckpointInfo(
                required=True,
                official_repo="https://github.com/abduallahmohamed/Social-STGCNN",
                download_instructions=(
                    "1. git clone https://github.com/abduallahmohamed/Social-STGCNN.git\n"
                    "2. Pretrained checkpoints ship in that repo's checkpoint/ directory "
                    "(one per ETH/UCY scene: eth, hotel, univ, zara1, zara2).\n"
                    "3. NOT usable yet: this repo does not vendor the ST-GCNN/TXP-CNN "
                    "model code, so selecting this model raises NotImplementedError "
                    "regardless of checkpoint path."
                ),
                default_config={"obs_len": 8, "pred_len": 12, "frame_rate_hz": 2.5, "coordinate_system": "relative_meters"},
                checkpoint_format="pytorch_full_model",
                notes="Not implemented in this repository yet (see pipeline.tp_adapters.SocialSTGCNNAdapter).",
            ),
        )

        cls.register_model(
            model_name="trajectron_pp",
            adapter_class=TP_ADAPTER_REGISTRY["trajectron_pp"],
            checkpoint_info=ModelCheckpointInfo(
                required=True,
                official_repo="https://github.com/StanfordASL/Trajectron-plus-plus",
                download_instructions=(
                    "1. git clone https://github.com/StanfordASL/Trajectron-plus-plus.git\n"
                    "2. Follow its README setup; pretrained models live under "
                    "experiments/<dataset>/models/ as a directory (config.json + checkpoint).\n"
                    "3. NOT usable yet: this repo does not vendor Trajectron++'s Scene/Node "
                    "pipeline or CVAE decoder, so selecting this model raises "
                    "NotImplementedError regardless of checkpoint path."
                ),
                default_config={"obs_len": 8, "pred_len": 12, "frame_rate_hz": 10.0, "coordinate_system": "absolute_meters", "num_modes": 5},
                checkpoint_format="model_directory",
                notes="Not implemented in this repository yet (see pipeline.tp_adapters.TrajectronPPAdapter).",
            ),
        )


TPModelRegistry.initialize_builtin_models()
