# Task: Implement TP Model Registry, Checkpoint Management, and Neural Model Adapters

## Context
The "TP Analysis" tab is already implemented with:
- CSV loading and validation
- PedPy overview plot (left panel)
- Interactive prediction preview with slider (right panel)
- UI controls for model selection, modes, prediction length
- ConstantVelocityAdapter (dummy baseline, no checkpoint required)

Now I need to extend this with:
1. Proper model registry with checkpoint validation
2. Real neural TP model adapters (Social GAN, Social STGCNN, Trajectron++)
3. Checkpoint download helpers and documentation
4. Error handling for missing/incompatible checkpoints
5. Model metadata and configuration management

## Critical: Checkpoint Handling

Unlike YOLOv8 detection models where `.pt` files are standardized, TP model checkpoints:
- Require specific model architecture code from official repositories
- Have repository-specific formats (state_dict, model, etc.)
- Depend on exact training configurations (obs_len, pred_len, frame_rate, normalization)
- May include optimizer state, epoch info, hyperparameters
- Are NOT interchangeable between implementations

**DO:**
- Document exact official repository for each model
- Validate checkpoint format before loading
- Raise clear errors with download instructions when checkpoint is missing
- Support model-specific configuration requirements
- Store metadata about checkpoint source and training setup

**DO NOT:**
- Assume generic checkpoint filenames exist
- Create fake checkpoint files
- Use checkpoints without documenting their source
- Ignore configuration mismatches (obs_len, pred_len, normalization)

## Requirements

### 1. Model Registry with Checkpoint Validation
Create `pedestrian_analysis/pipeline/tp_model_registry.py`:

```python
from typing import Dict, Type, Optional, List, Any
from pathlib import Path
from dataclasses import dataclass, field
from .tp_adapters import (
    TrajectoryPredictorAdapter,
    ConstantVelocityAdapter,
)


@dataclass
class ModelCheckpointInfo:
    """Metadata about a TP model checkpoint."""
    required: bool
    official_repo: str
    download_instructions: str
    default_config: Dict[str, Any]
    supported_configs: List[Dict[str, Any]] = field(default_factory=list)
    checkpoint_format: str = "pytorch_state_dict"  # pytorch_state_dict, tensorflow, etc.
    notes: str = ""


class TPModelRegistry:
    """
    Registry for trajectory prediction models with checkpoint management.
    
    Manages:
    - Available models and their adapter classes
    - Checkpoint requirements and download instructions
    - Default and supported configurations
    - Checkpoint validation
    """

    _models: Dict[str, Type[TrajectoryPredictorAdapter]] = {}
    _checkpoint_info: Dict[str, ModelCheckpointInfo] = {}

    @classmethod
    def register_model(
        cls,
        model_name: str,
        adapter_class: Type[TrajectoryPredictorAdapter],
        checkpoint_info: ModelCheckpointInfo,
    ) -> None:
        """
        Register a TP model with its adapter and checkpoint info.
        
        Args:
            model_name: Unique identifier for the model
            adapter_class: Adapter class implementing TrajectoryPredictorAdapter
            checkpoint_info: Metadata about checkpoint requirements
        """
        cls._models[model_name] = adapter_class
        cls._checkpoint_info[model_name] = checkpoint_info

    @classmethod
    def get_available_models(cls) -> List[str]:
        """Return list of available model names."""
        return list(cls._models.keys())

    @classmethod
    def get_model_class(cls, model_name: str) -> Type[TrajectoryPredictorAdapter]:
        """Get adapter class for model name."""
        if model_name not in cls._models:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available models: {cls.get_available_models()}"
            )
        return cls._models[model_name]

    @classmethod
    def get_checkpoint_info(cls, model_name: str) -> ModelCheckpointInfo:
        """Get checkpoint requirements for a model."""
        if model_name not in cls._checkpoint_info:
            raise ValueError(f"Unknown model: {model_name}")
        return cls._checkpoint_info[model_name]

    @classmethod
    def create_adapter(
        cls,
        model_name: str,
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: str = "cpu",
        **kwargs,
    ) -> TrajectoryPredictorAdapter:
        """
        Create and initialize a TP adapter with proper checkpoint validation.
        
        Args:
            model_name: Name of the model
            checkpoint_path: Path to checkpoint file
            config_path: Path to config file (optional)
            device: Device for inference ("cpu" or "cuda")
            **kwargs: Additional model-specific arguments
        
        Returns:
            Initialized adapter instance
        
        Raises:
            ValueError: If checkpoint required but not provided, or if config mismatch
            FileNotFoundError: If checkpoint file doesn't exist
            ImportError: If model dependencies not installed
        """
        adapter_class = cls.get_model_class(model_name)
        checkpoint_info = cls.get_checkpoint_info(model_name)

        # Check checkpoint requirement
        if checkpoint_info.required and checkpoint_path is None:
            raise ValueError(
                f"Model '{model_name}' requires a checkpoint file.\n\n"
                f"Download instructions:\n{checkpoint_info.download_instructions}\n\n"
                f"Official repository: {checkpoint_info.official_repo}"
            )

        # Validate checkpoint file exists
        if checkpoint_path:
            checkpoint_file = Path(checkpoint_path)
            if not checkpoint_file.exists():
                raise FileNotFoundError(
                    f"Checkpoint file not found: {checkpoint_path}\n\n"
                    f"Please verify the file path is correct."
                )

        # Create adapter
        adapter = adapter_class()

        # Load model with checkpoint
        try:
            adapter.load_model(
                checkpoint_path=checkpoint_path,
                config_path=config_path,
                device=device,
                **kwargs,
            )
        except Exception as e:
            raise type(e)(
                f"Failed to load model '{model_name}': {str(e)}\n\n"
                f"Checkpoint: {checkpoint_path}\n"
                f"Official repo: {checkpoint_info.official_repo}"
            ) from e

        return adapter

    @classmethod
    def initialize_builtin_models(cls) -> None:
        """Register all built-in TP models."""
        from .tp_adapters import (
            ConstantVelocityAdapter,
            SocialGANAdapter,
            SocialSTGCNNAdapter,
            TrajectronPPAdapter,
        )

        # Constant Velocity (no checkpoint)
        cls.register_model(
            model_name="constant_velocity",
            adapter_class=ConstantVelocityAdapter,
            checkpoint_info=ModelCheckpointInfo(
                required=False,
                official_repo="",
                download_instructions="",
                default_config={},
                notes="Baseline model using constant velocity extrapolation. No checkpoint required.",
            ),
        )

        # Social GAN
        cls.register_model(
            model_name="social_gan",
            adapter_class=SocialGANAdapter,
            checkpoint_info=ModelCheckpointInfo(
                required=True,
                official_repo="[https://github.com/agrimgupta92/sgan](https://github.com/agrimgupta92/sgan)",
                download_instructions=(
                    "1. Clone repository:\n"
                    "   git clone [https://github.com/agrimgupta92/sgan.git](https://github.com/agrimgupta92/sgan.git)\n"
                    "2. Download pretrained models:\n"
                    "   cd sgan && bash scripts/download_models.sh\n"
                    "3. Checkpoints will be in sgan-models/ directory\n"
                    "4. Select checkpoint file (e.g., eth_12.pt, hotel_12.pt)"
                ),
                default_config={
                    "obs_len": 8,
                    "pred_len": 12,
                    "frame_rate_hz": 2.5,
                    "embedding_dim": 64,
                    "num_modes": 20,
                    "coordinate_system": "relative_meters",
                },
                supported_configs=[
                    {"obs_len": 8, "pred_len": 12},
                    {"obs_len": 8, "pred_len": 20},
                ],
                notes=(
                    "Social GAN trained on ETH/UCY datasets. "
                    "Models are dataset-specific (eth, hotel, zara, etc.). "
                    "Frame rate is 2.5 Hz. Coordinates are relative meters."
                ),
            ),
        )

        # Social STGCNN
        cls.register_model(
            model_name="social_stgcnn",
            adapter_class=SocialSTGCNNAdapter,
            checkpoint_info=ModelCheckpointInfo(
                required=True,
                official_repo="[https://github.com/abduallahmohamed/Social-STGCNN](https://github.com/abduallahmohamed/Social-STGCNN)",
                download_instructions=(
                    "1. Clone repository:\n"
                    "   git clone [https://github.com/abduallahmohamed/Social-STGCNN.git](https://github.com/abduallahmohamed/Social-STGCNN.git)\n"
                    "2. Pretrained models are included in checkpoint/ directory\n"
                    "3. Models available for ETH, HOTEL, UNIV, ZARA1, ZARA2 datasets\n"
                    "4. Select appropriate checkpoint file"
                ),
                default_config={
                    "obs_len": 8,
                    "pred_len": 12,
                    "frame_rate_hz": 2.5,
                    "coordinate_system": "relative_meters",
                },
                supported_configs=[
                    {"obs_len": 8, "pred_len": 12},
                ],
                notes=(
                    "Social-STGCNN uses spatio-temporal graph convolutions. "
                    "Pretrained models available for ETH and UCY datasets. "
                    "Model consists of ST-GCNN and TXP-CNN components."
                ),
            ),
        )

        # Trajectron++
        cls.register_model(
            model_name="trajectron_pp",
            adapter_class=TrajectronPPAdapter,
            checkpoint_info=ModelCheckpointInfo(
                required=True,
                official_repo="[https://github.com/StanfordASL/Trajectron-plus-plus](https://github.com/StanfordASL/Trajectron-plus-plus)",
                download_instructions=(
                    "1. Clone repository:\n"
                    "   git clone [https://github.com/StanfordASL/Trajectron-plus-plus.git](https://github.com/StanfordASL/Trajectron-plus-plus.git)\n"
                    "2. Follow setup instructions in README\n"
                    "3. Pretrained models available for nuScenes, ETH, etc.\n"
                    "4. Models stored in experiments/<dataset>/models/ directory"
                ),
                default_config={
                    "obs_len": 8,
                    "pred_len": 12,
                    "frame_rate_hz": 10.0,
                    "coordinate_system": "absolute_meters",
                    "num_modes": 5,
                },
                supported_configs=[
                    {"obs_len": 8, "pred_len": 12},
                    {"obs_len": 8, "pred_len": 24},
                ],
                notes=(
                    "Trajectron++ supports heterogeneous agents and dynamic graphs. "
                    "More complex setup than Social GAN/STGCNN. "
                    "Requires model directory with config.json and checkpoint files."
                ),
            ),
        )


# Initialize builtin models on module load
TPModelRegistry.initialize_builtin_models()
```

### 2. Social GAN Adapter Implementation
Create/extend `pedestrian_analysis/pipeline/tp_adapters.py` with Social GAN:

```python
import torch
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path
from .tp_adapters import TrajectoryPredictorAdapter, PredictionResult
import time


class SocialGANAdapter(TrajectoryPredictorAdapter):
    """
    Adapter for Social GAN trajectory prediction model.
    
    Official repository: [https://github.com/agrimgupta92/sgan](https://github.com/agrimgupta92/sgan)
    
    Architecture:
    - Sequence-to-sequence LSTM encoder-decoder
    - Social pooling layer for interaction modeling
    - GAN discriminator for realistic trajectory generation
    - Multimodal predictions via noise injection
    
    Training configuration:
    - obs_len: typically 8 frames
    - pred_len: typically 12 or 20 frames
    - frame_rate: 2.5 Hz (ETH/UCY datasets)
    - coordinates: relative meters (normalized)
    - datasets: ETH, HOTEL, UNIV, ZARA1, ZARA2
    
    Checkpoint format:
    - PyTorch state_dict
    - May include: state_dict, model, optimizer_state_dict, epoch
    - Dataset-specific models (eth_12.pt, hotel_12.pt, etc.)
    """

    def __init__(self):
        self.model = None
        self.device = "cpu"
        self.config = None
        self._model_loaded = False

    @property
    def model_name(self) -> str:
        return "social_gan"

    @property
    def requires_checkpoint(self) -> bool:
        return True

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "obs_len": 8,
            "pred_len": 12,
            "frame_rate_hz": 2.5,
            "embedding_dim": 64,
            "num_modes": 20,
            "coordinate_system": "relative_meters",
            "normalization": "dataset_specific",
        }

    def load_model(
        self,
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        if checkpoint_path is None:
            raise ValueError(
                f"{self.model_name} requires a checkpoint file. "
                f"Download from: [https://github.com/agrimgupta92/sgan](https://github.com/agrimgupta92/sgan)"
            )

        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            raise FileNotFoundError(
                f"Checkpoint file not found: {checkpoint_path}. "
                f"Please download Social GAN checkpoints from the official repository."
            )

        self.device = torch.device(device)

        # Load configuration
        self.config = self.get_default_config()
        if config_path:
            config_file = Path(config_path)
            if config_file.exists():
                import json
                with open(config_file, 'r') as f:
                    self.config.update(json.load(f))

        # Import model architecture from official repo
        # NOTE: This requires the sgan repository to be in Python path
        try:
            # Option 1: Import from installed sgan package
            from sgan.models import SocialGAN
            self.model = SocialGAN(
                obs_len=self.config["obs_len"],
                pred_len=self.config["pred_len"],
                embedding_dim=self.config["embedding_dim"],
            )
        except ImportError:
            try:
                # Option 2: Import from local clone
                import sys
                sgan_path = Path(__file__).parent.parent.parent / "external" / "sgan"
                if sgan_path.exists():
                    sys.path.insert(0, str(sgan_path))
                from sgan.models import SocialGAN
                self.model = SocialGAN(
                    obs_len=self.config["obs_len"],
                    pred_len=self.config["pred_len"],
                    embedding_dim=self.config["embedding_dim"],
                )
            except ImportError as e:
                raise ImportError(
                    f"Social GAN repository not found. "
                    f"Install from: [https://github.com/agrimgupta92/sgan](https://github.com/agrimgupta92/sgan)\n\n"
                    f"Either:\n"
                    f"1. pip install sgan (if available)\n"
                    f"2. Clone repo to external/sgan/ directory\n"
                    f"3. Add sgan to PYTHONPATH"
                ) from e

        # Load checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load checkpoint {checkpoint_path}: {str(e)}"
            ) from e
        
        # Handle different checkpoint formats
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "generator" in checkpoint:
            # Some checkpoints store generator separately
            state_dict = checkpoint["generator"]
        else:
            # Assume checkpoint is the state_dict itself
            state_dict = checkpoint

        # Load weights
        try:
            self.model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load state_dict: {str(e)}\n"
                f"Checkpoint may be incompatible with this model architecture."
            ) from e

        self.model.to(self.device)
        self.model.eval()
        self._model_loaded = True

    def predict(
        self,
        positions: np.ndarray,  # (N, obs_len, 2)
        track_ids: np.ndarray,  # (N,)
        current_frame: int,
        frame_rate: float,
        num_modes: int = 5,
        pred_len: int = 30,
    ) -> PredictionResult:
        start_time = time.time()

        if not self._model_loaded:
            raise RuntimeError(f"Model not loaded. Call load_model() first.")

        # Validate input dimensions match training config
        if positions.shape != self.config["obs_len"]:
            raise ValueError(
                f"Input observation length ({positions.shape}) doesn't match "
                f"model training configuration (obs_len={self.config['obs_len']}).\n\n"
                f"Social GAN was trained with fixed observation length. "
                f"Either adjust your observation window to {self.config['obs_len']} frames "
                f"or retrain the model with different settings."
            )

        if pred_len != self.config["pred_len"]:
            print(
                f"Warning: Requested pred_len={pred_len} but model was trained with "
                f"pred_len={self.config['pred_len']}. Predictions may be unreliable."
            )

        N = positions.shape

        # Preprocess: convert to Social GAN input format
        # Social GAN expects: (seq_len, batch, 2)
        model_input = torch.from_numpy(positions).float()  # (N, obs_len, 2)
        model_input = model_input.permute(1, 0, 2)  # (obs_len, N, 2)
        model_input = model_input.to(self.device)

        # Run prediction
        with torch.no_grad():
            try:
                # Social GAN forward pass
                # Returns: (pred_len, batch, 2) or list of predictions
                pred_output = self.model(
                    model_input,
                    num_samples=num_modes,
                    pred_len=pred_len,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Social GAN inference failed: {str(e)}\n"
                    f"Input shape: {model_input.shape}, "
                    f"Expected: ({self.config['obs_len']}, N, 2)"
                ) from e

        # Postprocess: convert back to (N, num_modes, pred_len, 2)
        if isinstance(pred_output, (list, tuple)):
            # Multiple predictions returned
            pred_output = torch.stack(pred_output, dim=0)  # (num_modes, pred_len, N, 2)
            pred_output = pred_output.permute(2, 0, 1, 3)  # (N, num_modes, pred_len, 2)
        else:
            # Single prediction tensor
            if pred_output.dim() == 3:
                # (pred_len, batch, 2) - unimodal
                pred_output = pred_output.unsqueeze(1)  # (pred_len, 1, batch, 2)
                pred_output = pred_output.permute(2, 1, 0, 3)  # (N, 1, pred_len, 2)
                pred_output = pred_output.expand(N, num_modes, pred_len, 2)
            elif pred_output.dim() == 4:
                # Already multimodal
                pred_output = pred_output.permute(2, 0, 1, 3)  # (N, num_modes, pred_len, 2)
            else:
                raise ValueError(f"Unexpected prediction output shape: {pred_output.shape}")

        predictions = pred_output.cpu().numpy()

        # Social GAN doesn't output mode probabilities - use uniform
        probabilities = np.ones((N, num_modes)) / num_modes

        inference_time_ms = (time.time() - start_time) * 1000

        return PredictionResult(
            track_ids=track_ids,
            origin_frame=current_frame,
            future_frames=np.arange(current_frame + 1, current_frame + pred_len + 1),
            predictions=predictions,
            probabilities=probabilities,
            model_name=self.model_name,
            inference_time_ms=inference_time_ms,
            metadata={
                "config": self.config,
                "checkpoint_format": "pytorch_state_dict",
            },
        )
```

### 3. Social STGCNN Adapter
Add to `pedestrian_analysis/pipeline/tp_adapters.py`:

```python
class SocialSTGCNNAdapter(TrajectoryPredictorAdapter):
    """
    Adapter for Social-STGCNN trajectory prediction model.
    
    Official repository: [https://github.com/abduallahmohamed/Social-STGCNN](https://github.com/abduallahmohamed/Social-STGCNN)
    
    Architecture:
    - ST-GCNN: Spatio-temporal graph convolutional network
    - TXP-CNN: Time extrapolator CNN for future prediction
    - Graph-based interaction modeling
    - Non-generative (unimodal or simple multimodal)
    
    Training configuration:
    - obs_len: 8 frames
    - pred_len: 12 frames
    - frame_rate: 2.5 Hz (ETH/UCY datasets)
    - coordinates: relative meters
    
    Checkpoint format:
    - PyTorch model files in checkpoint/ directory
    - Dataset-specific: eth.pth, hotel.pth, univ.pth, zara1.pth, zara2.pth
    - Contains complete model state
    """

    def __init__(self):
        self.model = None
        self.device = "cpu"
        self.config = None
        self._model_loaded = False

    @property
    def model_name(self) -> str:
        return "social_stgcnn"

    @property
    def requires_checkpoint(self) -> bool:
        return True

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "obs_len": 8,
            "pred_len": 12,
            "frame_rate_hz": 2.5,
            "coordinate_system": "relative_meters",
            "graph_radius": 3.0,  # meters
        }

    def load_model(
        self,
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        if checkpoint_path is None:
            raise ValueError(
                f"{self.model_name} requires a checkpoint file. "
                f"Download from: [https://github.com/abduallahmohamed/Social-STGCNN](https://github.com/abduallahmohamed/Social-STGCNN)"
            )

        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            raise FileNotFoundError(
                f"Checkpoint file not found: {checkpoint_path}"
            )

        self.device = torch.device(device)
        self.config = self.get_default_config()

        # Import model architecture
        try:
            import sys
            stgcnn_path = Path(__file__).parent.parent.parent / "external" / "Social-STGCNN"
            if stgcnn_path.exists():
                sys.path.insert(0, str(stgcnn_path))
            
            from models.social_stgcnn import SocialSTGCNN
            self.model = SocialSTGCNN(
                obs_len=self.config["obs_len"],
                pred_len=self.config["pred_len"],
            )
        except ImportError as e:
            raise ImportError(
                f"Social-STGCNN repository not found. "
                f"Clone from: [https://github.com/abduallahmohamed/Social-STGCNN](https://github.com/abduallahmohamed/Social-STGCNN)\n\n"
                f"Add to external/Social-STGCNN/ or install separately."
            ) from e

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self._model_loaded = True

    def predict(
        self,
        positions: np.ndarray,
        track_ids: np.ndarray,
        current_frame: int,
        frame_rate: float,
        num_modes: int = 5,
        pred_len: int = 30,
    ) -> PredictionResult:
        start_time = time.time()

        if not self._model_loaded:
            raise RuntimeError("Model not loaded")

        if positions.shape != self.config["obs_len"]:
            raise ValueError(
                f"Expected obs_len={self.config['obs_len']}, "
                f"got {positions.shape}"
            )

        N = positions.shape

        # Convert to tensor: (batch, obs_len, 2)
        model_input = torch.from_numpy(positions).float().to(self.device)

        with torch.no_grad():
            # Social-STGCNN forward pass
            # Returns: (batch, pred_len, 2)
            pred_output = self.model(model_input)

        # Handle output format
        if pred_output.dim() == 2:
            # Unimodal: (batch, pred_len * 2)
            pred_output = pred_output.view(N, self.config["pred_len"], 2)
            pred_output = pred_output.unsqueeze(1)  # (N, 1, pred_len, 2)
            pred_output = pred_output.expand(N, num_modes, pred_len, 2)
        elif pred_output.dim() == 3:
            # (batch, pred_len, 2)
            pred_output = pred_output.unsqueeze(1)  # (N, 1, pred_len, 2)
            pred_output = pred_output.expand(N, num_modes, pred_len, 2)

        predictions = pred_output.cpu().numpy()
        probabilities = np.ones((N, num_modes)) / num_modes

        inference_time_ms = (time.time() - start_time) * 1000

        return PredictionResult(
            track_ids=track_ids,
            origin_frame=current_frame,
            future_frames=np.arange(current_frame + 1, current_frame + pred_len + 1),
            predictions=predictions,
            probabilities=probabilities,
            model_name=self.model_name,
            inference_time_ms=inference_time_ms,
            metadata={"config": self.config},
        )
```

### 4. Trajectron++ Adapter
Add to `pedestrian_analysis/pipeline/tp_adapters.py`:

```python
class TrajectronPPAdapter(TrajectoryPredictorAdapter):
    """
    Adapter for Trajectron++ trajectory prediction model.
    
    Official repository: [https://github.com/StanfordASL/Trajectron-plus-plus](https://github.com/StanfordASL/Trajectron-plus-plus)
    
    Architecture:
    - Encoder: LSTM/GRU for temporal encoding
    - Graph attention for spatial interactions
    - CVAE decoder for multimodal predictions
    - Supports heterogeneous agents (pedestrians, vehicles, etc.)
    - Dynamically feasible trajectory generation
    
    Training configuration:
    - obs_len: 8 frames (configurable)
    - pred_len: 12 or 24 frames
    - frame_rate: 10 Hz (nuScenes) or 2.5 Hz (ETH/UCY)
    - coordinates: absolute or relative meters
    
    Checkpoint format:
    - Model directory with config.json and checkpoint files
    - More complex than Social GAN/STGCNN
    - Requires loading both model architecture and hyperparameters
    """

    def __init__(self):
        self.model = None
        self.device = "cpu"
        self.config = None
        self._model_loaded = False

    @property
    def model_name(self) -> str:
        return "trajectron_pp"

    @property
    def requires_checkpoint(self) -> bool:
        return True

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "obs_len": 8,
            "pred_len": 12,
            "frame_rate_hz": 10.0,
            "coordinate_system": "absolute_meters",
            "num_modes": 5,
            "node_type": "PEDESTRIAN",
        }

    def load_model(
        self,
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        if checkpoint_path is None:
            raise ValueError(
                f"{self.model_name} requires a model directory. "
                f"Download from: [https://github.com/StanfordASL/Trajectron-plus-plus](https://github.com/StanfordASL/Trajectron-plus-plus)"
            )

        model_dir = Path(checkpoint_path)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {checkpoint_path}"
            )

        self.device = torch.device(device)

        # Load config from model directory
        config_file = model_dir / "config.json"
        if config_file.exists():
            import json
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = self.get_default_config()

        # Import Trajectron++
        try:
            import sys
            trajpp_path = Path(__file__).parent.parent.parent / "external" / "Trajectron-plus-plus"
            if trajpp_path.exists():
                sys.path.insert(0, str(trajpp_path))
            
            from trajectron.model.model import Model
            from trajectron.model.components.utils import load_model
            
            # Load model from directory
            self.model = load_model(
                str(model_dir),
                device=self.device,
            )
        except ImportError as e:
            raise ImportError(
                f"Trajectron++ repository not found. "
                f"Clone from: [https://github.com/StanfordASL/Trajectron-plus-plus](https://github.com/StanfordASL/Trajectron-plus-plus)\n\n"
                f"Add to external/Trajectron-plus-plus/ or install separately."
            ) from e

        self.model.eval()
        self._model_loaded = True

    def predict(
        self,
        positions: np.ndarray,
        track_ids: np.ndarray,
        current_frame: int,
        frame_rate: float,
        num_modes: int = 5,
        pred_len: int = 30,
    ) -> PredictionResult:
        start_time = time.time()

        if not self._model_loaded:
            raise RuntimeError("Model not loaded")

        N = positions.shape

        # Trajectron++ requires specific input format (Scene, Node, etc.)
        # This is a simplified wrapper - full implementation needs proper data structures
        
        # For now, raise NotImplementedError with instructions
        raise NotImplementedError(
            f"Trajectron++ adapter requires complex input formatting.\n\n"
            f"See official documentation: "
            f"[https://github.com/StanfordASL/Trajectron-plus-plus](https://github.com/StanfordASL/Trajectron-plus-plus)\n\n"
            f"Input must be converted to Trajectron Scene/Node format with:\n"
            f"- Dynamic state (position, velocity, etc.)\n"
            f"- Static state (map information)\n"
            f"- Control inputs (if available)\n\n"
            f"This adapter is a placeholder for future implementation."
        )
```

### 5. Update UI Controls to Use Registry
Update `pedestrian_analysis/ui/widgets/tp_controls.py`:

```python
from ...pipeline.tp_model_registry import TPModelRegistry, ModelCheckpointInfo


class TPControlPanel(ttk.Frame):
    """Control panel for trajectory prediction settings."""

    def __init__(
        self,
        parent,
        on_model_change: Callable[[str], None],
        on_predict: Callable[[], None],
        **kwargs,
    ):
        super().__init__(parent, **kwargs)
        self.on_model_change = on_model_change
        self.on_predict = on_predict

        self._create_widgets()
        self._update_model_info()

    def _create_widgets(self):
        # Model selection
        ttk.Label(self, text="TP Model:").grid(
            row=0, column=0, padx=5, pady=5, sticky="w"
        )

        available_models = TPModelRegistry.get_available_models()
        self.model_var = tk.StringVar(value="constant_velocity")
        self.model_combo = ttk.Combobox(
            self,
            textvariable=self.model_var,
            values=available_models,
            state="readonly",
            width=20,
        )
        self.model_combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_changed)

        # ... rest of widgets ...

        # Checkpoint requirements label
        self.checkpoint_info_text = tk.Text(
            self,
            wrap="word",
            height=6,
            width=50,
            state="disabled",
            font=("TkDefaultFont", 9),
        )
        self.checkpoint_info_text.grid(
            row=8, column=0, columnspan=3, padx=5, pady=5, sticky="ew"
        )

    def _on_model_changed(self, event=None):
        model_name = self.model_var.get()
        self.on_model_change(model_name)
        self._update_model_info()

    def _update_model_info(self):
        model_name = self.model_var.get()
        
        try:
            info = TPModelRegistry.get_checkpoint_info(model_name)
        except ValueError:
            self.checkpoint_info_text.config(state="normal")
            self.checkpoint_info_text.delete("1.0", tk.END)
            self.checkpoint_info_text.insert(
                tk.END,
                f"Model '{model_name}' not registered in TPModelRegistry."
            )
            self.checkpoint_info_text.config(state="disabled")
            return

        # Update checkpoint requirements display
        self.checkpoint_info_text.config(state="normal")
        self.checkpoint_info_text.delete("1.0", tk.END)

        if info.required:
            self.checkpoint_info_text.insert(
                tk.END,
                f"✓ Checkpoint Required\n\n"
                f"Official Repository:\n{info.official_repo}\n\n"
                f"Download Instructions:\n{info.download_instructions}\n\n"
                f"Notes: {info.notes}"
            )
            self.checkpoint_entry.config(state="normal")
            self.browse_btn.config(state="normal")
        else:
            self.checkpoint_info_text.insert(
                tk.END,
                f"✓ No Checkpoint Required\n\n"
                f"{info.notes}"
            )
            self.checkpoint_entry.config(state="disabled")
            self.browse_btn.config(state="disabled")

        self.checkpoint_info_text.config(state="disabled")

    def get_settings(self) -> dict:
        """Get current TP settings."""
        return {
            "model_name": self.model_var.get(),
            "checkpoint_path": self.checkpoint_var.get() or None,
            "num_modes": self.num_modes_var.get(),
            "pred_len": self.pred_len_var.get(),
            "obs_len": self.obs_len_var.get(),
            "show_all_modes": self.show_all_modes_var.get(),
        }
```

### 6. Checkpoint Download Helper Script
Create `pedestrian_analysis/scripts/download_tp_models.py`:

```python
#!/usr/bin/env python3
"""
Helper script to download pretrained TP model checkpoints.

Usage:
    python -m pedestrian_analysis.scripts.download_tp_models \
        --model social_gan \
        --output-dir models/trajectory_prediction/

    python -m pedestrian_analysis.scripts.download_tp_models \
        --model social_stgcnn \
        --output-dir models/trajectory_prediction/
"""

import argparse
import subprocess
import shutil
from pathlib import Path
import sys


def download_social_gan(output_dir: Path) -> None:
    """Download Social GAN checkpoints."""
    print("Downloading Social GAN models...")
    
    # Clone repository
    sgan_dir = output_dir / "sgan_temp"
    if sgan_dir.exists():
        shutil.rmtree(sgan_dir)
    
    subprocess.run(
        ["git", "clone", "[https://github.com/agrimgupta92/sgan.git](https://github.com/agrimgupta92/sgan.git)", str(sgan_dir)],
        check=True,
    )
    
    # Run download script
    subprocess.run(
        ["bash", "scripts/download_models.sh"],
        cwd=sgan_dir,
        check=True,
    )
    
    # Copy checkpoints to output directory
    models_dir = output_dir / "social_gan"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    sgan_models = sgan_dir / "sgan-models"
    if sgan_models.exists():
        for checkpoint in sgan_models.glob("*.pt"):
            shutil.copy2(checkpoint, models_dir / checkpoint.name)
        print(f"Checkpoints copied to: {models_dir}")
    else:
        print("Warning: sgan-models directory not found")
    
    # Cleanup
    shutil.rmtree(sgan_dir)
    print("Social GAN download complete.")


def download_social_stgcnn(output_dir: Path) -> None:
    """Download Social-STGCNN checkpoints."""
    print("Downloading Social-STGCNN models...")
    
    stgcnn_dir = output_dir / "Social-STGCNN"
    if stgcnn_dir.exists():
        shutil.rmtree(stgcnn_dir)
    
    subprocess.run(
        ["git", "clone", "[https://github.com/abduallahmohamed/Social-STGCNN.git](https://github.com/abduallahmohamed/Social-STGCNN.git)", str(stgcnn_dir)],
        check=True,
    )
    
    # Checkpoints are in checkpoint/ directory
    checkpoint_dir = stgcnn_dir / "checkpoint"
    if checkpoint_dir.exists():
        models_dir = output_dir / "social_stgcnn"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        for checkpoint in checkpoint_dir.glob("*.pth"):
            shutil.copy2(checkpoint, models_dir / checkpoint.name)
        print(f"Checkpoints copied to: {models_dir}")
    else:
        print("Warning: checkpoint directory not found")
    
    print("Social-STGCNN download complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Download pretrained TP model checkpoints"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["social_gan", "social_stgcnn", "all"],
        help="Model to download",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/trajectory_prediction"),
        help="Output directory for checkpoints",
    )
    
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.model in ["social_gan", "all"]:
        download_social_gan(args.output_dir)
    
    if args.model in ["social_stgcnn", "all"]:
        download_social_stgcnn(args.output_dir)
    
    print("\nDownload complete!")
    print(f"Checkpoints stored in: {args.output_dir}")
    print("\nTo use in the app:")
    print("1. Go to TP Analysis tab")
    print("2. Select model from dropdown")
    print("3. Browse to checkpoint file")
    print("4. Run prediction")


if __name__ == "__main__":
    main()
```

### 7. Update requirements.txt
Add to `requirements.txt`:

```txt
# Existing dependencies
...

# PedPy for trajectory analysis
pedpy>=1.5.0

# PyTorch for TP models
torch>=2.0.0
torchvision>=0.15.0

# Optional: Social GAN (if installing from pip)
# sgan @ git+[https://github.com/agrimgupta92/sgan.git](https://github.com/agrimgupta92/sgan.git)

# Optional: Social-STGCNN dependencies
networkx>=2.5
tqdm>=4.60.0
```

### 8. Documentation in README.md
Add section to `README.md`:

```markdown
## Trajectory Prediction Models

The TP Analysis tab supports multiple trajectory prediction models. Some models require pretrained checkpoints.

### Constant Velocity (Baseline)
- **Checkpoint required:** No
- **Description:** Simple baseline using constant velocity extrapolation
- **Usage:** Select "constant_velocity" in the model dropdown

### Social GAN
- **Checkpoint required:** Yes
- **Official repository:** [https://github.com/agrimgupta92/sgan](https://github.com/agrimgupta92/sgan)
- **Download:**
  ```bash
  git clone [https://github.com/agrimgupta92/sgan.git](https://github.com/agrimgupta92/sgan.git)
  cd sgan
  bash scripts/download_models.sh
  ```
- **Checkpoints location:** `sgan-models/` directory
- **Training config:** obs_len=8, pred_len=12, frame_rate=2.5 Hz
- **Datasets:** ETH, HOTEL, UNIV, ZARA1, ZARA2

### Social-STGCNN
- **Checkpoint required:** Yes
- **Official repository:** [https://github.com/abduallahmohamed/Social-STGCNN](https://github.com/abduallahmohamed/Social-STGCNN)
- **Download:**
  ```bash
  git clone [https://github.com/abduallahmohamed/Social-STGCNN.git](https://github.com/abduallahmohamed/Social-STGCNN.git)
  ```
- **Checkpoints location:** `checkpoint/` directory (included in repo)
- **Training config:** obs_len=8, pred_len=12, frame_rate=2.5 Hz

### Download Helper Script
```bash
python -m pedestrian_analysis.scripts.download_tp_models \
    --model social_gan \
    --output-dir models/trajectory_prediction/
```

### Using Checkpoints in the App
1. Download checkpoint files using instructions above
2. Open TP Analysis tab
3. Select model from dropdown
4. Click "Browse..." and select checkpoint file
5. Adjust observation/prediction lengths if needed
6. Click "Run Prediction"
```

## Testing Requirements
- Test ConstantVelocityAdapter (no checkpoint)
- Test SocialGANAdapter with downloaded checkpoint
- Test error messages when checkpoint is missing
- Test model switching in UI
- Verify checkpoint info display updates correctly
- Test download helper script
- Verify model registry initialization

## Deliverables
1. `pedestrian_analysis/pipeline/tp_model_registry.py` - Model registry with checkpoint validation
2. `pedestrian_analysis/pipeline/tp_adapters.py` - Extended with SocialGANAdapter, SocialSTGCNNAdapter, TrajectronPPAdapter
3. `pedestrian_analysis/ui/widgets/tp_controls.py` - Updated to use registry
4. `pedestrian_analysis/scripts/download_tp_models.py` - Checkpoint download helper
5. Updated `requirements.txt`
6. Updated `README.md` with TP model documentation