# Task: Integrate State-of-the-Art Trajectory Prediction Models into school-zone-ped

## Context
I have a working Python desktop application (`school-zone-ped`) that analyzes pedestrian trajectories from top-down drone videos. The app currently:
- Extracts trajectories using detection + tracking (BoT-SORT, ByteTrack, OC-SORT)
- Converts pixel coordinates to metric world coordinates via homography
- Exports trajectory CSVs and annotated videos
- Provides analysis tabs with plots and statistics

**Repository structure:**
school-zone-ped/
├── app.py
├── requirements.txt
├── pedestrian_analysis/
│ ├── app.py
│ ├── config.py
│ ├── pipeline/
│ │ ├── calibration.py
│ │ ├── tracker_adapters.py
│ │ ├── tracking.py
│ │ ├── trajectory_io.py
│ │ └── analysis.py
│ ├── ui/
│ │ ├── tabs_calibration.py
│ │ ├── tabs_extraction.py
│ │ └── tabs_analysis.py
│ ├── visualization/
│ ├── utils/
│ └── tests/

text

## Goal
Extend the application with a **Trajectory Prediction (TP) module** that:
1. Integrates multiple SOTA trajectory prediction models via an adapter pattern (similar to existing tracker adapters)
2. Displays predicted trajectories in the preview visualization (alongside tracked trajectories)
3. Exports annotated videos showing both observed and predicted trajectories
4. Provides quantitative performance analysis comparing different TP models

## SOTA Trajectory Prediction Models to Support

Implement adapters for the following model families (prioritize lightweight, inference-ready implementations):

### Tier 1: Classical & Well-Established
- **Social-LSTM** (Alahi et al., 2016) - social pooling layer for pedestrian interactions
- **Social-GAN** (Gupta et al., 2018) - LSTM + GAN for multimodal predictions
- **Social-STGCNN** - spatiotemporal graph convolutional networks
- **Trajectron++** - dynamic graph-based multimodal prediction

### Tier 2: Transformer-Based (Modern SOTA)
- **Social-Transformer** / **Agent-Transformer** - attention-based interaction modeling
- **STGAT** (Spatiotemporal Graph Attention Network)
- **Goal-LSTM** or **PECNet** - goal-conditioned prediction

### Tier 3: Lightweight / Real-Time Capable
- **BiTraP-DGF** (2026) - dual-branch gated fusion for autonomous driving scenes
- **MEE-LSTM** (2026) - robust LSTM with minimum error entropy loss
- **SR-LSTM** - state refinement for LSTM

**Selection criteria:**
- Prefer models with PyTorch implementations available on GitHub
- Prioritize models that can run locally (no cloud API dependencies)
- Support both unimodal and multimodal prediction (K=5-20 trajectories per pedestrian)
- Input: observed trajectory (2D metric coordinates, 1-3 seconds, 10-30 Hz)
- Output: predicted trajectory (2D metric coordinates, 2-5 seconds future)

## Architecture Requirements

### 1. Adapter Pattern (mirror existing tracker adapters)
Create `pedestrian_analysis/pipeline/tp_adapters.py`:

```python
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np

class TrajectoryPredictorAdapter(ABC):
    """Base adapter for trajectory prediction models."""
    
    @abstractmethod
    def predict(
        self,
        observed_trajectories: np.ndarray,  # shape: (num_pedestrians, obs_len, 2)
        num_modes: int = 5,
        **kwargs
    ) -> np.ndarray:
        """
        Predict future trajectories.
        
        Returns:
            predictions: shape (num_pedestrians, num_modes, pred_len, 2)
        """
        pass
    
    @abstractmethod
    def load_model(self, checkpoint_path: Optional[str] = None) -> None:
        pass
    
    @property
    def model_name(self) -> str:
        pass
```

Implement concrete adapters:
- `SocialLSTMAdapter`
- `SocialGANAdapter`
- `TransformerTPAdapter`
- `BiTraPDGFAdapter`
- `DummyTPAdapter` (for testing, predicts constant velocity)

### 2. Configuration Extension
Extend `pedestrian_analysis/config.py`:

```python
TP_MODEL_DEFAULTS = {
    "social_lstm": {
        "checkpoint": "path/to/social_lstm.pth",
        "obs_len": 20,  # frames (2s at 10Hz)
        "pred_len": 30,  # frames (3s at 10Hz)
        "embedding_dim": 64,
        "num_modes": 5,
    },
    "social_gan": {
        "checkpoint": "path/to/social_gan.pth",
        "obs_len": 20,
        "pred_len": 30,
        "num_modes": 20,
    },
    "transformer": {
        "checkpoint": "path/to/agent_transformer.pth",
        "obs_len": 20,
        "pred_len": 30,
        "num_modes": 10,
        "d_model": 128,
        "nhead": 8,
    },
}

TP_MODEL_GROUPS = {
    "research_classical": ["social_lstm", "social_gan"],
    "research_transformer": ["transformer", "stgat"],
    "research_lightweight": ["bitrap_dgf", "mee_lstm"],
}
```

### 3. Pipeline Integration
Extend `pedestrian_analysis/pipeline/tracking.py`:

```python
def extract_trajectories_with_prediction(
    video_path: str,
    H: np.ndarray,
    detector_type: str = "yolov8",
    tracker_type: str = "bot_sort",
    tp_model_type: str = "social_lstm",  # NEW
    tp_config: Optional[dict] = None,
    obs_duration_sec: float = 2.0,
    pred_duration_sec: float = 3.0,
    **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract trajectories AND generate predictions.
    
    Returns:
        df_observed: existing trajectory DataFrame
        df_predictions: new DataFrame with predicted trajectories
    """
    # ... existing tracking logic ...
    
    # After extracting observed trajectories:
    tp_adapter = get_tp_adapter(tp_model_type, tp_config)
    predictions = tp_adapter.predict(observed_trajectories_tensor)
    
    # Convert predictions to DataFrame format matching observed trajectories
    df_predictions = format_predictions_as_dataframe(predictions, track_ids, timestamps)
    
    return df_observed, df_predictions
```

### 4. UI Extension (Tkinter)
Extend `pedestrian_analysis/ui/tabs_extraction.py`:

Add new controls in the extraction tab:
- **TP Model Selection** dropdown (concrete models + research groups)
- **Observation Duration** slider (0.5-5 seconds)
- **Prediction Duration** slider (1-6 seconds)
- **Number of Modes** input (1-20, for multimodal models)
- **Show Predictions in Preview** checkbox
- **Prediction Color Scheme** selector (e.g., gradient by probability, distinct colors per mode)

Update preview rendering to overlay:
- Observed trajectories: solid lines (existing)
- Predicted trajectories: dashed/dotted lines, semi-transparent, color-coded by mode
- Optional: prediction cones or uncertainty ellipses for multimodal outputs

### 5. Video Export Extension
Extend `pedestrian_analysis/visualization/video_export.py`:

```python
def export_annotated_video_with_predictions(
    video_path: str,
    df_observed: pd.DataFrame,
    df_predictions: pd.DataFrame,  # NEW
    output_path: str,
    H: np.ndarray,
    calibration: dict,
    show_predictions: bool = True,
    prediction_display_mode: str = "all_modes",  # or "best_mode", "top_k"
    top_k: int = 3,
    **kwargs
) -> str:
    """
    Export video with both observed and predicted trajectories.
    
    Predictions rendered as:
    - Dashed lines in distinct colors per mode
    - Optional: probability labels, confidence bands
    """
    # ... existing rendering logic ...
    
    if show_predictions:
        for track_id, pred_rows in df_predictions.groupby("track_id"):
            for mode_idx, mode_data in pred_rows.groupby("mode"):
                # Draw dashed prediction line from last observed point
                draw_prediction_line(
                    frame,
                    last_observed_point,
                    mode_data[["x_m", "y_m"]].values,
                    color=get_mode_color(mode_idx),
                    style="dashed",
                    alpha=0.6,
                )
```

### 6. Performance Analysis Module
Create `pedestrian_analysis/pipeline/tp_evaluation.py`:

```python
from typing import Dict, List
import numpy as np
import pandas as pd

class TPEvaluation:
    """Evaluate trajectory prediction models against ground truth."""
    
    @staticmethod
    def compute_ade(
        predictions: np.ndarray,  # (num_ped, num_modes, pred_len, 2)
        ground_truth: np.ndarray,  # (num_ped, pred_len, 2)
    ) -> np.ndarray:
        """Average Displacement Error per pedestrian per mode."""
        # ADE = mean L2 distance across all predicted timesteps
        pass
    
    @staticmethod
    def compute_fde(
        predictions: np.ndarray,
        ground_truth: np.ndarray,
    ) -> np.ndarray:
        """Final Displacement Error (error at last timestep)."""
        pass
    
    @staticmethod
    def compute_min_ade(
        predictions: np.ndarray,
        ground_truth: np.ndarray,
    ) -> np.ndarray:
        """minADE: best ADE across all modes."""
        pass
    
    @staticmethod
    def compute_min_fde(
        predictions: np.ndarray,
        ground_truth: np.ndarray,
    ) -> np.ndarray:
        """minFDE: best FDE across all modes."""
        pass
    
    @staticmethod
    def compute_mr(
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        threshold: float = 2.0,
    ) -> float:
        """Miss Rate: % of predictions where minFDE > threshold."""
        pass
    
    @staticmethod
    def generate_comparison_report(
        results_by_model: Dict[str, dict],
        output_path: str,
    ) -> None:
        """Generate CSV + PDF report comparing all TP models."""
        pass
```

### 7. Analysis Tab Extension
Extend `pedestrian_analysis/ui/tabs_analysis.py`:

Add a new sub-tab or section for **Trajectory Prediction Analysis**:
- Load multiple prediction CSVs (one per TP model)
- Load optional ground-truth future trajectories (for evaluation)
- Display metrics table:
  | Model | minADE (m) | minFDE (m) | MR@2m (%) | Inference Time (ms) |
  |-------|------------|------------|-----------|---------------------|
  | Social-LSTM | ... | ... | ... | ... |
  | Social-GAN | ... | ... | ... | ... |
  | Transformer | ... | ... | ... | ... |

- Plot predicted vs. actual trajectories for qualitative inspection
- Plot error distributions (histograms of ADE/FDE per model)
- Export comparison report as PDF

## Data Formats

### Observed Trajectories (existing CSV format)
```csv
track_id,frame,timestamp,x_pixel,y_pixel,x_m,y_m,vx_m_s,vy_m_s
1,100,10.0,450.2,320.1,2.3,1.8,0.5,0.3
1,101,10.1,452.1,321.0,2.4,1.85,0.5,0.3
...
```

### Predicted Trajectories (new CSV format)
```csv
track_id,mode,frame_offset,timestamp_future,x_m_pred,y_m_pred,probability
1,0,1,10.1,2.35,1.82,0.35
1,0,2,10.2,2.40,1.85,0.35
1,1,1,10.1,2.30,1.90,0.25
...
```

## Implementation Priorities

**Phase 1 (MVP):**
1. Implement `DummyTPAdapter` (constant velocity extrapolation)
2. Add TP adapter interface + config structure
3. Extend extraction pipeline to call TP adapter
4. Add TP toggle + model selector to UI
5. Render dummy predictions in preview

**Phase 2 (Real Models):**
1. Integrate 2-3 SOTA models with pre-trained checkpoints (Social-LSTM, Social-GAN, one transformer-based)
2. Implement model loading + inference wrappers
3. Support multimodal predictions (K modes)
4. Extend video export to include predictions

**Phase 3 (Evaluation):**
1. Implement ADE/FDE/minADE/minFDE/MR metrics
2. Add ground-truth loading for evaluation scenarios
3. Create comparison report generator
4. Add TP analysis tab with metrics + plots

## Dependencies to Add (requirements.txt)
Trajectory prediction
torch>=2.0
torchvision>=0.15
social-lstm-pytorch # or install from git
trajectory-prediction-toolkit # if available
transformers>=4.30 # for transformer-based models

text

## Testing Strategy
- Unit tests for TP adapters (mock observed trajectories, verify output shapes)
- Integration tests: run full pipeline on short video clip, verify predictions are generated
- Visual tests: manually inspect preview + exported video for correct rendering
- Metrics tests: synthetic data with known ground truth, verify ADE/FDE calculations

## Deliverables
1. `pedestrian_analysis/pipeline/tp_adapters.py` - adapter interface + implementations
2. `pedestrian_analysis/pipeline/tp_evaluation.py` - metrics + report generation
3. Extended `pedestrian_analysis/config.py` - TP model defaults
4. Extended `pedestrian_analysis/pipeline/tracking.py` - TP integration
5. Extended `pedestrian_analysis/ui/tabs_extraction.py` - TP controls
6. Extended `pedestrian_analysis/visualization/video_export.py` - prediction rendering
7. Extended `pedestrian_analysis/ui/tabs_analysis.py` - TP analysis tab
8. Updated `requirements.txt`
9. Documentation in README.md for TP features
10. Example scripts for running TP evaluation on benchmark datasets

## Notes
- Keep the existing tracker-adapter pattern as a reference for TP adapters
- Ensure backward compatibility: TP features should be optional (app works without TP)
- Prioritize models that can run on CPU or modest GPU (research desktop context)
- Use metric world coordinates (not pixels) for TP input/output
- Support variable observation/prediction lengths via config