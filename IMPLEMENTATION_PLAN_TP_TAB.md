# Task: Add New Tab "TP Analysis" to school-zone-ped App

## Context
The existing `school-zone-ped` application is a Tkinter-based desktop tool for analyzing pedestrian trajectories from top-down drone videos. It currently has tabs for:
- Calibration (homography)
- Trajectory Extraction (tracking)
- Analysis (plots and statistics from CSV)

The app already exports trajectory CSVs with columns like:
```csv
track_id,frame,timestamp,x_pixel,y_pixel,x_m,y_m,vx_m_s,vy_m_s
```

I now want to add a **new tab** called "TP Analysis" (Trajectory Prediction Analysis) that:
1. Loads a previously exported trajectory CSV
2. Displays all trajectories using PedPy in a static matplotlib plot
3. Provides an interactive matplotlib-based preview with a frame slider
4. Shows current waypoints, observed history, and trajectory predictions
5. Allows selecting TP models, number of modes, and prediction horizon via UI controls

## Goal
Implement a fully functional tab that integrates PedPy for trajectory visualization and provides an interactive prediction preview with configurable TP models.

## Requirements

### 1. New Tab Structure
Create `pedestrian_analysis/ui/tabs_tp_analysis.py` with:
- A main container frame with two side-by-side panels:
  - **Left Panel**: Static PedPy trajectory overview (all trajectories, full video duration)
  - **Right Panel**: Interactive prediction preview with slider and controls

### 2. CSV Loading
Add a file picker button to load trajectory CSV files:
- Accept CSVs with columns: `track_id`, `frame`, `timestamp`, `x_m`, `y_m` (metric coordinates required)
- Validate the CSV format and show an error if required columns are missing
- Convert to internal DataFrame format for both PedPy and TP modules

Example loading logic:
```python
def load_trajectory_csv(csv_path: str) -> pd.DataFrame:
    required_cols = {"track_id", "frame", "x_m", "y_m"}
    df = pd.read_csv(csv_path)
    
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"Missing columns: {required_cols - set(df.columns)}")
    
    # Keep only needed columns and sort
    df = df[["track_id", "frame", "timestamp", "x_m", "y_m"]].dropna()
    df = df.sort_values(["frame", "track_id"]).reset_index(drop=True)
    
    return df
```

### 3. PedPy Integration (Left Panel)
Use PedPy to display the full trajectory overview:

```python
from pedpy import TrajectoryData, plot_trajectories
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def create_pedpy_overview(df: pd.DataFrame, frame_rate: float = 10.0):
    # Convert to PedPy format
    pedpy_df = df.rename(
        columns={"track_id": "id", "x_m": "x", "y_m": "y"}
    )[["id", "frame", "x", "y"]].copy()
    
    pedpy_df["id"] = pedpy_df["id"].astype(int)
    pedpy_df["frame"] = pedpy_df["frame"].astype(int)
    
    trajectory_data = TrajectoryData(data=pedpy_df, frame_rate=frame_rate)
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_trajectories(traj=trajectory_data, axes=ax, traj_alpha=0.4, traj_width=1.2)
    ax.set_title("All Observed Trajectories (PedPy)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    
    return fig, ax, trajectory_data
```

Embed this figure in the left panel using `FigureCanvasTkAgg`.

### 4. Interactive Prediction Preview (Right Panel)
Create an interactive matplotlib figure with:

#### a) Frame Slider
- Use `matplotlib.widgets.Slider` to navigate through all frames
- Display current frame number and timestamp
- Update the plot on slider change

#### b) Trajectory Visualization
For each frame, display:
- **Current waypoints**: Large filled circles at current positions (`x_m`, `y_m`)
- **Track IDs**: Text labels next to each waypoint
- **Observed history**: Solid lines showing past positions (e.g., last 20 frames)
- **Predicted trajectories**: Dashed lines showing future predictions (configurable length)
- **Ground truth** (optional): Different color/style if available

#### c) TP Model Controls
Add UI controls (comboboxes, spinboxes, checkboxes):
- **TP Model Selection**: Dropdown with available models:
  - `constant_velocity` (baseline)
  - `social_lstm`
  - `social_gan`
  - `social_stgcnn`
  - `trajectron_pp`
  - `transformer`
- **Number of Modes**: Spinbox (1–20, default 5) for multimodal predictions
- **Prediction Horizon**: Spinbox in frames (10–60, default 30 frames = 3s at 10Hz)
- **Observation Window**: Spinbox in frames (5–40, default 20 frames = 2s at 10Hz)
- **Show Ground Truth**: Checkbox
- **Show All Modes**: Checkbox (vs. show only best mode)
- **Top-K Modes**: Spinbox (1–10, default 3)

#### d) Prediction Execution
When the slider moves or parameters change:
1. Extract observation window for each track (e.g., last 20 frames)
2. Check if enough history exists for prediction
3. Run selected TP model to generate predictions
4. Update the plot with new predictions

Example prediction loop:
```python
def update_prediction_preview(frame_idx: int):
    # Get current positions
    current_df = df[df["frame"] == frame_idx]
    
    # Get history for each track
    history_dict = {}
    for track_id in current_df["track_id"].unique():
        history = df[
            (df["track_id"] == track_id) &
            (df["frame"] <= frame_idx) &
            (df["frame"] > frame_idx - obs_window)
        ].sort_values("frame")
        
        if len(history) >= min_obs_frames:
            history_dict[track_id] = history[["x_m", "y_m"]].values
    
    # Convert to numpy array for TP model
    if len(history_dict) > 0:
        positions = np.stack(list(history_dict.values()))  # (N, obs_len, 2)
        track_ids = np.array(list(history_dict.keys()))
        
        # Run TP model
        predictions = tp_adapter.predict(
            positions=positions,
            track_ids=track_ids,
            current_frame=frame_idx,
            frame_rate=fps,
        )
        
        # Update plot with predictions
        draw_predictions(ax, predictions, track_ids, num_modes, show_all_modes)
    
    # Draw current waypoints and history
    draw_current_frame(ax, current_df, history_dict)
    
    canvas.draw()
```

### 5. TP Adapter Integration
Reuse the existing TP adapter pattern from the pipeline:

```python
from pedestrian_analysis.pipeline.tp_adapters import (
    TrajectoryPredictorAdapter,
    ConstantVelocityAdapter,
    SocialLSTMAdapter,
    SocialGANAdapter,
    # ... etc
)

TP_MODEL_REGISTRY = {
    "constant_velocity": ConstantVelocityAdapter,
    "social_lstm": SocialLSTMAdapter,
    "social_gan": SocialGANAdapter,
    "social_stgcnn": SocialSTGCNNAdapter,
    "trajectron_pp": TrajectronPPAdapter,
    "transformer": TransformerAdapter,
}

def get_tp_adapter(model_name: str, checkpoint_path: Optional[str] = None) -> TrajectoryPredictorAdapter:
    adapter_class = TP_MODEL_REGISTRY.get(model_name)
    if adapter_class is None:
        raise ValueError(f"Unknown TP model: {model_name}")
    
    adapter = adapter_class()
    adapter.load_model(checkpoint_path)
    return adapter
```

### 6. Visualization Functions
Implement helper functions for drawing:

```python
def draw_current_frame(ax, current_df: pd.DataFrame, history_dict: dict):
    """Draw current waypoints and observed history."""
    for _, row in current_df.iterrows():
        track_id = row["track_id"]
        x, y = row["x_m"], row["y_m"]
        
        # Draw history
        if track_id in history_dict:
            history = history_dict[track_id]
            ax.plot(history[:, 0], history[:, 1], color="black", linewidth=1.0, alpha=0.6)
        
        # Draw current waypoint
        ax.scatter(x, y, s=80, color="red", edgecolors="white", zorder=5)
        ax.text(x, y, f" {int(track_id)}", fontsize=8, color="black")


def draw_predictions(
    ax,
    predictions: np.ndarray,  # (N, num_modes, pred_len, 2)
    track_ids: np.ndarray,
    num_modes: int,
    show_all_modes: bool = True,
    top_k: int = 3,
):
    """Draw predicted trajectories as dashed lines."""
    for i, track_id in enumerate(track_ids):
        track_preds = predictions[i]  # (num_modes, pred_len, 2)
        
        modes_to_show = range(num_modes) if show_all_modes else range(min(top_k, num_modes))
        
        for mode_idx in modes_to_show:
            future = track_preds[mode_idx]  # (pred_len, 2)
            
            alpha = 0.3 if mode_idx > 0 else 0.8
            linewidth = 1.0 if mode_idx > 0 else 2.0
            color = plt.cm.tab10(mode_idx % 10)
            
            ax.plot(
                future[:, 0],
                future[:, 1],
                linestyle="--",
                linewidth=linewidth,
                color=color,
                alpha=alpha,
            )
```

### 7. Performance Considerations
- Cache predictions when parameters haven't changed
- Use blitting for faster slider updates if possible
- Limit the number of tracks displayed if too many (>50)
- Show a loading indicator during prediction inference
- Display inference time per frame in the UI

### 8. Error Handling
- Show clear error messages if CSV loading fails
- Handle cases where no predictions can be made (insufficient history)
- Gracefully handle TP model loading failures (fallback to constant velocity)
- Warn if frame rate is not specified (default to 10 Hz)

### 9. Tab Integration
In `pedestrian_analysis/app.py`, add the new tab:

```python
from pedestrian_analysis.ui.tabs_tp_analysis import TPTab

class App:
    def __init__(self, root):
        self.notebook = ttk.Notebook(root)
        
        self.calib_tab = CalibrationTab(self.notebook)
        self.extraction_tab = ExtractionTab(self.notebook)
        self.analysis_tab = AnalysisTab(self.notebook)
        self.tp_tab = TPTab(self.notebook)  # NEW
        
        self.notebook.add(self.calib_tab.frame, text="Calibration")
        self.notebook.add(self.extraction_tab.frame, text="Extraction")
        self.notebook.add(self.analysis_tab.frame, text="Analysis")
        self.notebook.add(self.tp_tab.frame, text="TP Analysis")  # NEW
        
        self.notebook.pack(fill="both", expand=True)
```

### 10. File Structure
Create the following new files:
pedestrian_analysis/
├── ui/
│ ├── tabs_tp_analysis.py # NEW: Main TP tab implementation
│ └── widgets/
│ ├── prediction_controls.py # NEW: TP control widgets (comboboxes, sliders)
│ └── trajectory_preview.py # NEW: Interactive matplotlib preview
├── pipeline/
│ ├── tp_adapters.py # Existing or create new
│ └── tp_windowing.py # NEW: Observation window extraction
└── visualization/
├── pedpy_plots.py # NEW: PedPy-based static plots
└── prediction_plots.py # NEW: Prediction visualization helpers

text

### 11. Dependencies
Add to `requirements.txt`:
Existing dependencies
...

PedPy for trajectory analysis
pedpy>=1.5.0

Matplotlib (should already be present)
matplotlib>=3.7.0

For TP models (optional, install as needed)
torch>=2.0
social-lstm-pytorch # or install from git

text

### 12. Testing
- Test with the provided CSV file (`2_20260911_115552.csv`)
- Verify that all trajectories are displayed correctly in PedPy plot
- Test slider navigation through all frames
- Verify that predictions are updated when changing model/parameters
- Test edge cases: empty CSV, single track, very short videos

### 13. UI Layout Mockup
┌─────────────────────────────────────────────────────────────────────┐
│ TP Analysis │
├─────────────────────────────────────────────────────────────────────┤
│ │
│ [Load CSV] │
│ │
│ ┌──────────────────────┐ ┌─────────────────────────────────────┐ │
│ │ │ │ Frame: [====|====] 150 / 300 │ │
│ │ PedPy Overview │ │ │ │
│ │ (all trajectories) │ │ [Interactive Preview Plot] │ │
│ │ │ │ │ │
│ │ │ │ - Current waypoints (red circles) │ │
│ │ │ │ - Track IDs (text labels) │ │
│ │ │ │ - History (black lines) │ │
│ │ │ │ - Predictions (dashed colored) │ │
│ │ │ │ │ │
│ └──────────────────────┘ └─────────────────────────────────────┘ │
│ │
│ TP Model: [Social-LSTM ▼] Modes: Pred Frames: │

│ Obs Window: Show All Modes: [✓] Top-K: │

│ │
│ [Play] [Pause] Speed: [1x ▼] Inference Time: 45 ms │
│ │
└─────────────────────────────────────────────────────────────────────┘

text

## Deliverables
1. `pedestrian_analysis/ui/tabs_tp_analysis.py` - Complete tab implementation
2. `pedestrian_analysis/ui/widgets/prediction_controls.py` - Reusable control widgets
3. `pedestrian_analysis/ui/widgets/trajectory_preview.py` - Interactive preview widget
4. `pedestrian_analysis/pipeline/tp_windowing.py` - Observation window extraction utilities
5. `pedestrian_analysis/visualization/pedpy_plots.py` - PedPy visualization helpers
6. `pedestrian_analysis/visualization/prediction_plots.py` - Prediction drawing helpers
7. Updated `requirements.txt` with PedPy dependency
8. Updated `pedestrian_analysis/app.py` to include the new tab
9. Brief documentation in README.md about the TP Analysis tab

## Notes
- Keep the existing tab structure and styling consistent
- Use the same color scheme and font sizes as other tabs
- Ensure the tab works independently (doesn't require calibration or extraction tabs to be used first)
- The CSV must contain metric coordinates (`x_m`, `y_m`), not pixel coordinates
- Frame rate should be configurable (default 10 Hz if not specified)
- Prioritize responsiveness: predictions should update within 100-200 ms when slider moves
- For initial implementation, use `ConstantVelocityAdapter` as the default model
- Add placeholder adapters for other models that raise clear errors if not yet implemented