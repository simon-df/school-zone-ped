# school-zone-ped

School-zone-ped is a desktop-oriented Python application for analyzing pedestrian trajectories from top-down drone videos. The project focuses on robust person detection, stable track IDs across frames, homography-based metric conversion, and downstream behavioral and swarm analysis.

## Research and application focus

The project targets pedestrian safety research in urban crossing contexts. The workflow is designed for top-down, bird's-eye-view videos where people appear as small, partially occluded targets and where ID consistency is critical for extracting reliable trajectories.

## Main capabilities

- Calibrate a top-down scene from a reference image and compute a homography
- Extract trajectories from drone video using local detection and tracking
- Keep track IDs stable across frames via a modular tracking adapter layer
- Convert pixel foot points to metric coordinates and analyze kinematics
- Label behaviors such as waiting, approaching, crossing and crossed
- Detect groups and compute swarm-style metrics for research reporting

## Architecture overview

The codebase is organized around the following layers:

- UI: Tkinter/ttk desktop interface with calibration, extraction and analysis tabs
- Pipeline: calibration, tracking, trajectory I/O, and analysis modules
- Visualization: trajectory, behavior, group and swarm plots
- Persistence: JSON/CSV/NumPy file helpers and project directory management
- Configuration: central defaults for model selection, tracker choice and analysis thresholds
- Tests: focused regression tests for the core analysis logic

## Repository structure

```text
school-zone-ped/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── pedestrian_analysis/
│   ├── app.py
│   ├── config.py
│   ├── pipeline/
│   ├── ui/
│   ├── visualization/
│   ├── utils/
│   ├── data/
│   └── tests/
```

## Installation

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the application

Start the local desktop app with:

```bash
python app.py
```

## Example workflow

1. Load a calibration image and click four reference points.
2. Enter the corresponding world-space metre coordinates.
3. Compute and save the homography.
4. Load a drone video and select the saved calibration.
5. Choose a tracker mode (BoT-SORT is the default), a model and confidence threshold.
6. Run tracking to export annotated video and trajectory CSV output.
7. Open the analysis tab to compute behavior, group and swarm metrics.

## Tracking modes

The tracking pipeline is designed around adapter-based backends:

- bot_sort: default local mode using the supervision BoT-SORT adapter
- byte_track: lightweight supervision ByteTrack alternative
- ocsort / deep_ocsort: experimental placeholders for future integration
- pb_evformer: experimental stub for future BEV-based tracking

## Data and output folders

The application uses the following folders under `pedestrian_analysis/`:

- `data/videos/` for input videos
- `data/calibration/` for saved homographies and metadata
- `data/trajectories/` for exported CSV trajectories
- `data/previews/` for preview frames
- `outputs/figures/`, `outputs/reports/`, `outputs/videos/` for generated outputs

## Current status and roadmap

The current codebase already provides a solid local desktop workflow with:

- Tkinter-based calibration and extraction UI
- Homography-based metric conversion
- trajectory CSV I/O and analysis helpers
- modular tracker adapter hooks for future expansion

Planned work focuses on deeper tracker validation, experiment-specific adapters and more robust metric reporting for research use.

## Experimental components

The PBEVFormer integration and the OC-SORT / Deep OC-SORT adapters are clearly marked as experimental. They are included as scaffolding and raise explicit errors unless the required dependencies and implementation work are added.

## License

This repository is currently provided as a research prototype. Add an appropriate license file before public distribution if you plan to release it more broadly.
