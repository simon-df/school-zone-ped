# school-zone-ped

school-zone-ped is a local Python desktop application for analyzing pedestrian trajectories from top-down drone videos. It focuses on reliable person detection, stable track IDs across frames, homography-based metric conversion, and downstream behavior, group and swarm analysis.

## Purpose

The project is aimed at pedestrian safety research and applied traffic analysis in bird's-eye-view recordings. Because people are small, close together and frequently partially occluded in this view, ID consistency is the core requirement for extracting usable trajectories.

## Main Features

- Tkinter + ttk desktop UI with three tabs for calibration, trajectory extraction and analysis
- Homography calibration from four manually selected image points
- Tracking-by-detection pipeline with adapter-based tracker selection
- Metric conversion from pixel foot points to metre coordinates
- Trajectory export as CSV and optional annotated video export
- Behavior labeling, group analysis, swarm metrics and plotting utilities

## Architecture

The repository is split into clear layers:

- UI for the local workflow and worker-thread progress updates
- Pipeline for calibration, tracker adapters, tracking, trajectory I/O and analysis
- Visualization for PNG/PDF export and plot generation
- Utilities for file validation, paths, images, video IO and threading helpers
- Configuration for defaults, tracker selection, thresholds and project paths
- Tests for the core numerical and labeling logic

## Repository Layout

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

Python 3.11 or newer is required.

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Start

Launch the desktop application locally with:

```bash
python app.py
```

## Workflow

1. Open a calibration image and select four pixel points.
2. Enter the matching world points in metres.
3. Compute and save the homography plus calibration metadata.
4. Load a top-down drone video and the saved calibration file.
5. Choose a tracker mode, model, confidence threshold and frame skip.
6. Run tracking to generate preview frames, an annotated video and a trajectory CSV.
7. Load the CSV in the analysis tab to create plots and summary statistics.

## Tracker Modes

The tracking backend is adapter-based and selectable from the UI:

- `bot_sort`: default local mode using the supervision BoT-SORT adapter
- `byte_track`: lightweight supervision ByteTrack alternative
- `ocsort`: OC-SORT adapter (optional dependency: `boxmot`)
- `deep_ocsort`: experimental placeholder for future integration
- `pb_evformer`: experimental stub for future BEV-based tracking

Research-focused tracker groups (ordered fallback):

- `research_top_down_general`: `bot_sort` -> `ocsort` -> `byte_track`
- `research_top_down_occlusion`: `bot_sort` -> `ocsort` -> `byte_track`
- `research_top_down_small_targets`: `bot_sort` -> `byte_track` -> `ocsort`

You can select either a concrete tracker or a group name via `tracker_type`.

Experimental adapters raise explicit errors unless the required implementation is added.

## Data and Outputs

The application uses the following folders under `pedestrian_analysis/`:

- `data/videos/` for input videos
- `data/calibration/` for saved homographies and calibration metadata
- `data/trajectories/` for exported trajectory CSV files
- `data/previews/` for preview images
- `outputs/figures/`, `outputs/reports/`, `outputs/videos/` for generated outputs

## Status and Roadmap

Current status:

- local desktop UI is available via Tkinter
- calibration and trajectory analysis are wired into separate tabs
- tracker selection is adapter-based and no longer tied to a single backend

Open roadmap items:

- deeper validation of experimental tracker adapters
- optional GT-based tracking metrics such as IDF1, HOTA or MOTA
- richer export options for reports and figures

## License

This repository is currently a research prototype. Add a license file before broader public distribution.
