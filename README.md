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

## Current Detection Path

The default detector path is currently:

- default checkpoint: `yolov8n.pt` (`pedestrian_analysis/config.py`)
- default detector type: `yolov8`
- inference entrypoints: `extract_trajectories_from_video(...)` and `run_tracking_with_preview(...)`
- `supervision.Detections` creation: the detector wrapper calls the Ultralytics model and converts `results[0]` with `sv.Detections.from_ultralytics(...)`
- coordinate contract: detections use `xyxy` pixel boxes (`x1, y1, x2, y2`)
- score contract: detection confidence is preserved in `detections.confidence`
- class contract: detection class IDs are preserved in `detections.class_id`, then optionally filtered to selected classes before tracking

This keeps the tracker adapters unchanged: BoT-SORT, ByteTrack and OC-SORT still receive the same `supervision.Detections` object with `xyxy + confidence + class_id`.

### Current baseline limitations for small pedestrians / children

- `yolov8n.pt` is a lightweight baseline and tends to trade recall for speed.
- The baseline uses generic COCO person detection; children remain implicit small `person` targets rather than a specialized class.
- There is no dedicated small-object head or P2-style checkpoint in the repository by default.
- Standard 640px-style inference can miss tiny top-down pedestrians when they occupy only a few pixels.
- The tracker can stabilize IDs only after a detection exists; it cannot recover pedestrians that were never detected.

## Detector Options

Detector selection is now configurable via detector type plus checkpoint path:

- `yolov8`: existing YOLOv8 baseline (`yolov8n.pt` by default)
- `yolov8_large`: larger Ultralytics checkpoint preset (`yolov8l.pt` by default)
- `small_object_yolo`: small-object-oriented wrapper with lower-confidence / higher-resolution defaults for dense tiny targets; for best results, point this to a user-supplied small-object or P2-style Ultralytics checkpoint
- `rtdetr`: DETR-style wrapper using Ultralytics RT-DETR (`rtdetr-l.pt` by default, or a user-provided checkpoint path)

In the desktop extraction tab you can now choose:

- **Detector type**
- **Model name** (checkpoint name/path)
- **Tracked classes** (comma-separated, for example `person` or `person,child`)

Programmatic usage keeps the existing API and adds optional detector arguments:

```python
from pipeline.tracker import extract_trajectories_from_video

df = extract_trajectories_from_video(
    video_path="scene.mp4",
    H=homography,
    detector_type="small_object_yolo",
    model_name="path/to/your/small-object-checkpoint.pt",
    detector_classes="person,child",
    detector_kwargs={"imgsz": 1280, "nms_iou": 0.6},
    tracker_type="research_top_down_small_targets",
)
```

## Local Detector Evaluation

Use the local-only comparison script to run representative BEV frames/videos through one or many detectors, save overlays, and summarize counts/recall:

```bash
python pedestrian_analysis/scripts/evaluate_detectors.py \
  path/to/scene_a.mp4 path/to/scene_b.mp4 \
  --detector all \
  --class-filter person \
  --frame-step 30 \
  --max-frames 20 \
  --output-dir pedestrian_analysis/outputs/detector_eval
```

Outputs:

- `frame_summary.csv`: per-frame detector counts and optional recall estimates
- `scene_summary.csv`: per-scene totals and mean detections/frame
- `recall_summary.csv`: average recall estimate when annotations are available
- `overlays/<detector>/<scene>/frame_*.jpg`: qualitative overlays with boxes, labels and scores

Optional annotation CSV support:

- approximate recall from manual counts with columns like `scene,frame,gt_count`
- box-level recall with columns like `scene,frame,bbox_x1,bbox_y1,bbox_x2,bbox_y2`

### Pretrained recommendation

Start with `yolov8_large` for a stronger drop-in baseline, then compare it against `small_object_yolo` using a higher `imgsz` and a user-supplied tiny-target checkpoint if available. `rtdetr` is worth testing when crowded scenes cause NMS-related misses, but it should still be validated locally on your BEV footage.

## Optional Fine-Tuning Scaffold

Only use this if the pretrained detectors still miss too many children / pedestrians in your own scenes.

- dataset template: `pedestrian_analysis/training/pedestrian_small_target_template.yaml`
- local training entrypoint: `python pedestrian_analysis/scripts/train_detector.py --data /absolute/path/to/data.yaml`

Recommended starting settings for small top-down pedestrians:

- single class (`person`) unless you truly have reliable child-specific labels
- high input resolution (`imgsz` around 1280 or higher if hardware allows)
- prefer augmentations that preserve tiny targets; avoid aggressive zoom-out or copy-paste settings that make people even smaller
- start from the best-performing pretrained checkpoint from the local evaluation script

## Data and Outputs

The application uses the following folders under `pedestrian_analysis/`:

- `data/videos/` for input videos
- `data/calibration/` for saved homographies and calibration metadata
- `data/trajectories/` for exported trajectory CSV files
- `data/previews/` for preview images
- `outputs/figures/`, `outputs/reports/`, `outputs/videos/` for generated outputs

## Trajectory Prediction (TP) Models

The **TP Analyse** tab lets you load a trajectory CSV, view all trajectories via PedPy, and step through frames to preview predicted future trajectories. Models are managed by `pipeline.tp_model_registry.TPModelRegistry`, which tracks checkpoint requirements per model and validates checkpoint paths before creating an adapter.

| UI model name       | Checkpoint | Status |
|----------------------|:---------:|--------|
| `constant_velocity`  | not required | Implemented (`DummyTPAdapter`) |
| `social_lstm`        | optional  | Implemented, from-scratch reimplementation (`pipeline.tp_models.SocialLSTMNet`); runs with random weights unless you train and pass your own checkpoint |
| `social_gan`         | optional  | Implemented, from-scratch reimplementation (`pipeline.tp_models.SocialGANNet`); **not** compatible with the official [agrimgupta92/sgan](https://github.com/agrimgupta92/sgan) checkpoints (different architecture/state_dict keys) |
| `transformer`        | optional  | Implemented, custom architecture (`pipeline.tp_models.TransformerTPNet`); no published checkpoint exists for it |
| `social_stgcnn`      | **required** | Not implemented — selecting it raises a clear error. Official repo: [abduallahmohamed/Social-STGCNN](https://github.com/abduallahmohamed/Social-STGCNN) |
| `trajectron_pp`      | **required** | Not implemented — selecting it raises a clear error. Official repo: [StanfordASL/Trajectron-plus-plus](https://github.com/StanfordASL/Trajectron-plus-plus) |

Important: the official Social-GAN / Social-STGCNN / Trajectron++ checkpoints were all trained on ETH/UCY (or nuScenes) pedestrian data from oblique/side-view cameras at low frame rates (2.5–10 Hz), not top-down drone footage — even a fully integrated model would likely need fine-tuning on your own data to perform well here.

### Fetching official reference checkpoints (optional)

A helper script can clone the official repos and copy their bundled checkpoints for inspection, e.g. if you plan to vendor the official model code yourself:

```bash
python -m scripts.download_tp_models --model social_gan --output-dir models/trajectory_prediction/
python -m scripts.download_tp_models --model social_stgcnn --output-dir models/trajectory_prediction/
```

These checkpoints will **not** load into this repo's built-in adapters — see the table above.

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
