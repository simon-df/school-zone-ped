"""Extraction tab: video selection, tracking execution and frame-by-frame preview."""

from __future__ import annotations

import queue
from pathlib import Path
from typing import Optional

import numpy as np
import tkinter as tk
from tkinter import ttk

from config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_FPS,
    DEFAULT_FRAME_SKIP,
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_TRAJECTORY_DIR,
    DEFAULT_OUTPUT_VIDEO_DIR,
    DEFAULT_TRACKER_TYPE,
    TRACKER_CHOICES,
)
from ui.dialogs import ask_open_file, ask_save_file, show_error, show_info, show_warning
from ui.widgets import append_log, make_label_entry, make_scrolled_text
from utils.image_utils import bgr_to_pil, pil_to_tkimage, scale_image_for_canvas
from utils.paths import make_output_path
from utils.threading_utils import WorkerTask, poll_queue

import logging
logger = logging.getLogger(__name__)

_PREVIEW_W = 640
_PREVIEW_H = 360


class ExtractionTab(ttk.Frame):
    """UI tab for trajectory extraction.

    The tab runs tracking work in a background thread and streams progress,
    status messages and preview frames back to the Tkinter main thread via a
    queue. This keeps the desktop UI responsive while long tracking jobs run.
    """

    def __init__(self, parent: tk.Widget, app_state) -> None:
        super().__init__(parent)
        self._state = app_state
        self._worker: Optional[WorkerTask] = None
        self._result_queue: queue.Queue = queue.Queue()
        self._trajectories = None
        self._tk_preview = None

        self._build_ui()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(2, weight=1)

        file_frame = ttk.LabelFrame(self, text="Input Files", padding=6)
        file_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=6, pady=4)
        file_frame.columnconfigure(1, weight=1)

        ttk.Button(file_frame, text="Select Video", command=self._on_select_video).grid(row=0, column=0, padx=4)
        self._video_path_var = tk.StringVar(value="")
        ttk.Entry(file_frame, textvariable=self._video_path_var, state="readonly", width=55).grid(
            row=0, column=1, padx=4, sticky="ew"
        )

        ttk.Button(file_frame, text="Select Calibration", command=self._on_select_calibration).grid(
            row=1, column=0, padx=4, pady=2
        )
        self._calib_path_var = tk.StringVar(value="")
        ttk.Entry(file_frame, textvariable=self._calib_path_var, state="readonly", width=55).grid(
            row=1, column=1, padx=4, pady=2, sticky="ew"
        )

        param_frame = ttk.LabelFrame(self, text="Parameters", padding=6)
        param_frame.grid(row=1, column=0, sticky="nsew", padx=6, pady=4)
        param_frame.columnconfigure(1, weight=1)

        self._model_entry = make_label_entry(param_frame, "Model name:", default=DEFAULT_MODEL_NAME, row=0)
        self._conf_entry = make_label_entry(param_frame, "Confidence:", default=str(DEFAULT_CONFIDENCE), row=1)
        self._frame_skip_entry = make_label_entry(param_frame, "Frame skip:", default=str(DEFAULT_FRAME_SKIP), row=2)
        self._fps_entry = make_label_entry(param_frame, "FPS:", default=str(DEFAULT_FPS), row=3)

        ttk.Label(param_frame, text="Tracker type:", width=20, anchor="w").grid(row=4, column=0, padx=4, pady=2, sticky="w")
        self._tracker_type_var = tk.StringVar(value=DEFAULT_TRACKER_TYPE)
        self._tracker_type_cb = ttk.Combobox(
            param_frame,
            textvariable=self._tracker_type_var,
            state="readonly",
            values=list(TRACKER_CHOICES),
            width=28,
        )
        self._tracker_type_cb.grid(row=4, column=1, padx=4, pady=2, sticky="ew")

        self._street_start_entry = make_label_entry(param_frame, "Street start y (m):", default="2.0", row=5)
        self._street_end_entry = make_label_entry(param_frame, "Street end y (m):", default="6.0", row=6)
        self._speed_thresh_entry = make_label_entry(param_frame, "Speed threshold (m/s):", default="0.3", row=7)

        ttk.Label(param_frame, text="Output video:", width=20, anchor="w").grid(row=8, column=0, padx=4, pady=2, sticky="w")
        self._output_video_var = tk.StringVar(value="")
        ttk.Entry(param_frame, textvariable=self._output_video_var, width=28).grid(row=8, column=1, padx=4, pady=2, sticky="ew")
        ttk.Button(param_frame, text="Browse", command=self._on_browse_output_video).grid(row=8, column=2, padx=4, pady=2)

        ttk.Label(param_frame, text="Output CSV:", width=20, anchor="w").grid(row=9, column=0, padx=4, pady=2, sticky="w")
        self._output_csv_var = tk.StringVar(value="")
        ttk.Entry(param_frame, textvariable=self._output_csv_var, width=28).grid(row=9, column=1, padx=4, pady=2, sticky="ew")
        ttk.Button(param_frame, text="Browse", command=self._on_browse_output_csv).grid(row=9, column=2, padx=4, pady=2)

        btn_row = ttk.Frame(param_frame)
        btn_row.grid(row=10, column=0, columnspan=3, pady=6)
        self._start_btn = ttk.Button(btn_row, text="Start Extraction", command=self._on_start)
        self._start_btn.pack(side="left", padx=4)
        self._stop_btn = ttk.Button(btn_row, text="Stop", command=self._on_stop, state="disabled")
        self._stop_btn.pack(side="left", padx=4)

        self._progress = ttk.Progressbar(param_frame, mode="determinate", maximum=100)
        self._progress.grid(row=11, column=0, columnspan=3, sticky="ew", padx=4, pady=4)

        log_frame = ttk.LabelFrame(self, text="Log", padding=4)
        log_frame.grid(row=2, column=0, sticky="nsew", padx=6, pady=4)
        self._log_text = make_scrolled_text(log_frame, height=10, state="disabled")

        preview_frame = ttk.LabelFrame(self, text="Preview", padding=4)
        preview_frame.grid(row=1, column=1, rowspan=2, sticky="nsew", padx=6, pady=4)
        self._preview_canvas = tk.Canvas(preview_frame, width=_PREVIEW_W, height=_PREVIEW_H, bg="#1a1a1a")
        self._preview_canvas.pack(fill="both", expand=True)

        exp_frame = ttk.Frame(self)
        exp_frame.grid(row=3, column=0, columnspan=2, pady=4)
        ttk.Button(exp_frame, text="Save Annotated Video", command=self._on_save_video).pack(side="left", padx=6)
        ttk.Button(exp_frame, text="Save Trajectories CSV", command=self._on_save_csv).pack(side="left", padx=6)

    def _on_select_video(self) -> None:
        path = ask_open_file(
            "Select source video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")],
        )
        if path:
            self._video_path_var.set(path)
            self._state.video_path = Path(path)
            stem = Path(path).stem
            if not self._output_video_var.get():
                self._output_video_var.set(str(make_output_path(DEFAULT_OUTPUT_VIDEO_DIR, stem, ".mp4")))
            if not self._output_csv_var.get():
                self._output_csv_var.set(str(make_output_path(DEFAULT_OUTPUT_TRAJECTORY_DIR, stem, ".csv")))

    def _on_select_calibration(self) -> None:
        from config import CALIBRATION_DIR
        from pipeline.calibration import load_calibration

        path = ask_open_file(
            "Select calibration file",
            filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")],
            initialdir=str(CALIBRATION_DIR),
        )
        if not path:
            return
        try:
            H = load_calibration(path)
            self._state.homography = H
            self._state.calibration_path = Path(path)
            self._calib_path_var.set(path)
        except Exception as exc:
            show_error(f"Failed to load calibration: {exc}")

    def _on_start(self) -> None:
        video_path = self._video_path_var.get()
        if not video_path:
            show_error("Please select a video file first.")
            return
        if self._state.homography is None:
            show_warning("No calibration loaded. Metre coordinates will be invalid.")

        try:
            confidence = float(self._conf_entry.get())
            frame_skip = int(self._frame_skip_entry.get())
        except ValueError as exc:
            show_error(f"Invalid parameter: {exc}")
            return

        model_name = self._model_entry.get().strip() or DEFAULT_MODEL_NAME
        tracker_type = self._tracker_type_var.get().strip() or DEFAULT_TRACKER_TYPE
        output_video_path = self._output_video_var.get().strip()
        output_csv_path = self._output_csv_var.get().strip()
        H = self._state.homography if self._state.homography is not None else np.eye(3)

        self._result_queue = queue.Queue()
        self._start_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        self._progress["value"] = 0

        from pipeline.tracker import run_tracking_with_preview

        self._worker = WorkerTask(
            target=run_tracking_with_preview,
            kwargs={
                "video_path": video_path,
                "H": H,
                "result_queue": self._result_queue,
                "model_name": model_name,
                "confidence": confidence,
                "frame_skip": frame_skip,
                "output_video_path": output_video_path or None,
                "output_csv_path": output_csv_path or None,
                "cancelled_fn": lambda: self._worker.is_cancelled() if self._worker else False,
                "tracker_type": tracker_type,
            },
            result_queue=self._result_queue,
        )
        self._worker.start()
        self._poll_queue()

    def _on_stop(self) -> None:
        if self._worker and self._worker.is_alive():
            self._worker.cancel()
        self._start_btn.config(state="normal")
        self._stop_btn.config(state="disabled")

    def _on_save_video(self) -> None:
        if self._output_video_var.get().strip():
            show_info(f"Annotated video will be saved to:\n{self._output_video_var.get().strip()}")
        else:
            show_warning("No output video path selected yet.")

    def _on_save_csv(self) -> None:
        if self._trajectories is None:
            show_error("No trajectories available. Please run extraction first.")
            return
        from config import TRAJECTORIES_DIR
        from pipeline.trajectory_io import save_trajectories

        path = ask_save_file(
            "Save trajectories CSV",
            filetypes=[("CSV files", "*.csv")],
            default_extension=".csv",
            initialdir=str(TRAJECTORIES_DIR),
        )
        if not path:
            return
        try:
            save_trajectories(self._trajectories, path)
            self._state.trajectories = self._trajectories
            self._state.trajectory_path = Path(path)
            show_info(f"Trajectories saved to:\n{path}")
        except Exception as exc:
            show_error(f"Failed to save CSV: {exc}")

    def _poll_queue(self) -> None:
        def handle(msg: dict) -> None:
            mtype = msg["type"]
            payload = msg["payload"]
            if mtype == "status":
                append_log(self._log_text, str(payload))
            elif mtype == "progress":
                self._progress["value"] = int(float(payload) * 100)
            elif mtype == "preview":
                self._display_preview(payload)
            elif mtype == "result":
                self._trajectories = payload
                append_log(self._log_text, f"Extraction complete. {len(payload)} rows collected.")
            elif mtype == "error":
                exc, tb = payload
                show_error(f"Worker error: {exc}\n\n{tb}")
                self._start_btn.config(state="normal")
                self._stop_btn.config(state="disabled")
            elif mtype == "done":
                self._start_btn.config(state="normal")
                self._stop_btn.config(state="disabled")
                return

        poll_queue(self._result_queue, handle)
        if self._worker and self._worker.is_alive():
            self.after(100, self._poll_queue)

    def _display_preview(self, frame: np.ndarray) -> None:
        pil_img = bgr_to_pil(frame)
        w = self._preview_canvas.winfo_width() or _PREVIEW_W
        h = self._preview_canvas.winfo_height() or _PREVIEW_H
        scaled, _ = scale_image_for_canvas(pil_img, w, h)
        self._tk_preview = pil_to_tkimage(scaled)
        self._preview_canvas.delete("all")
        self._preview_canvas.create_image(0, 0, anchor="nw", image=self._tk_preview)

    def _toggle_pbev(self) -> None:
        if self._pbev_var.get():
            self._pbev_frame.grid()
        else:
            self._pbev_frame.grid_remove()

    def _browse_entry(self, entry: ttk.Entry) -> None:
        path = ask_open_file("Select file")
        if path:
            entry.delete(0, "end")
            entry.insert(0, path)

    def _on_browse_output_video(self) -> None:
        path = ask_save_file(
            "Save annotated video",
            filetypes=[("MP4 video", "*.mp4"), ("All files", "*.*")],
            default_extension=".mp4",
        )
        if path:
            self._output_video_var.set(path)

    def _on_browse_output_csv(self) -> None:
        path = ask_save_file(
            "Save trajectories CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            default_extension=".csv",
        )
        if path:
            self._output_csv_var.set(path)
