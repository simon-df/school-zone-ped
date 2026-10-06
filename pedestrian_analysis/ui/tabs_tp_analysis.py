"""TP Analysis tab: PedPy trajectory overview + interactive TP prediction preview."""
from __future__ import annotations

import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
import pandas as pd

from pipeline.trajectory_io import load_trajectory_csv  # noqa: F401  (re-exported for backward compatibility)
from pipeline.tp_windowing import extract_observation_windows, get_current_positions, stack_observations
from ui.dialogs import ask_open_file, show_error, show_warning
from ui.tp_controls import TPControls
from ui.tp_preview import TrajectoryPreviewCanvas
from visualization.pedpy_plots import create_pedpy_overview_figure
from visualization.prediction_plots import draw_current_frame, draw_predictions

import logging
logger = logging.getLogger(__name__)

_DEFAULT_FPS = 10.0
_MIN_OBS_FRAMES = 2


class TPAnalysisTab(ttk.Frame):
    """UI tab: PedPy trajectory overview + interactive TP prediction preview."""

    def __init__(self, parent: tk.Widget, app_state) -> None:
        super().__init__(parent)
        self._state = app_state
        self._df: pd.DataFrame | None = None
        self._fps = _DEFAULT_FPS
        self._pedpy_canvas: object | None = None
        self._warned_checkpoint_settings: set[tuple] = set()
        self._build_ui()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

        top = ttk.Frame(self)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=6, pady=4)
        ttk.Button(top, text="Load CSV", command=self._on_load_csv).pack(side="left", padx=4)
        self._path_var = tk.StringVar(value="No trajectory CSV loaded.")
        ttk.Label(top, textvariable=self._path_var).pack(side="left", padx=8)

        left = ttk.LabelFrame(self, text="PedPy Overview (all trajectories)", padding=4)
        left.grid(row=1, column=0, sticky="nsew", padx=6, pady=4)
        self._left_container = left

        right = ttk.Frame(self)
        right.grid(row=1, column=1, sticky="nsew", padx=6, pady=4)
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self._preview = TrajectoryPreviewCanvas(right, on_frame_committed=self._on_frame_committed)
        self._preview.frame.grid(row=0, column=0, sticky="nsew")

        self._controls = TPControls(right, on_change=self._on_controls_changed)
        self._controls.frame.grid(row=1, column=0, sticky="ew", pady=(4, 0))

        self._render_pedpy_placeholder()

    # ------------------------------------------------------------------
    # CSV loading
    # ------------------------------------------------------------------

    def _on_load_csv(self) -> None:
        path = ask_open_file("Select trajectory CSV", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            df = load_trajectory_csv(path, fps=self._fps)
        except Exception as exc:
            show_error(f"Failed to load trajectory CSV: {exc}")
            return

        if df.empty:
            show_warning("The loaded CSV contains no valid trajectory rows.")

        self._df = df
        self._path_var.set(Path(path).name)
        self._render_pedpy_overview()

        available_frames = sorted(df["frame"].unique().tolist())
        self._preview.set_available_frames(available_frames)
        self._on_frame_committed(self._preview.current_frame())

    # ------------------------------------------------------------------
    # PedPy static overview (left panel)
    # ------------------------------------------------------------------

    def _render_pedpy_placeholder(self) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title("Load a trajectory CSV to see the overview")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        self._pedpy_canvas = FigureCanvasTkAgg(fig, master=self._left_container)
        self._pedpy_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _render_pedpy_overview(self) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        if self._pedpy_canvas is not None:
            self._pedpy_canvas.get_tk_widget().destroy()
            plt.close(self._pedpy_canvas.figure)

        fig, _ax = create_pedpy_overview_figure(self._df, frame_rate=self._fps)
        self._pedpy_canvas = FigureCanvasTkAgg(fig, master=self._left_container)
        self._pedpy_canvas.get_tk_widget().pack(fill="both", expand=True)

    # ------------------------------------------------------------------
    # Interactive prediction preview (right panel)
    # ------------------------------------------------------------------

    def _on_controls_changed(self) -> None:
        frame_idx = self._preview.current_frame()
        if frame_idx is not None:
            self._on_frame_committed(frame_idx)

    def _warn_checkpoint_settings_mismatch(
        self, adapter_name: str, checkpoint_path: str | None, obs_window: int, pred_len: int
    ) -> None:
        """Warn (once per combination) when UI settings differ from the checkpoint's training settings."""
        from pipeline.tp_model_registry import checkpoint_settings_warnings, load_checkpoint_metadata

        key = (adapter_name, checkpoint_path, obs_window, pred_len, self._fps)
        if not checkpoint_path or key in self._warned_checkpoint_settings:
            return
        self._warned_checkpoint_settings.add(key)
        warnings = checkpoint_settings_warnings(
            load_checkpoint_metadata(checkpoint_path),
            obs_len=obs_window,
            pred_len=pred_len,
            fps=self._fps,
            model_name=adapter_name,
        )
        if warnings:
            show_warning(
                "TP checkpoint settings differ from the current settings:\n- "
                + "\n- ".join(warnings)
                + "\n\nPredictions may be unreliable; match the Obs window / Pred frames to the training setup."
            )

    def _on_frame_committed(self, frame_idx: int | None) -> None:
        if self._df is None or frame_idx is None:
            return

        current_df = get_current_positions(self._df, frame_idx)
        obs_window = int(self._controls.obs_window_var.get())
        history = extract_observation_windows(self._df, frame_idx, obs_window, min_obs_frames=_MIN_OBS_FRAMES)

        num_modes = int(self._controls.num_modes_var.get())
        pred_len = int(self._controls.pred_len_var.get())
        adapter_name = self._controls.selected_adapter_name()
        checkpoint_path = self._controls.checkpoint_path()

        predictions: dict[int, np.ndarray] = {}
        elapsed_ms: float | None = None

        if adapter_name is None:
            show_warning(f"TP model '{self._controls.model_var.get()}' is not registered.")
        elif history:
            track_ids, positions = stack_observations(history)
            if positions.shape[1] >= 2:
                try:
                    from pipeline.tp_model_registry import TPModelRegistry

                    adapter = TPModelRegistry.create_adapter(
                        adapter_name, checkpoint_path=checkpoint_path, pred_len=pred_len
                    )
                    self._warn_checkpoint_settings_mismatch(adapter_name, checkpoint_path, obs_window, pred_len)
                    start = time.perf_counter()
                    batch_predictions = adapter.predict(positions, num_modes=num_modes, pred_len=pred_len)
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    predictions = {int(tid): batch_predictions[i] for i, tid in enumerate(track_ids)}
                except Exception as exc:
                    show_error(f"TP prediction failed ({adapter_name}): {exc}")

        self._controls.inference_time_var.set(
            f"Inference time: {elapsed_ms:.1f} ms" if elapsed_ms is not None else "Inference time: – ms"
        )

        show_all_modes = bool(self._controls.show_all_modes_var.get())
        top_k = int(self._controls.top_k_var.get())

        def draw(ax) -> None:
            draw_current_frame(ax, current_df, history)
            if predictions:
                draw_predictions(ax, predictions, num_modes=num_modes, show_all_modes=show_all_modes, top_k=top_k)
            ax.set_title(f"Frame {frame_idx}")

        self._preview.redraw(draw)
