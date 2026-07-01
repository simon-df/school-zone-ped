"""Analysis tab: load CSV, compute statistics, show and export plots."""

from __future__ import annotations

import logging
import queue
from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import ttk
from PIL import Image as PILImage

from ui.dialogs import ask_open_file, ask_save_file, show_error, show_info, show_warning
from ui.widgets import make_label_entry, append_log, make_scrolled_text
from utils.threading_utils import WorkerTask, poll_queue
from utils.image_utils import pil_to_tkimage, scale_image_for_canvas

logger = logging.getLogger(__name__)

_PREVIEW_W = 700
_PREVIEW_H = 500


class AnalysisTab(ttk.Frame):
    """UI tab for trajectory analysis and plot generation."""

    def __init__(self, parent: tk.Widget, app_state) -> None:
        super().__init__(parent)
        self._state = app_state
        self._worker: Optional[WorkerTask] = None
        self._result_queue: queue.Queue = queue.Queue()
        self._tk_preview = None
        self._current_plot_paths: list[Path] = []

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(1, weight=1)

        # --- Left panel: controls ---
        left = ttk.Frame(self)
        left.grid(row=0, column=0, rowspan=3, sticky="nsew", padx=6, pady=4)
        left.columnconfigure(1, weight=1)

        file_frame = ttk.LabelFrame(left, text="Data", padding=6)
        file_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=4)
        file_frame.columnconfigure(1, weight=1)

        ttk.Button(file_frame, text="Load CSV", command=self._on_load_csv).grid(row=0, column=0, padx=4)
        self._csv_path_var = tk.StringVar(value="")
        ttk.Entry(file_frame, textvariable=self._csv_path_var, state="readonly", width=40).grid(
            row=0, column=1, padx=4, sticky="ew"
        )

        param_frame = ttk.LabelFrame(left, text="Parameters", padding=6)
        param_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=4)
        param_frame.columnconfigure(1, weight=1)

        self._fps_entry = make_label_entry(param_frame, "FPS:", default="25.0", row=0)
        self._street_start_entry = make_label_entry(param_frame, "Street start X (m):", default="2.0", row=1)
        self._street_end_entry = make_label_entry(param_frame, "Street end X (m):", default="6.0", row=2)

        btn_frame = ttk.LabelFrame(left, text="Actions", padding=6)
        btn_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=4)

        buttons = [
            ("Plot Trajectories", self._on_plot_trajectories),
            ("Plot Speed", self._on_plot_speed),
            ("Plot Behavior", self._on_plot_behavior),
            ("Plot Groups", self._on_plot_groups),
            ("Plot Swarm", self._on_plot_swarm),
            ("Compute Statistics", self._on_compute_stats),
            ("Export All Plots", self._on_export_all),
        ]
        for i, (label, cmd) in enumerate(buttons):
            ttk.Button(btn_frame, text=label, command=cmd, width=22).grid(row=i, column=0, pady=2, padx=4, sticky="ew")

        # --- Statistics text ---
        stats_frame = ttk.LabelFrame(left, text="Statistics", padding=4)
        stats_frame.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=4)
        left.rowconfigure(3, weight=1)
        self._stats_text = make_scrolled_text(stats_frame, height=12, state="disabled")

        # --- Right panel: plot preview ---
        preview_frame = ttk.LabelFrame(self, text="Plot Preview", padding=4)
        preview_frame.grid(row=0, column=1, rowspan=3, sticky="nsew", padx=6, pady=4)
        self._preview_canvas = tk.Canvas(preview_frame, width=_PREVIEW_W, height=_PREVIEW_H, bg="#1a1a1a")
        self._preview_canvas.pack(fill="both", expand=True)

        # Navigation buttons for multiple plots
        nav_frame = ttk.Frame(self)
        nav_frame.grid(row=3, column=1, pady=4)
        self._prev_btn = ttk.Button(nav_frame, text="◀ Prev", command=self._on_prev_plot, state="disabled")
        self._prev_btn.pack(side="left", padx=4)
        self._plot_idx_var = tk.StringVar(value="0 / 0")
        ttk.Label(nav_frame, textvariable=self._plot_idx_var).pack(side="left", padx=4)
        self._next_btn = ttk.Button(nav_frame, text="Next ▶", command=self._on_next_plot, state="disabled")
        self._next_btn.pack(side="left", padx=4)
        self._plot_view_idx = 0

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_load_csv(self) -> None:
        from config import TRAJECTORIES_DIR
        from pipeline.trajectory_io import load_trajectories

        path = ask_open_file(
            "Load trajectory CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=str(TRAJECTORIES_DIR),
        )
        if not path:
            return
        try:
            df = load_trajectories(path)
            self._state.trajectories = df
            self._state.trajectory_path = Path(path)
            self._csv_path_var.set(path)
            show_info(f"Loaded {len(df)} rows from:\n{path}")
        except Exception as exc:
            show_error(f"Failed to load CSV: {exc}")

    def _get_df(self):
        if self._state.trajectories is None:
            show_error("Please load a trajectory CSV first.")
            return None
        return self._state.trajectories

    def _get_params(self) -> tuple[float, float, float] | None:
        try:
            fps = float(self._fps_entry.get())
            street_start = float(self._street_start_entry.get())
            street_end = float(self._street_end_entry.get())
            return fps, street_start, street_end
        except ValueError as exc:
            show_error(f"Invalid parameter: {exc}")
            return None

    def _on_plot_trajectories(self) -> None:
        df = self._get_df()
        params = self._get_params()
        if df is None or params is None:
            return
        fps, start, end = params
        from config import FIGURES_DIR
        from visualization.plot_trajectories import plot_trajectories
        from utils.paths import make_output_path

        try:
            out = make_output_path(FIGURES_DIR, "trajectories", ".png")
            p = plot_trajectories(df, start, end, output_path=out)
            if p:
                self._show_plot(p)
        except Exception as exc:
            show_error(f"Plot failed: {exc}")

    def _on_plot_speed(self) -> None:
        df = self._get_df()
        params = self._get_params()
        if df is None or params is None:
            return
        fps, _, _ = params
        if "speed_ms" not in df.columns:
            from pipeline.pedpy_analysis import compute_kinematics
            try:
                df = compute_kinematics(df, fps)
                self._state.trajectories = df
            except Exception as exc:
                show_error(f"Kinematics failed: {exc}")
                return
        from config import FIGURES_DIR
        from visualization.plot_swarm import plot_swarm_metrics
        from utils.paths import make_output_path

        try:
            out = make_output_path(FIGURES_DIR, "speed", ".png")
            p = plot_swarm_metrics(df, fps, output_path=out, title="Speed Over Time")
            if p:
                self._show_plot(p)
        except Exception as exc:
            show_error(f"Plot failed: {exc}")

    def _on_plot_behavior(self) -> None:
        df = self._get_df()
        params = self._get_params()
        if df is None or params is None:
            return
        fps, start, end = params
        if "behavior" not in df.columns:
            show_warning("No behavior column found. Run behavior labeling first or load a CSV with behavior data.")
            return
        from config import FIGURES_DIR
        from visualization.plot_behavior import plot_behavior_timeline
        from utils.paths import make_output_path

        try:
            out = make_output_path(FIGURES_DIR, "behavior", ".png")
            p = plot_behavior_timeline(df, fps, output_path=out)
            if p:
                self._show_plot(p)
        except Exception as exc:
            show_error(f"Plot failed: {exc}")

    def _on_plot_groups(self) -> None:
        df = self._get_df()
        if df is None:
            return
        if "group_id" not in df.columns:
            show_warning("No group_id column found. Run group analysis first.")
            return
        from config import FIGURES_DIR
        from pipeline.group_analysis import compute_group_statistics
        from visualization.plot_groups import plot_group_statistics
        from utils.paths import make_output_path

        try:
            stats = compute_group_statistics(df)
            out = make_output_path(FIGURES_DIR, "groups", ".png")
            p = plot_group_statistics(stats, output_path=out)
            if p:
                self._show_plot(p)
        except Exception as exc:
            show_error(f"Plot failed: {exc}")

    def _on_plot_swarm(self) -> None:
        df = self._get_df()
        params = self._get_params()
        if df is None or params is None:
            return
        fps, _, _ = params
        from config import FIGURES_DIR
        from visualization.plot_swarm import plot_swarm_metrics
        from utils.paths import make_output_path

        try:
            out = make_output_path(FIGURES_DIR, "swarm", ".png")
            p = plot_swarm_metrics(df, fps, output_path=out)
            if p:
                self._show_plot(p)
            else:
                show_warning("No swarm metric columns found in the loaded data.")
        except Exception as exc:
            show_error(f"Plot failed: {exc}")

    def _on_compute_stats(self) -> None:
        df = self._get_df()
        params = self._get_params()
        if df is None or params is None:
            return
        fps, start, end = params

        lines = [f"Rows: {len(df)}", f"Pedestrians: {df['id'].nunique()}"]

        if "speed_ms" in df.columns:
            lines.append(f"Mean speed: {df['speed_ms'].mean():.3f} m/s")
            lines.append(f"Max speed:  {df['speed_ms'].max():.3f} m/s")

        if "behavior" in df.columns:
            for b, cnt in df["behavior"].value_counts().items():
                lines.append(f"  {b}: {cnt} frames")

        if "group_id" in df.columns:
            from pipeline.group_analysis import compute_group_statistics
            try:
                stats = compute_group_statistics(df)
                lines.append(f"Group count: {stats['group_count']}")
                lines.append(f"Split events: {stats['split_events']}")
                mwt = stats.get("median_waiting_time_s")
                if mwt is not None and mwt == mwt:  # not NaN
                    lines.append(f"Median wait: {mwt:.2f} s")
            except Exception as exc:
                lines.append(f"Group stats error: {exc}")

        text = "\n".join(lines)
        self._stats_text.config(state="normal")
        self._stats_text.delete("1.0", "end")
        self._stats_text.insert("end", text)
        self._stats_text.config(state="disabled")

    def _on_export_all(self) -> None:
        df = self._get_df()
        params = self._get_params()
        if df is None or params is None:
            return
        fps, start, end = params
        from config import FIGURES_DIR
        from visualization.export import export_all_plots

        try:
            paths = export_all_plots(df, FIGURES_DIR, fps=fps, street_start_m=start, street_end_m=end)
            self._current_plot_paths = paths
            self._plot_view_idx = 0
            self._update_plot_nav()
            if paths:
                self._show_plot(paths[0])
            show_info(f"Exported {len(paths)} plot(s) to:\n{FIGURES_DIR}")
        except Exception as exc:
            show_error(f"Export failed: {exc}")

    # ------------------------------------------------------------------
    # Plot navigation
    # ------------------------------------------------------------------

    def _show_plot(self, path: Path) -> None:
        if path not in self._current_plot_paths:
            self._current_plot_paths.append(path)
            self._plot_view_idx = len(self._current_plot_paths) - 1
        else:
            self._plot_view_idx = self._current_plot_paths.index(path)
        self._update_plot_nav()
        self._render_plot(path)

    def _render_plot(self, path: Path) -> None:
        try:
            img = PILImage.open(str(path))
            w = self._preview_canvas.winfo_width() or _PREVIEW_W
            h = self._preview_canvas.winfo_height() or _PREVIEW_H
            scaled, _ = scale_image_for_canvas(img, w, h)
            self._tk_preview = pil_to_tkimage(scaled)
            self._preview_canvas.delete("all")
            self._preview_canvas.create_image(0, 0, anchor="nw", image=self._tk_preview)
        except Exception as exc:
            logger.error("Cannot display plot '%s': %s", path, exc)

    def _on_prev_plot(self) -> None:
        if self._plot_view_idx > 0:
            self._plot_view_idx -= 1
            self._render_plot(self._current_plot_paths[self._plot_view_idx])
            self._update_plot_nav()

    def _on_next_plot(self) -> None:
        if self._plot_view_idx < len(self._current_plot_paths) - 1:
            self._plot_view_idx += 1
            self._render_plot(self._current_plot_paths[self._plot_view_idx])
            self._update_plot_nav()

    def _update_plot_nav(self) -> None:
        n = len(self._current_plot_paths)
        idx = self._plot_view_idx
        self._plot_idx_var.set(f"{idx + 1} / {n}" if n else "0 / 0")
        self._prev_btn.config(state="normal" if idx > 0 else "disabled")
        self._next_btn.config(state="normal" if idx < n - 1 else "disabled")
