"""Interactive matplotlib preview widget for the TP Analysis tab.

Uses a ``ttk.Scale`` (rather than ``matplotlib.widgets.Slider``) for the frame
slider so the control looks and behaves consistently with the rest of the
Tkinter UI, per the existing app's styling conventions.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class TrajectoryPreviewCanvas:
    """Embeds a matplotlib figure plus a frame slider for the interactive TP preview."""

    def __init__(self, parent: tk.Widget, on_frame_committed: Callable[[int], None]) -> None:
        self.frame = ttk.Frame(parent)
        self._on_frame_committed = on_frame_committed
        self._frames: list[int] = []

        slider_row = ttk.Frame(self.frame)
        slider_row.pack(fill="x", padx=4, pady=(4, 0))
        self.frame_label_var = tk.StringVar(value="Frame: – / –")
        ttk.Label(slider_row, textvariable=self.frame_label_var, width=22).pack(side="left", padx=4)

        self._frame_var = tk.DoubleVar(value=0)
        self._scale = ttk.Scale(slider_row, from_=0, to=0, variable=self._frame_var, command=self._on_drag)
        self._scale.pack(side="left", fill="x", expand=True, padx=4)
        self._scale.bind("<ButtonRelease-1>", lambda _e: self._commit())

        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)

    def set_available_frames(self, frames: list[int]) -> None:
        """Reset the slider range to the given sorted list of available frame indices."""
        self._frames = sorted(frames)
        if self._frames:
            self._scale.config(from_=0, to=len(self._frames) - 1)
            self._frame_var.set(0)
        else:
            self._scale.config(from_=0, to=0)
        self._update_label()

    def current_frame(self) -> int | None:
        """Return the currently selected (real) frame index, or ``None`` if no data is loaded."""
        if not self._frames:
            return None
        idx = min(max(int(round(self._frame_var.get())), 0), len(self._frames) - 1)
        return self._frames[idx]

    def _on_drag(self, _value: str) -> None:
        self._update_label()

    def _commit(self) -> None:
        frame_idx = self.current_frame()
        if frame_idx is not None:
            self._on_frame_committed(frame_idx)

    def _update_label(self) -> None:
        if not self._frames:
            self.frame_label_var.set("Frame: – / –")
            return
        idx = min(max(int(round(self._frame_var.get())), 0), len(self._frames) - 1)
        self.frame_label_var.set(f"Frame: {self._frames[idx]} ({idx + 1}/{len(self._frames)})")

    def redraw(self, draw_fn: Callable[[plt.Axes], None]) -> None:
        """Clear the axes, call ``draw_fn(ax)`` to repopulate it, then refresh the canvas."""
        self.ax.clear()
        draw_fn(self.ax)
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")
        self.ax.set_aspect("equal", adjustable="datalim")
        self.canvas.draw_idle()
