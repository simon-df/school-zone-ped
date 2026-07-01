"""Calibration tab: image selection, 4-point click, homography computation and saving."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk

from ui.dialogs import ask_open_file, ask_save_file, show_error, show_info, show_warning
from ui.widgets import make_label_entry, append_log
from utils.image_utils import bgr_to_pil, scale_image_for_canvas, canvas_coords_to_image_coords, draw_points_on_image

logger = logging.getLogger(__name__)

_CANVAS_W = 640
_CANVAS_H = 480


class CalibrationTab(ttk.Frame):
    """UI tab for homography calibration.

    Allows the user to:
    * Load a reference image.
    * Click four points on the image canvas.
    * Enter corresponding real-world metre coordinates.
    * Compute the homography.
    * Save the result as ``.npy`` + ``.json``.
    """

    def __init__(self, parent: tk.Widget, app_state) -> None:
        super().__init__(parent)
        self._state = app_state
        self._image_path: Optional[str] = None
        self._original_image: Optional[np.ndarray] = None
        self._scale: float = 1.0
        self._offset_x: float = 0.0
        self._offset_y: float = 0.0
        self._pixel_points: list[tuple[float, float]] = []
        self._tk_image = None  # keep reference to avoid GC

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

        # --- Top control row ---
        ctrl = ttk.LabelFrame(self, text="Image & Points", padding=6)
        ctrl.grid(row=0, column=0, columnspan=2, sticky="ew", padx=6, pady=4)
        ctrl.columnconfigure(1, weight=1)

        ttk.Button(ctrl, text="Select Image", command=self._on_select_image).grid(row=0, column=0, padx=4, pady=2)
        self._img_path_var = tk.StringVar(value="<no image selected>")
        ttk.Entry(ctrl, textvariable=self._img_path_var, state="readonly", width=50).grid(
            row=0, column=1, padx=4, pady=2, sticky="ew"
        )
        ttk.Button(ctrl, text="Reset Points", command=self._on_reset_points).grid(row=0, column=2, padx=4, pady=2)

        # --- Canvas ---
        canvas_frame = ttk.LabelFrame(self, text="Image (click to set pixel points)", padding=4)
        canvas_frame.grid(row=1, column=0, sticky="nsew", padx=6, pady=4)
        self._canvas = tk.Canvas(canvas_frame, width=_CANVAS_W, height=_CANVAS_H, bg="#2b2b2b", cursor="crosshair")
        self._canvas.pack(fill="both", expand=True)
        self._canvas.bind("<Button-1>", self._on_canvas_click)

        # --- Coordinate tables ---
        coord_frame = ttk.LabelFrame(self, text="Coordinates", padding=6)
        coord_frame.grid(row=1, column=1, sticky="nsew", padx=6, pady=4)
        coord_frame.columnconfigure(1, weight=1)
        coord_frame.columnconfigure(3, weight=1)

        ttk.Label(coord_frame, text="Pixel points (px, py)", font=("", 10, "bold")).grid(
            row=0, column=0, columnspan=2, pady=2
        )
        ttk.Label(coord_frame, text="World points (x_m, y_m)", font=("", 10, "bold")).grid(
            row=0, column=2, columnspan=2, pady=2
        )

        self._px_entries: list[tuple[ttk.Entry, ttk.Entry]] = []
        self._wx_entries: list[tuple[ttk.Entry, ttk.Entry]] = []

        for i in range(4):
            ttk.Label(coord_frame, text=f"P{i + 1}").grid(row=i + 1, column=0, padx=4, sticky="e")
            px_e = ttk.Entry(coord_frame, width=7)
            py_e = ttk.Entry(coord_frame, width=7)
            px_e.grid(row=i + 1, column=1, padx=2, pady=2)
            py_e.grid(row=i + 1, column=1, padx=(72, 2), pady=2)
            self._px_entries.append((px_e, py_e))

            ttk.Label(coord_frame, text=f"P{i + 1}").grid(row=i + 1, column=2, padx=4, sticky="e")
            wx_e = ttk.Entry(coord_frame, width=7)
            wy_e = ttk.Entry(coord_frame, width=7)
            wx_e.grid(row=i + 1, column=3, padx=2, pady=2)
            wy_e.grid(row=i + 1, column=3, padx=(72, 2), pady=2)
            self._wx_entries.append((wx_e, wy_e))

        # Grid spacing
        ttk.Label(coord_frame, text="Grid spacing (m):").grid(row=5, column=0, columnspan=2, pady=4, sticky="e")
        self._grid_spacing_var = tk.StringVar(value="1.0")
        ttk.Entry(coord_frame, textvariable=self._grid_spacing_var, width=8).grid(row=5, column=2, pady=4, sticky="w")

        # Action buttons
        btn_frame = ttk.Frame(coord_frame)
        btn_frame.grid(row=6, column=0, columnspan=4, pady=6)
        ttk.Button(btn_frame, text="Compute Calibration", command=self._on_compute).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Save Calibration", command=self._on_save).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Load Calibration", command=self._on_load).pack(side="left", padx=4)

        # BEV preview
        bev_frame = ttk.LabelFrame(self, text="BEV Validation Preview", padding=4)
        bev_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=6, pady=4)
        self._bev_canvas = tk.Canvas(bev_frame, width=_CANVAS_W, height=200, bg="#1a1a1a")
        self._bev_canvas.pack()

        self._saved_path_var = tk.StringVar(value="")
        ttk.Label(bev_frame, textvariable=self._saved_path_var, foreground="green").pack(pady=2)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_select_image(self) -> None:
        path = ask_open_file(
            title="Select calibration image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("All files", "*.*")],
        )
        if not path:
            return
        self._image_path = path
        self._img_path_var.set(path)
        self._original_image = cv2.imread(path)
        if self._original_image is None:
            show_error(f"Cannot read image: {path}")
            self._image_path = None
            return
        self._pixel_points = []
        self._render_canvas()

    def _on_canvas_click(self, event: tk.Event) -> None:
        if self._original_image is None:
            return
        if len(self._pixel_points) >= 4:
            show_warning("Already 4 points selected. Press 'Reset Points' to start over.")
            return
        img_x, img_y = canvas_coords_to_image_coords(event.x, event.y, self._scale, self._offset_x, self._offset_y)
        self._pixel_points.append((img_x, img_y))
        idx = len(self._pixel_points) - 1
        # Update entry fields
        px_e, py_e = self._px_entries[idx]
        px_e.delete(0, "end")
        px_e.insert(0, f"{img_x:.1f}")
        py_e.delete(0, "end")
        py_e.insert(0, f"{img_y:.1f}")
        self._render_canvas()

    def _on_reset_points(self) -> None:
        self._pixel_points = []
        for px_e, py_e in self._px_entries:
            px_e.delete(0, "end")
            py_e.delete(0, "end")
        self._render_canvas()

    def _on_compute(self) -> None:
        if self._original_image is None:
            show_error("Please load an image first.")
            return
        try:
            pixel_pts = self._read_pixel_points()
            world_pts = self._read_world_points()
            grid_spacing = float(self._grid_spacing_var.get())
        except ValueError as exc:
            show_error(f"Invalid input: {exc}")
            return

        from pipeline.calibration import compute_homography, validate_calibration

        try:
            H = compute_homography(np.array(pixel_pts), np.array(world_pts))
        except Exception as exc:
            show_error(f"Calibration failed: {exc}")
            return

        self._state.homography = H
        logger.info("Homography computed successfully.")

        bev = validate_calibration(self._original_image, H, grid_spacing)
        self._render_bev(bev)
        show_info("Calibration computed successfully.")

    def _on_save(self) -> None:
        if self._state.homography is None:
            show_error("No calibration computed yet. Please run 'Compute Calibration' first.")
            return
        from config import CALIBRATION_DIR

        path = ask_save_file(
            title="Save calibration matrix",
            filetypes=[("NumPy files", "*.npy")],
            default_extension=".npy",
            initialdir=str(CALIBRATION_DIR),
        )
        if not path:
            return

        from pipeline.calibration import save_calibration, save_calibration_metadata

        try:
            save_calibration(self._state.homography, path)
            json_path = Path(path).with_suffix(".json")
            save_calibration_metadata(
                json_path=json_path,
                original_image_path=self._image_path or "",
                pixel_points=[list(p) for p in self._read_pixel_points()],
                meter_points=[list(p) for p in self._read_world_points()],
                grid_spacing_m=float(self._grid_spacing_var.get()),
                homography_file=path,
            )
            self._state.calibration_path = Path(path)
            self._saved_path_var.set(f"Saved: {path}")
            self._persist_last_calibration(path, json_path)
            show_info(f"Calibration saved:\n{path}\n{json_path}")
        except Exception as exc:
            show_error(f"Failed to save calibration: {exc}")

    def _on_load(self) -> None:
        from config import CALIBRATION_DIR
        from pipeline.calibration import load_calibration

        path = ask_open_file(
            title="Load calibration matrix",
            filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")],
            initialdir=str(CALIBRATION_DIR),
        )
        if not path:
            return
        try:
            H = load_calibration(path)
            self._state.homography = H
            self._state.calibration_path = Path(path)
            self._saved_path_var.set(f"Loaded: {path}")
            show_info(f"Calibration loaded from:\n{path}")
        except Exception as exc:
            show_error(f"Failed to load calibration: {exc}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _read_pixel_points(self) -> list[tuple[float, float]]:
        pts = []
        for i, (px_e, py_e) in enumerate(self._px_entries):
            px_val = px_e.get().strip()
            py_val = py_e.get().strip()
            if not px_val or not py_val:
                raise ValueError(f"Pixel point P{i + 1} is incomplete.")
            pts.append((float(px_val), float(py_val)))
        return pts

    def _read_world_points(self) -> list[tuple[float, float]]:
        pts = []
        for i, (wx_e, wy_e) in enumerate(self._wx_entries):
            wx_val = wx_e.get().strip()
            wy_val = wy_e.get().strip()
            if not wx_val or not wy_val:
                raise ValueError(f"World point P{i + 1} is incomplete.")
            pts.append((float(wx_val), float(wy_val)))
        return pts

    def _render_canvas(self) -> None:
        if self._original_image is None:
            self._canvas.delete("all")
            return
        annotated = draw_points_on_image(self._original_image, self._pixel_points)
        from PIL import Image as PILImage

        pil_img = bgr_to_pil(annotated)
        canvas_w = self._canvas.winfo_width() or _CANVAS_W
        canvas_h = self._canvas.winfo_height() or _CANVAS_H
        scaled, self._scale = scale_image_for_canvas(pil_img, canvas_w, canvas_h)
        self._offset_x = (canvas_w - scaled.width) / 2
        self._offset_y = (canvas_h - scaled.height) / 2
        from utils.image_utils import pil_to_tkimage

        self._tk_image = pil_to_tkimage(scaled)
        self._canvas.delete("all")
        self._canvas.create_image(self._offset_x, self._offset_y, anchor="nw", image=self._tk_image)

    def _render_bev(self, bev: np.ndarray) -> None:
        pil_img = bgr_to_pil(bev)
        w = self._bev_canvas.winfo_width() or _CANVAS_W
        h = self._bev_canvas.winfo_height() or 200
        scaled, _ = scale_image_for_canvas(pil_img, w, h)
        from utils.image_utils import pil_to_tkimage

        self._tk_bev_image = pil_to_tkimage(scaled)
        self._bev_canvas.delete("all")
        self._bev_canvas.create_image(0, 0, anchor="nw", image=self._tk_bev_image)

    def _persist_last_calibration(self, npy_path: str, json_path: Path) -> None:
        """Save the last calibration paths for auto-reload at next startup."""
        from config import CALIBRATION_DIR
        from utils.io_utils import save_json

        record = {"npy_path": str(npy_path), "json_path": str(json_path)}
        try:
            save_json(record, CALIBRATION_DIR / "last_calibration.json")
        except Exception as exc:
            logger.warning("Could not persist last calibration: %s", exc)

    def load_last_calibration(self) -> None:
        """Try to restore the last calibration from disk (called at startup)."""
        from config import CALIBRATION_DIR
        from utils.io_utils import load_json
        from pipeline.calibration import load_calibration

        last_path = CALIBRATION_DIR / "last_calibration.json"
        if not last_path.is_file():
            return
        try:
            record = load_json(last_path)
            npy_path = record.get("npy_path", "")
            if npy_path and Path(npy_path).is_file():
                H = load_calibration(npy_path)
                self._state.homography = H
                self._state.calibration_path = Path(npy_path)
                self._saved_path_var.set(f"Auto-loaded: {npy_path}")
                logger.info("Auto-loaded last calibration from '%s'", npy_path)
        except Exception as exc:
            logger.warning("Could not auto-load last calibration: %s", exc)
