"""Main application window with ttk.Notebook and three tabs."""

from __future__ import annotations

import logging
import sys
import tkinter as tk
from tkinter import ttk

from ui.state import AppState

logger = logging.getLogger(__name__)


class PedestrianAnalysisApp:
    """Root Tkinter application.

    Creates the main window, sets up the :class:`ttk.Notebook` with three tabs,
    and holds the shared :class:`~ui.state.AppState`.
    """

    def __init__(self) -> None:
        self._root = tk.Tk()
        self._root.title("Pedestrian Crossing Trajectory Analysis")
        self._root.minsize(1100, 700)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._state = AppState()
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # Notebook
        self._notebook = ttk.Notebook(self._root)
        self._notebook.pack(fill="both", expand=True, padx=4, pady=4)

        # Lazy import to avoid circular issues
        from ui.calibration_tab import CalibrationTab
        from ui.extraction_tab import ExtractionTab
        from ui.analysis_tab import AnalysisTab

        self._calib_tab = CalibrationTab(self._notebook, self._state)
        self._extract_tab = ExtractionTab(self._notebook, self._state)
        self._analysis_tab = AnalysisTab(self._notebook, self._state)

        self._notebook.add(self._calib_tab, text="  Kalibrierung  ")
        self._notebook.add(self._extract_tab, text="  Trajektorienextraktion  ")
        self._notebook.add(self._analysis_tab, text="  Analyse  ")

        # Status bar
        self._status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(
            self._root,
            textvariable=self._status_var,
            relief="sunken",
            anchor="w",
            padding=(4, 2),
        )
        status_bar.pack(side="bottom", fill="x")

        # Try to auto-load the last saved calibration
        self._root.after(200, self._calib_tab.load_last_calibration)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_status(self, message: str) -> None:
        """Update the status bar text.

        Args:
            message: Short human-readable status string.
        """
        self._status_var.set(message)

    def run(self) -> None:
        """Start the Tkinter main loop."""
        logger.info("Starting main loop.")
        self._root.mainloop()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_close(self) -> None:
        logger.info("Application closing.")
        self._root.destroy()
        sys.exit(0)
