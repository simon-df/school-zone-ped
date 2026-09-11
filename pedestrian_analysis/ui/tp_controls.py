"""Reusable TP (trajectory-prediction) control widgets for the TP Analysis tab.

Named ``tp_controls.py`` rather than ``widgets/prediction_controls.py`` because
``ui/widgets.py`` already exists as a plain module in this package; a
``ui/widgets/`` sub-package with the same name would collide with it.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

TP_MODEL_UI_CHOICES: tuple[str, ...] = (
    "constant_velocity",
    "social_lstm",
    "social_gan",
    "social_stgcnn",
    "trajectron_pp",
    "transformer",
)

# Maps UI-facing model names to pipeline.tp_adapters registry names.
# Models absent from this map are recognized in the UI but not yet implemented.
TP_MODEL_ADAPTER_NAMES: dict[str, str] = {
    "constant_velocity": "dummy",
    "social_lstm": "social_lstm",
    "social_gan": "social_gan",
    "transformer": "transformer",
}


class TPControls:
    """Builds and holds the TP model/parameter control widgets."""

    def __init__(self, parent: tk.Widget, on_change: Callable[[], None]) -> None:
        self.frame = ttk.LabelFrame(parent, text="TP Model Controls", padding=6)
        self._on_change = on_change

        ttk.Label(self.frame, text="TP model:").grid(row=0, column=0, padx=4, pady=2, sticky="w")
        self.model_var = tk.StringVar(value=TP_MODEL_UI_CHOICES[0])
        model_cb = ttk.Combobox(
            self.frame,
            textvariable=self.model_var,
            state="readonly",
            values=list(TP_MODEL_UI_CHOICES),
            width=18,
        )
        model_cb.grid(row=0, column=1, padx=4, pady=2, sticky="ew")
        model_cb.bind("<<ComboboxSelected>>", lambda _e: self._on_change())

        self.num_modes_var = tk.IntVar(value=5)
        self._make_spinbox("Num modes:", self.num_modes_var, 1, 20, row=0, col=2)

        self.pred_len_var = tk.IntVar(value=30)
        self._make_spinbox("Pred frames:", self.pred_len_var, 10, 60, row=1, col=0)

        self.obs_window_var = tk.IntVar(value=20)
        self._make_spinbox("Obs window:", self.obs_window_var, 5, 40, row=1, col=2)

        self.show_all_modes_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            self.frame, text="Show all modes", variable=self.show_all_modes_var, command=self._on_change
        ).grid(row=2, column=0, columnspan=2, padx=4, pady=2, sticky="w")

        self.top_k_var = tk.IntVar(value=3)
        self._make_spinbox("Top-K modes:", self.top_k_var, 1, 10, row=2, col=2)

        # Ground-truth loading is not implemented yet; this is a UI placeholder per the plan.
        self.show_ground_truth_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self.frame,
            text="Show ground truth (not yet implemented)",
            variable=self.show_ground_truth_var,
            command=self._on_change,
        ).grid(row=3, column=0, columnspan=2, padx=4, pady=2, sticky="w")

        self.inference_time_var = tk.StringVar(value="Inference time: – ms")
        ttk.Label(self.frame, textvariable=self.inference_time_var).grid(
            row=3, column=2, columnspan=2, padx=4, pady=2, sticky="w"
        )

    def _make_spinbox(self, label: str, var: tk.IntVar, lo: int, hi: int, row: int, col: int) -> None:
        ttk.Label(self.frame, text=label).grid(row=row, column=col, padx=4, pady=2, sticky="w")
        spin = tk.Spinbox(self.frame, from_=lo, to=hi, textvariable=var, width=6, command=self._on_change)
        spin.grid(row=row, column=col + 1, padx=4, pady=2, sticky="w")
        spin.bind("<Return>", lambda _e: self._on_change())
        spin.bind("<FocusOut>", lambda _e: self._on_change())

    def selected_adapter_name(self) -> str | None:
        """Return the pipeline adapter registry name, or ``None`` if not yet implemented."""
        return TP_MODEL_ADAPTER_NAMES.get(self.model_var.get())
