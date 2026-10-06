"""Reusable TP (trajectory-prediction) control widgets for the TP Analysis tab.

Named ``tp_controls.py`` rather than ``widgets/prediction_controls.py`` because
``ui/widgets.py`` already exists as a plain module in this package; a
``ui/widgets/`` sub-package with the same name would collide with it.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

from pipeline.tp_model_registry import TPModelRegistry
from ui.dialogs import ask_open_file

TP_MODEL_UI_CHOICES: tuple[str, ...] = (
    "constant_velocity",
    "social_lstm",
    "social_gan",
    "social_stgcnn",
    "trajectron_pp",
    "transformer",
)

# Maps UI-facing model names to pipeline.tp_model_registry / tp_adapters names.
TP_MODEL_ADAPTER_NAMES: dict[str, str] = {
    "constant_velocity": "dummy",
    "social_lstm": "social_lstm",
    "social_gan": "social_gan",
    "social_stgcnn": "social_stgcnn",
    "trajectron_pp": "trajectron_pp",
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

        ttk.Label(self.frame, text="Checkpoint:").grid(row=4, column=0, padx=4, pady=2, sticky="w")
        self.checkpoint_path_var = tk.StringVar(value="")
        ttk.Entry(self.frame, textvariable=self.checkpoint_path_var, width=28).grid(
            row=4, column=1, columnspan=2, padx=4, pady=2, sticky="ew"
        )
        ttk.Button(self.frame, text="Browse...", command=self._on_browse_checkpoint).grid(
            row=4, column=3, padx=4, pady=2
        )

        self._info_text = tk.Text(self.frame, wrap="word", height=5, width=60, state="disabled")
        self._info_text.grid(row=5, column=0, columnspan=4, padx=4, pady=(4, 2), sticky="ew")

        model_cb.bind("<<ComboboxSelected>>", lambda _e: self._on_model_changed())
        self._on_model_changed()

    def _on_model_changed(self) -> None:
        self._update_checkpoint_info()
        self._on_change()

    def _on_browse_checkpoint(self) -> None:
        path = ask_open_file(
            "Select TP model checkpoint",
            filetypes=[("Checkpoint files", "*.pt *.pth"), ("All files", "*.*")],
        )
        if path:
            self.checkpoint_path_var.set(path)
            self._on_change()

    def _update_checkpoint_info(self) -> None:
        registry_name = self.selected_adapter_name()
        self._info_text.config(state="normal")
        self._info_text.delete("1.0", tk.END)
        try:
            info = TPModelRegistry.get_checkpoint_info(registry_name) if registry_name else None
        except ValueError:
            info = None

        if info is None:
            self._info_text.insert(tk.END, f"Model '{self.model_var.get()}' is not registered.")
        elif info.required:
            self._info_text.insert(
                tk.END,
                f"Checkpoint REQUIRED.\nOfficial repo: {info.official_repo}\n\n"
                f"{info.download_instructions}\n\nNotes: {info.notes}",
            )
        else:
            self._info_text.insert(
                tk.END,
                f"Checkpoint optional.\n{info.download_instructions}\n\nNotes: {info.notes}",
            )
        self._info_text.config(state="disabled")

    def _make_spinbox(self, label: str, var: tk.IntVar, lo: int, hi: int, row: int, col: int) -> None:
        ttk.Label(self.frame, text=label).grid(row=row, column=col, padx=4, pady=2, sticky="w")
        spin = tk.Spinbox(self.frame, from_=lo, to=hi, textvariable=var, width=6, command=self._on_change)
        spin.grid(row=row, column=col + 1, padx=4, pady=2, sticky="w")
        spin.bind("<Return>", lambda _e: self._on_change())
        spin.bind("<FocusOut>", lambda _e: self._on_change())

    def selected_adapter_name(self) -> str | None:
        """Return the pipeline/registry model name for the current UI selection."""
        return TP_MODEL_ADAPTER_NAMES.get(self.model_var.get())

    def checkpoint_path(self) -> str | None:
        """Return the checkpoint path entered by the user, or ``None`` if empty."""
        path = self.checkpoint_path_var.get().strip()
        return path or None
