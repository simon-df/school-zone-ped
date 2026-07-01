"""Reusable Tkinter widget helpers."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


def make_label_entry(
    parent: tk.Widget,
    label_text: str,
    default: str = "",
    row: int = 0,
    col: int = 0,
    label_width: int = 20,
    entry_width: int = 30,
) -> ttk.Entry:
    """Create a labelled :class:`ttk.Entry` widget pair in a grid layout.

    Args:
        parent: Parent widget.
        label_text: Text for the accompanying label.
        default: Default value for the entry.
        row: Grid row.
        col: Grid column for the label (entry placed at ``col + 1``).
        label_width: Width of the label in characters.
        entry_width: Width of the entry in characters.

    Returns:
        The created :class:`ttk.Entry` widget.
    """
    ttk.Label(parent, text=label_text, width=label_width, anchor="w").grid(
        row=row, column=col, padx=4, pady=2, sticky="w"
    )
    entry = ttk.Entry(parent, width=entry_width)
    entry.insert(0, default)
    entry.grid(row=row, column=col + 1, padx=4, pady=2, sticky="ew")
    return entry


def make_scrolled_text(
    parent: tk.Widget,
    height: int = 8,
    width: int = 60,
    state: str = "normal",
) -> tk.Text:
    """Create a :class:`tk.Text` widget with a vertical scrollbar.

    Args:
        parent: Parent widget.
        height: Height in lines.
        width: Width in characters.
        state: Initial state (``"normal"`` or ``"disabled"``).

    Returns:
        The :class:`tk.Text` widget (the scrollbar is packed internally).
    """
    frame = ttk.Frame(parent)
    frame.pack(fill="both", expand=True)
    scrollbar = ttk.Scrollbar(frame, orient="vertical")
    text = tk.Text(frame, height=height, width=width, yscrollcommand=scrollbar.set, state=state)
    scrollbar.config(command=text.yview)
    scrollbar.pack(side="right", fill="y")
    text.pack(side="left", fill="both", expand=True)
    return text


def append_log(text_widget: tk.Text, message: str) -> None:
    """Append *message* to a scrolled text widget and scroll to end.

    Must be called from the main Tkinter thread.

    Args:
        text_widget: The :class:`tk.Text` widget to write to.
        message: Text to append (a newline is added automatically).
    """
    text_widget.config(state="normal")
    text_widget.insert("end", message + "\n")
    text_widget.see("end")
    text_widget.config(state="disabled")


def make_readonly_entry(
    parent: tk.Widget,
    textvariable: tk.StringVar,
    row: int = 0,
    col: int = 0,
    width: int = 50,
) -> ttk.Entry:
    """Create a read-only entry bound to *textvariable*.

    Args:
        parent: Parent widget.
        textvariable: :class:`tk.StringVar` to bind.
        row: Grid row.
        col: Grid column.
        width: Width in characters.

    Returns:
        The read-only :class:`ttk.Entry` widget.
    """
    entry = ttk.Entry(parent, textvariable=textvariable, state="readonly", width=width)
    entry.grid(row=row, column=col, padx=4, pady=2, sticky="ew")
    return entry
