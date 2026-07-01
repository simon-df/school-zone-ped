"""Reusable dialog helpers."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, filedialog
from typing import Sequence


def ask_open_file(
    title: str = "Open file",
    filetypes: Sequence[tuple[str, str]] | None = None,
    initialdir: str = "",
) -> str:
    """Show an open-file dialog and return the selected path (empty string if cancelled).

    Args:
        title: Dialog window title.
        filetypes: Sequence of ``(description, pattern)`` tuples.
        initialdir: Initial directory to show.

    Returns:
        Selected file path, or ``""`` if the user cancelled.
    """
    if filetypes is None:
        filetypes = [("All files", "*.*")]
    return filedialog.askopenfilename(title=title, filetypes=filetypes, initialdir=initialdir)


def ask_save_file(
    title: str = "Save file",
    filetypes: Sequence[tuple[str, str]] | None = None,
    default_extension: str = "",
    initialdir: str = "",
) -> str:
    """Show a save-file dialog and return the selected path (empty string if cancelled).

    Args:
        title: Dialog window title.
        filetypes: Sequence of ``(description, pattern)`` tuples.
        default_extension: Default file extension.
        initialdir: Initial directory to show.

    Returns:
        Selected file path, or ``""`` if the user cancelled.
    """
    if filetypes is None:
        filetypes = [("All files", "*.*")]
    return filedialog.asksaveasfilename(
        title=title,
        filetypes=filetypes,
        defaultextension=default_extension,
        initialdir=initialdir,
    )


def show_error(message: str, title: str = "Error") -> None:
    """Show a modal error dialog.

    Args:
        message: Error message text.
        title: Dialog title.
    """
    messagebox.showerror(title, message)


def show_warning(message: str, title: str = "Warning") -> None:
    """Show a modal warning dialog.

    Args:
        message: Warning message text.
        title: Dialog title.
    """
    messagebox.showwarning(title, message)


def show_info(message: str, title: str = "Information") -> None:
    """Show a modal information dialog.

    Args:
        message: Information text.
        title: Dialog title.
    """
    messagebox.showinfo(title, message)
