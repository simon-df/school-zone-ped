"""Generic I/O helpers: JSON and NumPy file read/write."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import logging
logger = logging.getLogger(__name__)


def save_json(data: Any, path: str | Path) -> None:
    """Serialize *data* to JSON at *path*.

    Args:
        data: JSON-serialisable Python object.
        path: Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    logger.debug("Saved JSON: %s", path)


def load_json(path: str | Path) -> Any:
    """Load a JSON file and return the parsed content.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed Python object.

    Raises:
        FileNotFoundError: When *path* does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: '{path}'")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_npy(array: np.ndarray, path: str | Path) -> None:
    """Save a NumPy array to a ``.npy`` file.

    Args:
        array: Array to save.
        path: Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), array)
    logger.debug("Saved .npy: %s", path)


def load_npy(path: str | Path) -> np.ndarray:
    """Load a NumPy array from a ``.npy`` file.

    Args:
        path: Path to the ``.npy`` file.

    Returns:
        Loaded NumPy array.

    Raises:
        FileNotFoundError: When *path* does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f".npy file not found: '{path}'")
    return np.load(str(path))
