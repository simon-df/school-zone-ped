"""Root entry point for the local desktop application."""
from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info < (3, 11):
    raise SystemExit("This application requires Python 3.11 or newer.")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pedestrian_analysis.app import main


if __name__ == "__main__":
    main()
