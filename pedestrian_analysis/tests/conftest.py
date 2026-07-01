"""pytest configuration: add project root to sys.path."""

import sys
import os

# Allow bare imports like `from pipeline.calibration import ...`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
