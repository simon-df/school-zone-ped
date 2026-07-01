"""Entry point for the Pedestrian Crossing Trajectory Analysis desktop application.

Start the application with::

    python app.py
"""

import logging
import sys


def _configure_logging() -> None:
    """Set up the root logger with a sensible default format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def main() -> None:
    """Application entry point."""
    _configure_logging()

    # Ensure all project directories exist before any module touches them.
    from utils.paths import setup_project_directories

    setup_project_directories()

    # Launch the desktop UI
    from ui.main_window import PedestrianAnalysisApp

    app = PedestrianAnalysisApp()
    app.run()


if __name__ == "__main__":
    main()
