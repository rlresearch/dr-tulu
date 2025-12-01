"""
dr_agent_ui - Pre-compiled web interface for DR-Agent workflows.

This package contains the static web interface for DR-Agent workflows,
allowing for easy distribution and deployment without requiring a separate
Node.js frontend server.
"""

import os
from pathlib import Path

__version__ = "0.1.0.post1"

# Path to the static files directory
STATIC_DIR = Path(__file__).parent / "static"


def get_static_dir() -> Path:
    """
    Get the path to the static files directory.

    Returns:
        Path to the directory containing pre-compiled UI files.
    """
    return STATIC_DIR


def is_ui_available() -> bool:
    """
    Check if the pre-compiled UI files are available.

    Returns:
        True if UI files exist, False otherwise.
    """
    index_file = STATIC_DIR / "index.html"
    return index_file.exists()


from .server import mount_ui

__all__ = ["get_static_dir", "is_ui_available", "mount_ui", "__version__"]
