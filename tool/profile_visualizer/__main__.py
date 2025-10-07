#!/usr/bin/env python3
"""
Simple web-based profiling visualization tool.

Usage:
    # Start with an initial file
    python -m profile_visualizer <path_to_profiling_json>

    # Start without a file (uses parent directory for file search)
    python -m profile_visualizer

    Example:
    python -m profile_visualizer pie-metal/20251003_205526_profiling_result.json
    python -m profile_visualizer

Features:
    - Load files from server directory dropdown
    - Drag & drop JSON files
    - Click to browse local files
"""

import json
import sys
from pathlib import Path

from .server import start_server


def main():
    """Main entry point."""
    initial_data = None
    search_dir = None

    # Optional: Load initial file if provided
    if len(sys.argv) >= 2:
        json_path = Path(sys.argv[1])
        if not json_path.exists():
            print(f"Error: File not found: {json_path}")
            sys.exit(1)

        # Load profiling data
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                initial_data = json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error: Failed to parse JSON from {json_path}: {e}")
            sys.exit(1)

        # Set search directory to the parent directory of the provided file
        search_dir = str(json_path.parent.absolute())
    else:
        # No file provided, use parent directory of script location
        script_path = Path(__file__).resolve()
        search_dir = str(script_path.parent.parent.parent)  # Go up to workspace root

    # Get static files directory
    static_dir = str(Path(__file__).parent / "static")

    # Start the server
    start_server(
        port=8000,
        search_dir=search_dir,
        initial_data=initial_data,
        static_dir=static_dir,
    )


if __name__ == "__main__":
    main()
