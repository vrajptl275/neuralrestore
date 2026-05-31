#!/usr/bin/env python3
"""
run.py — NeuralRestore Entry Point
==================================
Launches the Gradio web interface.

Usage:
    python run.py              # Launch UI
    python run.py --port 8080  # Custom port
    python run.py --share      # Create public URL
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.app import create_ui


def main():
    parser = argparse.ArgumentParser(
        description="NeuralRestore — AI Image Restoration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Server port (default: 7860)"
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Create a public Gradio share link"
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="Don't auto-open browser"
    )
    args = parser.parse_args()

    demo = create_ui()
    demo.launch(
        server_name = "0.0.0.0",
        server_port = args.port,
        share       = args.share,
        show_error  = True,
        inbrowser   = not args.no_browser,
    )


if __name__ == "__main__":
    main()
