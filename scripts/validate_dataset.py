#!/usr/bin/env python3
"""
Standalone dataset validation script.

Usage:
    python scripts/validate_dataset.py [--data-dir data/fpv]

This script can be run without installing the fpv-vision package.
"""

import argparse
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fpv_vision.data.validator import DatasetValidator  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate YOLO FPV dataset structure.")
    parser.add_argument(
        "--data-dir",
        default="data/fpv",
        help="Path to dataset root (default: data/fpv)",
    )
    args = parser.parse_args()
    validator = DatasetValidator(args.data_dir)
    ok = validator.validate()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
