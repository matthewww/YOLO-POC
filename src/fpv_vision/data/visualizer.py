"""
Random labelled-sample visualiser for YOLO datasets.

Uses OpenCV + matplotlib to draw bounding boxes on random images
and display (or save) the result.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml

from fpv_vision.utils.logging import get_console, get_logger

logger = get_logger(__name__)
console = get_console()

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Deterministic colour palette (one colour per class id)
_PALETTE = [
    (255, 56, 56),
    (255, 157, 151),
    (255, 112, 31),
    (255, 178, 29),
    (207, 210, 49),
    (72, 249, 10),
    (146, 204, 23),
    (61, 219, 134),
    (26, 147, 52),
    (0, 212, 187),
    (44, 153, 168),
    (0, 194, 255),
    (52, 69, 147),
    (100, 115, 255),
    (0, 24, 236),
    (132, 56, 255),
    (82, 0, 133),
    (203, 56, 255),
    (255, 149, 200),
    (255, 55, 199),
]


def _colour(cls_id: int) -> tuple[int, int, int]:
    return _PALETTE[cls_id % len(_PALETTE)]


class DatasetVisualizer:
    """Draw random labelled samples from a YOLO dataset."""

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self._names = self._load_names()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show_samples(
        self,
        n: int = 9,
        split: str = "train",
        save_path: Optional[Path] = None,
    ) -> None:
        """
        Display *n* random labelled samples in a grid.

        Args:
            n: Number of images to show (capped at available images).
            split: Dataset split to sample from ("train" or "val").
            save_path: If provided, save the grid to this path instead of showing.
        """
        img_dir = self.data_dir / "images" / split
        lbl_dir = self.data_dir / "labels" / split
        if not img_dir.exists():
            console.print(f"[red]Image directory not found: {img_dir}[/red]")
            return

        images = [p for p in img_dir.iterdir() if p.suffix.lower() in _IMG_EXTS]
        if not images:
            console.print("[yellow]No images found.[/yellow]")
            return

        sample = random.sample(images, min(n, len(images)))
        cols = min(3, len(sample))
        rows = (len(sample) + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = np.array(axes).reshape(-1)

        for ax, img_path in zip(axes[: len(sample)], sample):
            annotated = self._draw_labels(img_path, lbl_dir)
            ax.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            ax.set_title(img_path.stem, fontsize=8)
            ax.axis("off")

        for ax in axes[len(sample) :]:
            ax.axis("off")

        plt.suptitle(f"FPV Dataset – {split} split ({len(images)} images)", fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            console.print(f"[green]Grid saved to {save_path}[/green]")
        else:
            plt.show()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_names(self) -> dict[int, str]:
        yaml_path = self.data_dir / "data.yaml"
        if yaml_path.exists():
            with yaml_path.open() as f:
                cfg = yaml.safe_load(f)
            return cfg.get("names", {})
        return {}

    def _draw_labels(self, img_path: Path, lbl_dir: Path) -> np.ndarray:
        img = cv2.imread(str(img_path))
        if img is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        h, w = img.shape[:2]
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            return img
        with lbl_path.open() as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                colour = _colour(cls_id)
                cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
                label = self._names.get(cls_id, str(cls_id))
                cv2.putText(
                    img, label, (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA,
                )
        return img
