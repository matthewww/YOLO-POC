"""
Dataset structure validator for YOLO-format FPV datasets.

Expected layout::

    data/fpv/
        images/
            train/   *.jpg | *.png | *.jpeg
            val/     *.jpg | *.png | *.jpeg
        labels/
            train/   *.txt
            val/     *.txt
        data.yaml
"""

from __future__ import annotations

from pathlib import Path

import yaml
from rich.table import Table

from fpv_vision.utils.logging import get_console, get_logger

logger = get_logger(__name__)
console = get_console()

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class DatasetValidator:
    """Validates a YOLO-format dataset directory."""

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self) -> bool:
        """
        Run all validation checks.

        Returns:
            True if the dataset passes all checks, False otherwise.
        """
        console.rule("[bold cyan]Dataset Validation")
        ok = True
        ok &= self._check_yaml()
        ok &= self._check_splits()
        ok &= self._check_label_image_parity()
        self._report_class_balance()
        if ok:
            console.print("\n[bold green]✓ Dataset validation passed.[/bold green]")
        else:
            console.print("\n[bold red]✗ Dataset validation failed. See above.[/bold red]")
        return ok

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_yaml(self) -> bool:
        yaml_path = self.data_dir / "data.yaml"
        if not yaml_path.exists():
            console.print(f"[red]Missing data.yaml at {yaml_path}[/red]")
            return False
        with yaml_path.open() as f:
            cfg = yaml.safe_load(f)
        required = {"nc", "names"}
        missing = required - cfg.keys()
        if missing:
            console.print(f"[red]data.yaml missing keys: {missing}[/red]")
            return False
        console.print(f"[green]✓ data.yaml found – {cfg['nc']} classes[/green]")
        return True

    def _check_splits(self) -> bool:
        ok = True
        for split in ("train", "val"):
            img_dir = self.data_dir / "images" / split
            lbl_dir = self.data_dir / "labels" / split
            for d in (img_dir, lbl_dir):
                if not d.exists():
                    console.print(f"[red]Missing directory: {d}[/red]")
                    ok = False
                else:
                    console.print(f"[green]✓ {d.relative_to(self.data_dir.parent)}[/green]")
        return ok

    def _check_label_image_parity(self) -> bool:
        ok = True
        for split in ("train", "val"):
            img_dir = self.data_dir / "images" / split
            lbl_dir = self.data_dir / "labels" / split
            if not img_dir.exists() or not lbl_dir.exists():
                continue
            imgs = {p.stem for p in img_dir.iterdir() if p.suffix.lower() in _IMG_EXTS}
            lbls = {p.stem for p in lbl_dir.iterdir() if p.suffix == ".txt"}
            only_img = imgs - lbls
            only_lbl = lbls - imgs
            if only_img:
                console.print(
                    f"[yellow]⚠ {split}: {len(only_img)} images without labels[/yellow]"
                )
            if only_lbl:
                console.print(
                    f"[yellow]⚠ {split}: {len(only_lbl)} labels without images[/yellow]"
                )
            if not only_img and not only_lbl:
                console.print(
                    f"[green]✓ {split}: {len(imgs)} image/label pairs matched[/green]"
                )
            else:
                ok = False
        return ok

    def _report_class_balance(self) -> None:
        """Print a class-frequency table across all splits."""
        yaml_path = self.data_dir / "data.yaml"
        if not yaml_path.exists():
            return
        with yaml_path.open() as f:
            cfg = yaml.safe_load(f)
        names: dict[int, str] = cfg.get("names", {})

        counts: dict[int, int] = {}
        for split in ("train", "val"):
            lbl_dir = self.data_dir / "labels" / split
            if not lbl_dir.exists():
                continue
            for lbl_file in lbl_dir.glob("*.txt"):
                with lbl_file.open() as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            cls_id = int(parts[0])
                            counts[cls_id] = counts.get(cls_id, 0) + 1

        if not counts:
            console.print("[yellow]No label data found for class balance report.[/yellow]")
            return

        table = Table(title="Class Balance", show_header=True)
        table.add_column("ID", style="dim", justify="right")
        table.add_column("Class", style="cyan")
        table.add_column("Instances", justify="right")
        total = sum(counts.values())
        table.add_column("Share", justify="right")
        for cls_id in sorted(counts):
            name = names.get(cls_id, str(cls_id))
            share = f"{counts[cls_id] / total * 100:.1f}%"
            table.add_row(str(cls_id), name, str(counts[cls_id]), share)
        console.print(table)
