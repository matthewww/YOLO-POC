"""
YOLO training wrapper.

Wraps Ultralytics YOLO to train on the FPV dataset with settings
derived from the project configuration.
"""

from __future__ import annotations

from pathlib import Path

from fpv_vision.config import get_settings
from fpv_vision.utils.device import get_device
from fpv_vision.utils.logging import get_console, get_logger

logger = get_logger(__name__)
console = get_console()


class Trainer:
    """Thin wrapper around Ultralytics YOLO for FPV training."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def train(
        self,
        data_yaml: str | None = None,
        epochs: int | None = None,
        batch: int | None = None,
        imgsz: int | None = None,
        name: str | None = None,
        resume: bool = False,
    ) -> None:
        """
        Launch YOLO training.

        Args:
            data_yaml: Path to dataset YAML. Falls back to settings.
            epochs:    Override training epochs.
            batch:     Override batch size.
            imgsz:     Override image size.
            name:      Override run name.
            resume:    Resume from the last checkpoint.
        """
        # Lazy import to avoid mandatory GPU setup at import time
        from ultralytics import YOLO  # noqa: PLC0415

        cfg = self.settings
        device = get_device(cfg.model.device)

        weights = cfg.model.weights
        if resume:
            last = Path(cfg.training.project) / (name or cfg.training.name) / "weights" / "last.pt"
            if last.exists():
                weights = str(last)
                console.print(f"[cyan]Resuming from {weights}[/cyan]")
            else:
                console.print("[yellow]No checkpoint found, starting from scratch.[/yellow]")

        console.rule("[bold cyan]FPV Vision – Training")
        console.print(f"  Weights  : {weights}")
        console.print(f"  Device   : {device}")
        console.print(f"  Data     : {data_yaml or cfg.data.yaml}")
        console.print(f"  Epochs   : {epochs or cfg.training.epochs}")

        model = YOLO(weights)
        model.train(
            data=data_yaml or cfg.data.yaml,
            epochs=epochs or cfg.training.epochs,
            batch=batch or cfg.training.batch_size,
            imgsz=imgsz or cfg.model.imgsz,
            device=device,
            optimizer=cfg.training.optimizer,
            lr0=cfg.training.lr0,
            patience=cfg.training.patience,
            save_period=cfg.training.save_period,
            project=cfg.training.project,
            name=name or cfg.training.name,
            pretrained=cfg.training.pretrained,
            augment=cfg.training.augment,
            resume=resume,
            exist_ok=True,
        )
        console.print("[bold green]Training complete.[/bold green]")
