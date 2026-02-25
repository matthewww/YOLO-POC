"""
fpv-vision CLI

Entry-point: fpv-vision

Commands:
    fpv-vision train
    fpv-vision detect webcam
    fpv-vision detect image <path>
    fpv-vision detect video <path>
    fpv-vision dataset validate
    fpv-vision dataset visualize
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint

app = typer.Typer(
    name="fpv-vision",
    help="FPV drone component detection with YOLO.",
    add_completion=False,
)

# ── Sub-apps ──────────────────────────────────────────────────────────────────
detect_app = typer.Typer(help="Run detection from various sources.")
dataset_app = typer.Typer(help="Dataset utilities (validate, visualise).")

app.add_typer(detect_app, name="detect")
app.add_typer(dataset_app, name="dataset")


# ── train ─────────────────────────────────────────────────────────────────────
@app.command()
def train(
    data: Optional[str] = typer.Option(None, "--data", "-d", help="Path to data.yaml"),
    epochs: Optional[int] = typer.Option(None, "--epochs", "-e", help="Number of epochs"),
    batch: Optional[int] = typer.Option(None, "--batch", "-b", help="Batch size"),
    imgsz: Optional[int] = typer.Option(None, "--imgsz", help="Image size"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Run name"),
    resume: bool = typer.Option(False, "--resume", "-r", help="Resume training"),
) -> None:
    """Train the YOLO model on the FPV dataset."""
    from fpv_vision.training import Trainer  # noqa: PLC0415

    trainer = Trainer()
    trainer.train(
        data_yaml=data,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        name=name,
        resume=resume,
    )


# ── detect webcam ─────────────────────────────────────────────────────────────
@detect_app.command("webcam")
def detect_webcam(
    camera: int = typer.Option(0, "--camera", "-c", help="Camera index"),
    weights: Optional[str] = typer.Option(None, "--weights", "-w"),
    confidence: Optional[float] = typer.Option(None, "--conf", help="Confidence threshold"),
    save: bool = typer.Option(False, "--save", help="Save detections"),
    save_dir: Optional[str] = typer.Option(None, "--save-dir"),
) -> None:
    """Run real-time detection from a webcam."""
    from fpv_vision.inference import Detector  # noqa: PLC0415

    detector = Detector(weights=weights, confidence=confidence)
    detector.detect_webcam(camera_index=camera, show=True, save=save, save_dir=save_dir)


# ── detect image ─────────────────────────────────────────────────────────────
@detect_app.command("image")
def detect_image(
    path: Path = typer.Argument(..., help="Path to image or directory"),
    weights: Optional[str] = typer.Option(None, "--weights", "-w"),
    confidence: Optional[float] = typer.Option(None, "--conf"),
    no_show: bool = typer.Option(False, "--no-show", help="Suppress display"),
    save: bool = typer.Option(False, "--save"),
    save_dir: Optional[str] = typer.Option(None, "--save-dir"),
) -> None:
    """Run detection on a saved image or directory of images."""
    from fpv_vision.inference import Detector  # noqa: PLC0415

    detector = Detector(weights=weights, confidence=confidence)
    detector.detect_image(path=path, show=not no_show, save=save, save_dir=save_dir)


# ── detect video ──────────────────────────────────────────────────────────────
@detect_app.command("video")
def detect_video(
    path: Path = typer.Argument(..., help="Path to video file"),
    weights: Optional[str] = typer.Option(None, "--weights", "-w"),
    confidence: Optional[float] = typer.Option(None, "--conf"),
    no_show: bool = typer.Option(False, "--no-show"),
    save: bool = typer.Option(False, "--save"),
    save_dir: Optional[str] = typer.Option(None, "--save-dir"),
) -> None:
    """Run detection on a video file."""
    from fpv_vision.inference import Detector  # noqa: PLC0415

    detector = Detector(weights=weights, confidence=confidence)
    detector.detect_video(path=path, show=not no_show, save=save, save_dir=save_dir)


# ── dataset validate ──────────────────────────────────────────────────────────
@dataset_app.command("validate")
def dataset_validate(
    data_dir: str = typer.Option("data/fpv", "--data-dir", "-d", help="Dataset root"),
) -> None:
    """Validate dataset structure, label/image parity and class balance."""
    from fpv_vision.data import DatasetValidator  # noqa: PLC0415

    validator = DatasetValidator(data_dir)
    ok = validator.validate()
    raise typer.Exit(code=0 if ok else 1)


# ── dataset visualize ─────────────────────────────────────────────────────────
@dataset_app.command("visualize")
def dataset_visualize(
    data_dir: str = typer.Option("data/fpv", "--data-dir", "-d"),
    n: int = typer.Option(9, "--n", help="Number of samples to show"),
    split: str = typer.Option("train", "--split", help="Dataset split"),
    save: Optional[Path] = typer.Option(None, "--save", help="Save grid to file"),
) -> None:
    """Display random labelled samples from the dataset."""
    from fpv_vision.data import DatasetVisualizer  # noqa: PLC0415

    visualizer = DatasetVisualizer(data_dir)
    visualizer.show_samples(n=n, split=split, save_path=save)


if __name__ == "__main__":
    app()
