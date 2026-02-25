"""
YOLO inference engine supporting webcam, image, and video sources.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from fpv_vision.config import get_settings
from fpv_vision.utils.device import get_device
from fpv_vision.utils.logging import get_console, get_logger

logger = get_logger(__name__)
console = get_console()


class Detector:
    """Runs YOLO detection on a given source."""

    def __init__(
        self,
        weights: str | None = None,
        confidence: float | None = None,
        device: str | None = None,
    ) -> None:
        cfg = get_settings()
        self.weights = weights or cfg.model.weights
        self.confidence = confidence if confidence is not None else cfg.model.confidence
        self.device = get_device(device or cfg.model.device)
        self._model = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_webcam(
        self,
        camera_index: int = 0,
        show: bool = True,
        save: bool = False,
        save_dir: str | None = None,
    ) -> None:
        """Run real-time detection from a webcam."""
        self._run(
            source=camera_index,
            show=show,
            save=save,
            save_dir=save_dir,
            stream=True,
        )

    def detect_image(
        self,
        path: Union[str, Path],
        show: bool = True,
        save: bool = False,
        save_dir: str | None = None,
    ) -> None:
        """Run detection on a single image or directory of images."""
        self._run(
            source=str(path),
            show=show,
            save=save,
            save_dir=save_dir,
            stream=False,
        )

    def detect_video(
        self,
        path: Union[str, Path],
        show: bool = True,
        save: bool = False,
        save_dir: str | None = None,
    ) -> None:
        """Run detection on a video file."""
        self._run(
            source=str(path),
            show=show,
            save=save,
            save_dir=save_dir,
            stream=True,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_model(self):
        if self._model is None:
            from ultralytics import YOLO  # noqa: PLC0415

            self._model = YOLO(self.weights)
        return self._model

    def _run(
        self,
        source,
        show: bool,
        save: bool,
        save_dir: str | None,
        stream: bool,
    ) -> None:
        cfg = get_settings()
        effective_save_dir = save_dir or cfg.inference.save_dir
        model = self._load_model()

        console.rule("[bold cyan]FPV Vision – Detection")
        console.print(f"  Source     : {source}")
        console.print(f"  Weights    : {self.weights}")
        console.print(f"  Device     : {self.device}")
        console.print(f"  Confidence : {self.confidence}")

        results = model.predict(
            source=source,
            conf=self.confidence,
            device=self.device,
            show=show,
            save=save,
            project=effective_save_dir if save else None,
            stream=stream,
            verbose=False,
        )

        # Consume the generator so prediction actually runs
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes):
                logger.info(
                    "Detected %d object(s): %s",
                    len(boxes),
                    [result.names[int(c)] for c in boxes.cls.tolist()],
                )
