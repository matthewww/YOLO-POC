"""
Annotation drawing utilities using OpenCV.

These are thin helpers on top of OpenCV's drawing primitives.
For richer visualisation use the ``supervision`` library directly.
"""

from __future__ import annotations

import cv2
import numpy as np

# One BGR colour per class slot (up to 20 classes)
_PALETTE: list[tuple[int, int, int]] = [
    (56, 56, 255),
    (151, 157, 255),
    (31, 112, 255),
    (29, 178, 255),
    (49, 210, 207),
    (10, 249, 72),
    (23, 204, 146),
    (134, 219, 61),
    (52, 147, 26),
    (187, 212, 0),
    (168, 153, 44),
    (255, 194, 0),
    (147, 69, 52),
    (255, 115, 100),
    (236, 24, 0),
    (255, 56, 132),
    (133, 0, 82),
    (255, 56, 203),
    (200, 149, 255),
    (199, 55, 255),
]


def _colour(cls_id: int) -> tuple[int, int, int]:
    return _PALETTE[cls_id % len(_PALETTE)]


def draw_detections(
    image: np.ndarray,
    boxes_xyxy: np.ndarray,
    class_ids: list[int],
    class_names: dict[int, str],
    confidences: list[float] | None = None,
    thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """
    Draw bounding boxes and labels onto *image*.

    Args:
        image:       BGR image (H, W, 3) – modified in-place and returned.
        boxes_xyxy:  (N, 4) array of [x1, y1, x2, y2] in pixel coordinates.
        class_ids:   List of integer class IDs, length N.
        class_names: Mapping of class ID → class name string.
        confidences: Optional list of confidence scores, length N.
        thickness:   Box and text line thickness.
        font_scale:  OpenCV font scale for labels.

    Returns:
        Annotated BGR image.
    """
    for i, (box, cls_id) in enumerate(zip(boxes_xyxy, class_ids)):
        x1, y1, x2, y2 = map(int, box)
        colour = _colour(cls_id)
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, thickness)

        label = class_names.get(cls_id, str(cls_id))
        if confidences is not None:
            label = f"{label} {confidences[i]:.2f}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw + 2, y1), colour, -1)
        cv2.putText(
            image,
            label,
            (x1 + 1, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return image
