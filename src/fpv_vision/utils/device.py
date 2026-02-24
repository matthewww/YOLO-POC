"""Device selection utilities."""

from __future__ import annotations


def get_device(preferred: str = "") -> str:
    """
    Return the best available compute device.

    Args:
        preferred: Explicit device string ("cpu", "cuda", "mps").
                   Empty string triggers auto-detection.

    Returns:
        Device string suitable for Ultralytics / PyTorch.
    """
    if preferred:
        return preferred
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"
