"""Tests for device selection utilities."""

from fpv_vision.utils.device import get_device


def test_get_device_explicit_cpu():
    assert get_device("cpu") == "cpu"


def test_get_device_explicit_cuda():
    assert get_device("cuda") == "cuda"


def test_get_device_auto_returns_string():
    device = get_device()
    assert isinstance(device, str)
    assert device in {"cpu", "cuda", "mps"}
