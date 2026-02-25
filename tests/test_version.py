"""Smoke test – package can be imported and version is set."""

import fpv_vision


def test_version_is_set():
    assert hasattr(fpv_vision, "__version__")
    assert fpv_vision.__version__ != ""
