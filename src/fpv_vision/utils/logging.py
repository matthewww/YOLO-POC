"""Logging and Rich console helpers."""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

_console = Console()
_configured = False


def get_console() -> Console:
    """Return the shared Rich console instance."""
    return _console


def get_logger(name: str = "fpv_vision") -> logging.Logger:
    """Configure root logging with Rich (once) and return a named logger."""
    global _configured
    if not _configured:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=_console, rich_tracebacks=True)],
        )
        _configured = True
    return logging.getLogger(name)
