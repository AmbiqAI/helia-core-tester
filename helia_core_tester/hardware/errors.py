"""Errors the hardware CLI maps to exit codes."""

from __future__ import annotations


class RunRefused(RuntimeError):
    """Inputs do not fit; nothing ran."""
