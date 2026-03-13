"""FVP orchestration error types."""

from __future__ import annotations


class FvpScriptError(RuntimeError):
    """Typed script error carrying process exit code."""

    def __init__(self, message: str, exit_code: int = 2):
        super().__init__(message)
        self.exit_code = exit_code
