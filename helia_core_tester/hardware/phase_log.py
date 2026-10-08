"""Wall-clock phase marks, for timing evals.

Set HCT_PHASE_LOG to a file; each mark appends one JSON line
(t, pid, mark). Unset, marks cost one env lookup.
"""

from __future__ import annotations

import json
import os
import time


def mark(name: str) -> None:
    """Append one mark when enabled."""
    path = os.environ.get("HCT_PHASE_LOG")
    if path:
        with open(path, "a", encoding="utf-8") as out:
            out.write(json.dumps({"t": time.time(), "pid": os.getpid(), "mark": name}) + "\n")
