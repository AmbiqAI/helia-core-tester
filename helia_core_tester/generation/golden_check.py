"""Reject goldens too flat to catch bugs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np

# Descriptor key: reason a degenerate golden is intended.
EDGE_CASE_KEY = "degenerate_golden_reason"

# Too few elements to judge spread.
MIN_ELEMENTS = 8

# Share of outputs at the dtype bounds.
MAX_SATURATED_SHARE = 0.9

_BOUNDS = {"int8_t": (-128, 127), "int16_t": (-32768, 32767)}
_GOLDEN_RE = re.compile(
    r"static\s+const\s+(int8_t|int16_t|bool)\s+(\w+_expected_output)\s*\[\s*\]\s*=\s*\{([^}]*)\}"
)


def golden_problem(values: np.ndarray, c_type: str) -> str | None:
    """Say why a golden is degenerate, else None."""
    flat = np.asarray(values).reshape(-1)
    if flat.size < MIN_ELEMENTS:
        return None
    distinct = np.unique(flat).size
    if c_type == "bool":
        return "constant bool output" if distinct == 1 else None
    if distinct <= 2:
        return f"{distinct} distinct value(s) over {flat.size}"
    low, high = _BOUNDS[c_type]
    share = float(np.mean((flat == low) | (flat == high)))
    if share >= MAX_SATURATED_SHARE:
        return f"{share:.0%} of {flat.size} at dtype bounds"
    return None


def _parse_literal(body: str) -> np.ndarray:
    tokens = [t.strip() for t in body.replace("\n", " ").split(",") if t.strip()]
    mapped = [{"true": "1", "false": "0"}.get(t, t) for t in tokens]
    return np.array([int(t, 0) for t in mapped], dtype=np.int64)


def case_problems(case_dir: Path) -> list[str]:
    """List degenerate goldens in one case."""
    problems: list[str] = []
    for header in sorted((case_dir / "includes").glob("*.h")):
        for c_type, array_name, body in _GOLDEN_RE.findall(header.read_text()):
            problem = golden_problem(_parse_literal(body), c_type)
            if problem:
                problems.append(f"{array_name}: {problem}")
    return problems


def _status_only(desc: Mapping[str, Any]) -> bool:
    """True when the case checks a status."""
    status = str(desc.get("expected_status", "ARM_CMSIS_NN_SUCCESS"))
    return bool(desc.get("fault")) or status != "ARM_CMSIS_NN_SUCCESS"


def check_case_golden(case_dir: Path, desc: Mapping[str, Any]) -> None:
    """Fail on a degenerate golden without a reason."""
    if desc.get(EDGE_CASE_KEY) or _status_only(desc):
        return
    problems = case_problems(case_dir)
    if problems:
        raise ValueError(
            f"{desc.get('name')}: degenerate golden ({'; '.join(problems)}); "
            f"fix the data or set {EDGE_CASE_KEY}"
        )
