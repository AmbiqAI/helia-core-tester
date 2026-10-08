"""Campaign YAML: what to optimize, where, and the limits."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from helia_core_tester.hardware.boards import UnknownBoardError, resolve_board
from helia_core_tester.hardware.nsx_app import PLACEMENTS

NAME_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,47}")
WORD_RE = re.compile(r"[A-Za-z0-9_]+")
CASE_RE = re.compile(r"[A-Za-z0-9_.-]+")
# Hidden shapes exist only for these.
HIDDEN_OPS = ("Convolve", "DepthwiseConv")
HIDDEN_DTYPES = ("S8",)


class ConfigError(ValueError):
    """The campaign file is unusable."""


@dataclass(frozen=True)
class Campaign:
    """One validated campaign."""

    name: str
    board: str
    op: str
    dtype: str
    kernels_repo: Path
    base_ref: str
    secrets_dir: Path
    legs: tuple[str, ...] = ("tcm", "mram")
    case_ids: tuple[str, ...] = ()
    bench_id: str = ""
    evals: int = 12
    cost_usd: float = 25.0
    model: str = "claude-opus-5-5"
    hidden_shapes: int = 12
    repeats: int = 3
    start_patch: Optional[Path] = None
    start_notes: str = ""
    min_score: Optional[float] = None
    lock_timeout_s: int = 3600
    eval_timeout_s: int = 1800
    retries: int = 1
    max_infra_errors: int = 5

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        return {k: str(v) if isinstance(v, Path) else list(v) if isinstance(v, tuple) else v for k, v in out.items()}


# YAML key -> (type, required).
_TOP = {
    "name": (str, True), "board": (str, True), "bench_id": (str, False), "legs": (list, False),
    "target": (dict, True), "kernels": (dict, True), "evals": (int, False), "cost_usd": ((int, float), False),
    "model": (str, False), "hidden_shapes": (int, False), "repeats": (int, False), "start_patch": (str, False),
    "start_notes": (str, False), "secrets_dir": (str, True), "min_score": ((int, float), False),
    "lock_timeout_s": (int, False), "eval_timeout_s": (int, False), "retries": (int, False),
    "max_infra_errors": (int, False),
}
_TARGET = {"op": (str, True), "dtype": (str, True), "case_ids": (list, False)}
_KERNELS = {"repo": (str, True), "ref": (str, True)}


def _keys(data: Any, spec: dict, where: str) -> dict:
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a mapping")
    unknown = sorted(set(data) - set(spec))
    if unknown:
        raise ConfigError(f"{where}: unknown key {unknown[0]!r}")
    for key, (kind, required) in spec.items():
        if key not in data or data[key] is None:
            if required:
                raise ConfigError(f"{where}: missing {key!r}")
            continue
        # bool is an int; reject it.
        if isinstance(data[key], bool) or not isinstance(data[key], kind):
            raise ConfigError(f"{where}.{key}: wrong type")
    return {k: v for k, v in data.items() if v is not None}


def _path(raw: str, base: Path) -> Path:
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _at_least(data: dict, key: str, low: int) -> None:
    if key in data and data[key] < low:
        raise ConfigError(f"{key}: must be >= {low}")


def parse_campaign(data: Any, base: Path) -> Campaign:
    """Validate a loaded YAML mapping."""
    top = _keys(data, _TOP, "campaign")
    target = _keys(top["target"], _TARGET, "target")
    kernels = _keys(top["kernels"], _KERNELS, "kernels")
    if not NAME_RE.fullmatch(top["name"]):
        raise ConfigError("name: use lowercase letters, digits and dashes")
    try:
        board = resolve_board(top["board"])
    except UnknownBoardError as exc:
        raise ConfigError(f"board: {exc}") from exc
    legs = tuple(top.get("legs") or (("tcm", "mram") if board.has_mram else ("tcm",)))
    if not legs or len(set(legs)) != len(legs) or any(leg not in PLACEMENTS for leg in legs):
        raise ConfigError(f"legs: unique values from {', '.join(PLACEMENTS)}")
    if "mram" in legs and not board.has_mram:
        raise ConfigError(f"legs: {board.id} has no cached MRAM")
    for key in ("op", "dtype"):
        if not WORD_RE.fullmatch(target[key]):
            raise ConfigError(f"target.{key}: one word")
    case_ids = tuple(target.get("case_ids") or ())
    if not all(isinstance(c, str) and CASE_RE.fullmatch(c) for c in case_ids):
        raise ConfigError("target.case_ids: plain case ids")
    for key, low in (("evals", 1), ("hidden_shapes", 0), ("repeats", 1), ("lock_timeout_s", 1),
                     ("eval_timeout_s", 60), ("retries", 0), ("max_infra_errors", 1)):
        _at_least(top, key, low)
    cost = float(top.get("cost_usd", 25.0))
    if not math.isfinite(cost) or cost <= 0:
        raise ConfigError("cost_usd: must be positive")
    hidden = top.get("hidden_shapes", 12)
    if hidden and (target["op"] not in HIDDEN_OPS or target["dtype"] not in HIDDEN_DTYPES):
        raise ConfigError(f"hidden_shapes: only {'/'.join(HIDDEN_OPS)} S8; set 0")
    min_score = top.get("min_score")
    if min_score is not None and not math.isfinite(float(min_score)):
        raise ConfigError("min_score: must be finite")
    if not WORD_RE.fullmatch(top.get("bench_id", board.id).replace("-", "_")):
        raise ConfigError("bench_id: one word")
    return Campaign(
        name=top["name"], board=board.id, op=target["op"], dtype=target["dtype"],
        kernels_repo=_path(kernels["repo"], base), base_ref=kernels["ref"], secrets_dir=_path(top["secrets_dir"], base),
        legs=legs, case_ids=case_ids, bench_id=top.get("bench_id", board.id), evals=top.get("evals", 12),
        cost_usd=cost, model=top.get("model", "claude-opus-5-5"), hidden_shapes=hidden,
        repeats=top.get("repeats", 3),
        start_patch=_path(top["start_patch"], base) if top.get("start_patch") else None,
        start_notes=top.get("start_notes", "").strip(), min_score=None if min_score is None else float(min_score),
        lock_timeout_s=top.get("lock_timeout_s", 3600), eval_timeout_s=top.get("eval_timeout_s", 1800),
        retries=top.get("retries", 1), max_infra_errors=top.get("max_infra_errors", 5),
    )


def load_campaign(path: Path) -> Campaign:
    """Read and validate a campaign YAML."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    return parse_campaign(data, path.resolve().parent)


def from_json(data: dict) -> Campaign:
    """Campaign saved in a workspace."""
    paths = ("kernels_repo", "secrets_dir", "start_patch")
    fixed = {k: (Path(v) if k in paths and v else v) for k, v in data.items()}
    fixed["legs"], fixed["case_ids"] = tuple(fixed["legs"]), tuple(fixed["case_ids"])
    return Campaign(**fixed)
