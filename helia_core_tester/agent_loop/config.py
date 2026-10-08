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
from helia_core_tester.hardware.toolchain import DEFAULT_TOOLCHAIN, TOOLCHAINS, toolchain_spec

NAME_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,47}")
WORD_RE = re.compile(r"[A-Za-z0-9_]+")
CASE_RE = re.compile(r"[A-Za-z0-9_.-]+")


# Agent Bash cap 600 s, less margin.
MAX_DEADLINE_S = 570


class ConfigError(ValueError):
    """The campaign file is unusable."""


@dataclass(frozen=True)
class Leg:
    """One eval leg: placement and toolchain."""

    name: str
    placement: str
    toolchain: str


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
    lock_timeout_s: int = 600
    eval_timeout_s: int = 300
    # Under the agent's 10 min Bash cap.
    submit_deadline_s: int = 540
    retries: int = 1
    max_infra_errors: int = 5
    toolchains: tuple[str, ...] = (DEFAULT_TOOLCHAIN,)

    @property
    def runs(self) -> tuple[Leg, ...]:
        """Every leg; gcc keeps bare names."""
        return tuple(Leg(p + toolchain_spec(t).dir_suffix, p, t) for t in self.toolchains for p in self.legs)

    @property
    def leg_names(self) -> tuple[str, ...]:
        return tuple(leg.name for leg in self.runs)

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        # gcc-only files stay unchanged.
        if self.toolchains == (DEFAULT_TOOLCHAIN,):
            del out["toolchains"]
        return {k: str(v) if isinstance(v, Path) else list(v) if isinstance(v, tuple) else v for k, v in out.items()}


# Copied as is when present.
_PLAIN = ("evals", "model", "hidden_shapes", "repeats", "lock_timeout_s", "eval_timeout_s", "submit_deadline_s",
          "retries", "max_infra_errors", "bench_id")
DEFAULTS = Campaign.__dataclass_fields__
# YAML key -> (type, required).
_TOP = {
    "name": (str, True), "board": (str, True), "bench_id": (str, False), "legs": (list, False),
    "toolchains": (list, False),
    "target": (dict, True), "kernels": (dict, True), "evals": (int, False), "cost_usd": ((int, float), False),
    "model": (str, False), "hidden_shapes": (int, False), "repeats": (int, False), "start_patch": (str, False),
    "start_notes": (str, False), "secrets_dir": (str, True), "min_score": ((int, float), False),
    "lock_timeout_s": (int, False), "eval_timeout_s": (int, False), "submit_deadline_s": (int, False),
    "retries": (int, False),
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
    if not all(isinstance(leg, str) for leg in legs):
        raise ConfigError(f"legs: unique values from {', '.join(PLACEMENTS)}")
    if not legs or len(set(legs)) != len(legs) or any(leg not in PLACEMENTS for leg in legs):
        raise ConfigError(f"legs: unique values from {', '.join(PLACEMENTS)}")
    if "mram" in legs and not board.has_mram:
        raise ConfigError(f"legs: {board.id} has no cached MRAM")
    toolchains = tuple(top.get("toolchains") or (DEFAULT_TOOLCHAIN,))
    if (not all(isinstance(t, str) for t in toolchains) or len(set(toolchains)) != len(toolchains)
            or any(t not in TOOLCHAINS for t in toolchains)):
        raise ConfigError(f"toolchains: unique values from {', '.join(TOOLCHAINS)}")
    for key in toolchains:
        try:
            toolchain_spec(key).require()
        except FileNotFoundError as exc:
            raise ConfigError(f"toolchains: {exc}") from exc
    for key in ("op", "dtype"):
        if not WORD_RE.fullmatch(target[key]):
            raise ConfigError(f"target.{key}: one word")
    case_ids = tuple(target.get("case_ids") or ())
    if not all(isinstance(c, str) and CASE_RE.fullmatch(c) for c in case_ids):
        raise ConfigError("target.case_ids: plain case ids")
    if top.get("submit_deadline_s", 0) > MAX_DEADLINE_S:
        raise ConfigError(f"submit_deadline_s: at most {MAX_DEADLINE_S}")
    for key, low in (("evals", 1), ("hidden_shapes", 0), ("repeats", 1), ("lock_timeout_s", 1),
                     ("eval_timeout_s", 60), ("submit_deadline_s", 120), ("retries", 0), ("max_infra_errors", 1)):
        _at_least(top, key, low)
    cost = float(top.get("cost_usd", DEFAULTS["cost_usd"].default))
    if not math.isfinite(cost) or cost <= 0:
        raise ConfigError("cost_usd: must be positive")
    hidden = top.get("hidden_shapes", DEFAULTS["hidden_shapes"].default)
    if hidden:
        from helia_core_tester.generation.random_shapes import select_ops

        try:
            select_ops(target["op"], target["dtype"])
        except ValueError as exc:
            raise ConfigError(f"hidden_shapes: {exc}; or set 0") from exc
    min_score = top.get("min_score")
    if min_score is not None and not math.isfinite(float(min_score)):
        raise ConfigError("min_score: must be finite")
    top.setdefault("bench_id", board.id)
    if not WORD_RE.fullmatch(top["bench_id"].replace("-", "_")):
        raise ConfigError("bench_id: one word")
    return Campaign(
        name=top["name"], board=board.id, op=target["op"], dtype=target["dtype"],
        kernels_repo=_path(kernels["repo"], base), base_ref=kernels["ref"], secrets_dir=_path(top["secrets_dir"], base),
        legs=legs, toolchains=toolchains, case_ids=case_ids, cost_usd=cost,
        start_patch=_path(top["start_patch"], base) if top.get("start_patch") else None,
        start_notes=top.get("start_notes", "").strip(), min_score=None if min_score is None else float(min_score),
        **{key: top[key] for key in _PLAIN if key in top},
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
    fixed["toolchains"] = tuple(fixed.get("toolchains") or (DEFAULT_TOOLCHAIN,))
    return Campaign(**fixed)
