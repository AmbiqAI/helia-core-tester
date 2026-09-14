"""Board table for the hardware CLI (assets/hardware_boards.yaml).

`--board <id>` is the single identity flag on every `helia_core_tester hardware`
command. Everything else that used to be an independent flag that could disagree
with it (`--cpu`, `--chip-name`, `--speed-khz`, the NSX board name) is derived
from the matching :class:`BoardSpec` row here instead.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

_TABLE_RELATIVE_PATH = Path("assets/hardware_boards.yaml")
_EXPECTED_SCHEMA = "hct.hardware_boards"
_EXPECTED_SCHEMA_VERSION = 1
_PMU_TIERS = ("dwt", "armv8m")

DEFAULT_BOARD_ID = "apollo510_evb"
BOARD_ENV_VAR = "HPX_BOARD"


class UnknownBoardError(ValueError):
    """Raised when a `--board` id has no row in the board table."""


@dataclass(frozen=True)
class BoardSpec:
    id: str
    nsx_board: str
    cpu: str
    pmu_tier: str
    has_mve: bool
    jlink_device: str
    swd_speed_khz: int
    workspace_bytes: int

    def build_dir(self, repo_root: Path) -> Path:
        """Board-keyed benchmark-server CMake build directory."""
        return repo_root / "build" / "perf_stream" / self.id

    def target_info(self) -> dict:
        """The `target` block written into every result bundle's session manifest."""
        return {
            "board": self.id,
            "nsx_board": self.nsx_board,
            "cpu": self.cpu,
            "pmu_tier": self.pmu_tier,
            "has_mve": self.has_mve,
            "jlink_device": self.jlink_device,
            "transport": "jlink-rtt",
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_row(row: dict, path: Path) -> BoardSpec:
    required = ("id", "nsx_board", "cpu", "pmu_tier", "has_mve", "jlink_device", "swd_speed_khz", "workspace_bytes")
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"{path}: board row {row.get('id', '?')!r} is missing field(s): {', '.join(missing)}")
    pmu_tier = str(row["pmu_tier"])
    if pmu_tier not in _PMU_TIERS:
        raise ValueError(f"{path}: board {row['id']!r} has pmu_tier {pmu_tier!r}; expected one of {_PMU_TIERS}")
    return BoardSpec(
        id=str(row["id"]),
        nsx_board=str(row["nsx_board"]),
        cpu=str(row["cpu"]),
        pmu_tier=pmu_tier,
        has_mve=bool(row["has_mve"]),
        jlink_device=str(row["jlink_device"]),
        swd_speed_khz=int(row["swd_speed_khz"]),
        workspace_bytes=int(row["workspace_bytes"]),
    )


def load_board_table(path: Optional[Path] = None) -> tuple[BoardSpec, ...]:
    """Parse the board table. Rows keep their file order; ids must be unique."""
    path = path or (_repo_root() / _TABLE_RELATIVE_PATH)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != _EXPECTED_SCHEMA:
        raise ValueError(f"{path}: expected a mapping with schema {_EXPECTED_SCHEMA!r}")
    if data.get("schema_version") != _EXPECTED_SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported schema_version {data.get('schema_version')!r}")
    rows = data.get("boards")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: 'boards' must be a non-empty list")
    specs = tuple(_parse_row(row, path) for row in rows)
    seen: set[str] = set()
    for spec in specs:
        if spec.id in seen:
            raise ValueError(f"{path}: duplicate board id {spec.id!r}")
        seen.add(spec.id)
    return specs


def board_ids(path: Optional[Path] = None) -> tuple[str, ...]:
    return tuple(spec.id for spec in load_board_table(path))


def resolve_board(board_id: str, path: Optional[Path] = None) -> BoardSpec:
    """Return the row for `board_id`, or raise UnknownBoardError naming the known ids."""
    table = load_board_table(path)
    for spec in table:
        if spec.id == board_id:
            return spec
    known = ", ".join(spec.id for spec in table)
    raise UnknownBoardError(f"Unknown board {board_id!r}. Known boards: {known} (see assets/hardware_boards.yaml).")


def default_board_id(env: Optional[dict] = None) -> str:
    """`--board` default: $HPX_BOARD when set, else apollo510_evb."""
    env = os.environ if env is None else env
    value = env.get(BOARD_ENV_VAR, "").strip()
    return value or DEFAULT_BOARD_ID


def default_session_id(board: BoardSpec, now: Optional[datetime] = None) -> str:
    """`<board>-<UTC timestamp YYYYmmddTHHMMSSZ>`; also the result-bundle directory name."""
    now = now or datetime.now(timezone.utc)
    return f"{board.id}-{now.strftime('%Y%m%dT%H%M%SZ')}"
