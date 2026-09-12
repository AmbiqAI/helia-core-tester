"""Board table (assets/hardware_boards.yaml) loading and --board resolution."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from helia_core_tester.perf_stream.boards import (
    BoardSpec,
    UnknownBoardError,
    board_ids,
    default_board_id,
    default_session_id,
    load_board_table,
    resolve_board,
)


def test_board_table_seeds_apollo510_evb() -> None:
    table = load_board_table()
    assert [b.id for b in table] == ["apollo510_evb"]
    spec = resolve_board("apollo510_evb")
    assert spec == BoardSpec(
        id="apollo510_evb",
        nsx_board="apollo510_evb",
        cpu="cortex-m55",
        pmu_tier="armv8m",
        has_mve=True,
        jlink_device="AP510NFA-CBR",
        swd_speed_khz=4000,
        workspace_bytes=114688,
    )


def test_unknown_board_error_lists_known_ids() -> None:
    with pytest.raises(UnknownBoardError) as excinfo:
        resolve_board("apollo9000_evb")
    message = str(excinfo.value)
    assert "apollo9000_evb" in message
    for known in board_ids():
        assert known in message


def test_board_derived_values(tmp_path: Path) -> None:
    spec = resolve_board("apollo510_evb")
    assert spec.build_dir(tmp_path) == tmp_path / "build" / "perf_stream" / "apollo510_evb"
    assert spec.target_info()["board"] == "apollo510_evb"
    assert spec.target_info()["cpu"] == "cortex-m55"
    assert spec.target_info()["transport"] == "jlink-rtt"
    stamp = datetime(2026, 9, 12, 13, 4, 5, tzinfo=timezone.utc)
    assert default_session_id(spec, now=stamp) == "apollo510_evb-20260912T130405Z"


def test_default_board_id_prefers_env() -> None:
    assert default_board_id(env={}) == "apollo510_evb"
    assert default_board_id(env={"HPX_BOARD": "apollo510_evb"}) == "apollo510_evb"
    assert default_board_id(env={"HPX_BOARD": "  "}) == "apollo510_evb"
    assert default_board_id(env={"HPX_BOARD": "other_board"}) == "other_board"


def test_malformed_table_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "boards.yaml"
    bad.write_text(
        "schema: hct.hardware_boards\nschema_version: 1\nboards:\n"
        "  - id: x\n    nsx_board: x\n    cpu: cortex-m55\n    pmu_tier: bogus\n    has_mve: true\n"
        "    jlink_device: X\n    swd_speed_khz: 1\n    workspace_bytes: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="pmu_tier"):
        load_board_table(bad)
    missing = tmp_path / "missing.yaml"
    missing.write_text("schema: hct.hardware_boards\nschema_version: 1\nboards:\n  - id: y\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing field"):
        load_board_table(missing)
