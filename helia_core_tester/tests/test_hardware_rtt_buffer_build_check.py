"""#188: a firmware build without the HCTP RTT buffer sizes fails before it is flashed."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from helia_core_tester.hardware import firmware_build, nsx_app, nsx_cli, session, transport
from helia_core_tester.hardware.transport import RttBufferSizeError
from helia_core_tester.tests.test_hardware_nsx_build import BOARD, SERIAL, nsx  # noqa: F401

UP, DOWN = nsx_app.RTT_BUFFER_SIZE_UP, nsx_app.RTT_BUFFER_SIZE_DOWN


def _cache(build_dir: Path, **values: object) -> None:
    lines = [f"{name}:STRING={value}" for name, value in values.items()]
    (build_dir / "CMakeCache.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _symbols(monkeypatch, sizes: dict[str, int]) -> None:
    monkeypatch.setattr(transport, "elf_symbol_sizes", lambda elf: dict(sizes))


@pytest.mark.parametrize(
    "values",
    [
        {},
        {"NSX_SEGGER_RTT_BUFFER_SIZE_UP": UP},
        {"NSX_SEGGER_RTT_BUFFER_SIZE_DOWN": DOWN},
        {"NSX_SEGGER_RTT_BUFFER_SIZE_UP": "big", "NSX_SEGGER_RTT_BUFFER_SIZE_DOWN": DOWN},
        {"NSX_SEGGER_RTT_BUFFER_SIZE_UP": 1, "NSX_SEGGER_RTT_BUFFER_SIZE_DOWN": DOWN},
        {"NSX_SEGGER_RTT_BUFFER_SIZE_UP": "", "NSX_SEGGER_RTT_BUFFER_SIZE_DOWN": DOWN},
    ],
    ids=["none", "no-down", "no-up", "not-a-number", "too-small", "empty"],
)
def test_missing_or_bad_cache_values_are_refused(tmp_path: Path, values) -> None:
    _cache(tmp_path, **values)
    with pytest.raises(RttBufferSizeError, match="SEGGER's 1 KiB / 16 B RTT buffers"):
        firmware_build.configured_rtt_buffer_sizes(tmp_path)


def test_no_cache_at_all_is_refused(tmp_path: Path) -> None:
    with pytest.raises(RttBufferSizeError, match="NSX_SEGGER_RTT_BUFFER_SIZE_UP=None"):
        firmware_build.configured_rtt_buffer_sizes(tmp_path)


def test_configured_sizes_come_from_the_build_dir(tmp_path: Path) -> None:
    _cache(tmp_path, NSX_SEGGER_RTT_BUFFER_SIZE_UP=4096, NSX_SEGGER_RTT_BUFFER_SIZE_DOWN=256)
    assert firmware_build.configured_rtt_buffer_sizes(tmp_path) == (4096, 256)


@pytest.mark.parametrize(
    "sizes",
    [
        {"_acUpBuffer": UP, "_acDownBuffer": DOWN},
        # Rounded up to a cache line by SEGGER_RTT__ROUND_UP_2_CACHE_LINE_SIZE.
        {"_acUpBuffer": UP + 32, "_acDownBuffer": DOWN + 32},
    ],
    ids=["exact", "cache-line-rounded"],
)
def test_elf_with_the_configured_arrays_passes(tmp_path: Path, monkeypatch, sizes) -> None:
    _cache(tmp_path, NSX_SEGGER_RTT_BUFFER_SIZE_UP=UP, NSX_SEGGER_RTT_BUFFER_SIZE_DOWN=DOWN)
    _symbols(monkeypatch, sizes)
    assert firmware_build.check_rtt_buffers(tmp_path) == (UP, DOWN)


@pytest.mark.parametrize(
    ("sizes", "message"),
    [
        # SEGGER's defaults compiled in although the cache says otherwise.
        ({"_acUpBuffer": 1024, "_acDownBuffer": DOWN}, "_acUpBuffer is 1024 bytes but NSX_SEGGER_RTT_BUFFER_SIZE_UP"),
        ({"_acUpBuffer": UP, "_acDownBuffer": 16}, "_acDownBuffer is 16 bytes but NSX_SEGGER_RTT_BUFFER_SIZE_DOWN"),
        ({"_acUpBuffer": UP - 1, "_acDownBuffer": DOWN}, f"_acUpBuffer is {UP - 1} bytes"),
        ({"_acUpBuffer": UP}, "has no _acDownBuffer symbol"),
        ({}, "has no _acUpBuffer symbol"),
    ],
    ids=["segger-up", "segger-down", "one-short", "no-down-symbol", "no-symbols"],
)
def test_elf_without_the_configured_arrays_fails(tmp_path: Path, monkeypatch, sizes, message) -> None:
    _cache(tmp_path, NSX_SEGGER_RTT_BUFFER_SIZE_UP=UP, NSX_SEGGER_RTT_BUFFER_SIZE_DOWN=DOWN)
    _symbols(monkeypatch, sizes)
    with pytest.raises(RttBufferSizeError, match=message):
        firmware_build.check_rtt_buffers(tmp_path)


NM_OUTPUT = """\
20000000 00000060 B _SEGGER_RTT
20000060 00002000 b _acUpBuffer.lto_priv.0
20002060 00000200 b _acDownBuffer
00001000 00000010 T main
20003000 B _end
"""


def test_elf_symbol_sizes_parses_nm_and_lto_names(monkeypatch) -> None:
    seen = []

    def fake_run(cmd, **kwargs):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=NM_OUTPUT, stderr="")

    monkeypatch.setattr(transport.subprocess, "run", fake_run)
    from helia_core_tester.hardware import toolchain

    monkeypatch.setattr(toolchain, "arm_tool", lambda name: name)
    sizes = transport.elf_symbol_sizes("fw.elf")
    assert seen == [["arm-none-eabi-nm", "-S", "fw.elf"]]
    assert sizes["_acUpBuffer"] == 0x2000 and sizes["_acUpBuffer.lto_priv.0"] == 0x2000
    assert sizes["_acDownBuffer"] == 0x200
    assert "_end" not in sizes  # unsized symbols are skipped


def test_build_without_rtt_sizes_fails_and_is_not_recorded(tmp_path: Path, nsx, monkeypatch) -> None:  # noqa: F811
    def configure_without_rtt(app_dir, board, *, build_dir, frozen=False):
        (build_dir / "build.ninja").write_text("", encoding="utf-8")
        (build_dir / "CMakeCache.txt").write_text(
            f"CMAKE_HOME_DIRECTORY:INTERNAL={app_dir.resolve()}\nNSX_BOARD:STRING={board}\n", encoding="utf-8",
        )

    monkeypatch.setattr(nsx_cli, "configure_app", configure_without_rtt)
    with pytest.raises(RttBufferSizeError, match="NSX_SEGGER_RTT_BUFFER_SIZE_UP"):
        firmware_build.build_firmware(BOARD, build_dir=tmp_path)
    assert firmware_build.built_record(firmware_build.nsx_app_dir(tmp_path)) == {}


def test_mis_sized_build_is_never_probed_or_flashed(tmp_path: Path, nsx, monkeypatch) -> None:  # noqa: F811
    _symbols(monkeypatch, {"_acUpBuffer": 1024, "_acDownBuffer": 16})
    monkeypatch.setattr(nsx_cli, "flash_app", lambda *a, **k: pytest.fail("flashed a mis-sized build"))

    def reader(*args):
        pytest.fail("probed the board for a mis-sized build")

    with pytest.raises(RttBufferSizeError, match="_acUpBuffer is 1024 bytes"):
        firmware_build.flash_firmware(BOARD, SERIAL, build_dir=tmp_path, board_build_id_reader=reader)


def test_build_id_probe_expects_the_build_dir_sizes(tmp_path: Path, monkeypatch) -> None:
    _cache(tmp_path, NSX_SEGGER_RTT_BUFFER_SIZE_UP=4096, NSX_SEGGER_RTT_BUFFER_SIZE_DOWN=256)
    made = {}

    class _Transport:
        def __init__(self, **kwargs):
            made.update(kwargs)

        def close(self):
            made["closed"] = True

    monkeypatch.setattr(transport, "symbol_address_from_elf", lambda *a: 0x20000000)
    monkeypatch.setattr(transport, "JLinkRttTransport", _Transport)
    monkeypatch.setattr(session, "read_target_info", lambda t: type("Info", (), {"build_id": "abc"})())
    assert firmware_build.board_build_id(BOARD, SERIAL, tmp_path) == "abc"
    assert (made["expected_up_size"], made["expected_down_size"]) == (4096, 256)
    assert made["closed"] is True
