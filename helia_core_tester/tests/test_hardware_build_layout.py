"""Hardware build output paths have one home: firmware_build.

Every consumer of the linked server image (ELF/bin/map) and the build-id stamp
must go through the `firmware_build` helpers so a layout change lands once.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from helia_core_tester.hardware import firmware_build
from helia_core_tester.hardware.firmware_build import bin_path, build_id_path, elf_path, map_path

HARDWARE_PKG = Path(firmware_build.__file__).parent

# Either quote style, so a rewrite cannot dodge the guard.
_Q = r"""["']"""

# Spellings that would bypass the helpers.
_PATH_LITERALS = (
    # build_dir / "hardware" / f"{SERVER_TARGET}.elf"
    re.compile(rf"{_Q}hardware{_Q}\s*/\s*f?{_Q}\{{?(SERVER_TARGET|hct_benchmark_server)"),
    # f"{build_dir}/hardware/hct_benchmark_server.elf"
    re.compile(r"/hardware/(\{SERVER_TARGET\}|hct_benchmark_server)"),
    # build_dir / "hct_build_id.txt"
    re.compile(rf"/\s*{_Q}hct_build_id\.txt{_Q}"),
    # f"{build_dir}/hct_build_id.txt"
    re.compile(rf"/hct_build_id\.txt{_Q}"),
)


def _offending_lines(source: str) -> list[str]:
    return [
        f"{lineno}: {line.strip()}"
        for lineno, line in enumerate(source.splitlines(), start=1)
        if any(pattern.search(line) for pattern in _PATH_LITERALS)
    ]


def test_helpers_agree_on_layout(tmp_path: Path) -> None:
    build_dir = tmp_path / "build"
    image_dir = build_dir / "hardware"
    assert elf_path(build_dir) == image_dir / "hct_benchmark_server.elf"
    assert bin_path(build_dir) == image_dir / "hct_benchmark_server.bin"
    assert map_path(build_dir) == image_dir / "hct_benchmark_server.map"
    assert build_id_path(build_dir) == build_dir / "hct_build_id.txt"


@pytest.mark.parametrize("module", sorted(p.name for p in HARDWARE_PKG.glob("*.py") if p.name != "firmware_build.py"))
def test_no_module_spells_build_paths(module: str) -> None:
    offenders = _offending_lines((HARDWARE_PKG / module).read_text(encoding="utf-8"))
    assert not offenders, f"{module} must use firmware_build helpers:\n" + "\n".join(offenders)


@pytest.mark.parametrize(
    "line",
    [
        'build_dir / "hardware" / f"{SERVER_TARGET}.elf"',
        "build_dir / 'hardware' / f'{SERVER_TARGET}.elf'",
        'build_dir / "hardware" / "hct_benchmark_server.map"',
        "build_dir / 'hardware' / 'hct_benchmark_server.bin'",
        'f"{build_dir}/hardware/{SERVER_TARGET}.elf"',
        "f'{build_dir}/hardware/hct_benchmark_server.elf'",
        'build_dir / "hct_build_id.txt"',
        "build_dir / 'hct_build_id.txt'",
        'f"{build_dir}/hct_build_id.txt"',
        "f'{build_dir}/hct_build_id.txt'",
    ],
)
def test_guard_bites_every_spelling(line: str) -> None:
    assert _offending_lines(f"path = {line}") == [f"1: path = {line}"]


@pytest.mark.parametrize(
    "line",
    [
        "    Preflight: the build dir must carry `hct_build_id.txt` so every session",
        '    "Stream even when the build dir has no hct_build_id.txt (firmware built before"',
        'SERVER_TARGET = "hct_benchmark_server"',
        'display_path(out_dir / f"{SIZE_PROBE_TARGET}.map", project_root)',
        'return repo_root / "build" / "hardware" / self.id',
    ],
)
def test_guard_ignores_prose(line: str) -> None:
    assert _offending_lines(line) == []
