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

# Spellings that would bypass the helpers.
_PATH_LITERALS = (
    re.compile(r'"hardware"\s*/\s*f?"\{?(SERVER_TARGET|hct_benchmark_server)'),
    re.compile(r'/\s*"hct_build_id\.txt"'),
)


def test_helpers_agree_on_layout(tmp_path: Path) -> None:
    build_dir = tmp_path / "build"
    image_dir = build_dir / "hardware"
    assert elf_path(build_dir) == image_dir / "hct_benchmark_server.elf"
    assert bin_path(build_dir) == image_dir / "hct_benchmark_server.bin"
    assert map_path(build_dir) == image_dir / "hct_benchmark_server.map"
    assert build_id_path(build_dir) == build_dir / "hct_build_id.txt"


@pytest.mark.parametrize("module", sorted(p.name for p in HARDWARE_PKG.glob("*.py") if p.name != "firmware_build.py"))
def test_no_module_spells_build_paths(module: str) -> None:
    source = (HARDWARE_PKG / module).read_text(encoding="utf-8")
    offenders = [
        f"{module}:{lineno}: {line.strip()}"
        for lineno, line in enumerate(source.splitlines(), start=1)
        if any(pattern.search(line) for pattern in _PATH_LITERALS)
    ]
    assert not offenders, "use firmware_build helpers:\n" + "\n".join(offenders)
