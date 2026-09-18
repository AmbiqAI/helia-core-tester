"""The flash path: the NSX J-Link recipe, its validation, and the JLinkExe driver.

No hardware and no JLinkExe: every test drives `flash_recipe`/`jlink_cli` with a
fake `subprocess.run` that returns recorded J-Link output.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from helia_core_tester.hardware import flash_recipe, jlink_cli
from helia_core_tester.hardware.boards import resolve_board
from helia_core_tester.hardware.flash_recipe import FlashRecipeError

BOARD = resolve_board("apollo510_evb")
SERIAL = 1160003180

#: Exactly what `nsx_finalize_app()` writes (cmake/segger/templates/flash_cmds.jlink.in),
#: license header included -- the shape the validator must accept unchanged.
NSX_RECIPE = """// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Ambiq
ExitOnError 1
Reset
LoadFile "{binary}", 0x00410000
Reset
Go
Exit
"""

#: A real Apollo510 flash, trimmed to the lines the host reads.
JLINK_FLASH_OUTPUT = """SEGGER J-Link Commander V8.10 (Compiled Nov 20 2025 12:00:00)
Connecting to target via SWD
Found SW-DP with ID 0x6BA02477
Found Cortex-M55 r1p1, Little endian.
J-Link: Flash download: Bank 0 @ 0x00410000: 1 range affected (761856 bytes)
J-Link: Flash download: Total: 6.079s (Prepare: 0.121s, Compare: 0.000s, Erase: 0.000s, Program & Verify: 5.834s, Restore: 0.123s)
O.K.
"""


def _proc(stdout: str = "", returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["JLinkExe"], returncode=returncode, stdout=stdout, stderr=stderr)


def _recipe(tmp_path: Path, text: str | None = None, *, binary_bytes: bytes = b"image") -> tuple[Path, Path]:
    binary = tmp_path / "hct_benchmark_server.bin"
    binary.write_bytes(binary_bytes)
    script = tmp_path / "jlink" / "hct_benchmark_server" / "flash_cmds.jlink"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(NSX_RECIPE.format(binary=binary) if text is None else text, encoding="utf-8")
    return script, binary


# --- recipe validation -------------------------------------------------------------


def test_the_recipe_nsx_generates_is_accepted_verbatim(tmp_path: Path) -> None:
    script, binary = _recipe(tmp_path)
    address = flash_recipe.recipe_load_address(
        script.read_text(), script_path=script, bin_path=binary
    )
    assert address == 0x00410000


def test_hand_written_variants_the_commander_accepts_are_accepted(tmp_path: Path) -> None:
    """Quoting, a `noreset` tail and a trailing comment are all valid to JLinkExe."""
    binary = tmp_path / "hct_benchmark_server.bin"
    binary.write_bytes(b"image")
    for line in (
        f'LoadFile "{binary}", 0x410000',
        f"LoadFile {binary}, 0x410000",
        f'LoadFile "{binary}", 0x410000, noreset',
        f'LoadFile "{binary}", 0x410000 // the app',
    ):
        script = tmp_path / "variant.jlink"
        script.write_text(f"ExitOnError 1\nReset\n{line}\nExit\n", encoding="utf-8")
        assert flash_recipe.recipe_load_address(script.read_text(), script_path=script, bin_path=binary) == 0x410000


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        # No fail-fast: JLinkExe can fail a command and still exit zero.
        ('Reset\nLoadFile "{binary}", 0x410000\nExit\n', "ExitOnError 1"),
        # Fail-fast armed only after the LoadFile protects nothing.
        ('LoadFile "{binary}", 0x410000\nExitOnError 1\nExit\n', "after its first `LoadFile`"),
        # Addressless LoadFile: programs flash somewhere unverifiable.
        ('ExitOnError 1\nLoadFile "{binary}"\nExit\n', "read a destination out of"),
        # No LoadFile at all (a bare LoadBin, say): no address to verify against.
        ('ExitOnError 1\nLoadBin "{binary}", 0x410000\nExit\n', "no `LoadFile` command at all"),
        # A recipe left behind by another build, naming another image.
        ('ExitOnError 1\nLoadFile "/elsewhere/other.bin", 0x410000\nExit\n', "not this build's image"),
    ],
)
def test_recipes_this_tool_refuses_to_run(tmp_path: Path, body: str, expected: str) -> None:
    script, binary = _recipe(tmp_path, "placeholder")
    script.write_text(body.format(binary=binary), encoding="utf-8")
    with pytest.raises(FlashRecipeError) as excinfo:
        flash_recipe.recipe_load_address(script.read_text(), script_path=script, bin_path=binary)
    message = str(excinfo.value)
    assert expected in message
    # Every refusal has to say the board was left alone: "refused" and "failed
    # halfway through programming" call for opposite next steps.
    assert flash_recipe.NOTHING_PROGRAMMED in message


def test_missing_recipe_and_missing_image_name_hardware_build(tmp_path: Path) -> None:
    script, binary = _recipe(tmp_path)
    with pytest.raises(FlashRecipeError, match="No NSX flash recipe"):
        flash_recipe.read_recipe(tmp_path / "absent.jlink", binary)
    binary.unlink()
    with pytest.raises(FlashRecipeError, match="this build's image"):
        flash_recipe.read_recipe(script, binary)


def test_recipe_path_is_the_one_nsx_writes(tmp_path: Path) -> None:
    from helia_core_tester.hardware.firmware_build import output_dir

    assert flash_recipe.recipe_path(tmp_path, BOARD) == (
        output_dir(tmp_path, BOARD) / "jlink" / "hct_benchmark_server" / "flash_cmds.jlink"
    )


# --- bank verification -------------------------------------------------------------


def test_bank_verification_reads_jlinks_own_confirmation() -> None:
    lines: list[str] = []
    assert flash_recipe.verify_flash_bank(
        JLINK_FLASH_OUTPUT, expected_addr=0x00410000, echo=lines.append
    ) == [0x00410000]
    assert lines == []


def test_bank_verification_accepts_the_already_matching_skip() -> None:
    output = "J-Link: Flash download: Bank 0 @ 0x00018000: Skipped. Contents already match\n"
    assert flash_recipe.verify_flash_bank(output, expected_addr=0x18000, echo=print) == [0x18000]


def test_a_flash_into_another_bank_is_an_error() -> None:
    with pytest.raises(FlashRecipeError, match="0x00018000"):
        flash_recipe.verify_flash_bank(
            "J-Link: Flash download: Bank 0 @ 0x00018000: 1 range affected\n",
            expected_addr=0x00410000, echo=print,
        )


def test_no_bank_line_warns_instead_of_blocking_the_flash() -> None:
    """The bank line corroborates the exit status; a J-Link rewording must not stop a flash."""
    lines: list[str] = []
    assert flash_recipe.verify_flash_bank("Flash download: Total: 1.0s\n", expected_addr=0x410000, echo=lines.append) == []
    assert any("UNVERIFIED FLASH DESTINATION" in line for line in lines)


def test_error_and_info_lines_that_merely_name_a_bank_do_not_count() -> None:
    """Only `Flash download: Bank N @ ...` is a programming confirmation."""
    noise = (
        "Error while determining flash info (Bank 0 @ 0x00410000)\n"
        "Start of determining flash info (Bank 0 @ 0x00410000)\n"
    )
    lines: list[str] = []
    assert flash_recipe.verify_flash_bank(noise, expected_addr=0x00410000, echo=lines.append) == []
    assert any("UNVERIFIED FLASH DESTINATION" in line for line in lines)


# --- the flash itself --------------------------------------------------------------


def test_flash_image_runs_the_recipe_verbatim_and_verifies_the_bank(tmp_path: Path, monkeypatch) -> None:
    script, binary = _recipe(tmp_path)
    monkeypatch.setattr(jlink_cli, "resolve_exe", lambda: "/opt/SEGGER/JLinkExe")
    seen: dict = {}

    def _runner(argv, **kwargs):
        seen["argv"] = argv
        seen["script"] = kwargs["input"]
        return _proc(JLINK_FLASH_OUTPUT)

    address = flash_recipe.flash_image(
        script_path=script, bin_path=binary, device=BOARD.jlink_device, serial_no=SERIAL,
        speed_khz=BOARD.swd_speed_khz, echo=lambda _m: None, runner=_runner,
    )
    assert address == 0x00410000
    # Verbatim: the bytes piped to JLinkExe are the recipe's own.
    assert seen["script"] == script.read_text()
    assert seen["argv"][0] == "/opt/SEGGER/JLinkExe"
    assert "-SelectEmuBySN" in seen["argv"] and str(SERIAL) in seen["argv"]
    assert seen["argv"][seen["argv"].index("-device") + 1] == BOARD.jlink_device


def test_flash_without_a_recognized_confirmation_is_a_failure(tmp_path: Path, monkeypatch) -> None:
    """A silent no-op flash is the worst failure this path has; a bare `O.K.` is not a flash."""
    script, binary = _recipe(tmp_path)
    monkeypatch.setattr(jlink_cli, "resolve_exe", lambda: "/opt/SEGGER/JLinkExe")
    with pytest.raises(FlashRecipeError, match="no recognized flash confirmation"):
        flash_recipe.flash_image(
            script_path=script, bin_path=binary, device=BOARD.jlink_device, serial_no=SERIAL,
            echo=lambda _m: None, runner=lambda argv, **kw: _proc("Connecting to target\nO.K.\n"),
        )


def test_a_nonzero_jlinkexe_exit_reports_its_output(tmp_path: Path, monkeypatch) -> None:
    script, binary = _recipe(tmp_path)
    monkeypatch.setattr(jlink_cli, "resolve_exe", lambda: "/opt/SEGGER/JLinkExe")
    with pytest.raises(jlink_cli.JLinkCliError, match="Could not connect to target"):
        flash_recipe.flash_image(
            script_path=script, bin_path=binary, device=BOARD.jlink_device, serial_no=SERIAL,
            echo=lambda _m: None,
            runner=lambda argv, **kw: _proc("Could not connect to target.\n", returncode=1),
        )


def test_describe_recipe_reports_presence_for_doctor(tmp_path: Path, monkeypatch) -> None:
    from helia_core_tester.hardware.firmware_build import output_dir

    assert "missing" in flash_recipe.describe_recipe(tmp_path, BOARD)
    path = flash_recipe.recipe_path(tmp_path, BOARD)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(NSX_RECIPE.format(binary=output_dir(tmp_path, BOARD) / "x.bin"), encoding="utf-8")
    assert "loads at 0x00410000" in flash_recipe.describe_recipe(tmp_path, BOARD)


# --- the JLinkExe driver ------------------------------------------------------------


def test_reset_is_a_commander_script_not_a_pylink_reset(monkeypatch) -> None:
    """`r` / `g` / `exit`: the commander's exit releases the probe, which is what lets
    the Apollo510 secure bootloader start the application."""
    monkeypatch.setattr(jlink_cli, "resolve_exe", lambda: "/opt/SEGGER/JLinkExe")
    seen: dict = {}

    def _runner(argv, **kwargs):
        seen["argv"], seen["script"] = argv, kwargs["input"]
        return _proc("Reset delay: 0 ms\nO.K.\n")

    jlink_cli.reset_target(device=BOARD.jlink_device, serial_no=SERIAL, speed_khz=4000, runner=_runner)
    assert seen["script"] == "r\ng\nexit\n" == jlink_cli.RESET_SCRIPT
    assert seen["argv"][1:] == [
        "-NoGui", "1", "-device", BOARD.jlink_device, "-if", "SWD", "-speed", "4000",
        "-autoconnect", "1", "-SelectEmuBySN", str(SERIAL),
    ]


def test_jlinkexe_resolution_honours_the_shared_precedence(tmp_path: Path, monkeypatch) -> None:
    """One resolution path for flash, reset and RTT: $JLINK_PATH, then next to the
    library $HPX_JLINK_DLL names, then PATH (see jlink_library)."""
    install = tmp_path / "segger"
    install.mkdir()
    exe = install / "JLinkExe"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    library = install / "libjlinkarm.so"
    library.write_text("")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    other_exe = elsewhere / "JLinkExe"
    other_exe.write_text("#!/bin/sh\n")
    other_exe.chmod(0o755)

    monkeypatch.delenv("HPX_JLINK_DLL", raising=False)
    monkeypatch.setenv("JLINK_PATH", str(exe))
    assert jlink_cli.resolve_exe() == str(exe)

    # $HPX_JLINK_DLL alone is enough: the binary is looked up beside the library.
    monkeypatch.delenv("JLINK_PATH")
    monkeypatch.setenv("HPX_JLINK_DLL", str(library))
    assert jlink_cli.resolve_exe() == str(exe)

    # Nothing configured: PATH.
    monkeypatch.delenv("HPX_JLINK_DLL")
    monkeypatch.setenv("PATH", str(elsewhere))
    assert jlink_cli.resolve_exe() == str(other_exe)

    # A misconfigured $HPX_JLINK_DLL is an error, not a silent fall-through to PATH.
    monkeypatch.setenv("HPX_JLINK_DLL", str(tmp_path / "gone.so"))
    with pytest.raises(jlink_cli.JLinkCliError, match="does not exist"):
        jlink_cli.resolve_exe()


def test_a_timeout_says_which_operation_hung(monkeypatch) -> None:
    monkeypatch.setattr(jlink_cli, "resolve_exe", lambda: "/opt/SEGGER/JLinkExe")

    def _runner(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    with pytest.raises(jlink_cli.JLinkCliError, match="JLinkExe reset timed out"):
        jlink_cli.reset_target(device=BOARD.jlink_device, serial_no=SERIAL, runner=_runner)


def test_version_banner_is_parsed_for_doctor(monkeypatch) -> None:
    banner = jlink_cli.version(
        exe="/opt/SEGGER/JLinkExe",
        runner=lambda argv, **kw: _proc(
            "SEGGER J-Link Commander V8.10 (Compiled Nov 20 2025 12:00:00)\nDLL version V8.10\n"
        ),
    )
    assert banner == "SEGGER J-Link Commander V8.10 (Compiled Nov 20 2025 12:00:00)"
    assert jlink_cli.version(exe="/nope/JLinkExe", runner=lambda argv, **kw: _proc("garbage")) is None
