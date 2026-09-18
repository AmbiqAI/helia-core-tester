"""The one place this package shells out to `JLinkExe`.

The board is reached two ways, exactly as heliaPROFILER reaches it:

* in-process through **pylink** for runtime data (RTT rings, memory peeks) --
  see `transport.JLinkRttTransport`;
* through the **JLinkExe commander** for script operations: flashing (the
  NSX-generated recipe, see `flash_recipe`), target reset, and probe
  inspection.

Reset in particular has to be the commander, not pylink: the Apollo510 secure
bootloader checks for an attached debugger on the boot that follows a reset, and
only the commander's exit releases the probe in time. `pylink.JLink.reset()`
leaves the SBL in a state where the application never starts, which is why hpx
resets through a `r` / `g` / `exit` script and this module does the same.

The binary itself comes from `jlink_library.find_jlink_exe()`, so `$JLINK_PATH`,
the directory of the library `$HPX_JLINK_DLL` names, and `JLinkExe` on PATH all
select the *same* install for flashing, resetting and RTT.
"""

from __future__ import annotations

import subprocess
from typing import Optional, Sequence

from .jlink_library import JLinkLibraryError, find_jlink_exe, missing_library_hint

#: `r` (reset), `g` (go), `exit`. The commander releases the probe on exit,
#: which is what lets the Apollo510 SBL hand control to the application.
RESET_SCRIPT = "r\ng\nexit\n"

#: Wall-clock budget for one commander invocation. Generous enough to absorb a
#: slow USB enumeration; flashing passes its own, larger value.
DEFAULT_TIMEOUT_S = 30.0
FLASH_TIMEOUT_S = 180.0


class JLinkCliError(RuntimeError):
    """`JLinkExe` is missing, timed out, or exited non-zero."""


def resolve_exe() -> str:
    """Path of the `JLinkExe` binary every J-Link operation in this package uses.

    Raises JLinkCliError when none of `$JLINK_PATH`, the resolved J-Link
    library's directory or PATH yields one -- the same order
    `firmware_build._prepare_probe_env` exports to NSX as `$JLINK_PATH`.
    """
    try:
        found = find_jlink_exe()
    except JLinkLibraryError as exc:  # $HPX_JLINK_DLL names a missing file
        raise JLinkCliError(str(exc)) from exc
    if found is None:
        raise JLinkCliError(f"JLinkExe not found. {missing_library_hint()}")
    return found.path


def build_argv(
    *,
    device: str,
    serial_no: Optional[int] = None,
    speed_khz: int = 4000,
    interface: str = "SWD",
    exe: Optional[str] = None,
) -> list[str]:
    """The commander argv for a target-connected session.

    `-SelectEmuBySN` rather than `-USB`: both select a probe by serial, but the
    former is the documented flag and is what hpx passes, so a probe that works
    for one tool works for the other.
    """
    argv = [
        exe or resolve_exe(),
        "-NoGui", "1",
        "-device", device,
        "-if", interface,
        "-speed", str(speed_khz),
        "-autoconnect", "1",
    ]
    if serial_no is not None:
        argv += ["-SelectEmuBySN", str(serial_no)]
    return argv


def run_script(
    script: str,
    *,
    device: str,
    serial_no: Optional[int] = None,
    speed_khz: int = 4000,
    interface: str = "SWD",
    timeout_s: float = DEFAULT_TIMEOUT_S,
    op_label: str = "JLinkExe",
    check: bool = True,
    runner=subprocess.run,
) -> subprocess.CompletedProcess:
    """Run one commander script (which must end in `exit`) and return the process.

    `check=False` returns a non-zero exit to the caller instead of raising, for
    operations whose own side effect interrupts the debug session.
    """
    argv = build_argv(device=device, serial_no=serial_no, speed_khz=speed_khz, interface=interface)
    try:
        result = runner(argv, input=script, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise JLinkCliError(
            f"{op_label} timed out after {timeout_s:.0f}s. Check that the probe is connected "
            "and not in use by another process (another hardware run, JLinkExe, Ozone)."
        ) from exc
    except FileNotFoundError as exc:
        raise JLinkCliError(f"JLinkExe not found ({argv[0]}). {missing_library_hint()}") from exc
    if check and result.returncode != 0:
        # The commander reports most failures (cannot connect, LoadFile errors,
        # script refusals) on stdout with an empty stderr, so a stderr-only hint
        # renders blank exactly when it is needed.
        stderr = (result.stderr or "").strip()
        detail = stderr if stderr else (result.stdout or "").strip()[-800:]
        raise JLinkCliError(f"{op_label} failed (rc={result.returncode}): {detail[:800]}")
    return result


def reset_target(
    *,
    device: str,
    serial_no: Optional[int] = None,
    speed_khz: int = 4000,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    runner=subprocess.run,
) -> subprocess.CompletedProcess:
    """Reset the target and let it run, through the commander (see module docstring)."""
    return run_script(
        RESET_SCRIPT,
        device=device,
        serial_no=serial_no,
        speed_khz=speed_khz,
        timeout_s=timeout_s,
        op_label="JLinkExe reset",
        runner=runner,
    )


def version(*, exe: Optional[str] = None, timeout_s: float = 15.0, runner=subprocess.run) -> Optional[str]:
    """The commander's version banner, or None when it cannot be asked.

    Runs with no `-device`, so the commander prints its banner and exits without
    touching a target; used by `doctor`, which passes the binary it just
    resolved rather than letting this resolve a second, possibly different one.
    """
    try:
        exe = exe or resolve_exe()
    except JLinkCliError:
        return None
    try:
        result = runner([exe, "-NoGui", "1"], input="exit\n", capture_output=True, text=True, timeout=timeout_s)
    except (subprocess.TimeoutExpired, OSError):
        return None
    return _first_version_line((result.stdout or "") + "\n" + (result.stderr or ""))


def _first_version_line(output: str) -> Optional[str]:
    for line in output.splitlines():
        text = line.strip()
        if text.startswith("SEGGER J-Link Commander"):
            return text
    return None


def detected_cores(output: str) -> Sequence[int]:
    """Cortex-M core numbers named in commander output (`Found Cortex-M55`)."""
    import re

    return tuple(int(m) for m in re.findall(r"Found\s+Cortex-M(\d+)", output, re.IGNORECASE))
