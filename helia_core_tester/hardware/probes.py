"""J-Link probe enumeration and serial-number resolution for the hardware CLI.

Resolution order for every command that needs a probe:

1. the explicit `--serial-no` flag,
2. the `HPX_JLINK_SERIAL` environment variable,
3. enumeration of connected probes through pylink (the same J-Link DLL the RTT
   transport uses) -- exactly one connected probe is used as-is. An empty first
   enumeration is retried once after a short pause: a probe that was just
   plugged in (or a J-Link OB whose board was just powered) can be missing from
   the first USB scan and present on the next.
4. with several probes connected *and* a board to match against, each probe is
   asked which core it reaches (hpx's rule: auto-select only when exactly one
   probe reports the board's core). Anything else -- no match, or two probes
   behind the same core -- is an error listing what was found and asking for
   `--serial-no`.

Enumeration itself never shells out to JLinkExe; only the disambiguation in
step 4 does, and only on the path that would otherwise have been a hard error.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, Optional

from .jlink_library import JLinkLibraryError, missing_library_hint, open_jlink

SERIAL_ENV_VAR = "HPX_JLINK_SERIAL"
ENUMERATION_RETRY_DELAY_S = 1.0


class ProbeResolutionError(RuntimeError):
    """Raised when no single J-Link probe can be chosen."""


@dataclass(frozen=True)
class ProbeInfo:
    serial: int
    product: str = ""

    def describe(self) -> str:
        return f"{self.serial} ({self.product})" if self.product else str(self.serial)


def _decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").rstrip("\x00").strip()
    return str(value or "").strip()


def list_probes() -> list[ProbeInfo]:
    """Enumerate USB-connected J-Link probes via pylink's JLINKARM_EMU_GetList.

    Opens pylink through `jlink_library.open_jlink` exactly like
    `transport.JLinkRttTransport`, so the library discovery ($HPX_JLINK_DLL,
    $JLINK_PATH, JLinkExe on PATH, then pylink's default) stays shared.
    Raises ProbeResolutionError when the J-Link DLL itself cannot be loaded.
    """
    import pylink

    try:
        jlink = open_jlink(pylink)
    except JLinkLibraryError as exc:  # $HPX_JLINK_DLL names a missing file: say exactly that
        raise ProbeResolutionError(str(exc)) from exc
    except Exception as exc:  # pylink raises TypeError when it cannot find the DLL
        raise ProbeResolutionError(
            f"Cannot load the SEGGER J-Link library through pylink ({exc}). {missing_library_hint()} "
            "Alternatively pass --serial-no explicitly."
        ) from exc
    try:
        emulators = jlink.connected_emulators()
    except Exception as exc:
        raise ProbeResolutionError(f"J-Link probe enumeration failed: {exc}") from exc
    finally:
        try:
            jlink.close()
        except Exception:
            pass
    probes = [ProbeInfo(serial=int(info.SerialNumber), product=_decode(getattr(info, "acProduct", b""))) for info in emulators]
    return sorted(probes, key=lambda p: p.serial)


def _parse_serial(raw: str, source: str) -> int:
    text = raw.strip()
    if not text.isdigit():
        raise ProbeResolutionError(f"{source} must be a decimal J-Link serial number (got {raw!r}).")
    return int(text)


def expected_core(cpu: str) -> Optional[int]:
    """The Cortex-M number a board's `cpu` field names (`cortex-m55` -> 55)."""
    text = str(cpu).strip().lower()
    prefix = "cortex-m"
    if not text.startswith(prefix):
        return None
    digits = text[len(prefix):].split("+")[0].strip()
    return int(digits) if digits.isdigit() else None


def probe_core(probe: ProbeInfo, *, device: str) -> Optional[int]:
    """The Cortex-M core `probe` reaches for `device`, or None when it reaches none.

    One `JLinkExe` connect per probe (`exit` straight away); the commander names
    what it found as `Found Cortex-M55`.
    """
    from . import jlink_cli

    try:
        result = jlink_cli.run_script(
            "exit\n", device=device, serial_no=probe.serial,
            op_label=f"JLinkExe probe inspection ({probe.serial})", check=False,
        )
    except jlink_cli.JLinkCliError:
        return None
    cores = jlink_cli.detected_cores((result.stdout or "") + "\n" + (result.stderr or ""))
    return cores[0] if cores else None


def resolve_serial(
    explicit: Optional[int] = None,
    *,
    board=None,
    env: Optional[dict] = None,
    enumerate_probes: Callable[[], list[ProbeInfo]] = list_probes,
    retry_delay_s: float = ENUMERATION_RETRY_DELAY_S,
    sleep: Callable[[float], None] = time.sleep,
    inspect_probe: Callable[..., Optional[int]] = probe_core,
) -> int:
    """Apply the flag > $HPX_JLINK_SERIAL > enumeration resolution order.

    Enumeration is re-run once, after `retry_delay_s`, when the first pass finds
    no probe at all (see the module docstring). When it finds several and `board`
    is given, the probes are asked which core they reach and a unique match for
    the board's CPU wins; without a board, or without a unique match, the
    ambiguity is reported."""
    if explicit is not None:
        return int(explicit)
    env = os.environ if env is None else env
    from_env = env.get(SERIAL_ENV_VAR)
    if from_env is not None and from_env.strip():
        return _parse_serial(from_env, f"${SERIAL_ENV_VAR}")

    probes = enumerate_probes()
    if not probes:
        sleep(retry_delay_s)
        probes = enumerate_probes()
    if len(probes) == 1:
        return probes[0].serial
    if not probes:
        raise ProbeResolutionError(
            "No connected J-Link probes detected. Connect the board, or pass --serial-no "
            f"(or set ${SERIAL_ENV_VAR}) explicitly."
        )
    listing = ", ".join(p.describe() for p in probes)
    wanted = expected_core(board.cpu) if board is not None else None
    if wanted is None:
        raise ProbeResolutionError(
            f"Multiple J-Link probes detected: {listing}. Pass --serial-no (or set "
            f"${SERIAL_ENV_VAR}) to select one."
        )
    cores = {probe.serial: inspect_probe(probe, device=board.jlink_device) for probe in probes}
    matches = [probe for probe in probes if cores[probe.serial] == wanted]
    if len(matches) == 1:
        return matches[0].serial
    seen = ", ".join(
        f"{p.describe()} -> {'Cortex-M%d' % cores[p.serial] if cores[p.serial] else 'no target'}"
        for p in probes
    )
    what = (
        f"{len(matches)} of them reach a Cortex-M{wanted}"
        if matches
        else f"none of them reaches a Cortex-M{wanted} ({board.id})"
    )
    raise ProbeResolutionError(
        f"Multiple J-Link probes detected and {what}: {seen}. Pass --serial-no (or set "
        f"${SERIAL_ENV_VAR}) to select one."
    )
