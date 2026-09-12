"""J-Link probe enumeration and serial-number resolution for the hardware CLI.

Resolution order for every command that needs a probe:

1. the explicit `--serial-no` flag,
2. the `HPX_JLINK_SERIAL` environment variable,
3. enumeration of connected probes through pylink (the same J-Link DLL the RTT
   transport uses) -- exactly one connected probe is used as-is; zero or several
   is an error that lists what was found and asks for `--serial-no`.

Enumeration never shells out to JLinkExe.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional

SERIAL_ENV_VAR = "HPX_JLINK_SERIAL"


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

    Uses a bare `pylink.JLink()` exactly like `transport.JLinkRttTransport`, so the
    DLL discovery (including a custom `JLINK_PATH`/library location) stays shared.
    Raises ProbeResolutionError when the J-Link DLL itself cannot be loaded.
    """
    import pylink

    try:
        jlink = pylink.JLink()
    except Exception as exc:  # pylink raises TypeError when it cannot find the DLL
        raise ProbeResolutionError(
            f"Cannot load the SEGGER J-Link library through pylink ({exc}). "
            "Install the SEGGER J-Link software or pass --serial-no explicitly."
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


def resolve_serial(
    explicit: Optional[int] = None,
    *,
    env: Optional[dict] = None,
    enumerate_probes: Callable[[], list[ProbeInfo]] = list_probes,
) -> int:
    """Apply the flag > $HPX_JLINK_SERIAL > enumeration resolution order."""
    if explicit is not None:
        return int(explicit)
    env = os.environ if env is None else env
    from_env = env.get(SERIAL_ENV_VAR)
    if from_env is not None and from_env.strip():
        return _parse_serial(from_env, f"${SERIAL_ENV_VAR}")

    probes = enumerate_probes()
    if len(probes) == 1:
        return probes[0].serial
    if not probes:
        raise ProbeResolutionError(
            "No connected J-Link probes detected. Connect the board, or pass --serial-no "
            f"(or set ${SERIAL_ENV_VAR}) explicitly."
        )
    listing = ", ".join(p.describe() for p in probes)
    raise ProbeResolutionError(
        f"Multiple J-Link probes detected: {listing}. Pass --serial-no (or set "
        f"${SERIAL_ENV_VAR}) to select one."
    )
