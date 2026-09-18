"""J-Link serial resolution order: --serial-no > $HPX_JLINK_SERIAL > pylink enumeration."""

from __future__ import annotations

import pytest

from helia_core_tester.hardware import probes
from helia_core_tester.hardware.probes import ProbeInfo, ProbeResolutionError, resolve_serial


def _enumerator(found: list[ProbeInfo], *then: list[ProbeInfo]):
    """Fake enumerator returning `found` first, then each of `then` on later calls (last one repeats)."""
    calls = {"count": 0}
    answers = [found, *then]

    def _list() -> list[ProbeInfo]:
        index = min(calls["count"], len(answers) - 1)
        calls["count"] += 1
        return list(answers[index])

    _list.calls = calls  # type: ignore[attr-defined]
    return _list


def test_flag_wins_without_touching_env_or_probes() -> None:
    enumerate = _enumerator([ProbeInfo(1), ProbeInfo(2)])
    assert resolve_serial(1160002276, env={"HPX_JLINK_SERIAL": "99"}, enumerate_probes=enumerate) == 1160002276
    assert enumerate.calls["count"] == 0


def test_env_wins_over_enumeration() -> None:
    enumerate = _enumerator([ProbeInfo(1), ProbeInfo(2)])
    assert resolve_serial(None, env={"HPX_JLINK_SERIAL": " 1160002276 "}, enumerate_probes=enumerate) == 1160002276
    assert enumerate.calls["count"] == 0


def test_env_must_be_numeric() -> None:
    with pytest.raises(ProbeResolutionError, match="HPX_JLINK_SERIAL"):
        resolve_serial(None, env={"HPX_JLINK_SERIAL": "abc"}, enumerate_probes=_enumerator([]))


def test_single_connected_probe_is_used() -> None:
    enumerate = _enumerator([ProbeInfo(1160002276, "J-Link OB")])
    assert resolve_serial(None, env={}, enumerate_probes=enumerate) == 1160002276
    assert enumerate.calls["count"] == 1


def test_zero_probes_retries_once_then_errors_and_asks_for_serial() -> None:
    slept: list[float] = []
    enumerate = _enumerator([])
    with pytest.raises(ProbeResolutionError, match="No connected J-Link probes.*--serial-no"):
        resolve_serial(None, env={}, enumerate_probes=enumerate, retry_delay_s=0.25, sleep=slept.append)
    assert enumerate.calls["count"] == 2 and slept == [0.25]


def test_probe_missing_on_first_enumeration_is_found_on_the_retry() -> None:
    slept: list[float] = []
    enumerate = _enumerator([], [ProbeInfo(1160002276, "J-Link OB")])
    assert resolve_serial(None, env={}, enumerate_probes=enumerate, sleep=slept.append) == 1160002276
    assert enumerate.calls["count"] == 2 and slept == [probes.ENUMERATION_RETRY_DELAY_S]

    # A non-empty first enumeration is never retried, even when it is ambiguous.
    enumerate = _enumerator([ProbeInfo(1), ProbeInfo(2)], [ProbeInfo(1)])
    with pytest.raises(ProbeResolutionError, match="Multiple"):
        resolve_serial(None, env={}, enumerate_probes=enumerate, sleep=slept.append)
    assert enumerate.calls["count"] == 1


def test_multiple_probes_error_lists_them() -> None:
    found = [ProbeInfo(1160002276, "J-Link OB"), ProbeInfo(1160001958, "J-Link OB")]
    with pytest.raises(ProbeResolutionError) as excinfo:
        resolve_serial(None, env={}, enumerate_probes=_enumerator(found))
    message = str(excinfo.value)
    assert "Multiple J-Link probes" in message
    assert "1160002276 (J-Link OB)" in message and "1160001958 (J-Link OB)" in message
    assert "--serial-no" in message


def test_list_probes_uses_pylink_enumeration(monkeypatch) -> None:
    class _Info:
        def __init__(self, serial: int, product: bytes) -> None:
            self.SerialNumber = serial
            self.acProduct = product

    class _FakeJLink:
        closed = False

        def __init__(self, lib=None) -> None:
            _FakeJLink.lib = lib

        def connected_emulators(self):
            return [_Info(2, b"J-Link OB\x00\x00"), _Info(1, b"J-Link Plus")]

        def close(self):
            _FakeJLink.closed = True

    import pylink

    monkeypatch.setattr(pylink, "JLink", _FakeJLink)
    assert probes.list_probes() == [ProbeInfo(1, "J-Link Plus"), ProbeInfo(2, "J-Link OB")]
    assert _FakeJLink.closed


def test_list_probes_reports_missing_dll(monkeypatch) -> None:
    import pylink

    def _boom(*args, **kwargs):
        raise TypeError("Expected to be given a valid DLL.")

    monkeypatch.setattr(pylink, "JLink", _boom)
    with pytest.raises(ProbeResolutionError, match="J-Link library"):
        probes.list_probes()
