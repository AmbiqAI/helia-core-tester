"""The NSX facade forwards arguments, enforces timeouts and translates errors.

Every neuralspotx.api entry point is monkeypatched; nothing here runs NSX.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
from neuralspotx._io import Event
from neuralspotx.api import NSXError

from helia_core_tester.hardware import nsx_cli
from helia_core_tester.hardware.nsx_cli import HardwareBuildError

APP = Path("/tmp/app")


def _capture(monkeypatch: pytest.MonkeyPatch, name: str, result: Any = None) -> dict[str, Any]:
    """Replace one NSX API function; return its captured kwargs."""
    seen: dict[str, Any] = {}

    def fake(app_dir: Any, **kwargs: Any) -> Any:
        seen["app_dir"] = app_dir
        seen.update(kwargs)
        return result

    monkeypatch.setattr(nsx_cli.nsx_api, name, fake)
    return seen


def test_lock_app_forwards_update_and_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _capture(monkeypatch, "lock_app", result="LOCK")
    assert nsx_cli.lock_app(APP, update=True, timeout_s=42) == "LOCK"
    assert seen["app_dir"] == APP
    assert seen["update"] is True
    assert seen["timeout_s"] == 42


def test_lock_app_default_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _capture(monkeypatch, "lock_app")
    nsx_cli.lock_app(APP)
    assert seen["timeout_s"] == nsx_cli.LOCK_TIMEOUT_S
    assert seen["update"] is False


def test_sync_app_is_unfrozen_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # A frozen sync cannot populate a fresh modules/ tree.
    seen = _capture(monkeypatch, "sync_app")
    nsx_cli.sync_app(APP)
    assert seen["frozen"] is False
    assert seen["force"] is False
    assert seen["timeout_s"] == nsx_cli.SYNC_TIMEOUT_S


def test_sync_app_forwards_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _capture(monkeypatch, "sync_app")
    nsx_cli.sync_app(APP, frozen=True, force=True, timeout_s=7)
    assert seen["frozen"] is True
    assert seen["force"] is True
    assert seen["timeout_s"] == 7


def test_configure_app_forwards_board_and_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _capture(monkeypatch, "configure_app")
    nsx_cli.configure_app(APP, "apollo510_evb", build_dir=Path("/tmp/b"), toolchain="gcc", frozen=True, timeout_s=9)
    assert seen["app_dir"] == APP
    assert seen["board"] == "apollo510_evb"
    assert seen["build_dir"] == Path("/tmp/b")
    assert seen["toolchain"] == "gcc"
    assert seen["frozen"] is True
    assert seen["timeout_s"] == 9


def test_configure_app_default_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _capture(monkeypatch, "configure_app")
    nsx_cli.configure_app(APP, "apollo510_evb")
    assert seen["timeout_s"] == nsx_cli.CONFIGURE_TIMEOUT_S
    assert seen["frozen"] is False


def test_build_app_forwards_jobs_and_target(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _capture(monkeypatch, "build_app")
    nsx_cli.build_app(APP, board="apollo510_evb", target="hct_benchmark_server", jobs=4, frozen=True, timeout_s=11)
    assert seen["board"] == "apollo510_evb"
    assert seen["target"] == "hct_benchmark_server"
    assert seen["jobs"] == 4
    assert seen["frozen"] is True
    assert seen["timeout_s"] == 11


def test_build_app_jobs_none_keeps_nsx_default(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _capture(monkeypatch, "build_app")
    nsx_cli.build_app(APP)
    assert "jobs" not in seen
    assert seen["frozen"] is False
    assert seen["timeout_s"] == nsx_cli.BUILD_TIMEOUT_S


def test_starter_profile_forwards_board(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake(board: str) -> dict[str, Any]:
        seen["board"] = board
        return {"modules": ["nsx-core"]}

    monkeypatch.setattr(nsx_cli.nsx_api, "starter_profile", fake)
    assert nsx_cli.starter_profile("apollo510_evb") == {"modules": ["nsx-core"]}
    assert seen["board"] == "apollo510_evb"


@pytest.mark.parametrize(
    "name, call",
    [
        ("lock_app", lambda: nsx_cli.lock_app(APP)),
        ("sync_app", lambda: nsx_cli.sync_app(APP)),
        ("configure_app", lambda: nsx_cli.configure_app(APP, "apollo510_evb")),
        ("build_app", lambda: nsx_cli.build_app(APP)),
    ],
)
def test_nsx_error_becomes_hardware_build_error(monkeypatch: pytest.MonkeyPatch, name: str, call: Any) -> None:
    def fake(*args: Any, **kwargs: Any) -> None:
        raise NSXError("modules/ drifted from nsx.lock")

    monkeypatch.setattr(nsx_cli.nsx_api, name, fake)
    with pytest.raises(HardwareBuildError, match=r"^nsx \w+ failed: modules/ drifted") as info:
        call()
    assert isinstance(info.value.__cause__, NSXError)


def test_called_process_error_becomes_hardware_build_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # NSX's build runner raises this unwrapped, with no stderr.
    def fake(*args: Any, **kwargs: Any) -> None:
        raise subprocess.CalledProcessError(2, ["ninja", "-C", "build"])

    monkeypatch.setattr(nsx_cli.nsx_api, "build_app", fake)
    with pytest.raises(HardwareBuildError, match=r"^nsx build failed: Command .*ninja.*exit status 2") as info:
        nsx_cli.build_app(APP)
    assert isinstance(info.value.__cause__, subprocess.CalledProcessError)


def test_verbosity_zero_swallows_nsx_notes(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    seen = _capture(monkeypatch, "sync_app")
    nsx_cli.sync_app(APP, verbosity=0)
    emit = seen["emit"]
    assert emit is not None
    emit(Event("line", "ninja: building 12 targets"))
    emit(Event("error", "boom"))
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_verbosity_one_uses_nsx_default_emitter(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _capture(monkeypatch, "lock_app")
    nsx_cli.lock_app(APP, verbosity=1)
    assert seen["emit"] is None
    assert "quiet" not in seen
