"""The FVP is launched with the glibc tunable that lets its model library get an executable stack."""

from __future__ import annotations

from pathlib import Path

import pytest

from helia_core_tester.fvp import env as fvp_env
from helia_core_tester.fvp import runner as fvp_runner

ALLOW = "glibc.rtld.execstack=2"


@pytest.mark.parametrize(
    "current, expected",
    [
        (None, ALLOW),
        ("", ALLOW),
        ("glibc.malloc.check=3", f"glibc.malloc.check=3:{ALLOW}"),
        ("glibc.malloc.check=3::glibc.pthread.rseq=0:", f"glibc.malloc.check=3:glibc.pthread.rseq=0:{ALLOW}"),
    ],
)
def test_linux_adds_the_tunable_and_keeps_the_others(current, expected) -> None:
    env = {"PATH": "/bin"} if current is None else {"PATH": "/bin", "GLIBC_TUNABLES": current}
    out = fvp_env.fvp_process_env(env, system="Linux")
    assert out["GLIBC_TUNABLES"] == expected
    assert out["PATH"] == "/bin"


@pytest.mark.parametrize("value", ["glibc.rtld.execstack=0", "glibc.malloc.check=3:glibc.rtld.execstack=1"])
def test_an_explicit_execstack_value_wins(value) -> None:
    out = fvp_env.fvp_process_env({"GLIBC_TUNABLES": value}, system="Linux")
    assert out["GLIBC_TUNABLES"] == value


def test_a_similarly_named_tunable_is_not_mistaken_for_it() -> None:
    out = fvp_env.fvp_process_env({"GLIBC_TUNABLES": "glibc.rtld.execstackx=0"}, system="Linux")
    assert out["GLIBC_TUNABLES"] == f"glibc.rtld.execstackx=0:{ALLOW}"


@pytest.mark.parametrize("system", ["Darwin", "Windows"])
def test_other_systems_are_left_alone(system) -> None:
    env = {"GLIBC_TUNABLES": "glibc.malloc.check=3"}
    assert fvp_env.fvp_process_env(env, system=system) == env
    assert fvp_env.fvp_process_env({}, system=system) == {}


def test_the_callers_env_is_not_mutated() -> None:
    env = {"GLIBC_TUNABLES": "glibc.malloc.check=3"}
    out = fvp_env.fvp_process_env(env, system="Linux")
    assert env == {"GLIBC_TUNABLES": "glibc.malloc.check=3"}
    assert out is not env


def test_run_fvp_process_launches_with_the_tunable(monkeypatch, tmp_path) -> None:
    seen = {}

    class FakePopen:
        pid = 4242
        returncode = 0

        def __init__(self, cmd, **kwargs):
            seen["cmd"] = cmd
            seen["env"] = kwargs["env"]

        def communicate(self, timeout=None):
            return "ok\n", None

    monkeypatch.setattr(fvp_env.platform, "system", lambda: "Linux")
    monkeypatch.setattr(fvp_runner.subprocess, "Popen", FakePopen)
    caller_env = {"PATH": "/bin"}
    result = fvp_runner.run_fvp_process(
        fvp_exe=Path("/opt/fvp/FVP_Corstone_SSE-300_Ethos-U55"),
        elf=tmp_path / "case.elf",
        timeout=5,
        verbosity=0,
        extra_args=[],
        env=caller_env,
        cpu="cortex-m55",
        descriptor_name="case",
    )
    assert result.exit_code == 0 and result.launch_error is None
    assert seen["env"]["GLIBC_TUNABLES"] == ALLOW
    assert caller_env == {"PATH": "/bin"}
