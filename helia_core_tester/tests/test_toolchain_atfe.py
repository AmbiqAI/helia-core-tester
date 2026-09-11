"""--toolchain atfe: Arm Toolchain for Embedded download, path detection and fvp CLI plumbing."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from helia_core_tester.fvp import cli as fvp_cli
from helia_core_tester.fvp import env as fvp_env
from helia_core_tester.fvp.errors import FvpScriptError
from helia_core_tester.scripts import setup_dependencies as sd


# --- setup_dependencies -------------------------------------------------------


@pytest.mark.parametrize(
    ("arch", "expected_tag"),
    [("x86_64", "x86_64"), ("aarch64", "AArch64")],
)
def test_atfe_download_url_and_pin(arch: str, expected_tag: str) -> None:
    url = sd.atfe_download_url(sd.ATFE_VERSION, arch)
    assert url == (
        "https://github.com/ARM-software/LLVM-embedded-toolchain-for-Arm/releases/download/"
        f"release-{sd.ATFE_VERSION}/LLVM-ET-Arm-{sd.ATFE_VERSION}-Linux-{expected_tag}.tar.xz"
    )
    assert ("atfe", sd.ATFE_VERSION, arch) in sd.PINNED_SHA256


def _install_fakes(monkeypatch: pytest.MonkeyPatch, calls: dict) -> None:
    def fake_download(url, dest, description, expected_sha256):
        calls["download"] = (url, expected_sha256)
        Path(dest).write_bytes(b"archive")

    def fake_extract(archive_path, extract_to, strip_components=0):
        toolchain = Path(extract_to) / f"LLVM-ET-Arm-{sd.ATFE_VERSION}-Linux-x86_64"
        (toolchain / "bin").mkdir(parents=True)
        (toolchain / "bin" / "clang").write_text("#!/bin/sh\n")

    monkeypatch.setattr(sd, "download_file", fake_download)
    monkeypatch.setattr(sd, "extract_tar_gz", fake_extract)
    monkeypatch.setattr(sd, "get_architecture", lambda: "x86_64")


def test_setup_atfe_fresh_install_writes_marker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: dict = {}
    _install_fakes(monkeypatch, calls)

    sd.setup_atfe(tmp_path)

    atfe_dir = tmp_path / sd.ATFE_DIRNAME
    assert (atfe_dir / "bin" / "clang").exists()
    assert sd.read_installed_atfe_version(atfe_dir) == sd.ATFE_VERSION
    url, digest = calls["download"]
    assert url == sd.atfe_download_url(sd.ATFE_VERSION, "x86_64")
    assert digest == sd.PINNED_SHA256[("atfe", sd.ATFE_VERSION, "x86_64")]


def test_setup_atfe_marker_match_returns_early_and_mismatch_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: dict = {}
    _install_fakes(monkeypatch, calls)
    sd.setup_atfe(tmp_path)
    calls.clear()

    sd.setup_atfe(tmp_path)
    assert "download" not in calls

    (tmp_path / sd.ATFE_DIRNAME / sd.ATFE_VERSION_MARKER).write_text("18.1.3\n")
    with pytest.raises(RuntimeError, match="18.1.3 is installed .* --force"):
        sd.setup_atfe(tmp_path)

    sd.setup_atfe(tmp_path, force=True)
    assert "download" in calls
    assert sd.read_installed_atfe_version(tmp_path / sd.ATFE_DIRNAME) == sd.ATFE_VERSION


def test_cli_with_atfe_is_opt_in(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: list[str] = []
    monkeypatch.setattr(sd, "setup_atfe", lambda downloads_dir, force=False: seen.append("atfe"))
    monkeypatch.setattr(sd, "get_architecture", lambda: "x86_64")
    monkeypatch.setattr(sd, "get_os", lambda: "linux")
    monkeypatch.delenv(sd.GCC_VERSION_ENV, raising=False)
    base = [
        "setup_dependencies.py",
        "--downloads-dir", str(tmp_path),
        "--skip-corstone", "--skip-gcc", "--skip-cmsis5", "--skip-ethos", "--skip-python", "--skip-nsx-sdk",
    ]

    monkeypatch.setattr(sd.sys, "argv", base)
    assert sd.main() == 0
    assert seen == []

    monkeypatch.setattr(sd.sys, "argv", [*base, "--with-atfe"])
    assert sd.main() == 0
    assert seen == ["atfe"]


# --- fvp CLI ------------------------------------------------------------------


def _parse(*argv: str):
    parser = fvp_cli.build_arg_parser(Path("dl"), Path("src"))
    return parser.parse_args(list(argv))


def test_resolve_toolchain_defaults_and_alias() -> None:
    assert fvp_cli.resolve_toolchain(_parse()) == "gcc"
    assert fvp_cli.resolve_toolchain(_parse("--toolchain", "atfe")) == "atfe"
    assert fvp_cli.resolve_toolchain(_parse("-a")) == "armclang"
    assert fvp_cli.resolve_toolchain(_parse("-a", "--toolchain", "armclang")) == "armclang"
    with pytest.raises(ValueError, match="conflicts"):
        fvp_cli.resolve_toolchain(_parse("-a", "--toolchain", "atfe"))


def test_fvp_cli_rejects_unknown_toolchain() -> None:
    with pytest.raises(SystemExit):
        _parse("--toolchain", "iar")


# --- detect_paths -------------------------------------------------------------


def _downloads(tmp_path: Path, with_atfe: bool) -> Path:
    dl = tmp_path / "downloads"
    (dl / "ethos-u-core-platform" / "cmake" / "toolchain").mkdir(parents=True)
    (dl / "CMSIS_5").mkdir()
    if with_atfe:
        (dl / fvp_env.ATFE_DIRNAME / "bin").mkdir(parents=True)
        (dl / fvp_env.ATFE_DIRNAME / "bin" / "clang").write_text("")
    return dl


def _args(dl: Path, toolchain: str) -> SimpleNamespace:
    return SimpleNamespace(
        downloads_dir=dl,
        ethos_path=None,
        cmsis5_path=None,
        toolchain=toolchain,
        use_arm_compiler=False,
        no_gcc_from_download=True,
        no_fvp_from_download=True,
        no_run=True,
    )


def test_detect_paths_atfe_uses_tester_toolchain_file(tmp_path: Path) -> None:
    dl = _downloads(tmp_path, with_atfe=True)
    ctx = fvp_env.detect_paths(_args(dl, "atfe"))

    assert ctx["toolchain_file"] == fvp_env.REPO_ROOT / "cmake" / "toolchain" / "atfe.cmake"
    assert ctx["toolchain_file"].exists()
    assert ctx["compiler_tag"] == "atfe"
    assert ctx["env"][fvp_env.ATFE_ROOT_ENV] == str(dl / fvp_env.ATFE_DIRNAME)
    assert ctx["env"]["PATH"].startswith(str(dl / fvp_env.ATFE_DIRNAME / "bin"))


def test_detect_paths_atfe_missing_install_points_at_setup(tmp_path: Path) -> None:
    dl = _downloads(tmp_path, with_atfe=False)
    with pytest.raises(FvpScriptError, match="--with-atfe"):
        fvp_env.detect_paths(_args(dl, "atfe"))


def test_call_setup_dependencies_requests_atfe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: list[list[str]] = []
    monkeypatch.setattr(fvp_env, "find_setup_dependencies_script", lambda root: tmp_path / "setup.py")
    (tmp_path / "setup.py").write_text("")
    monkeypatch.setattr(fvp_env.subprocess, "call", lambda cmd, cwd=None: seen.append(cmd) or 0)

    fvp_env.call_setup_dependencies(tmp_path, "atfe")
    fvp_env.call_setup_dependencies(tmp_path, "gcc")
    fvp_env.call_setup_dependencies(tmp_path)
    assert "--with-atfe" in seen[0]
    assert all("--with-atfe" not in cmd for cmd in seen[1:])
