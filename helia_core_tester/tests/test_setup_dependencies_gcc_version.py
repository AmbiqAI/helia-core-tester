from __future__ import annotations

from pathlib import Path

import pytest

from helia_core_tester.scripts import setup_dependencies as sd


PINNED_VERSIONS = ("13.2.rel1", "13.3.rel1", "14.2.rel1", "14.3.rel1", "15.2.rel1")
FAKE_DIGEST = "ab" * 32


@pytest.mark.parametrize("arch", ["x86_64", "aarch64"])
@pytest.mark.parametrize("version", PINNED_VERSIONS)
def test_download_url_per_version_and_arch(version: str, arch: str) -> None:
    url = sd.arm_gcc_download_url(version, arch)
    assert url == (
        f"https://developer.arm.com/-/media/Files/downloads/gnu/{version}/binrel/"
        f"arm-gnu-toolchain-{version}-{arch}-arm-none-eabi.tar.xz"
    )


def test_default_url_matches_pre_existing_hardcoded_url() -> None:
    assert sd.arm_gcc_download_url(sd.DEFAULT_GCC_VERSION, "x86_64") == (
        "https://developer.arm.com/-/media/Files/downloads/gnu/14.2.rel1/binrel/"
        "arm-gnu-toolchain-14.2.rel1-x86_64-arm-none-eabi.tar.xz"
    )


@pytest.mark.parametrize(
    "raw, expected",
    [("13.2.Rel1", "13.2.rel1"), ("  15.2.REL1 ", "15.2.rel1"), ("14.2.rel1", "14.2.rel1")],
)
def test_normalize_gcc_version_lowercases(raw: str, expected: str) -> None:
    assert sd.normalize_gcc_version(raw) == expected


@pytest.mark.parametrize("bad", ["14.2", "14.2.1", "gcc-14", "14.2.rel", "14.2.rel1-x86_64", "", "v14.2.rel1"])
def test_normalize_gcc_version_rejects_bad_format(bad: str) -> None:
    with pytest.raises(ValueError, match="Invalid ARM GCC version"):
        sd.normalize_gcc_version(bad)


def test_resolve_gcc_version_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(sd.GCC_VERSION_ENV, raising=False)
    assert sd.resolve_gcc_version(None) == sd.DEFAULT_GCC_VERSION == "14.2.rel1"

    monkeypatch.setenv(sd.GCC_VERSION_ENV, "13.3.Rel1")
    assert sd.resolve_gcc_version(None) == "13.3.rel1"

    assert sd.resolve_gcc_version("15.2.rel1") == "15.2.rel1"


def test_resolve_gcc_version_invalid_env_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(sd.GCC_VERSION_ENV, "latest")
    with pytest.raises(ValueError, match="Invalid ARM GCC version"):
        sd.resolve_gcc_version(None)


@pytest.mark.parametrize("arch", ["x86_64", "aarch64"])
@pytest.mark.parametrize("version", PINNED_VERSIONS)
def test_pinned_lookup_returns_table_digest_without_network(
    monkeypatch: pytest.MonkeyPatch, version: str, arch: str
) -> None:
    def _no_network(*_args, **_kwargs):
        raise AssertionError("pinned versions must not hit the sidecar")

    monkeypatch.setattr(sd, "fetch_sha256_sidecar", _no_network)
    digest = sd.resolve_gcc_sha256(version, arch)
    assert digest == sd.PINNED_SHA256[("arm_gcc", version, arch)]
    assert len(digest) == 64
    int(digest, 16)


def test_parse_sha256_sidecar_accepts_arm_format() -> None:
    text = f"{FAKE_DIGEST}  arm-gnu-toolchain-15.3.rel1-x86_64-arm-none-eabi.tar.xz\n"
    assert sd.parse_sha256_sidecar(text) == FAKE_DIGEST


def test_parse_sha256_sidecar_uppercase_and_leading_blank_lines() -> None:
    text = f"\n\n{FAKE_DIGEST.upper()}  file.tar.xz\n"
    assert sd.parse_sha256_sidecar(text) == FAKE_DIGEST


@pytest.mark.parametrize("bad", ["", "\n", "<html>404</html>", "deadbeef  file.tar.xz", "file.tar.xz " + FAKE_DIGEST])
def test_parse_sha256_sidecar_rejects_malformed(bad: str) -> None:
    with pytest.raises(ValueError):
        sd.parse_sha256_sidecar(bad)


def test_unpinned_version_fetches_sidecar_and_prints_note(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    version, arch = "15.3.rel1", "x86_64"
    assert ("arm_gcc", version, arch) not in sd.PINNED_SHA256
    expected_url = sd.arm_gcc_download_url(version, arch) + ".sha256asc"
    seen: list[str] = []

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return f"{FAKE_DIGEST}  arm-gnu-toolchain-{version}-{arch}-arm-none-eabi.tar.xz\n".encode()

    def fake_urlopen(url):
        seen.append(url)
        return _FakeResponse()

    monkeypatch.setattr(sd.urllib.request, "urlopen", fake_urlopen)
    assert sd.resolve_gcc_sha256(version, arch) == FAKE_DIGEST
    assert seen == [expected_url]
    out = capsys.readouterr().out
    assert "NOTE" in out
    assert "same origin" in out
    assert "--gcc-sha256" in out


def test_unpinned_version_sidecar_failure_points_at_override(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(url):
        raise OSError("HTTP Error 404: Not Found")

    monkeypatch.setattr(sd.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="--gcc-sha256"):
        sd.resolve_gcc_sha256("15.3.rel1", "x86_64")


def test_explicit_sha256_override_wins_over_table(monkeypatch: pytest.MonkeyPatch) -> None:
    def _no_network(*_args, **_kwargs):
        raise AssertionError("override must short-circuit the sidecar")

    monkeypatch.setattr(sd, "fetch_sha256_sidecar", _no_network)
    override = "F" * 64
    assert sd.resolve_gcc_sha256("14.2.rel1", "x86_64", override=override) == "f" * 64
    assert sd.resolve_gcc_sha256("14.2.rel1", "x86_64", override=override) != sd.PINNED_SHA256[
        ("arm_gcc", "14.2.rel1", "x86_64")
    ]


def test_explicit_sha256_override_rejects_non_hex() -> None:
    with pytest.raises(ValueError, match="--gcc-sha256"):
        sd.resolve_gcc_sha256("14.2.rel1", "x86_64", override="not-a-digest")


# --- setup_arm_gcc() install/early-return paths, no network -------------------


def _install_fakes(monkeypatch: pytest.MonkeyPatch, calls: dict) -> None:
    def fake_download(url, dest, description, expected_sha256):
        calls["download"] = (url, expected_sha256)
        Path(dest).write_bytes(b"archive")

    def fake_extract(archive_path, extract_to, strip_components=0):
        calls["extract"] = True
        toolchain = Path(extract_to) / "arm-gnu-toolchain-fake"
        (toolchain / "bin").mkdir(parents=True)
        (toolchain / "bin" / "arm-none-eabi-gcc").write_text("#!/bin/sh\n")

    monkeypatch.setattr(sd, "download_file", fake_download)
    monkeypatch.setattr(sd, "extract_tar_gz", fake_extract)
    monkeypatch.setattr(sd, "get_architecture", lambda: "x86_64")
    monkeypatch.delenv(sd.GCC_VERSION_ENV, raising=False)
    monkeypatch.delenv(sd.GCC_SHA256_ENV, raising=False)


def test_fresh_install_writes_marker_and_uses_pinned_digest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: dict = {}
    _install_fakes(monkeypatch, calls)

    sd.setup_arm_gcc(tmp_path, version="13.2.rel1")

    gcc_dir = tmp_path / "arm_gcc_download"
    assert (gcc_dir / "bin" / "arm-none-eabi-gcc").exists()
    assert (gcc_dir / sd.GCC_VERSION_MARKER).read_text().strip() == "13.2.rel1"
    assert sd.read_installed_gcc_version(gcc_dir) == "13.2.rel1"
    url, digest = calls["download"]
    assert url == sd.arm_gcc_download_url("13.2.rel1", "x86_64")
    assert digest == sd.PINNED_SHA256[("arm_gcc", "13.2.rel1", "x86_64")]


def test_marker_mismatch_raises_with_force_hint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: dict = {}
    _install_fakes(monkeypatch, calls)
    gcc_dir = tmp_path / "arm_gcc_download"
    gcc_dir.mkdir()
    (gcc_dir / sd.GCC_VERSION_MARKER).write_text("14.2.rel1\n")

    with pytest.raises(RuntimeError) as excinfo:
        sd.setup_arm_gcc(tmp_path, version="13.2.rel1")

    message = str(excinfo.value)
    assert "14.2.rel1" in message
    assert "13.2.rel1" in message
    assert "--force" in message
    assert "download" not in calls


def test_marker_match_returns_early(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: dict = {}
    _install_fakes(monkeypatch, calls)
    gcc_dir = tmp_path / "arm_gcc_download"
    gcc_dir.mkdir()
    (gcc_dir / sd.GCC_VERSION_MARKER).write_text("14.2.rel1\n")

    sd.setup_arm_gcc(tmp_path)

    assert "download" not in calls
    assert "14.2.rel1 already installed" in capsys.readouterr().out


def test_missing_marker_with_default_version_continues(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: dict = {}
    _install_fakes(monkeypatch, calls)
    gcc_dir = tmp_path / "arm_gcc_download"
    (gcc_dir / "bin").mkdir(parents=True)

    sd.setup_arm_gcc(tmp_path)

    assert "download" not in calls
    assert not (gcc_dir / sd.GCC_VERSION_MARKER).exists()
    assert f"assumed {sd.DEFAULT_GCC_VERSION}" in capsys.readouterr().out


def test_missing_marker_with_non_default_version_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: dict = {}
    _install_fakes(monkeypatch, calls)
    gcc_dir = tmp_path / "arm_gcc_download"
    (gcc_dir / "bin").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="cannot be confirmed as 13.2.rel1.*--force"):
        sd.setup_arm_gcc(tmp_path, version="13.2.rel1")

    assert "download" not in calls


def test_force_reinstalls_and_rewrites_marker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: dict = {}
    _install_fakes(monkeypatch, calls)
    gcc_dir = tmp_path / "arm_gcc_download"
    (gcc_dir / "bin").mkdir(parents=True)
    (gcc_dir / "bin" / "stale").write_text("old")
    (gcc_dir / sd.GCC_VERSION_MARKER).write_text("14.2.rel1\n")

    sd.setup_arm_gcc(tmp_path, force=True, version="15.2.rel1")

    assert calls["download"][0] == sd.arm_gcc_download_url("15.2.rel1", "x86_64")
    assert calls["extract"] is True
    assert not (gcc_dir / "bin" / "stale").exists()
    assert (gcc_dir / "bin" / "arm-none-eabi-gcc").exists()
    assert sd.read_installed_gcc_version(gcc_dir) == "15.2.rel1"


def test_setup_arm_gcc_honours_env_version_and_sha(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: dict = {}
    _install_fakes(monkeypatch, calls)
    monkeypatch.setenv(sd.GCC_VERSION_ENV, "15.3.Rel1")
    monkeypatch.setenv(sd.GCC_SHA256_ENV, FAKE_DIGEST)

    sd.setup_arm_gcc(tmp_path)

    url, digest = calls["download"]
    assert url == sd.arm_gcc_download_url("15.3.rel1", "x86_64")
    assert digest == FAKE_DIGEST
    assert sd.read_installed_gcc_version(tmp_path / "arm_gcc_download") == "15.3.rel1"


def test_cli_flags_reach_setup_arm_gcc(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict = {}

    def fake_setup_arm_gcc(downloads_dir, force=False, version=None, sha256=None):
        seen.update(downloads_dir=downloads_dir, force=force, version=version, sha256=sha256)

    monkeypatch.setattr(sd, "setup_arm_gcc", fake_setup_arm_gcc)
    monkeypatch.setattr(sd, "get_architecture", lambda: "x86_64")
    monkeypatch.setattr(sd, "get_os", lambda: "linux")
    monkeypatch.delenv(sd.GCC_VERSION_ENV, raising=False)
    monkeypatch.setattr(
        sd.sys,
        "argv",
        [
            "setup_dependencies.py",
            "--downloads-dir", str(tmp_path),
            "--gcc-version", "13.3.Rel1",
            "--gcc-sha256", FAKE_DIGEST,
            "--skip-corstone", "--skip-cmsis5", "--skip-ethos", "--skip-python", "--skip-nsx-sdk",
        ],
    )

    assert sd.main() == 0
    assert seen["version"] == "13.3.rel1"
    assert seen["sha256"] == FAKE_DIGEST
    assert seen["force"] is False


def test_cli_rejects_invalid_gcc_version(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr(sd, "get_architecture", lambda: "x86_64")
    monkeypatch.setattr(sd.sys, "argv", ["setup_dependencies.py", "--downloads-dir", str(tmp_path), "--gcc-version", "14"])
    assert sd.main() == 1
    assert "Invalid ARM GCC version" in capsys.readouterr().out
