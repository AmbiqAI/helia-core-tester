"""The Arm GCC pin: URL and digest per architecture, and an install that replaces stale toolchains."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from helia_core_tester.scripts import setup_dependencies as sd


def _archive_bytes(top: str = "arm-gnu-toolchain-14.3.rel1-aarch64-arm-none-eabi", with_bin: bool = True) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:xz") as tar:
        names = [f"{top}/bin/arm-none-eabi-gcc"] if with_bin else [f"{top}/share/readme"]
        for name in names:
            data = b"#!/bin/sh\n"
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


@pytest.fixture
def fake_download(monkeypatch):
    """Serves an archive from memory; records each request."""
    state = {"payload": _archive_bytes(), "calls": [], "fail": None}

    def _download(url, dest, description, expected_sha256):
        state["calls"].append((url, expected_sha256))
        if state["fail"]:
            raise state["fail"]
        Path(dest).write_bytes(state["payload"])

    monkeypatch.setattr(sd, "download_file", _download)
    monkeypatch.setattr(sd, "get_architecture", lambda: "aarch64")
    return state


def _old_install(downloads: Path, version: str | None) -> Path:
    gcc = downloads / "arm_gcc_download"
    (gcc / "bin").mkdir(parents=True)
    (gcc / "bin" / "arm-none-eabi-gcc").write_text("old\n")
    if version is not None:
        (gcc / sd.ARM_GCC_VERSION_MARKER).write_text(version + "\n")
    return gcc


def test_pin_matches_heliaaot_and_has_a_digest_per_arch() -> None:
    assert sd.ARM_GCC_VERSION == "14.3.rel1"
    for arch in ("x86_64", "aarch64"):
        url = sd.arm_gcc_url(arch)
        assert url.endswith(f"/gnu/14.3.rel1/binrel/arm-gnu-toolchain-14.3.rel1-{arch}-arm-none-eabi.tar.xz")
        digest = sd.PINNED_SHA256[("arm_gcc", arch)]
        assert len(digest) == 64 and int(digest, 16) >= 0
    assert sd.PINNED_SHA256[("arm_gcc", "x86_64")] != sd.PINNED_SHA256[("arm_gcc", "aarch64")]


@pytest.mark.parametrize("arch", ["armv7l", "", "x86"])
def test_unsupported_arch_has_no_url(arch: str) -> None:
    with pytest.raises(RuntimeError, match="Unsupported architecture"):
        sd.arm_gcc_url(arch)


def test_fresh_install_downloads_the_pinned_archive_and_records_its_version(tmp_path, fake_download) -> None:
    sd.setup_arm_gcc(tmp_path)
    gcc = tmp_path / "arm_gcc_download"
    assert (gcc / "bin" / "arm-none-eabi-gcc").is_file()
    assert sd.installed_arm_gcc_version(gcc) == "14.3.rel1"
    assert fake_download["calls"] == [(sd.arm_gcc_url("aarch64"), sd.PINNED_SHA256[("arm_gcc", "aarch64")])]


def test_current_install_is_kept(tmp_path, fake_download) -> None:
    _old_install(tmp_path, sd.ARM_GCC_VERSION)
    sd.setup_arm_gcc(tmp_path)
    assert fake_download["calls"] == []


@pytest.mark.parametrize("recorded", [None, "14.2.rel1", "", "garbage"])
def test_stale_or_unrecorded_install_is_replaced(tmp_path, fake_download, recorded) -> None:
    gcc = _old_install(tmp_path, recorded)
    sd.setup_arm_gcc(tmp_path)
    assert len(fake_download["calls"]) == 1
    assert sd.installed_arm_gcc_version(gcc) == "14.3.rel1"
    assert (gcc / "bin" / "arm-none-eabi-gcc").read_text() == "#!/bin/sh\n"


def test_force_reinstalls_a_current_install(tmp_path, fake_download) -> None:
    _old_install(tmp_path, sd.ARM_GCC_VERSION)
    sd.setup_arm_gcc(tmp_path, force=True)
    assert len(fake_download["calls"]) == 1


def test_failed_download_keeps_the_old_toolchain_marked_stale(tmp_path, fake_download) -> None:
    gcc = _old_install(tmp_path, "14.2.rel1")
    fake_download["fail"] = sd.ChecksumMismatchError("digest mismatch")
    with pytest.raises(sd.ChecksumMismatchError):
        sd.setup_arm_gcc(tmp_path)
    assert (gcc / "bin" / "arm-none-eabi-gcc").read_text() == "old\n"
    assert sd.installed_arm_gcc_version(gcc) == "14.2.rel1"


def test_archive_without_bin_is_refused_and_old_install_kept(tmp_path, fake_download) -> None:
    gcc = _old_install(tmp_path, None)
    fake_download["payload"] = _archive_bytes(with_bin=False)
    with pytest.raises(RuntimeError, match="no bin/"):
        sd.setup_arm_gcc(tmp_path)
    assert (gcc / "bin" / "arm-none-eabi-gcc").read_text() == "old\n"
    assert sd.installed_arm_gcc_version(gcc) is None


def test_archive_with_two_top_dirs_is_refused(tmp_path, fake_download, monkeypatch) -> None:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:xz") as tar:
        for top in ("a", "b"):
            info = tarfile.TarInfo(f"{top}/bin/x")
            info.size = 0
            tar.addfile(info, io.BytesIO(b""))
    fake_download["payload"] = buf.getvalue()
    with pytest.raises(RuntimeError, match="Expected one toolchain directory"):
        sd.setup_arm_gcc(tmp_path)
    assert not (tmp_path / "arm_gcc_download").exists()


def test_hardware_entry_point_replaces_a_stale_install(tmp_path, fake_download, monkeypatch) -> None:
    from helia_core_tester.hardware import firmware_build

    downloads = tmp_path / firmware_build.DOWNLOADS_DIR
    gcc = _old_install(downloads, None)
    monkeypatch.setattr(firmware_build, "add_toolchain_to_path", lambda root: False)
    firmware_build.ensure_build_tools(tmp_path)
    assert sd.installed_arm_gcc_version(gcc) == sd.ARM_GCC_VERSION
    assert len(fake_download["calls"]) == 1


def test_hardware_entry_point_keeps_a_current_install(tmp_path, fake_download, monkeypatch) -> None:
    from helia_core_tester.hardware import firmware_build

    gcc = _old_install(tmp_path / firmware_build.DOWNLOADS_DIR, sd.ARM_GCC_VERSION)
    monkeypatch.setattr(firmware_build, "add_toolchain_to_path", lambda root: False)
    firmware_build.ensure_build_tools(tmp_path)
    assert (gcc / "bin" / "arm-none-eabi-gcc").read_text() == "old\n"
    assert fake_download["calls"] == []
