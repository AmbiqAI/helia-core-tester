from __future__ import annotations

from pathlib import Path

from helia_core_tester.fvp.cmake import cmake_configure


def _stub_configure_subprocess(monkeypatch) -> None:
    """cmake_configure only needs a successful subprocess.call for this test;
    the real CMake invocation is exercised elsewhere."""
    monkeypatch.setattr(
        "helia_core_tester.fvp.cmake.subprocess.call",
        lambda *args, **kwargs: 0,
    )


def test_cmake_configure_clears_stale_gcda_when_coverage_enabled(tmp_path: Path, monkeypatch) -> None:
    _stub_configure_subprocess(monkeypatch)

    build_dir = tmp_path / "build"
    nested = build_dir / "CMakeFiles" / "cmsis-nn.dir" / "Source" / "BasicMathFunctions"
    nested.mkdir(parents=True, exist_ok=True)
    stale_gcda = nested / "arm_elementwise_add_f16.c.gcda"
    stale_gcda.write_bytes(b"stale-coverage-data")

    cmake_configure(
        source_dir=tmp_path / "src",
        build_dir=build_dir,
        toolchain_file=tmp_path / "toolchain.cmake",
        cpu="cortex-m55",
        cmsis5=tmp_path / "CMSIS_5",
        optimization="-Ofast",
        extra_defs=[],
        generator=None,
        generated_tests_dir=None,
        enable_coverage=True,
        verbosity=0,
        env={},
    )

    assert not stale_gcda.exists()


def test_cmake_configure_keeps_gcda_when_coverage_disabled(tmp_path: Path, monkeypatch) -> None:
    _stub_configure_subprocess(monkeypatch)

    build_dir = tmp_path / "build"
    nested = build_dir / "CMakeFiles" / "cmsis-nn.dir" / "Source" / "BasicMathFunctions"
    nested.mkdir(parents=True, exist_ok=True)
    existing_gcda = nested / "arm_elementwise_add_f16.c.gcda"
    existing_gcda.write_bytes(b"unrelated-data")

    cmake_configure(
        source_dir=tmp_path / "src",
        build_dir=build_dir,
        toolchain_file=tmp_path / "toolchain.cmake",
        cpu="cortex-m55",
        cmsis5=tmp_path / "CMSIS_5",
        optimization="-Ofast",
        extra_defs=[],
        generator=None,
        generated_tests_dir=None,
        enable_coverage=False,
        verbosity=0,
        env={},
    )

    assert existing_gcda.exists()
