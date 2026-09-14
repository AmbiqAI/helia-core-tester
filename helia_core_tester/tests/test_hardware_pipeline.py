"""Host-only pieces of `hardware run`: --precision rules, the flash decision
(ELF-hash stamp plus the board's own build id), and the --json summary shape
(driven through the fake target)."""

from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path

import pytest

from helia_core_tester.perf_stream import firmware_build
from helia_core_tester.perf_stream.boards import resolve_board
from helia_core_tester.perf_stream.case_bundle import build_abs_s8_case_bundle, load_case_bundle
from helia_core_tester.perf_stream.fake_target import FakeTargetTransport
from helia_core_tester.perf_stream.firmware_build import (
    FlashDecision,
    build_id_path,
    confirm_board_build_id,
    decide_flash,
    elf_path,
    flash_stamp_path,
    read_build_id,
    record_flash,
)
from helia_core_tester.perf_stream.hardware_pipeline import (
    StreamOptions,
    apply_precision,
    float_precision_for,
    generate_tests_for_board,
    parse_pmu_groups,
    run_hardware_pipeline,
    validate_fvp_gate,
)
from helia_core_tester.perf_stream.run_summary import build_json_summary
from helia_core_tester.perf_stream.session import HostSession, read_hello

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BOARD = resolve_board("apollo510_evb")
SERIAL = 1160002276


def _load_build_id_script():
    path = PROJECT_ROOT / "scripts" / "patch_build_id.py"
    spec = importlib.util.spec_from_file_location("patch_build_id", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- --precision -------------------------------------------------------------------


def test_precision_forces_float_suite_and_suffix_filter() -> None:
    assert apply_precision("fp16", "int", None) == ("float", "_f16")
    assert apply_precision("FP32", "float", None) == ("float", "_f32")
    assert apply_precision(None, "both", "conv") == ("both", "conv")


def test_precision_rejects_suite_both() -> None:
    with pytest.raises(ValueError, match=r"--precision cannot be combined with --suite both \(it selects float cases only\)\."):
        apply_precision("fp16", "both", None)


def test_precision_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match=r"--precision must be 'fp16' or 'fp32' \(got 'fp64'\)\."):
        apply_precision("fp64", "int", None)


def test_precision_rejects_test_name() -> None:
    with pytest.raises(ValueError, match=r"--precision and --test-name cannot be combined \(both filter via a single substring match\)\."):
        apply_precision("fp32", "float", "reshape")


def test_precision_maps_to_the_generate_steps_float_precision() -> None:
    assert float_precision_for(None) is None
    assert float_precision_for("fp16") == "f16"
    assert float_precision_for("FP32") == "f32"


def test_precision_is_an_explicit_override_for_generation(monkeypatch) -> None:
    """`--precision fp16` must generate _f16 artifacts even when the environment or
    TOML pins float_precision to f32; without --precision the env value stands."""
    import helia_core_tester.core.steps as steps

    captured: list = []

    class _FakeGenerateStep:
        def __init__(self, config) -> None:
            captured.append(config)

        def execute(self):
            class _Result:
                success = True
                skipped = False
                message = ""

            return _Result()

    monkeypatch.setattr(steps, "GenerateStep", _FakeGenerateStep)
    monkeypatch.setenv("HELIA_CORE_TESTER_FLOAT_PRECISION", "f32")
    monkeypatch.delenv("HELIA_CORE_TESTER_CONFIG", raising=False)

    generate_tests_for_board(PROJECT_ROOT, BOARD, "float", float_precision="f16")
    assert captured[-1].float_precision == "f16" and captured[-1].suite == "float" and captured[-1].cpu == "cortex-m55"

    generate_tests_for_board(PROJECT_ROOT, BOARD, "float")
    assert captured[-1].float_precision == "f32"


def test_fvp_gate_and_pmu_groups_parsing() -> None:
    validate_fvp_gate(None)
    validate_fvp_gate("advisory")
    with pytest.raises(ValueError, match="--fvp-gate must be one of"):
        validate_fvp_gate("maybe")
    assert parse_pmu_groups("cpu, memory,,mve ") == ("cpu", "memory", "mve")


# --- flash-only-if-changed ---------------------------------------------------------


def _write_elf(build_dir: Path, payload: bytes, build_id: str | None = None) -> Path:
    elf = elf_path(build_dir)
    elf.parent.mkdir(parents=True, exist_ok=True)
    elf.write_bytes(payload)
    if build_id is not None:
        build_id_path(build_dir).write_text(build_id + "\n")
    return elf


def _silent_board(*_args):
    raise AssertionError("the board must not be asked in this test")


def _board_running(build_id: str):
    asked: list[tuple[int, Path]] = []

    def _reader(board, serial, build_dir):
        asked.append((serial, build_dir))
        return build_id

    _reader.asked = asked  # type: ignore[attr-defined]
    return _reader


def test_flash_decision_follows_elf_hash(tmp_path: Path) -> None:
    build_dir = tmp_path / "build" / "perf_stream" / "apollo510_evb"
    _write_elf(build_dir, b"firmware-v1", "hct-v1")
    serial = 1160002276

    first = decide_flash(build_dir, serial)
    assert first.needed and "no flash stamp" in first.reason
    assert first.build_id == "hct-v1"

    stamp = record_flash(build_dir, serial, first.digest)
    assert stamp == flash_stamp_path(build_dir, serial) == build_dir / f".flashed-{serial}.sha256"
    assert stamp.read_text().strip() == first.digest

    unchanged = decide_flash(build_dir, serial)
    assert not unchanged.needed and "unchanged" in unchanged.reason
    # The skip message names the stamp file it trusted.
    assert str(stamp) in unchanged.reason

    # The stamp is keyed by serial: another probe still needs a flash.
    assert decide_flash(build_dir, 1160001958).needed

    assert decide_flash(build_dir, serial, force=True).needed

    _write_elf(build_dir, b"firmware-v2")
    changed = decide_flash(build_dir, serial)
    assert changed.needed and "changed" in changed.reason and changed.digest != first.digest


def test_flash_decision_requires_built_elf(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        decide_flash(tmp_path, 1)


@pytest.fixture
def fake_toolchain(monkeypatch):
    """Stub configure/build so flash_firmware runs without CMake; returns the list of built targets."""
    built: list[str] = []
    monkeypatch.setattr(firmware_build, "configure", lambda *a, **k: None)
    monkeypatch.setattr(firmware_build, "build", lambda build_dir, target, jobs: built.append(target))
    return built


def test_flash_firmware_skips_flash_target_when_unchanged_and_board_confirms(tmp_path: Path, fake_toolchain) -> None:
    build_dir = tmp_path / "bd"
    _write_elf(build_dir, b"firmware", "hct-abc")
    board = _board_running("hct-abc")

    first = firmware_build.flash_firmware(BOARD, 7, build_dir=build_dir, board_build_id_reader=_silent_board)
    assert first.needed
    assert fake_toolchain == [firmware_build.SERVER_TARGET, firmware_build.FLASH_TARGET]

    fake_toolchain.clear()
    second = firmware_build.flash_firmware(BOARD, 7, build_dir=build_dir, board_build_id_reader=board)
    assert not second.needed
    assert fake_toolchain == [firmware_build.SERVER_TARGET]
    assert board.asked == [(7, build_dir)]
    assert "board confirmed build id hct-abc" in second.reason and str(flash_stamp_path(build_dir, 7)) in second.reason
    assert second.build_id == second.board_build_id == "hct-abc"

    fake_toolchain.clear()
    forced = firmware_build.flash_firmware(BOARD, 7, build_dir=build_dir, force=True, board_build_id_reader=_silent_board)
    assert forced.needed and "--force" in forced.reason
    assert fake_toolchain == [firmware_build.SERVER_TARGET, firmware_build.FLASH_TARGET]


def test_flash_skip_is_refused_when_another_build_dir_flashed_the_probe(tmp_path: Path, fake_toolchain) -> None:
    """The reviewer's scenario: build dir A stamps the probe, build dir B flashes the same
    probe with a byte-identical ELF but its own build id, then A decides again. The host
    stamp still says "unchanged"; the board says otherwise, so A must flash."""
    build_a = tmp_path / "a"
    build_b = tmp_path / "b"
    _write_elf(build_a, b"same-firmware", "hct-aaa")
    _write_elf(build_b, b"same-firmware", "hct-bbb")

    firmware_build.flash_firmware(BOARD, SERIAL, build_dir=build_a, board_build_id_reader=_silent_board)
    firmware_build.flash_firmware(BOARD, SERIAL, build_dir=build_b, board_build_id_reader=_silent_board)
    assert flash_stamp_path(build_a, SERIAL).read_text() == flash_stamp_path(build_b, SERIAL).read_text()
    assert not decide_flash(build_a, SERIAL).needed  # the stamp alone would skip

    fake_toolchain.clear()
    board_runs_b = _board_running("hct-bbb")
    decision = firmware_build.flash_firmware(BOARD, SERIAL, build_dir=build_a, board_build_id_reader=board_runs_b)
    assert decision.needed
    assert fake_toolchain == [firmware_build.SERVER_TARGET, firmware_build.FLASH_TARGET]
    assert "board reports build id hct-bbb, expected hct-aaa" in decision.reason
    assert decision.build_id == "hct-aaa" and decision.board_build_id == "hct-bbb"

    # And the reverse: the board really does run A's build, so A skips.
    fake_toolchain.clear()
    again = firmware_build.flash_firmware(BOARD, SERIAL, build_dir=build_a, board_build_id_reader=_board_running("hct-aaa"))
    assert not again.needed and fake_toolchain == [firmware_build.SERVER_TARGET]


def test_confirm_board_build_id_flashes_when_board_is_silent_or_unstamped(tmp_path: Path) -> None:
    build_dir = tmp_path / "bd"
    _write_elf(build_dir, b"fw", "hct-abc")
    unchanged = FlashDecision(False, "digest", "unchanged", "hct-abc")

    def _no_hello(board, serial, build_dir):
        raise RuntimeError("Transport stalled before a complete HELLO frame arrived.")

    silent = confirm_board_build_id(BOARD, SERIAL, build_dir, unchanged, reader=_no_hello)
    assert silent.needed and "did not confirm build id hct-abc" in silent.reason and "RuntimeError" in silent.reason

    unstamped = confirm_board_build_id(BOARD, SERIAL, build_dir, FlashDecision(False, "digest", "unchanged", None), reader=_silent_board)
    assert unstamped.needed and "hct_build_id.txt is missing" in unstamped.reason

    # A decision that already says "flash" is passed through untouched.
    forced = FlashDecision(True, "digest", "--force given", "hct-abc")
    assert confirm_board_build_id(BOARD, SERIAL, build_dir, forced, reader=_silent_board) is forced


def test_read_build_id_handles_missing_and_blank_files(tmp_path: Path) -> None:
    assert read_build_id(tmp_path) is None
    build_id_path(tmp_path).write_text("  \n")
    assert read_build_id(tmp_path) is None
    build_id_path(tmp_path).write_text("hct-0123\n")
    assert read_build_id(tmp_path) == "hct-0123"


# --- post-link build id (scripts/patch_build_id.py) ------------------------------------


def _synthetic_elf(segments: list[tuple[int, bytes]]) -> bytes:
    """A minimal ELF32 little-endian file with one PT_LOAD segment per (lma, payload)."""
    ehsize, phentsize = 52, 32
    data_offset = ehsize + phentsize * len(segments)
    phdrs, body = bytearray(), bytearray()
    for lma, payload in segments:
        # p_type=PT_LOAD, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz, p_flags, p_align
        phdrs += struct.pack("<IIIIIIII", 1, data_offset + len(body), lma, lma, len(payload), len(payload), 5, 4)
        body += payload
    ident = b"\x7fELF" + bytes([1, 1, 1, 0]) + bytes(8)
    header = ident + struct.pack("<HHIIIIIHHHHHH", 2, 40, 1, segments[0][0], ehsize, 0, 0, ehsize, phentsize, len(segments), 0, 0, 0)
    return bytes(header + phdrs + body)


def _synthetic_firmware(build_dir: Path, *, server: bytes, library: bytes, gap: int = 16) -> tuple[Path, Path]:
    """Write an ELF + .bin pair whose flash image is `server` (holding the build-id slot),
    a zero gap, then `library` in a second PT_LOAD segment -- standing in for the NSX
    libraries and linker layout the old object hash never saw."""
    script = _load_build_id_script()
    assert script.MARKER in server
    base = 0x00410000
    elf, binary = elf_path(build_dir), build_dir / "perf_stream" / "hct_benchmark_server.bin"
    elf.parent.mkdir(parents=True, exist_ok=True)
    elf.write_bytes(_synthetic_elf([(base, server), (base + len(server) + gap, library)]))
    binary.write_bytes(server + bytes(gap) + library)  # what objcopy -O binary emits
    return elf, binary


def _stamp(build_dir: Path) -> str:
    script = _load_build_id_script()
    elf, binary = elf_path(build_dir), build_dir / "perf_stream" / "hct_benchmark_server.bin"
    assert script.main(["--elf", str(elf), "--bin", str(binary), "--output-txt", str(build_id_path(build_dir))]) == 0
    return read_build_id(build_dir)


def _embedded_id(path: Path) -> str:
    script = _load_build_id_script()
    data = path.read_bytes()
    assert data.count(script.MARKER) == 1
    start = data.index(script.MARKER) + len(script.MARKER)
    return data[start:data.index(b"\0", start)].decode("ascii")


def test_post_link_build_id_covers_the_whole_image(tmp_path: Path) -> None:
    script = _load_build_id_script()
    slot = script.MARKER + bytes(script.ID_AREA)
    server = b"server-code-" + slot + b"-more-server-code"
    library = b"nsx-board+core+perf+startup+cmsis-nn+linker-layout"

    # Same image from two build dirs -> the same id, and the host-side txt equals the
    # string embedded in both the ELF and the .bin.
    build_a, build_b = tmp_path / "a", tmp_path / "b"
    _synthetic_firmware(build_a, server=server, library=library)
    _synthetic_firmware(build_b, server=server, library=library)
    id_a, id_b = _stamp(build_a), _stamp(build_b)
    assert id_a == id_b
    assert id_a.startswith("hct-") and len(id_a) == 4 + script.BUILD_ID_HEX_CHARS
    assert _embedded_id(elf_path(build_a)) == _embedded_id(build_a / "perf_stream" / "hct_benchmark_server.bin") == id_a
    assert elf_path(build_a).read_bytes() == elf_path(build_b).read_bytes()

    # A byte that only a linked library changes -> a different id (the old object hash missed this).
    build_c = tmp_path / "c"
    _synthetic_firmware(build_c, server=server, library=library[:-1] + b"X")
    id_c = _stamp(build_c)
    assert id_c != id_a
    # ...and so does a byte in the server's own code.
    build_d = tmp_path / "d"
    _synthetic_firmware(build_d, server=server.replace(b"more", b"MORE"), library=library)
    assert _stamp(build_d) not in {id_a, id_c}

    # Re-running on an already patched image is a no-op with the same id.
    before = elf_path(build_a).read_bytes()
    assert _stamp(build_a) == id_a and elf_path(build_a).read_bytes() == before


def test_post_link_build_id_rejects_unpatchable_images(tmp_path: Path, capsys) -> None:
    script = _load_build_id_script()
    slot = script.MARKER + bytes(script.ID_AREA)

    def _run(build_dir: Path) -> int:
        elf, binary = elf_path(build_dir), build_dir / "perf_stream" / "hct_benchmark_server.bin"
        return script.main(["--elf", str(elf), "--bin", str(binary), "--output-txt", str(build_id_path(build_dir))])

    no_marker = tmp_path / "no_marker"
    _synthetic_firmware(no_marker, server=b"code" + slot, library=b"lib")
    elf_path(no_marker).write_bytes(_synthetic_elf([(0x410000, b"code-without-a-slot")]))
    assert _run(no_marker) == 1 and "marker" in capsys.readouterr().err

    twice = tmp_path / "twice"
    _synthetic_firmware(twice, server=b"code" + slot + slot, library=b"lib")
    assert _run(twice) == 1 and "more than once" in capsys.readouterr().err

    # The .bin must be the image assembled from the ELF, byte for byte.
    stale_bin = tmp_path / "stale_bin"
    _synthetic_firmware(stale_bin, server=b"code" + slot, library=b"lib")
    (stale_bin / "perf_stream" / "hct_benchmark_server.bin").write_bytes(b"code" + slot + bytes(16) + b"lib-old")
    assert _run(stale_bin) == 1 and "does not match" in capsys.readouterr().err
    assert not build_id_path(stale_bin).exists()

    assert script.main(["--elf", str(tmp_path / "missing.elf"), "--output-txt", str(tmp_path / "x.txt")]) == 1


# --- HELLO build id verification ------------------------------------------------------


def test_session_verifies_hello_build_id(tmp_path: Path) -> None:
    bundle = load_case_bundle(build_abs_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path, case_id="abs_id").manifest_path)

    result = HostSession(FakeTargetTransport(build_id="hct-aaa")).run_many([bundle], expected_build_id="hct-aaa")
    assert result.build_id == "hct-aaa" and result.cases[0].comparison.passed

    # Legacy callers that pass nothing still get the id reported, never a failure.
    assert HostSession(FakeTargetTransport(build_id="hct-aaa")).run_many([bundle]).build_id == "hct-aaa"

    with pytest.raises(RuntimeError, match=r"build id mismatch.*reports 'hct-bbb'.*expects 'hct-aaa'.*--force-flash"):
        HostSession(FakeTargetTransport(build_id="hct-bbb")).run_many([bundle], expected_build_id="hct-aaa")


def test_read_hello_returns_the_full_payload_without_acknowledging() -> None:
    transport = FakeTargetTransport(build_id="hct-xyz")
    hello = read_hello(transport)
    assert hello.build_id == "hct-xyz" and hello.board_id == "fake_board" and hello.target_cpu == "fake-cpu"
    assert hello.max_frame_payload == 64 and hello.runtime_arena_capacity == 4096
    assert transport.read() == b""  # nothing else was sent: the fake is still waiting for HELLO_ACK


# --- configure flags ---------------------------------------------------------------


@pytest.fixture
def captured_cmake(monkeypatch, tmp_path: Path):
    """Run `configure()` without CMake or the dependency fetch; returns the argv it would run."""
    calls: list[list[str]] = []
    monkeypatch.setattr(firmware_build, "ensure_hardware_dependencies", lambda repo_root: None)
    monkeypatch.setattr(firmware_build, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(firmware_build.subprocess, "run", lambda cmd, **kwargs: calls.append(list(cmd)))
    return calls


def test_configure_passes_the_board_row_to_cmake(captured_cmake, monkeypatch, tmp_path: Path) -> None:
    from dataclasses import replace

    monkeypatch.setattr(firmware_build, "find_jlink_exe", lambda: None)
    firmware_build.configure(tmp_path / "bd", BOARD, force=False)
    [cmd] = captured_cmake
    assert cmd[:4] == ["cmake", "-S", str(tmp_path), "-B", str(tmp_path / "bd")][:4]
    assert "-DHELIA_HARDWARE_BOARD=apollo510_evb" in cmd and "-DTARGET_CPU=cortex-m55" in cmd
    # apollo510_evb's workspace must stay at the historical 114688 so the memory report is unchanged.
    assert BOARD.workspace_bytes == 114688 and "-DHCT_SERVER_WORKSPACE_BYTES=114688" in cmd

    captured_cmake.clear()
    other = replace(BOARD, id="other_evb", workspace_bytes=65536)
    firmware_build.configure(tmp_path / "bd2", other, force=False)
    assert "-DHCT_SERVER_WORKSPACE_BYTES=65536" in captured_cmake[0]


def test_configure_forwards_the_resolved_jlinkexe_to_the_flash_target(captured_cmake, monkeypatch, tmp_path: Path) -> None:
    from helia_core_tester.perf_stream.jlink_library import JLinkExecutable, JLinkLibraryError

    monkeypatch.setattr(firmware_build, "find_jlink_exe", lambda: JLinkExecutable("/opt/SEGGER/JLink/JLinkExe", "$JLINK_PATH"))
    firmware_build.configure(tmp_path / "bd", BOARD, force=False, serial_no=SERIAL)
    [cmd] = captured_cmake
    assert "-DNSX_JLINK_EXE=/opt/SEGGER/JLink/JLinkExe" in cmd
    assert f"-DNSX_JLINK_SERIAL={SERIAL}" in cmd

    # Nothing resolved: leave CMake's own find_program(JLinkExe) alone.
    captured_cmake.clear()
    monkeypatch.setattr(firmware_build, "find_jlink_exe", lambda: None)
    firmware_build.configure(tmp_path / "bd", BOARD, force=False)
    assert not any(arg.startswith("-DNSX_JLINK_EXE") for arg in captured_cmake[0])

    # A broken $HPX_JLINK_DLL is doctor's problem, not a reason to refuse `hardware build`.
    captured_cmake.clear()

    def _broken():
        raise JLinkLibraryError("$HPX_JLINK_DLL=/x/gone.so does not exist")

    monkeypatch.setattr(firmware_build, "find_jlink_exe", _broken)
    firmware_build.configure(tmp_path / "bd", BOARD, force=False)
    assert len(captured_cmake) == 1 and not any(arg.startswith("-DNSX_JLINK_EXE") for arg in captured_cmake[0])


# --- --json summary ----------------------------------------------------------------


class _SkippedTest:
    def __init__(self, name: str) -> None:
        self.name = name


def test_json_summary_shape_from_fake_target_session(tmp_path: Path) -> None:
    bundle = load_case_bundle(build_abs_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path, case_id="abs_json").manifest_path)
    result = HostSession(FakeTargetTransport(), requested_counter_groups=("cpu",)).run_many([bundle])
    skipped = [(_SkippedTest("conv_x"), "conv_x: operator='Foo' is not bridgeable (bridged today: ['Abs']).")]

    summary = build_json_summary(
        result, skipped, session_id="apollo510_evb-20260912T000000Z", board_id="apollo510_evb",
        bundle=tmp_path / "artifacts" / "reports" / "performance_stream" / "apollo510_evb-20260912T000000Z",
    )
    encoded = json.loads(json.dumps(summary))  # must be JSON-serialisable as-is

    assert set(encoded) == {"session_id", "board", "bundle", "totals", "cases"}
    assert encoded["session_id"] == "apollo510_evb-20260912T000000Z"
    assert encoded["board"] == "apollo510_evb"
    assert encoded["bundle"].endswith("apollo510_evb-20260912T000000Z")
    assert encoded["totals"] == {"ran": 1, "passed": 1, "failed": 0, "skipped": 1}
    ran, skip = encoded["cases"]
    assert set(ran) == {"case_id", "passed", "median_cycles", "skipped_reason"}
    assert ran == {"case_id": "abs_json", "passed": True, "median_cycles": ran["median_cycles"], "skipped_reason": None}
    assert isinstance(ran["median_cycles"], float)
    assert skip["case_id"] == "conv_x" and skip["passed"] is None and skip["median_cycles"] is None
    assert skip["skipped_reason"].startswith("operator='Foo' is not bridgeable")
    assert "bridged today" not in skip["skipped_reason"]


# --- orchestration order ----------------------------------------------------------


def test_run_hardware_pipeline_generates_flashes_then_streams(tmp_path: Path, monkeypatch) -> None:
    from helia_core_tester.perf_stream import hardware_pipeline

    board = resolve_board("apollo510_evb")
    order: list[str] = []

    def _generate(repo_root, spec, suite, float_precision=None):
        order.append(f"generate:{spec.cpu}:{suite}:{float_precision}")

    def _flash(spec, serial, *, build_dir, jobs, force_reconfigure, force):
        order.append(f"flash:{serial}:{build_dir.relative_to(tmp_path)}:force={force}")
        return firmware_build.FlashDecision(True, "abc", "test")

    def _stream(repo_root, spec, serial, *, build_dir, options, echo, progress_to_stderr):
        order.append(f"stream:{options.suite}:{options.test_name}")
        return hardware_pipeline.HardwareRunOutcome(session_id="s", result=None, bundle=tmp_path, skipped=[])

    monkeypatch.setattr(hardware_pipeline, "generate_tests_for_board", _generate)
    monkeypatch.setattr(hardware_pipeline, "flash_firmware", _flash)
    monkeypatch.setattr(hardware_pipeline, "stream_generated_tests", _stream)

    outcome = run_hardware_pipeline(
        tmp_path, board, 42, options=StreamOptions(suite="float", test_name="_f16", float_precision="f16"), echo=lambda _msg: None,
    )
    assert order == ["generate:cortex-m55:float:f16", "flash:42:build/perf_stream/apollo510_evb:force=False", "stream:float:_f16"]
    assert outcome.flash is not None and outcome.flash.needed

    order.clear()
    run_hardware_pipeline(
        tmp_path, board, 42, options=StreamOptions(), skip_generate=True, skip_flash=True, echo=lambda _msg: None,
    )
    assert order == ["stream:int:None"]

    order.clear()
    run_hardware_pipeline(
        tmp_path, board, 42, options=StreamOptions(), skip_generate=True, force_flash=True, echo=lambda _msg: None,
    )
    assert order == ["flash:42:build/perf_stream/apollo510_evb:force=True", "stream:int:None"]

    with pytest.raises(ValueError, match="--skip-flash and --force-flash"):
        run_hardware_pipeline(
            tmp_path, board, 42, options=StreamOptions(), skip_generate=True, skip_flash=True, force_flash=True, echo=lambda _msg: None,
        )


def test_stream_passes_build_dir_build_id_to_the_session(tmp_path: Path, monkeypatch) -> None:
    from helia_core_tester.perf_stream import hardware_pipeline

    seen: dict = {}

    def _session(repo_root, **kwargs):
        seen.update(kwargs)
        return object(), tmp_path / "bundle", []

    monkeypatch.setattr(hardware_pipeline, "make_live_progress_printer", lambda *a, **k: None)
    monkeypatch.setattr("helia_core_tester.perf_stream.hardware_run.build_generated_test_case_bundles", lambda *a, **k: ([], []))
    monkeypatch.setattr("helia_core_tester.perf_stream.hardware_run.run_apollo510_generated_test_session", _session)

    build_dir = tmp_path / "bd"
    _write_elf(build_dir, b"fw", "hct-stream")
    echoed: list[str] = []
    hardware_pipeline.stream_generated_tests(tmp_path, BOARD, 5, build_dir=build_dir, options=StreamOptions(), echo=echoed.append)
    assert seen["expected_build_id"] == "hct-stream"
    assert any("firmware build id hct-stream" in line for line in echoed)

    unstamped = tmp_path / "old"
    _write_elf(unstamped, b"fw")
    echoed.clear()
    hardware_pipeline.stream_generated_tests(tmp_path, BOARD, 5, build_dir=unstamped, options=StreamOptions(), echo=echoed.append)
    assert seen["expected_build_id"] is None
    assert any("WARNING" in line and "hct_build_id.txt" in line for line in echoed)


# --- --json keeps stdout clean ----------------------------------------------------


def test_stdout_to_stderr_covers_python_and_subprocess_output(capfd) -> None:
    import subprocess
    import sys

    from helia_core_tester.perf_stream.run_summary import stdout_to_stderr

    with stdout_to_stderr():
        print("python-line")
        subprocess.run([sys.executable, "-c", "print('child-line')"], check=True)
    print("after-line")
    out, err = capfd.readouterr()
    assert "python-line" in err and "child-line" in err
    assert "python-line" not in out and "child-line" not in out
    assert "after-line" in out
