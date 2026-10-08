"""Host-only pieces of `hardware run`: --precision rules, the flash decision
(ELF-hash stamp plus the board's own build id), and the --json summary shape
(driven through the fake target)."""

from __future__ import annotations

import importlib.util
import csv
import dataclasses
import json
import struct
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from helia_core_tester.hardware.errors import RunRefused
from helia_core_tester.hardware import firmware_build, nsx_cli
from helia_core_tester.hardware.boards import resolve_board
from helia_core_tester.hardware.case_bundle import build_abs_s8_case_bundle, load_case_bundle
from helia_core_tester.hardware.fake_target import FakeTargetTransport
from helia_core_tester.hardware.firmware_build import (
    FlashDecision,
    build_id_path,
    confirm_board_build_id,
    decide_flash,
    elf_path,
    flash_stamp_path,
    read_build_id,
    record_flash,
)
from helia_core_tester.hardware.hardware_pipeline import (
    StreamOptions,
    apply_precision,
    fit_to_board,
    float_precision_for,
    generate_tests_for_board,
    parse_pmu_counters,
    parse_pmu_groups,
    resolve_pmu_options,
    resolved_selection,
    run_hardware_pipeline,
    validate_fvp_gate,
)
from helia_core_tester.hardware.result_bundle import write_result_bundle
from helia_core_tester.hardware.run_summary import build_json_summary, print_run_report
from helia_core_tester.hardware.session import BootFailure, HostSession, read_target_info

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


def test_generation_exports_the_kernel_root(tmp_path: Path, monkeypatch) -> None:
    import helia_core_tester.core.steps.generate as generate

    envs: list = []
    monkeypatch.setattr(generate, "run_command", lambda cmd, cwd, verbosity, env: envs.append(env))
    monkeypatch.delenv("HELIA_CORE_TESTER_CONFIG", raising=False)
    generate_tests_for_board(PROJECT_ROOT, BOARD, "int", cmsis_nn_root=tmp_path)
    assert envs and all(env["CMSIS_NN_ROOT"] == str(tmp_path.resolve()) for env in envs)


def test_fvp_gate_and_pmu_groups_parsing() -> None:
    validate_fvp_gate(None)
    validate_fvp_gate("advisory")
    with pytest.raises(ValueError, match="--fvp-gate must be one of"):
        validate_fvp_gate("maybe")
    assert parse_pmu_groups("cpu, memory,,mve ") == ("cpu", "memory", "mve")


def test_pmu_counters_parsing_and_deprecated_groups_alias() -> None:
    assert parse_pmu_counters(["mve:all", "cpu:default"]) == {"mve": "all", "cpu": "default"}
    assert parse_pmu_counters(["mve:ARM_PMU_MVE_STALL, ARM_PMU_MVE_PRED"]) == {"mve": ["ARM_PMU_MVE_STALL", "ARM_PMU_MVE_PRED"]}
    # Every group at "all" plans 5 + 4 + 9 = 18 passes, within HCT_SERVER_MAX_PASSES;
    # a bare `all` is the same selection.
    every = {"cpu": "all", "memory": "all", "mve": "all"}
    assert parse_pmu_counters(["cpu:all", "memory:all", "mve:all"]) == every
    assert parse_pmu_counters(["all"]) == parse_pmu_counters([" ALL "]) == every
    assert resolve_pmu_options(["all"], None) == every
    for mixed in (["all", "mve:default"], ["cpu:default", "all"]):
        with pytest.raises(ValueError, match="bare 'all' already selects every group"):
            parse_pmu_counters(mixed)
    # An empty or blank name list is rejected rather than silently timing cycles only.
    for empty in (["mve:,"], ["mve: , "], ["mve:ARM_PMU_MVE_STALL,"], ["mve:ARM_PMU_MVE_STALL,,ARM_PMU_MVE_PRED"]):
        with pytest.raises(ValueError, match=r"--pmu-counters: .* names an empty counter for group 'mve'"):
            parse_pmu_counters(empty)
    with pytest.raises(ValueError, match="expects GROUP:SELECTION"):
        parse_pmu_counters(["mve:"])
    for bad, message in (
        (["mve"], "expects GROUP:SELECTION"),
        (["dsp:all"], "unknown group 'dsp'"),
        (["mve:ARM_PMU_NOPE"], "Unsupported counter 'ARM_PMU_NOPE' for group 'mve'. Valid names: ARM_PMU_MVE_INST_RETIRED"),
        (["cpu:all", "cpu:default"], "given more than once"),
    ):
        with pytest.raises(ValueError, match=message):
            parse_pmu_counters(bad)

    # Neither flag: every group at its default. The deprecated --pmu-groups maps each
    # listed group to GROUP:default and warns.
    assert resolve_pmu_options([], None) == {"cpu": "default", "memory": "default", "mve": "default"}
    warnings: list[str] = []
    assert resolve_pmu_options([], "mve,cpu", warn=warnings.append) == {"mve": "default", "cpu": "default"}
    assert warnings and "deprecated" in warnings[0] and "--pmu-counters mve:default" in warnings[0]
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_pmu_options(["mve:all"], "cpu")
    assert StreamOptions().pmu_counters == {"cpu": "default", "memory": "default", "mve": "default"}


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
    build_dir = tmp_path / "build" / "hardware" / "apollo510_evb"
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
    """Stub the NSX build and flash target; returns the list of built targets."""
    built: list[str] = []
    monkeypatch.setattr(firmware_build, "build_firmware", lambda *a, **k: built.append(firmware_build.SERVER_TARGET))
    monkeypatch.setattr(nsx_cli, "flash_app", lambda app_dir, **kwargs: built.append("flash"))
    return built


def test_flash_firmware_skips_flash_target_when_unchanged_and_board_confirms(tmp_path: Path, fake_toolchain) -> None:
    build_dir = tmp_path / "bd"
    _write_elf(build_dir, b"firmware", "hct-abc")
    board = _board_running("hct-abc")

    first = firmware_build.flash_firmware(BOARD, 7, build_dir=build_dir, board_build_id_reader=_silent_board)
    assert first.needed
    assert fake_toolchain == [firmware_build.SERVER_TARGET, "flash"]

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
    assert fake_toolchain == [firmware_build.SERVER_TARGET, "flash"]


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
    assert fake_toolchain == [firmware_build.SERVER_TARGET, "flash"]
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

    def _no_target_info(board, serial, build_dir):
        raise RuntimeError("Transport stalled before a complete TARGET_INFO frame arrived.")

    silent = confirm_board_build_id(BOARD, SERIAL, build_dir, unchanged, reader=_no_target_info)
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
    elf, binary = elf_path(build_dir), build_dir / "hardware" / "hct_benchmark_server.bin"
    elf.parent.mkdir(parents=True, exist_ok=True)
    elf.write_bytes(_synthetic_elf([(base, server), (base + len(server) + gap, library)]))
    binary.write_bytes(server + bytes(gap) + library)  # what objcopy -O binary emits
    return elf, binary


def _stamp(build_dir: Path) -> str:
    script = _load_build_id_script()
    elf, binary = elf_path(build_dir), build_dir / "hardware" / "hct_benchmark_server.bin"
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
    assert _embedded_id(elf_path(build_a)) == _embedded_id(build_a / "hardware" / "hct_benchmark_server.bin") == id_a
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
        elf, binary = elf_path(build_dir), build_dir / "hardware" / "hct_benchmark_server.bin"
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
    (stale_bin / "hardware" / "hct_benchmark_server.bin").write_bytes(b"code" + slot + bytes(16) + b"lib-old")
    assert _run(stale_bin) == 1 and "does not match" in capsys.readouterr().err
    assert not build_id_path(stale_bin).exists()

    assert script.main(["--elf", str(tmp_path / "missing.elf"), "--output-txt", str(tmp_path / "x.txt")]) == 1


# --- TARGET_INFO build id verification ------------------------------------------------------


def test_session_verifies_target_info_build_id(tmp_path: Path) -> None:
    bundle = load_case_bundle(build_abs_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path, case_id="abs_id").manifest_path)

    result = HostSession(FakeTargetTransport(build_id="hct-aaa")).run_many([bundle], expected_build_id="hct-aaa")
    assert result.build_id == "hct-aaa" and result.cases[0].comparison.passed

    # Legacy callers that pass nothing still get the id reported, never a failure.
    assert HostSession(FakeTargetTransport(build_id="hct-aaa")).run_many([bundle]).build_id == "hct-aaa"

    with pytest.raises(RuntimeError, match=r"build id mismatch.*reports 'hct-bbb'.*expects 'hct-aaa'.*--force-flash"):
        HostSession(FakeTargetTransport(build_id="hct-bbb")).run_many([bundle], expected_build_id="hct-aaa")


def test_session_refuses_failed_boot_before_any_case(tmp_path: Path) -> None:
    bundle = load_case_bundle(build_abs_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path, case_id="abs_boot").manifest_path)
    transport = FakeTargetTransport(boot_status=7, core_clock_hz=96_000_000)
    with pytest.raises(RuntimeError, match=r"^Board init failed: nsx_system_init status 7, core 96 MHz\.$"):
        HostSession(transport).run_many([bundle])
    # No ACK sent, so no catalog.
    assert transport.read() == b"" and transport.completed_case_count == 0


@pytest.mark.parametrize(
    "boot_status, clock_hz, refusal",
    [
        (0, 250_000_000, None),
        (0, 96_000_000, r"^Board core clock 96 MHz, expected 250 MHz\.$"),
        (0, 0, r"^Board core clock unknown, expected 250 MHz\.$"),
        (None, 96_000_000, None),
    ],
    ids=["match", "mismatch", "unknown", "not-reported"],
)
def test_session_checks_core_clock(tmp_path: Path, boot_status, clock_hz, refusal) -> None:
    bundle = load_case_bundle(build_abs_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path, case_id="abs_clock").manifest_path)
    transport = FakeTargetTransport(boot_status=boot_status, core_clock_hz=clock_hz)
    session = HostSession(transport)
    if refusal is None:
        session.handshake(expected_clock_hz=250_000_000)
        assert session.run_many([bundle]).cases[0].comparison.passed
        return
    with pytest.raises(BootFailure, match=refusal):
        session.handshake(expected_clock_hz=250_000_000)
    # Refused before ACK and any case.
    assert transport.read() == b"" and transport.completed_case_count == 0


@pytest.mark.parametrize(
    "boot_status, expected",
    [
        (0, {"status": 0, "core_clock_hz": 250_000_000, "fpscr_boot": 0x03040000, "fpscr": 0x00040000, "fp_mode": {"ahp": 0, "dn": 0, "fz": 0, "rmode": 0, "fz16": 0}}),
        (None, {"status": None, "core_clock_hz": None, "fpscr_boot": None, "fpscr": None, "fp_mode": None}),
    ],
    ids=["healthy", "old-firmware"],
)
def test_boot_health_is_stamped_in_bundle(tmp_path: Path, boot_status, expected, capsys) -> None:
    bundle = load_case_bundle(build_abs_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path, case_id="abs_boot").manifest_path)
    result = HostSession(FakeTargetTransport(boot_status=boot_status)).run_many([bundle])
    assert result.cases[0].comparison.passed
    bundle_root = write_result_bundle(result, session_id="boot", output_root=tmp_path, memory_report={}, kernel_catalog=[])
    manifest = json.loads((bundle_root / "session_manifest.json").read_text())
    assert manifest["boot"] == expected
    # Fake targets set no placement bit.
    assert manifest["target"]["placement"]["name"] == "tcm"
    print_run_report(result, [], bundle_root)
    line = "status 0, core 250 MHz, FPSCR 0x00040000" if boot_status == 0 else "not reported"
    assert f"Target boot: {line}" in capsys.readouterr().out


def test_bundle_names_the_timed_kernel(tmp_path: Path) -> None:
    bundle = load_case_bundle(build_abs_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path, case_id="abs_timed").manifest_path)
    result = HostSession(FakeTargetTransport()).run_many([bundle])
    catalog = json.loads((PROJECT_ROOT / "cmake" / "hardware" / "kernel_catalog.json").read_text())
    bundle_root = write_result_bundle(result, session_id="timed", output_root=tmp_path, memory_report={}, kernel_catalog=catalog)
    row = json.loads((bundle_root / "cases.json").read_text())[0]
    assert row["timed_symbol"] == "arm_abs_s8" and row["inner_symbol"] is None
    with (bundle_root / "case_summary.csv").open(encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
        assert row["timed_symbol"] == "arm_abs_s8" and row["inner_symbol"] == ""


def test_read_target_info_returns_the_full_payload_without_acknowledging() -> None:
    transport = FakeTargetTransport(build_id="hct-xyz")
    target_info = read_target_info(transport)
    assert target_info.build_id == "hct-xyz" and target_info.board_id == "fake_board" and target_info.target_cpu == "cortex-m55"
    assert target_info.max_frame_payload == 64 and target_info.runtime_arena_capacity == 4096
    assert transport.read() == b""  # nothing else was sent: the fake is still waiting for TARGET_INFO_ACK


# --- --json summary ----------------------------------------------------------------


class _SkippedTest:
    def __init__(self, name: str) -> None:
        self.name = name


def test_json_summary_shape_from_fake_target_session(tmp_path: Path) -> None:
    bundle = load_case_bundle(build_abs_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path, case_id="abs_json").manifest_path)
    result = HostSession(FakeTargetTransport()).run_many([bundle])
    skipped = [(_SkippedTest("conv_x"), "conv_x: operator='Foo' is not bridgeable (bridged today: ['Abs']).")]

    timing = {"generate_s": 0.0, "build_s": 2.5, "flash_s": 0.0, "stream_s": 1.25, "total_s": 3.75, "batch_count": 1, "cases": {"abs_json": 1.25}}
    summary = build_json_summary(
        result, skipped, session_id="apollo510_evb-20260912T000000Z", board_id="apollo510_evb",
        bundle=tmp_path / "artifacts" / "reports" / "hardware" / "apollo510_evb-20260912T000000Z",
        selection={"suite": "int"}, timing=timing,
    )
    encoded = json.loads(json.dumps(summary))  # must be JSON-serialisable as-is

    assert set(encoded) == {
        "schema", "schema_version", "generated_at", "session_id", "board", "toolchain", "boot", "bundle", "totals",
        "timing", "selection", "coverage", "github", "cases",
    }
    # No manifest: toolchain unknown.
    assert encoded["toolchain"] is None
    assert encoded["boot"] == {"status": 0, "core_clock_hz": 250_000_000, "fpscr_boot": 0x03040000, "fpscr": 0x00040000, "fp_mode": {"ahp": 0, "dn": 0, "fz": 0, "rmode": 0, "fz16": 0}}
    assert encoded["session_id"] == "apollo510_evb-20260912T000000Z"
    assert encoded["board"] == "apollo510_evb"
    assert encoded["bundle"].endswith("apollo510_evb-20260912T000000Z")
    assert encoded["totals"] == {"ran": 1, "passed": 1, "failed": 0, "skipped": 1}
    assert encoded["timing"] == timing
    ran, skip = encoded["cases"]
    assert set(ran) == {"case_id", "passed", "median_cycles", "valid_for_regression", "timing_status", "max_abs_diff", "diff_count",
                        "skipped_reason"}
    assert ran == {
        "case_id": "abs_json", "passed": True, "median_cycles": ran["median_cycles"], "valid_for_regression": True,
        "timing_status": "valid", "max_abs_diff": 0.0, "diff_count": 0, "skipped_reason": None,
    }
    assert isinstance(ran["median_cycles"], float)
    assert skip["case_id"] == "conv_x" and skip["passed"] is None and skip["median_cycles"] is None
    assert skip["skipped_reason"].startswith("operator='Foo' is not bridgeable")
    assert "bridged today" not in skip["skipped_reason"]


_GITHUB_ENV = {
    "GITHUB_RUN_ID": "123456", "GITHUB_RUN_ATTEMPT": "2", "GITHUB_EVENT_NAME": "schedule",
    "GITHUB_SHA": "a" * 40, "GITHUB_REF": "refs/heads/main", "GITHUB_REPOSITORY": "AmbiqAI/helia-core-tester",
    "GITHUB_SERVER_URL": "https://github.com",
}


def _run_document(tmp_path: Path, options: StreamOptions) -> dict:
    """A --json document from one fake case."""
    bundle = load_case_bundle(build_abs_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path, case_id="abs_doc").manifest_path)
    result = HostSession(FakeTargetTransport()).run_many([bundle])
    selection = resolved_selection(PROJECT_ROOT, resolve_board("apollo510_evb"), options)
    summary = build_json_summary(result, [], session_id="s", board_id="apollo510_evb", bundle=tmp_path, selection=selection)
    return json.loads(json.dumps(summary))


def test_json_summary_identifies_its_schema(tmp_path: Path, monkeypatch) -> None:
    for name in _GITHUB_ENV:
        monkeypatch.delenv(name, raising=False)
    options = StreamOptions(
        suite="float", family="ActivationFunctions", limit=2, float_precision="f32", pmu_counters={"cpu": "all"},
        fvp_gate="strict",
    )
    encoded = _run_document(tmp_path, options)

    assert encoded["schema"] == "hct.hardware.nightly_run"
    assert encoded["schema_version"] == 1
    assert datetime.fromisoformat(encoded["generated_at"]).utcoffset() == timedelta(0)
    assert encoded["selection"] == {
        "suite": "float", "limit": 2, "family": "ActivationFunctions", "test_name": None,
        "ops": [], "dtypes": [], "case_ids": [], "precision": "f32",
        "pmu_counters": {"cpu": "all"}, "fvp_gate": "strict",
        "compare": {"strict": False, "golden_from": None, "golden_session_id": None},
    }
    assert encoded["github"] is None


@pytest.mark.parametrize(
    ("board_id", "suite", "env_precision", "precision"),
    [
        ("apollo510_evb", "float", None, "both"),  # config default
        ("apollo510_evb", "float", "f16", "f16"),  # config from env
        ("apollo510_evb", "both", None, "both"),
        ("apollo3p_evb", "both", None, "f32"),  # M4 has no FP16
        ("apollo3p_evb", "float", None, "f32"),
        ("apollo510_evb", "int", None, None),  # no float cases
    ],
)
def test_selection_records_generation_precision(monkeypatch, board_id, suite, env_precision, precision) -> None:
    monkeypatch.delenv("HELIA_CORE_TESTER_FLOAT_PRECISION", raising=False)
    if env_precision:
        monkeypatch.setenv("HELIA_CORE_TESTER_FLOAT_PRECISION", env_precision)
    board = resolve_board(board_id)
    options = fit_to_board(board, StreamOptions(suite=suite), explicit_pmu=False)

    assert resolved_selection(PROJECT_ROOT, board, options)["precision"] == precision


def test_selection_records_default_fvp_gate() -> None:
    board = resolve_board("apollo510_evb")

    assert resolved_selection(PROJECT_ROOT, board, StreamOptions())["fvp_gate"] == "advisory"


def test_json_summary_records_github_run(tmp_path: Path, monkeypatch) -> None:
    for name, value in _GITHUB_ENV.items():
        monkeypatch.setenv(name, value)

    encoded = _run_document(tmp_path, StreamOptions())

    assert encoded["github"] == {
        "run_id": 123456, "run_attempt": 2, "event_name": "schedule", "sha": "a" * 40, "ref": "refs/heads/main",
        "repository": "AmbiqAI/helia-core-tester",
        "run_url": "https://github.com/AmbiqAI/helia-core-tester/actions/runs/123456",
    }


# --- orchestration order ----------------------------------------------------------


def test_run_hardware_pipeline_generates_flashes_then_streams(tmp_path: Path, monkeypatch) -> None:
    from helia_core_tester.hardware import hardware_pipeline

    board = resolve_board("apollo510_evb")
    order: list[str] = []

    def _generate(repo_root, spec, suite, float_precision=None, cmsis_nn_root=None, select=None):
        order.append(f"generate:{spec.cpu}:{suite}:{float_precision}:{cmsis_nn_root}")

    def _stage(spec, *, build_dir, options, force_sync, update_dependencies):
        order.append(f"stage:{build_dir.relative_to(tmp_path)}")
        return Path("/kernels")

    def _flash(spec, serial, *, build_dir, jobs, force_reconfigure, force, options, update_dependencies):
        order.append(f"flash:{serial}:{build_dir.relative_to(tmp_path)}:force={force}")
        return firmware_build.FlashDecision(True, "abc", "test")

    def _stream(repo_root, spec, serial, *, build_dir, options, echo, progress_to_stderr, allow_unverified_firmware, prepared,
                fresh_boot):
        order.append(f"stream:{options.suite}:{options.test_name}:unverified={allow_unverified_firmware}:fresh={fresh_boot}")
        return hardware_pipeline.HardwareRunOutcome(session_id="s", result=None, bundle=tmp_path, skipped=[])

    monkeypatch.setattr(hardware_pipeline, "generate_tests_for_board", _generate)
    monkeypatch.setattr(hardware_pipeline, "stage_kernels", _stage)
    monkeypatch.setattr(hardware_pipeline, "flash_firmware", _flash)
    monkeypatch.setattr(hardware_pipeline, "stream_generated_tests", _stream)

    outcome = run_hardware_pipeline(
        tmp_path, board, 42, options=StreamOptions(suite="float", test_name="_f16", float_precision="f16"), echo=lambda _msg: None,
    )
    assert order == [
        "stage:build/hardware/apollo510_evb", "generate:cortex-m55:float:f16:/kernels", "flash:42:build/hardware/apollo510_evb:force=False",
        "stream:float:_f16:unverified=False:fresh=True",
    ]
    assert outcome.flash is not None and outcome.flash.needed

    order.clear()
    run_hardware_pipeline(
        tmp_path, board, 42, options=StreamOptions(), skip_generate=True, skip_flash=True, echo=lambda _msg: None,
        allow_unverified_firmware=True,
    )
    assert order == ["stream:int:None:unverified=True:fresh=False"]

    order.clear()
    run_hardware_pipeline(
        tmp_path, board, 42, options=StreamOptions(), skip_generate=True, force_flash=True, echo=lambda _msg: None,
    )
    assert order == ["flash:42:build/hardware/apollo510_evb:force=True", "stream:int:None:unverified=False:fresh=True"]

    with pytest.raises(ValueError, match="--skip-flash and --force-flash"):
        run_hardware_pipeline(
            tmp_path, board, 42, options=StreamOptions(), skip_generate=True, skip_flash=True, force_flash=True, echo=lambda _msg: None,
        )


def test_run_generates_from_the_saved_kernels(tmp_path: Path, monkeypatch) -> None:
    """Generator and firmware use one resolution."""
    from helia_core_tester.hardware import hardware_pipeline, nsx_app
    from helia_core_tester.tests.test_hardware_nsx_app import make_checkout

    build_dir = tmp_path / "bd"
    saved = nsx_app.AppOptions(cmsis_nn_root=make_checkout(tmp_path / "kernels"), requantize_inline_asm=False)
    app_dir = firmware_build.nsx_app_dir(build_dir)
    app_dir.mkdir(parents=True)
    nsx_app.save_options(app_dir, saved)
    seen: dict = {}

    def _stage(spec, *, options, **kwargs):
        seen["stage"] = options
        return options.cmsis_nn_root

    def _flash(spec, serial, *, options, **kwargs):
        seen["flash"] = options
        return FlashDecision(False, "abc", "test")

    monkeypatch.setattr(hardware_pipeline, "generate_tests_for_board", lambda *a, cmsis_nn_root, **k: seen.update(generate=cmsis_nn_root))
    monkeypatch.setattr(hardware_pipeline, "stage_kernels", _stage)
    monkeypatch.setattr(hardware_pipeline, "flash_firmware", _flash)
    monkeypatch.setattr(
        hardware_pipeline, "stream_generated_tests",
        lambda *a, **k: hardware_pipeline.HardwareRunOutcome(session_id="s", result=None, bundle=tmp_path, skipped=[]),
    )
    run_hardware_pipeline(tmp_path, BOARD, SERIAL, options=StreamOptions(), build_dir=build_dir, echo=lambda _msg: None)
    # Saved kernels, default switches.
    wanted = dataclasses.replace(saved, requantize_inline_asm=True)
    assert seen == {"stage": wanted, "generate": saved.cmsis_nn_root, "flash": wanted}


def _skip_coverage(monkeypatch) -> None:
    """Fake results need no bundle."""
    monkeypatch.setattr("helia_core_tester.hardware.entry_coverage.build_coverage", lambda *a, **k: {})
    monkeypatch.setattr("helia_core_tester.hardware.entry_coverage.write_coverage", lambda *a, **k: None)


def test_stream_passes_build_dir_build_id_to_the_session(tmp_path: Path, monkeypatch) -> None:
    from helia_core_tester.hardware import hardware_pipeline

    (tmp_path / "bundle").mkdir()

    seen: dict = {}
    bridged: list[str] = []

    class _Bundle:
        case_id = "abs_default_s8_hw_generated"
        manifest = {"timing": {"samples": 5}}

    preview = ([_Bundle()], [("skipped-case", "reason")])

    def _bridge(*args, **kwargs):
        bridged.append("bridge")
        return preview

    def _session(repo_root, bundles, **kwargs):
        seen.update(kwargs, bundles=bundles)
        return SimpleNamespace(cases=[]), tmp_path / "bundle"

    monkeypatch.setattr(hardware_pipeline, "make_live_progress_printer", lambda *a, **k: None)
    _skip_coverage(monkeypatch)
    monkeypatch.setattr("helia_core_tester.hardware.session_runner.build_generated_test_case_bundles", _bridge)
    monkeypatch.setattr("helia_core_tester.hardware.session_runner.run_case_bundles", _session)

    build_dir = tmp_path / "bd"
    _write_elf(build_dir, b"fw", "hct-stream")
    echoed: list[str] = []
    outcome = hardware_pipeline.stream_generated_tests(tmp_path, BOARD, 5, build_dir=build_dir, options=StreamOptions(), echo=echoed.append)
    assert seen["expected_build_id"] == "hct-stream"
    assert any("firmware build id hct-stream" in line for line in echoed)
    # Bridged exactly once: the preview list is what the session runner gets.
    assert bridged == ["bridge"]
    assert seen["bundles"] is preview[0] and outcome.skipped is preview[1]


def test_strict_compare_drops_int_tolerance(tmp_path: Path, monkeypatch) -> None:
    from helia_core_tester.hardware import hardware_pipeline
    from helia_core_tester.hardware.case_bundle import CaseBundle

    seen: dict = {}
    tolerant = CaseBundle(tmp_path, tmp_path / "m.json", {
        "case_id": "conv", "timing": {"samples": 5}, "correctness_comparison": {"mode": "tolerant_int", "tolerance": 1},
    }, ())
    monkeypatch.setattr(hardware_pipeline, "make_live_progress_printer", lambda *a, **k: None)
    _skip_coverage(monkeypatch)
    monkeypatch.setattr("helia_core_tester.hardware.session_runner.build_generated_test_case_bundles", lambda *a, **k: ([tolerant], []))
    monkeypatch.setattr(
        "helia_core_tester.hardware.session_runner.run_case_bundles",
        lambda repo_root, bundles, **kwargs: (seen.update(kwargs, bundles=bundles), (SimpleNamespace(cases=[]), tmp_path))[1],
    )
    build_dir = tmp_path / "bd"
    _write_elf(build_dir, b"fw", "hct-strict")
    hardware_pipeline.stream_generated_tests(
        tmp_path, BOARD, 5, build_dir=build_dir, options=StreamOptions(strict_compare=True), echo=lambda _msg: None,
    )
    assert [bundle.comparison for bundle in seen["bundles"]] == [{"mode": "exact_int"}]
    assert seen["compare"] == {"strict": True, "golden_from": None, "golden_session_id": None}
    assert tolerant.comparison["mode"] == "tolerant_int"


def test_golden_from_judges_against_past_output(tmp_path: Path) -> None:
    from helia_core_tester.hardware.case_bundle import blob_numpy, golden_bundle

    bundle = load_case_bundle(build_abs_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path, case_id="abs_gold").manifest_path)
    golden_dir = tmp_path / "past"
    with pytest.raises(RuntimeError, match="No usable golden output for abs_gold"):
        golden_bundle(bundle, golden_dir)
    (golden_dir / "outputs").mkdir(parents=True)
    past = blob_numpy(bundle.expected_output).copy()
    (golden_dir / "outputs" / "abs_gold.bin").write_bytes(past.tobytes())
    assert HostSession(FakeTargetTransport()).run_many([golden_bundle(bundle, golden_dir)]).cases[0].comparison.passed
    past.flat[0] += 1
    (golden_dir / "outputs" / "abs_gold.bin").write_bytes(past.tobytes())
    case = HostSession(FakeTargetTransport()).run_many([golden_bundle(bundle, golden_dir)]).cases[0]
    assert (case.comparison.passed, case.comparison.diff_count, case.comparison.max_abs_diff) == (False, 1, 1.0)


def _write_golden(golden_dir: Path, bundle, *, output: bytes | None = None, **record) -> None:
    """A past run's output and record."""
    from helia_core_tester.hardware.case_bundle import blob_numpy, input_digest

    (golden_dir / "outputs").mkdir(parents=True, exist_ok=True)
    (golden_dir / "correctness").mkdir(exist_ok=True)
    payload = blob_numpy(bundle.expected_output).tobytes() if output is None else output
    (golden_dir / "outputs" / f"{bundle.case_id}.bin").write_bytes(payload)
    doc = {"passed": True, "input_digest": input_digest(bundle), **record}
    (golden_dir / "correctness" / f"{bundle.case_id}.json").write_text(json.dumps(doc))


def _stream_golden(tmp_path: Path, monkeypatch, bundles: list, golden_dir: Path, **flags) -> dict:
    """Stream bundles judged by golden_dir."""
    from helia_core_tester.hardware import hardware_pipeline

    seen: dict = {}
    monkeypatch.setattr(hardware_pipeline, "make_live_progress_printer", lambda *a, **k: None)
    _skip_coverage(monkeypatch)
    monkeypatch.setattr("helia_core_tester.hardware.session_runner.build_generated_test_case_bundles", lambda *a, **k: (bundles, []))
    monkeypatch.setattr(
        "helia_core_tester.hardware.session_runner.run_case_bundles",
        lambda repo_root, bundles, **kwargs: (seen.update(kwargs, bundles=bundles), (SimpleNamespace(cases=[]), tmp_path))[1],
    )
    build_dir = tmp_path / "bd"
    _write_elf(build_dir, b"fw", "hct-gold")
    options = StreamOptions(golden_from=golden_dir, **flags)
    hardware_pipeline.stream_generated_tests(tmp_path, BOARD, 5, build_dir=build_dir, options=options, echo=lambda _m: None)
    return seen


def _abs_bundles(tmp_path: Path, *names: str) -> list:
    return [
        load_case_bundle(build_abs_s8_case_bundle(PROJECT_ROOT, output_root=tmp_path / name, case_id=name).manifest_path)
        for name in names
    ]


def test_golden_from_refuses_failed_cases(tmp_path: Path, monkeypatch) -> None:
    (bundle,) = _abs_bundles(tmp_path, "abs_bad")
    golden_dir = tmp_path / "past"
    _write_golden(golden_dir, bundle, passed=False)
    with pytest.raises(RuntimeError, match="Golden run failed these cases: abs_bad"):
        _stream_golden(tmp_path, monkeypatch, [bundle], golden_dir)
    seen = _stream_golden(tmp_path, monkeypatch, [bundle], golden_dir, golden_allow_failed=True)
    assert [b.case_id for b in seen["bundles"]] == ["abs_bad"]


@pytest.mark.parametrize("record", ['{"passed": tr', "[true]"])
def test_golden_from_refuses_unreadable_records(tmp_path: Path, monkeypatch, record) -> None:
    (bundle,) = _abs_bundles(tmp_path, "abs_bad")
    golden_dir = tmp_path / "past"
    _write_golden(golden_dir, bundle)
    (golden_dir / "correctness" / "abs_bad.json").write_text(record)
    with pytest.raises(RuntimeError, match="Golden run failed these cases: abs_bad"):
        _stream_golden(tmp_path, monkeypatch, [bundle], golden_dir)
    # Unreadable records carry no digest.
    with pytest.raises(RuntimeError, match="no input digest for: abs_bad$"):
        _stream_golden(tmp_path, monkeypatch, [bundle], golden_dir, golden_allow_failed=True)


def test_golden_from_matches_inputs(tmp_path: Path, monkeypatch) -> None:
    import hashlib

    from helia_core_tester.hardware.case_bundle import blob_numpy

    (bundle,) = _abs_bundles(tmp_path, "abs_ok")
    golden_dir = tmp_path / "past"
    past = blob_numpy(bundle.expected_output).copy()
    past.flat[0] += 1
    _write_golden(golden_dir, bundle, output=past.tobytes())
    (golden_dir / "session_manifest.json").write_text(json.dumps({"session_id": "base-1"}))
    seen = _stream_golden(tmp_path, monkeypatch, [bundle], golden_dir)
    (judged,) = seen["bundles"]
    digest = hashlib.sha256(past.tobytes()).hexdigest()
    assert judged.expected_output.sha256 == digest
    assert judged.comparison == {"mode": "exact_int"}
    assert seen["compare"] == {"strict": True, "golden_from": str(golden_dir), "golden_session_id": "base-1"}


def test_golden_from_refuses_other_inputs(tmp_path: Path, monkeypatch) -> None:
    bundles = _abs_bundles(tmp_path, "abs_same", "abs_moved")
    golden_dir = tmp_path / "past"
    _write_golden(golden_dir, bundles[0])
    _write_golden(golden_dir, bundles[1], input_digest="0" * 64)
    with pytest.raises(RuntimeError, match="Golden run used other inputs for: abs_moved$"):
        _stream_golden(tmp_path, monkeypatch, bundles, golden_dir)


def test_golden_from_needs_input_digest(tmp_path: Path, monkeypatch) -> None:
    (bundle,) = _abs_bundles(tmp_path, "abs_old")
    golden_dir = tmp_path / "past"
    _write_golden(golden_dir, bundle, input_digest=None)
    with pytest.raises(RuntimeError, match="Golden run has no input digest for: abs_old$"):
        _stream_golden(tmp_path, monkeypatch, [bundle], golden_dir)


def test_golden_from_names_missing_cases(tmp_path: Path, monkeypatch) -> None:
    bundles = _abs_bundles(tmp_path, "abs_here", "abs_gone")
    golden_dir = tmp_path / "past"
    golden_dir.mkdir()
    with pytest.raises(RuntimeError, match="Golden bundle has no results"):
        _stream_golden(tmp_path, monkeypatch, bundles, golden_dir)
    _write_golden(golden_dir, bundles[0])
    with pytest.raises(RuntimeError, match="Golden run is missing these cases: abs_gone$"):
        _stream_golden(tmp_path, monkeypatch, bundles, golden_dir)
    # Disjoint selection: missing, not empty.
    with pytest.raises(RuntimeError, match="Golden run is missing these cases: abs_gone$"):
        _stream_golden(tmp_path, monkeypatch, bundles[1:], golden_dir)


def test_input_digest_ignores_golden(tmp_path: Path) -> None:
    from helia_core_tester.hardware.case_bundle import golden_bundle, input_digest

    (bundle,) = _abs_bundles(tmp_path, "abs_dig")
    assert bundle.manifest["input_digest"] == input_digest(bundle)
    golden_dir = tmp_path / "past"
    _write_golden(golden_dir, bundle, output=bytes(bundle.expected_output.byte_length))
    assert input_digest(golden_bundle(bundle, golden_dir)) == input_digest(bundle)
    (bundle.root_dir / "blobs" / "input_0.bin").write_bytes(bytes(bundle.input_blob.byte_length))
    assert input_digest(load_case_bundle(bundle.manifest_path)) != input_digest(bundle)


def test_golden_gaps_fail_before_flash(tmp_path: Path, monkeypatch) -> None:
    from helia_core_tester.hardware import hardware_pipeline
    from helia_core_tester.hardware.case_bundle import blob_numpy

    bundles = _abs_bundles(tmp_path, "abs_gone", "abs_short", "abs_unread")
    golden_dir = tmp_path / "past"
    for bundle in bundles:
        _write_golden(golden_dir, bundle)
    (golden_dir / "outputs" / "abs_gone.bin").unlink()
    (golden_dir / "outputs" / "abs_short.bin").write_bytes(blob_numpy(bundles[1].expected_output).tobytes()[:-1])
    # A directory cannot be read.
    (golden_dir / "outputs" / "abs_unread.bin").unlink()
    (golden_dir / "outputs" / "abs_unread.bin").mkdir()
    order: list[str] = []
    monkeypatch.setattr("helia_core_tester.hardware.session_runner.build_generated_test_case_bundles", lambda *a, **k: (bundles, []))
    monkeypatch.setattr(hardware_pipeline, "generate_tests_for_board", lambda *a, **k: order.append("generate"))
    monkeypatch.setattr(hardware_pipeline, "stage_kernels", lambda *a, **k: Path("/kernels"))
    monkeypatch.setattr(hardware_pipeline, "flash_firmware", lambda *a, **k: order.append("flash"))
    with pytest.raises(RuntimeError, match="for: abs_gone, abs_short, abs_unread$"):
        run_hardware_pipeline(
            tmp_path, BOARD, SERIAL, options=StreamOptions(golden_from=golden_dir), build_dir=tmp_path / "bd",
            app_options=object(), echo=lambda _msg: None,
        )
    assert order == ["generate"]


def test_stream_refuses_an_unstamped_build_dir_unless_opted_out(tmp_path: Path, monkeypatch) -> None:
    from helia_core_tester.hardware import hardware_pipeline

    (tmp_path / "bundle").mkdir()

    class _Bundle:
        case_id = "abs_default_s8_hw_generated"
        manifest = {"timing": {"samples": 5}}

    seen: dict = {}
    monkeypatch.setattr(hardware_pipeline, "make_live_progress_printer", lambda *a, **k: None)
    _skip_coverage(monkeypatch)
    monkeypatch.setattr("helia_core_tester.hardware.session_runner.build_generated_test_case_bundles", lambda *a, **k: ([_Bundle()], []))
    monkeypatch.setattr(
        "helia_core_tester.hardware.session_runner.run_case_bundles",
        lambda repo_root, bundles, **kwargs: (seen.update(kwargs), (SimpleNamespace(cases=[]), tmp_path / "bundle"))[1],
    )

    unstamped = tmp_path / "old"
    _write_elf(unstamped, b"fw")
    echoed: list[str] = []
    with pytest.raises(RuntimeError, match="hct_build_id.txt not found") as info:
        hardware_pipeline.stream_generated_tests(tmp_path, BOARD, 5, build_dir=unstamped, options=StreamOptions(), echo=echoed.append)
    assert "--allow-unverified-firmware" in str(info.value) and "hardware build --board apollo510_evb" in str(info.value)
    assert not seen and not echoed  # preflight: nothing streamed, nothing announced

    hardware_pipeline.stream_generated_tests(
        tmp_path, BOARD, 5, build_dir=unstamped, options=StreamOptions(), echo=echoed.append, allow_unverified_firmware=True,
    )
    assert seen["expected_build_id"] is None
    assert any("WARNING" in line and "hct_build_id.txt" in line and "unverified" in line for line in echoed)
    assert any("firmware build id unverified" in line for line in echoed)


def test_stream_refuses_results_over_the_outbox_before_the_probe(tmp_path: Path, monkeypatch) -> None:
    from helia_core_tester.hardware import hardware_pipeline
    from helia_core_tester.hardware.measurement import OutboxOverflowError

    class _Bundle:
        case_id = "abs_default_s8_hw_generated"
        manifest = {"timing": {"samples": 64}}

    monkeypatch.setattr(hardware_pipeline, "make_live_progress_printer", lambda *a, **k: None)
    monkeypatch.setattr("helia_core_tester.hardware.session_runner.build_generated_test_case_bundles", lambda *a, **k: ([_Bundle()], []))
    monkeypatch.setattr("helia_core_tester.hardware.session_runner.run_case_bundles", lambda *a, **k: pytest.fail("streamed"))

    build_dir = tmp_path / "bd"
    _write_elf(build_dir, b"fw", "hct-stream")
    options = StreamOptions(pmu_counters={"cpu": "all", "memory": "all", "mve": "all"})
    with pytest.raises(OutboxOverflowError, match=r"64 samples x 18 passes"):
        hardware_pipeline.stream_generated_tests(tmp_path, BOARD, 5, build_dir=build_dir, options=options, echo=lambda _: None)


# --- --json keeps stdout clean ----------------------------------------------------


def test_stdout_to_stderr_covers_python_and_subprocess_output(capfd) -> None:
    import subprocess
    import sys

    from helia_core_tester.hardware.run_summary import stdout_to_stderr

    with stdout_to_stderr():
        print("python-line")
        subprocess.run([sys.executable, "-c", "print('child-line')"], check=True)
    print("after-line")
    out, err = capfd.readouterr()
    assert "python-line" in err and "child-line" in err
    assert "python-line" not in out and "child-line" not in out
    assert "after-line" in out


def _skip_flash_run(tmp_path: Path, monkeypatch, build_dir: Path, **kwargs) -> dict:
    """Run --skip-flash; capture what generation saw."""
    from helia_core_tester.hardware import hardware_pipeline

    seen: dict = {}

    def _no_stage(*args, **kwargs):
        raise AssertionError("--skip-flash must not restage kernels")

    monkeypatch.setattr(hardware_pipeline, "stage_kernels", _no_stage)
    monkeypatch.setattr(hardware_pipeline, "generate_tests_for_board", lambda *a, cmsis_nn_root, **k: seen.update(generate=cmsis_nn_root))
    monkeypatch.setattr(
        hardware_pipeline, "stream_generated_tests",
        lambda *a, **k: hardware_pipeline.HardwareRunOutcome(session_id="s", result=None, bundle=tmp_path, skipped=[]),
    )
    run_hardware_pipeline(
        tmp_path, BOARD, SERIAL, options=StreamOptions(), build_dir=build_dir, skip_flash=True, echo=lambda _msg: None, **kwargs,
    )
    return seen


def _built_app(tmp_path: Path, monkeypatch, *, digest: str = "d1", current: bool = True):
    from helia_core_tester.hardware import firmware_build, nsx_cli

    build_dir = tmp_path / "bd"
    app_dir = firmware_build.nsx_app_dir(build_dir)
    app_dir.mkdir(parents=True)
    (app_dir / firmware_build.BUILT_LOCK).write_text('{"lock": "d1", "kernels": "k1"}', encoding="utf-8")
    monkeypatch.setattr(nsx_cli, "lock_digest", lambda _app: digest)
    monkeypatch.setattr(nsx_cli, "lock_is_current", lambda _app, _board: current)
    monkeypatch.setattr(nsx_cli, "tree_hash", lambda _root: "k1")
    return build_dir, app_dir


def test_skip_flash_generates_from_the_built_pinned_kernels(tmp_path: Path, monkeypatch) -> None:
    from helia_core_tester.hardware import nsx_app

    build_dir, app_dir = _built_app(tmp_path, monkeypatch)
    options = nsx_app.AppOptions(cmsis_nn_ref="v9")
    nsx_app.kernel_dir(app_dir, options).mkdir(parents=True)
    seen = _skip_flash_run(tmp_path, monkeypatch, build_dir, app_options=options)
    assert seen["generate"] == nsx_app.kernel_dir(app_dir, options)


def test_skip_flash_refuses_edits_in_the_synced_kernels(tmp_path: Path, monkeypatch) -> None:
    """Hand edits under modules/ are caught."""
    from helia_core_tester.hardware import nsx_app, nsx_cli

    build_dir, app_dir = _built_app(tmp_path, monkeypatch)
    options = nsx_app.AppOptions(cmsis_nn_ref="v9")
    nsx_app.kernel_dir(app_dir, options).mkdir(parents=True)
    monkeypatch.setattr(nsx_cli, "tree_hash", lambda _root: "k2")
    with pytest.raises(RunRefused, match="Kernels changed since the build"):
        _skip_flash_run(tmp_path, monkeypatch, build_dir, app_options=options)


@pytest.mark.parametrize("digest, current", [("d2", True), ("d1", False), (None, True)])
def test_skip_flash_refuses_a_moved_lock(tmp_path: Path, monkeypatch, digest, current) -> None:
    """A relock or resync after the build."""
    from helia_core_tester.hardware import nsx_app

    build_dir, _ = _built_app(tmp_path, monkeypatch, digest=digest, current=current)
    with pytest.raises(RunRefused, match="Kernels changed since the build"):
        _skip_flash_run(tmp_path, monkeypatch, build_dir, app_options=nsx_app.AppOptions())


def test_skip_flash_refuses_an_edited_checkout(tmp_path: Path, monkeypatch) -> None:
    """Edits after the build never reach the firmware."""
    import json

    from helia_core_tester.hardware import firmware_build, nsx_app, nsx_cli
    from helia_core_tester.tests.test_hardware_nsx_app import make_checkout

    from neuralspotx.nsx_lock import hash_tree

    build_dir, app_dir = _built_app(tmp_path, monkeypatch)
    root = make_checkout(tmp_path / "kernels")
    options = nsx_app.AppOptions(cmsis_nn_root=root)
    module = nsx_app.kernel_dir(app_dir, options)
    nsx_app.write_kernels(root, module)
    # Real hashes: the checkout comparison needs them.
    monkeypatch.setattr(nsx_cli, "tree_hash", hash_tree)
    record = json.dumps({"lock": "d1", "kernels": hash_tree(module)})
    (app_dir / firmware_build.BUILT_LOCK).write_text(record, encoding="utf-8")
    assert _skip_flash_run(tmp_path, monkeypatch, build_dir, app_options=options)["generate"] == options.cmsis_nn_root
    edited = next(path for path in (root / "Source").rglob("*") if path.is_file())
    edited.write_text(edited.read_text(encoding="utf-8") + "\n// edit\n", encoding="utf-8")
    with pytest.raises(RunRefused, match="Kernel checkout edited since the build"):
        _skip_flash_run(tmp_path, monkeypatch, build_dir, app_options=options)


def test_skip_flash_refuses_dependency_updates(tmp_path: Path, monkeypatch) -> None:
    from helia_core_tester.hardware import nsx_app

    build_dir, _ = _built_app(tmp_path, monkeypatch)
    with pytest.raises(RunRefused, match="cannot update dependencies"):
        _skip_flash_run(tmp_path, monkeypatch, build_dir, app_options=nsx_app.AppOptions(), update_dependencies=True)


def test_status_compare_has_no_output_diff() -> None:
    from helia_core_tester.hardware.comparison import compare_status

    result = compare_status(-1, {"mode": "exact_status", "expected_status": 0})
    assert (result.passed, result.mismatch_count, result.diff_count) == (False, 1, None)
