"""The Cortex-M4, DWT-only board (apollo3p_evb): render, gating, sizing.

`nsx_cli.starter_profile` is monkeypatched; nothing here reads the
installed NSX registry.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from helia_core_tester.hardware import generated_test_bridge as bridge
from helia_core_tester.hardware import memory_report, nsx_app, nsx_cli
from helia_core_tester.hardware.boards import BoardSpec, resolve_board
from helia_core_tester.hardware.hardware_pipeline import StreamOptions, fit_to_board

M4 = resolve_board("apollo3p_evb")
M55 = resolve_board("apollo510_evb")


@pytest.fixture(autouse=True)
def fake_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    profile = {"board": "x", "channel": "stable", "modules": ["nsx-core"]}
    monkeypatch.setattr(nsx_cli, "starter_profile", lambda board: profile)


def _cmakelists(board: BoardSpec, tmp_path: Path) -> str:
    return nsx_app.render_app(board, nsx_app.AppOptions(), tmp_path / board.id).cmakelists


def test_row_matches_nsx_and_hpx_facts() -> None:
    assert M4 == BoardSpec(
        id="apollo3p_evb",
        nsx_board="apollo3p_evb",
        soc="apollo3p",
        cpu="cortex-m4",
        pmu_tier="dwt",
        has_mve=False,
        jlink_device="AMA3B2KK-KBR",
        swd_speed_khz=4000,
        workspace_bytes=114688,
        flash_region="ROMEM",
        ram_region="RWMEM",
        core_clock_hz=48_000_000,
    )


def test_m4_render_adds_fp16_storage_and_empty_heap(tmp_path: Path) -> None:
    text = _cmakelists(M4, tmp_path)
    assert "-mfp16-format=ieee" in text
    assert text.index("-mfp16-format=ieee") < text.index("nsx_bootstrap_app(")
    assert "--defsym=__HeapBase=_ebss" in text and "--defsym=__HeapLimit=_ebss" in text
    assert "nsx::pmu_armv8m" not in text
    assert 'HCT_BENCHMARK_SERVER_TARGET_CPU="cortex-m4"' in text


def test_m55_render_has_neither(tmp_path: Path) -> None:
    text = _cmakelists(M55, tmp_path)
    assert "-mfp16-format" not in text and "--defsym" not in text


def test_dwt_board_defaults_to_cycles_only() -> None:
    options = fit_to_board(M4, StreamOptions(), explicit_pmu=False)
    assert options.pmu_counters == {"cpu": ["ARM_PMU_CPU_CYCLES"]}
    # PMU boards keep the default groups.
    assert fit_to_board(M55, StreamOptions(), explicit_pmu=False).pmu_counters == StreamOptions().pmu_counters


def test_dwt_board_refuses_event_counters() -> None:
    with pytest.raises(ValueError, match="no PMU"):
        fit_to_board(M4, StreamOptions(pmu_counters={"mve": "default"}), explicit_pmu=True)
    cycles = StreamOptions(pmu_counters={"cpu": ["ARM_PMU_CPU_CYCLES"]})
    assert fit_to_board(M4, cycles, explicit_pmu=True).pmu_counters == {"cpu": ["ARM_PMU_CPU_CYCLES"]}


def test_m4_refuses_fp16_and_narrows_float_to_f32() -> None:
    with pytest.raises(ValueError, match="no FP16"):
        fit_to_board(M4, StreamOptions(suite="float", float_precision="f16"), explicit_pmu=False)
    assert fit_to_board(M4, StreamOptions(suite="float"), explicit_pmu=False).float_precision == "f32"
    assert fit_to_board(M4, StreamOptions(suite="both"), explicit_pmu=False).float_precision is None
    assert fit_to_board(M4, StreamOptions(suite="int"), explicit_pmu=False).float_precision is None
    assert fit_to_board(M55, StreamOptions(suite="float"), explicit_pmu=False).float_precision is None


def test_s4_one_by_n_needs_im2col_without_mve() -> None:
    dims = dict(
        input_dims={"n": 1, "h": 1, "w": 8, "c": 4},
        filter_dims={"n": 2, "h": 1, "w": 3, "c": 4},
        output_dims={"n": 1, "h": 1, "w": 6, "c": 2},
        stride_h=1, stride_w=1, pad_h=0, pad_w=0, dilation_h=1, dilation_w=1,
    )
    assert bridge._calculate_convolve_s4_scratch_bytes(**dims) == 0
    # Covers arm_convolve_s4_get_buffer_size's 2 * rhs_cols * 2.
    assert bridge._calculate_convolve_s4_scratch_bytes(**dims, mve=False) >= 2 * (3 * 4) * 2


def test_linker_script_falls_back_to_the_plain_script(tmp_path: Path) -> None:
    gcc = tmp_path / "modules" / "nsx-core" / "src" / "apollo3p" / "gcc"
    assert memory_report.linker_script_path(M4, tmp_path).name == "linker_script_sbl.ld"
    gcc.mkdir(parents=True)
    (gcc / "linker_script.ld").write_text("MEMORY {}\n")
    assert memory_report.linker_script_path(M4, tmp_path) == gcc / "linker_script.ld"
    (gcc / "linker_script_sbl.ld").write_text("MEMORY {}\n")
    assert memory_report.linker_script_path(M4, tmp_path).name == "linker_script_sbl.ld"
