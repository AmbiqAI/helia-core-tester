"""Depthwise cases that call a named ns-cmsis-nn entry (`entry:`) instead of the wrapper."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.entry import MANIFEST_ENTRY_KEYS, EntryError, apply_entry_manifest, resolve_entry
from helia_core_tester.generation.kernel_dispatch import DEPTHWISE_CONV_S8_PLANAR_RULE

from helia_core_tester.generation.test_ops import _required_kernel_symbols, generate_test

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPT_SIZER = "arm_depthwise_conv_s8_opt_get_buffer_size"
SIZER_3X3 = "arm_depthwise_conv_s8_opt_3x3_get_buffer_size"
# Each pilot entry and the scratch query its manifest record binds to ctx.
DEPTHWISE_CONV_S8_ENTRIES = {"arm_depthwise_conv_s8_opt_3x3": SIZER_3X3,
                             "arm_depthwise_conv_s8_opt_3x3_c64_s1": SIZER_3X3,
                             "arm_depthwise_conv_s8_opt_planar": OPT_SIZER,
                             "arm_depthwise_conv_s8_opt_channelwise": OPT_SIZER}


def _descriptor(name: str) -> dict:
    return next(d for d in load_all_descriptors(str(_PROJECT_ROOT / "assets" / "descriptors")) if d["name"] == name)


def _source(name: str, tmp_path: Path, **overrides) -> str:
    generate_test({**_descriptor(name), **overrides}, str(tmp_path))
    case_dir = next(p.parent for p in tmp_path.rglob(f"{name}.reference.json"))
    return "".join(p.read_text() for p in case_dir.glob("*.c"))


def _resolve(entry: str, act: str = "S8", weight: str = "S8", **desc) -> dict:
    return resolve_entry("DepthwiseConv", entry, activation_dtype=act, weight_dtype=weight, cpu="cortex-m55",
                         desc={"name": "x", **desc})


def test_entries_resolve_with_the_scratch_query_of_their_manifest_record() -> None:
    for entry, sizer in DEPTHWISE_CONV_S8_ENTRIES.items():
        assert _resolve(entry) == {
            "kernel_fn": entry,
            "kernel_get_buffer_size_fn": sizer,
            "entry_family": "contract",
            "kernel_needs_layout": False,
            "buffer_size_needs_layout": False,
            "entry_implementation": entry.replace("arm_", "", 1),
            "entry_ctx_buffer": "scratch",
        }


@pytest.mark.parametrize(
    ("name", "cpu", "capacity"),
    [
        # 16 x (3 x 52 + 32) + W x C + 16 on every build.
        ("depthwise_conv_entry_3x3_25x5_c64_s8", "cortex-m55", 16 * (3 * 52 + 32) + 5 * 64 + 16),
        ("depthwise_conv_entry_3x3_declines_c15_s8", "cortex-m4", 16 * (3 * 52 + 32) + 5 * 15 + 16),
        # The opt scratch: 4 x 124 x 3 x 3 on MVE, C x 3 x 3 int16 on DSP, none on the plain build.
        ("depthwise_conv_entry_channelwise_25x5_c64_s8", "cortex-m55", 4 * 124 * 9),
        ("depthwise_conv_entry_channelwise_25x5_c64_s8", "cortex-m4", 64 * 9 * 2),
        ("depthwise_conv_entry_channelwise_25x5_c64_s8", "cortex-m0", 0),
        ("depthwise_conv_entry_planar_48x48_c8_s8", "cortex-m55", 4 * 124 * 9),
    ],
)
def test_entry_scratch_capacity_is_its_records_planned_size(tmp_path: Path, name: str, cpu: str, capacity: int) -> None:
    desc = _descriptor(name)
    generate_test(desc, str(tmp_path), cpu=cpu)
    source = "".join(p.read_text() for pattern in ("*.h", "*.c") for p in tmp_path.rglob(pattern))
    assert re.findall(rf"#define {name.upper()}_BUFFER_SIZE_MAX (\d+)", source) == [str(capacity)]


def test_an_entry_sizer_unlike_its_records_query_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    import helia_core_tester.generation.entry as entry_module

    stamped = apply_entry_manifest(_descriptor("depthwise_conv_entry_3x3_25x5_c64_s8"))
    monkeypatch.setattr(entry_module, "apply_entry_manifest", lambda desc, contracts=None: {**stamped,
                                                                                           "entry_sizer": OPT_SIZER})
    with pytest.raises(EntryError, match="is not the size query of record depthwise_conv_s8_opt_3x3's ctx buffer"):
        _resolve("arm_depthwise_conv_s8_opt_3x3")


@pytest.mark.parametrize(
    ("entry", "act", "message"),
    [
        ("arm_depthwise_conv_s8_opt_3x4", "S8", "is not a public function"),
        ("arm_depthwise_conv_s8_opt_3x3", "S16", "input_data"),
    ],
)
def test_unknown_or_mismatched_entries_are_rejected(entry, act, message) -> None:
    with pytest.raises(EntryError, match=message):
        _resolve(entry, act)


def test_entry_and_planar_rule_gate_the_case_on_the_checkout() -> None:
    desc = apply_entry_manifest(_descriptor("depthwise_conv_entry_planar_48x48_c8_s8"))

    assert _required_kernel_symbols(desc) == ["arm_depthwise_conv_s8_opt_planar", OPT_SIZER,
                                              DEPTHWISE_CONV_S8_PLANAR_RULE]
    assert _required_kernel_symbols(_descriptor("depthwise_conv_dilated_1d_k7_d2_c24_s8")) == []


def test_3x3_entry_gates_on_the_query_its_record_names() -> None:
    desc = apply_entry_manifest(_descriptor("depthwise_conv_entry_3x3_25x5_c64_s8"))
    assert _required_kernel_symbols(desc) == ["arm_depthwise_conv_s8_opt_3x3", SIZER_3X3]
    assert desc["required_capabilities"] == ["mve"] and desc["autovectorize_declines"] is True
    assert desc["entry_implementation"] == "depthwise_conv_s8_opt_3x3"


def test_a_decline_case_runs_everywhere_and_never_expects_the_full_result() -> None:
    desc = apply_entry_manifest(_descriptor("depthwise_conv_entry_3x3_declines_c15_s8"))
    assert desc["entry_sizer"] == SIZER_3X3
    assert "required_capabilities" not in desc and "autovectorize_declines" not in desc


@pytest.mark.parametrize("key, value", [("entry_sizer", OPT_SIZER), ("entry_scratch", "none"),
                                        ("required_capabilities", ["mve"]), ("autovectorize_declines", True)])
def test_a_covered_entry_refuses_the_terms_its_record_decides(key: str, value) -> None:
    assert key in MANIFEST_ENTRY_KEYS
    with pytest.raises(EntryError, match=rf"record depthwise_conv_s8_opt_3x3 .* decides \['{key}'\]"):
        _resolve("arm_depthwise_conv_s8_opt_3x3", **{key: value})


def test_a_sizer_list_is_refused() -> None:
    with pytest.raises(EntryError, match="must name one scratch-size query"):
        _resolve("arm_depthwise_conv_s16", "S16", entry_sizer=["arm_depthwise_conv_s16_get_buffer_size"])


def test_entry_case_calls_the_entry_with_weight_sums_and_its_scratch_query(tmp_path: Path) -> None:
    name = "depthwise_conv_entry_3x3_25x5_c64_s8"
    source = _source(name, tmp_path)

    # The calls are bound from the kernel contract, which names each argument in a comment.
    code = re.sub(r"/\*.*?\*/|//[^\n]*", " ", source, flags=re.S)
    call = re.search(r"return (\w+)\(\s*&\w+_ctx,\s*&(\w+)_weight_sum_ctx,", code)
    assert call and call.group(1) == "arm_depthwise_conv_s8_opt_3x3"
    assert re.search(r"arm_depthwise_conv_s8_opt_3x3_get_buffer_size\(\s*&\w+_input_dims\s*\)", code)
    assert "arm_depthwise_conv_s8_opt_get_buffer_size(" not in code
    assert "arm_depthwise_conv_wrapper_s8(" not in source
    assert "HELIA_VALIDATE_OUTPUTS(" in source


@pytest.mark.parametrize(
    ("name", "sizer"),
    [
        ("depthwise_conv_entry_3x3_28x28_c64_stride2_s8", SIZER_3X3),
        ("depthwise_conv_entry_3x3_c64_s1_14x28_s8", SIZER_3X3),
        ("depthwise_conv_entry_channelwise_25x5_c64_s8", OPT_SIZER),
    ],
)
def test_each_entry_queries_only_the_size_its_record_names(tmp_path: Path, name: str, sizer: str) -> None:
    code = re.sub(r"/\*.*?\*/|//[^\n]*", " ", _source(name, tmp_path), flags=re.S)

    queried = set(re.findall(r"(arm_depthwise_conv_s8_opt(?:_3x3)?_get_buffer_size)\(", code))
    assert queried == {sizer}


def test_declined_case_checks_the_status_and_an_untouched_output(tmp_path: Path) -> None:
    name = "depthwise_conv_entry_3x3_declines_c15_s8"
    desc = _descriptor(name)
    assert desc["input_shape"][-1] == desc["filter_shape"][2] == 15
    source = _source(name, tmp_path)

    assert "HELIA_GUARD_CHECK_UNTOUCHED(" in source
    assert re.search(r"HELIA_VALIDATE_EXPECTED_STATUS\([^;]*ARM_CMSIS_NN_NO_IMPL_ERROR", source)
    assert "HELIA_VALIDATE_OUTPUTS(" not in source


def test_planar_case_checks_the_rule(tmp_path: Path) -> None:
    source = _source("depthwise_conv_entry_planar_declines_25x5_c64_s8", tmp_path)

    assert re.search(rf"{DEPTHWISE_CONV_S8_PLANAR_RULE}\(", source)
    assert "planar_supported != 0" in source


def test_entry_with_a_fault_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="supports only fault: invalid_layout"):
        _source(
            "depthwise_conv_entry_3x3_25x5_c64_s8",
            tmp_path,
            fault="null_ctx_buf",
            expected_status="ARM_CMSIS_NN_ARG_ERROR",
        )


@pytest.mark.parametrize("name", ["depthwise_conv_float_2x5_f16", "depthwise_conv_dilated_1d_k7_d2_c24_s8"])
def test_cases_without_an_entry_render_no_entry_code(name: str, tmp_path: Path) -> None:
    source = _source(name, tmp_path)

    assert DEPTHWISE_CONV_S8_PLANAR_RULE not in source
    assert "planar_supported" not in source
    assert "HELIA_GUARD_CHECK_UNTOUCHED(" not in source


@pytest.mark.parametrize(
    "name",
    ["depthwise_conv_entry_3x3_14x10_c36_stride2_s8", "depthwise_conv_entry_3x3_c64_s1_14x28_s8"],
)
def test_3x3_build_decline_uses_the_core_contract(name: str, tmp_path: Path) -> None:
    source = _source(name, tmp_path)
    blocks = re.findall(
        r"#if defined\(HELIA_CMSIS_NN_INT_AUTOVECTORIZE\) && "
        r"defined\(ARM_NN_DEPTHWISE_CONV_S8_OPT_3X3_AUTOVECTORIZE_DECLINES\)(.*?)#else(.*?)#endif",
        source,
        re.S,
    )
    assert len(blocks) == 2
    assert "true /* the entry declines on this build: poison */" in blocks[0][0]
    declined, full = blocks[1]
    assert "ARM_CMSIS_NN_NO_IMPL_ERROR" in declined
    assert "HELIA_GUARD_CHECK_UNTOUCHED(" in declined
    assert "HELIA_VALIDATE_OUTPUTS(" not in declined
    assert "HELIA_VALIDATE_STATUS(" in full and "HELIA_VALIDATE_OUTPUTS(" in full


@pytest.mark.parametrize(
    "name",
    ["depthwise_conv_entry_channelwise_25x5_c64_s8", "depthwise_conv_entry_planar_48x48_c8_s8",
     "depthwise_conv_entry_3x3_declines_c15_s8"],
)
def test_other_entry_contracts_do_not_depend_on_the_3x3_build(name: str, tmp_path: Path) -> None:
    assert "ARM_NN_DEPTHWISE_CONV_S8_OPT_3X3_AUTOVECTORIZE_DECLINES" not in _source(name, tmp_path)
