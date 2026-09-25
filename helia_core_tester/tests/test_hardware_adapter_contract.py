"""Adapter bodies that render their kernel calls from the ns-cmsis-nn kernel contract.

The pilot is the elementwise add/sub/mul calls of `run_elementwise_binary_once`: the
argument order comes from the export, a parameter the body forgets is a render error, and
the checks below keep every adapter's declared session scalars and every contract-bound
argument honest -- a waiver in `FirmwareAdapterSpec.unmarshalled` must say why."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from helia_core_tester.contract.ir import CONTRACT_RELPATH, ContractError, ContractSet, load_contract_set
from helia_core_tester.contract.render import ContractRenderError
from helia_core_tester.generation.utils.temp_sizer_probe import resolve_cmsis_nn_root
from helia_core_tester.hardware.adapter_specs import (
    FIRMWARE_ADAPTERS,
    FirmwareAdapterSpec,
    contract_calls_in,
    render_adapter_body,
    render_generated_adapters_source,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GENERATOR_SCRIPT = PROJECT_ROOT / "scripts" / "generate_hardware_adapters.py"
ADAPTERS_C_PATH = PROJECT_ROOT / "cmake" / "hardware" / "benchmark_server_adapters.gen.c"
PILOT_CONTRACT_ROOT = Path(__file__).parent / "fixtures" / "contract" / "hardware_pilot"
PILOT_SYMBOLS = (
    "arm_add_s8", "arm_sub_s8", "arm_mul_s8", "arm_add_s16", "arm_sub_s16", "arm_mul_s16",
    "arm_elementwise_add_f32", "arm_elementwise_sub_f32", "arm_elementwise_mul_f32",
    "arm_elementwise_add_f16", "arm_elementwise_sub_f16", "arm_elementwise_mul_f16",
)
# Reset for every case by handle_case_meta() and shared by all adapters (see the
# adapter_specs module docstring); never listed in scalar_fields.
ALWAYS_PRESENT_FIELDS = {"output_h", "output_w", "output_c", "input_offset", "output_offset",
                         "activation_min", "activation_max"}
LITERAL_RE = re.compile(r"^(?:-?\d+(?:\.\d+)?[uUlLfF]*|NULL|true|false|\{.*\}|\".*\")$")


@pytest.fixture(scope="module")
def pilot() -> ContractSet:
    contracts = load_contract_set(PILOT_CONTRACT_ROOT)
    assert contracts.present
    return contracts


def _by_name(name: str) -> FirmwareAdapterSpec:
    return next(adapter for adapter in FIRMWARE_ADAPTERS if adapter.function_name == name)


def _waivers(adapter: FirmwareAdapterSpec) -> dict[str, str]:
    return dict(adapter.unmarshalled)


# --- the pilot -------------------------------------------------------------------------


def test_pilot_body_binds_every_parameter_of_each_call(pilot: ContractSet) -> None:
    calls = contract_calls_in(_by_name("run_elementwise_binary_once"), pilot)
    assert [symbol for symbol, _ in calls] == list(PILOT_SYMBOLS)
    for symbol, args in calls:
        assert list(args) == [param.name for param in pilot.require(symbol).params], symbol


def test_pilot_render_matches_the_committed_file(pilot: ContractSet) -> None:
    assert render_generated_adapters_source(pilot) == ADAPTERS_C_PATH.read_text(encoding="utf-8")


def test_pilot_arguments_come_from_session_or_blob_state(pilot: ContractSet) -> None:
    """A contract call may only bind a parameter to a constant when `unmarshalled` says
    why (hard-coding like PReLUScalar's scalar_is_input must be visible, not implicit)."""
    for adapter in FIRMWARE_ADAPTERS:
        waivers = _waivers(adapter)
        for symbol, args in contract_calls_in(adapter, pilot):
            for name, expression in args.items():
                if LITERAL_RE.match(expression.strip()):
                    assert f"{symbol}.{name}" in waivers, (
                        f"{adapter.function_name}: {symbol}.{name} is bound to the constant "
                        f"{expression!r} without an unmarshalled reason"
                    )


def test_every_declared_scalar_field_is_read_by_its_body_or_waived() -> None:
    """The `null_arg_mask` case: a field the host sends and the firmware parses but no
    body reads is drift unless a waiver names it."""
    for adapter in FIRMWARE_ADAPTERS:
        reads = set(re.findall(r"session->(\w+)", adapter.c_body))
        waivers = _waivers(adapter)
        for field in adapter.scalar_fields:
            if field in ALWAYS_PRESENT_FIELDS or field in reads:
                continue
            assert field in waivers, f"{adapter.function_name}: scalar_fields lists {field!r} but the body never reads it"


def test_waivers_are_explained_and_not_stale(pilot: ContractSet) -> None:
    for adapter in FIRMWARE_ADAPTERS:
        bound = {f"{symbol}.{name}" for symbol, args in contract_calls_in(adapter, pilot) for name in args}
        reads = set(re.findall(r"session->(\w+)", adapter.c_body))
        for name, reason in adapter.unmarshalled:
            assert isinstance(reason, str) and len(reason.split()) >= 4, f"{adapter.function_name}: waiver {name!r} needs a reason"
            if "." in name:
                assert name in bound, f"{adapter.function_name}: waiver {name!r} names no contract-bound argument"
            else:
                assert name in adapter.scalar_fields, f"{adapter.function_name}: waiver {name!r} is not a scalar field"
                assert name not in reads, f"{adapter.function_name}: waiver {name!r} is stale, the body reads it"


def test_null_arg_mask_is_the_one_documented_unread_field() -> None:
    unread = {(a.function_name, name) for a in FIRMWARE_ADAPTERS for name, _ in a.unmarshalled if "." not in name}
    assert unread == {("run_data_movement_once", "null_arg_mask")}


# --- rendering rules -------------------------------------------------------------------


def _spec(body: str, *, templated: bool) -> FirmwareAdapterSpec:
    return FirmwareAdapterSpec(label="fx", function_name="run_fx_once", guard=None, kernel_ids=(),
                               scalar_fields=(), c_body=body, templated=templated)


def test_plain_bodies_are_emitted_verbatim_even_with_jinja_tokens(pilot: ContractSet) -> None:
    body = "static int x = {{ 1 }}; /* {% not a tag %} */"
    assert render_adapter_body(_spec(body, templated=False), pilot) == body
    assert contract_calls_in(_spec(body, templated=False), pilot) == []


def test_templated_body_renders_the_call_in_prototype_order(pilot: ContractSet) -> None:
    body = ('return {{ contract_call("arm_elementwise_add_f32", {"block_size": "n", "output": "o", '
            '"out_activation_max": "hi", "out_activation_min": "lo", "input_2_vect": "b", "input_1_vect": "a"}, '
            'indent="    ") }};')
    rendered = render_adapter_body(_spec(body, templated=True), pilot)
    assert rendered == ("return arm_elementwise_add_f32(\n    a, /* input_1_vect */\n    b, /* input_2_vect */\n"
                        "    o, /* output */\n    lo, /* out_activation_min */\n    hi, /* out_activation_max */\n"
                        "    n /* block_size */\n);")


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ('{{ contract_call("arm_mul_s8", {"input1_data": "a"}) }}', "arm_mul_s8: call arguments do not match the contract (missing"),
        ('{{ contract_call("arm_elementwise_add_f16", {"input_1_vect": "a", "input_2_vect": "b", "output": "o", '
         '"out_activation_min": "lo", "out_activation_max": "hi", "block_size": "n", "layout": "x"}) }}',
         "unknown ['layout']"),
        ('{{ contract_call("arm_fx_missing", {}) }}', "arm_fx_missing: not in the kernel contract"),
    ],
)
def test_templated_body_fails_closed_on_a_bad_call(pilot: ContractSet, body: str, message: str) -> None:
    with pytest.raises(ContractError, match=re.escape(message)):
        render_adapter_body(_spec(body, templated=True), pilot)


def test_templated_body_rejects_an_unknown_template_name(pilot: ContractSet) -> None:
    import jinja2

    with pytest.raises(jinja2.exceptions.UndefinedError):
        render_adapter_body(_spec("{{ not_a_global }}", templated=True), pilot)


def test_render_without_a_contract_fails_naming_the_export(tmp_path: Path) -> None:
    bare = tmp_path / "old-checkout"
    (bare / "Include").mkdir(parents=True)
    absent = load_contract_set(bare)
    assert not absent.present
    with pytest.raises(ContractRenderError, match=re.escape("arm_add_s8: this template renders its call from the kernel contract")):
        render_generated_adapters_source(absent)


def test_generator_script_fails_closed_without_a_contract(tmp_path: Path) -> None:
    bare = tmp_path / "old-checkout"
    (bare / "Include").mkdir(parents=True)
    before = ADAPTERS_C_PATH.read_bytes()
    result = subprocess.run([sys.executable, str(GENERATOR_SCRIPT), "--cmsis-nn-root", str(bare)],
                            cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode == 1
    assert "cannot render -- arm_add_s8" in result.stderr and "kernel_contracts.json" in result.stderr
    assert ADAPTERS_C_PATH.read_bytes() == before


def test_generator_script_reports_a_corrupt_contract(tmp_path: Path) -> None:
    root = tmp_path / "ns-cmsis-nn"
    (root / CONTRACT_RELPATH).parent.mkdir(parents=True)
    (root / CONTRACT_RELPATH).write_text("{not json")
    result = subprocess.run([sys.executable, str(GENERATOR_SCRIPT), "--check", "--cmsis-nn-root", str(root)],
                            cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode == 1 and "not valid JSON" in result.stderr


# --- the fixture against the real export -------------------------------------------------


def test_pilot_fixture_matches_the_real_export(pilot: ContractSet) -> None:
    """The committed pilot contract is a copy of twelve records of the ns-cmsis-nn export;
    when the checkout is reachable they must still agree, or the pilot renders a call the
    real headers no longer declare."""
    root = resolve_cmsis_nn_root()
    real = load_contract_set(root)
    if not real.present:
        if os.environ.get("HELIA_CORE_TESTER_REQUIRE_CONTRACT"):
            pytest.fail(f"HELIA_CORE_TESTER_REQUIRE_CONTRACT is set but {root} has no kernel contract")
        pytest.skip("no ns-cmsis-nn checkout with a kernel contract")
    assert set(pilot.functions) == set(PILOT_SYMBOLS)
    for symbol in PILOT_SYMBOLS:
        expected = replace(real.require(symbol), line=pilot.require(symbol).line)
        assert pilot.require(symbol) == expected, f"{symbol}: refresh the fixture from {real.path}"
    assert render_generated_adapters_source(real) == render_generated_adapters_source(pilot)
