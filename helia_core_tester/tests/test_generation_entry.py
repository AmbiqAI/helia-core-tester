"""resolve_entry: every `entry:` resolves from the kernel contract, only for operators that bind
their call from it; the shipped descriptors name their scratch query explicitly where the entry
has none of its own."""

from __future__ import annotations

import pytest

from helia_core_tester.contract.ir import STATUS_ABSENT, STATUS_PRESENT, ContractSet, FunctionDecl, ParamDecl
from helia_core_tester.generation import entry as entry_module
from helia_core_tester.generation.entry import EntryError, entry_scratch_bytes, resolve_entry
from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.test_ops import _required_kernel_symbols


def _decl(name: str, *params: tuple[str, str]) -> FunctionDecl:
    return FunctionDecl(name=name, header="Include/arm_nnfunctions.h", line=1, guards=(),
                        returns="int32_t" if "_get_" in name else "arm_cmsis_nn_status",
                        params=tuple(ParamDecl(n, t, "in") for n, t in params))


KERNEL = _decl("arm_fx_kernel_s16", ("ctx", "const cmsis_nn_context *"), ("input_data", "const int16_t *"),
               ("filter_data", "const int8_t *"), ("bias_data", "const int64_t *"), ("output_data", "int16_t *"))
CONTRACTS = ContractSet(status=STATUS_PRESENT, root=None, path=None, functions={d.name: d for d in (
    KERNEL,
    _decl("arm_fx_kernel_s16_get_buffer_size", ("input_dims", "const cmsis_nn_dims *")),
    _decl("arm_fx_kernel_s16_get_buffer_size_mve", ("input_dims", "const cmsis_nn_dims *")),
    _decl("arm_fx_nosizer_s16", ("input_data", "const int16_t *"), ("output_data", "int16_t *")),
    _decl("arm_fx_other_get_buffer_size", ("input_dims", "const cmsis_nn_dims *")),
    _decl("arm_nn_fx_helper", ("x", "int32_t")),
)})


@pytest.fixture
def bound(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(entry_module, "CONTRACT_BOUND_OPERATORS", frozenset({"FxOp"}))


def _resolve(entry: str, desc: dict | None = None, *, operator: str = "FxOp", act: str = "S16", weight: str = "S8",
             cpu: str = "cortex-m4", contracts: ContractSet = CONTRACTS) -> dict:
    return resolve_entry(operator, entry, activation_dtype=act, weight_dtype=weight, cpu=cpu,
                         desc={"name": "fx_case", **(desc or {})}, contracts=contracts)


def test_every_shipped_entry_resolves_from_the_contract_with_its_declared_scratch() -> None:
    """No per-entry table remains: each descriptor's entry is a public kernel of the contract,
    and one without a same-named sizer names its family's query as entry_sizer."""
    from pathlib import Path

    from helia_core_tester.contract.render import load_current_contracts

    contracts = load_current_contracts()
    root = Path(__file__).resolve().parents[2] / "assets" / "descriptors"
    entries = [d for d in load_all_descriptors(str(root)) if d.get("entry")]
    assert len(entries) >= 50
    for desc in entries:
        entry, sizer = desc["entry"], desc.get("entry_sizer")
        own = contracts.sizer_for(entry, "cortex-m55")
        assert own is not None or sizer is not None or desc.get("entry_scratch") is not None, (
            f"{desc['name']}: {entry} has no sizer of its own and declares none")
        if contracts.find(entry) is None:
            continue  # gated on the checkout, which the pure-Python fixture may not carry
        got = resolve_entry(desc["operator"], entry, activation_dtype=desc.get("activation_dtype", "S8"),
                            weight_dtype=desc.get("weight_dtype", "S8"), cpu="cortex-m55", desc=desc,
                            contracts=contracts)
        assert got["kernel_fn"] == entry and got["entry_family"] == "contract"
        if sizer is not None:
            assert got["kernel_get_buffer_size_fn"] == sizer


def test_operators_outside_the_contract_path_are_refused() -> None:
    with pytest.raises(EntryError, match=r"AvgPool does not resolve entries from the kernel contract"):
        _resolve("arm_fx_kernel_s16", operator="AvgPool")


def test_convolve_binds_its_call_from_the_contract() -> None:
    assert {"Convolve", "DepthwiseConv", "FullyConnected", "TransposeConv"} <= entry_module.CONTRACT_BOUND_OPERATORS


def test_contract_entry_uses_its_own_sizer_per_cpu(bound) -> None:
    assert _resolve("arm_fx_kernel_s16") == {"kernel_fn": "arm_fx_kernel_s16", "entry_family": "contract",
                                             "kernel_get_buffer_size_fn": "arm_fx_kernel_s16_get_buffer_size",
                                             "kernel_needs_layout": False, "buffer_size_needs_layout": False}
    assert _resolve("arm_fx_kernel_s16", cpu="cortex-m55")["kernel_get_buffer_size_fn"] == \
        "arm_fx_kernel_s16_get_buffer_size_mve"


def test_entry_sizer_and_entry_scratch(bound) -> None:
    assert _resolve("arm_fx_nosizer_s16", {"entry_sizer": "arm_fx_other_get_buffer_size"})[
        "kernel_get_buffer_size_fn"] == "arm_fx_other_get_buffer_size"
    none = _resolve("arm_fx_nosizer_s16", {"entry_scratch": "none"})
    assert none["kernel_get_buffer_size_fn"] is None and none["entry_scratch_bytes"] == 0
    assert _resolve("arm_fx_nosizer_s16", {"entry_scratch": 256})["entry_scratch_bytes"] == 256


@pytest.mark.parametrize("desc, message", [
    ({}, "declares no arm_fx_nosizer_s16_get_buffer_size; set entry_sizer"),
    ({"entry_sizer": "arm_fx_missing_get_buffer_size"}, "is not a public function"),
    ({"entry_sizer": "arm_fx_kernel_s16"}, "is a kernel, not a scratch-size query"),
    ({"entry_sizer": "arm_fx_other_get_buffer_size", "entry_scratch": "none"}, "not both"),
    ({"entry_scratch": "some"}, "must be 'none' or a non-negative byte count"),
    ({"entry_scratch": -1}, "must be 'none' or a non-negative byte count"),
    ({"entry_scratch": True}, "must be 'none' or a non-negative byte count"),
])
def test_scratch_errors_fail_closed(bound, desc: dict, message: str) -> None:
    with pytest.raises(EntryError, match=message):
        _resolve("arm_fx_nosizer_s16", desc)


def test_contract_entry_must_be_a_declared_kernel_of_the_right_precision(bound) -> None:
    with pytest.raises(EntryError, match="is not a public function of this ns-cmsis-nn checkout"):
        _resolve("arm_fx_missing_s16")
    with pytest.raises(EntryError, match="is a sizer, not a kernel"):
        _resolve("arm_fx_kernel_s16_get_buffer_size")
    with pytest.raises(EntryError, match="is a support, not a kernel"):
        _resolve("arm_nn_fx_helper")
    with pytest.raises(EntryError, match="input_data is 'const int16_t \\*' but the input dtype is S8"):
        _resolve("arm_fx_kernel_s16", act="S8")
    with pytest.raises(EntryError, match="bias_data is 'const int64_t \\*' but the bias dtype is S32"):
        resolve_entry("FxOp", "arm_fx_kernel_s16", activation_dtype="S16", weight_dtype="S8", cpu="cortex-m4",
                      desc={"name": "x"}, extra_roles={"bias": "S32"}, contracts=CONTRACTS)


def test_contract_entry_without_a_contract_fails_closed(bound) -> None:
    absent = ContractSet(status=STATUS_ABSENT, root=None, path=None)
    with pytest.raises(EntryError, match="has no Tests/KernelContracts/kernel_contracts.json"):
        _resolve("arm_fx_kernel_s16", contracts=absent)


def test_entry_scratch_bytes() -> None:
    assert entry_scratch_bytes("NONE", "x") == 0 and entry_scratch_bytes(0, "x") == 0


def test_entry_sizer_gates_the_case_like_the_entry() -> None:
    assert _required_kernel_symbols({"entry": "arm_fx_nosizer_s16", "entry_sizer": "arm_fx_other_get_buffer_size"}) == [
        "arm_fx_nosizer_s16", "arm_fx_other_get_buffer_size"]
