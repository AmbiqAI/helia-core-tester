"""generation/kernel_gate.py: gating a case on the signature export and the implementation manifest."""

from __future__ import annotations

from pathlib import Path

import pytest

from helia_core_tester.contract.ir import (
    STATUS_ABSENT,
    STATUS_PRESENT,
    ContractSet,
    FunctionDecl,
    ImplementationDecl,
    load_contract_set,
    load_contracts_package,
)
from helia_core_tester.generation.kernel_gate import (
    BUILD_DECIDED_FEATURES,
    CPU_DECIDED_FEATURES,
    NO_IMPL,
    cpu_features,
    gate_kernel_symbols,
    unmet_features,
)
from helia_core_tester.tests.contract_checkout import function, implementation, write_checkout
from helia_core_tester.tests.contract_fallback import FIXTURE_ROOT


def _decl(name: str) -> FunctionDecl:
    return FunctionDecl(name=name, header="Include/arm_nnfunctions.h", line=1, guards=(),
                        returns="arm_cmsis_nn_status", params=())


def _contracts(*impls: ImplementationDecl, symbols=("arm_k_s8", "arm_k_s8_get_buffer_size"),
               library_version="7.47.0") -> ContractSet:
    return ContractSet(status=STATUS_PRESENT, root=None, path=None,
                       functions={name: _decl(name) for name in symbols},
                       implementations={impl.id: impl for impl in impls}, library_version=library_version)


def _impl(impl_id="k_s8", entry="arm_k_s8", requires=(), since="7.0.0", wrapper=None,
          declines_on_build=False) -> ImplementationDecl:
    return ImplementationDecl(id=impl_id, entry=entry, wrapper=wrapper, requires=frozenset(requires), since=since,
                              declines_on_build=declines_on_build)


@pytest.mark.parametrize("cpu, expected", [
    ("cortex-m55", {"mvei", "mvef", "mve_f16", "dsp", "f32_api", "f16_api"}),
    ("cortex-m55-dsp", {"dsp", "f32_api", "f16_api"}),
    ("cortex-m4", {"dsp", "f32_api"}),
    ("cortex-m0", {"f32_api"}),
])
def test_cpu_features_follow_the_cpu_profile(cpu: str, expected: set[str]) -> None:
    assert cpu_features(cpu) == expected


def test_cpu_features_reject_an_unknown_cpu() -> None:
    with pytest.raises(ValueError, match="Unsupported CPU target"):
        cpu_features("cortex-m85")


def test_every_manifest_token_of_the_reader_is_mapped() -> None:
    vocabulary = load_contracts_package(FIXTURE_ROOT).vocabulary
    assert set(vocabulary.BASE_FEATURES) == CPU_DECIDED_FEATURES | BUILD_DECIDED_FEATURES
    assert not CPU_DECIDED_FEATURES & BUILD_DECIDED_FEATURES
    for token in vocabulary.FEATURES:
        unmet_features([token], cpu_features("cortex-m55"))


@pytest.mark.parametrize("requires, cpu, unmet", [
    ((), "cortex-m0", ()),
    (("mvei", "dsp"), "cortex-m55", ()),
    (("mvei", "dsp"), "cortex-m4", ("mvei",)),
    (("mvei", "dsp"), "cortex-m0", ("dsp", "mvei")),
    (("!mvei",), "cortex-m55", ("!mvei",)),
    (("!mvei",), "cortex-m55-dsp", ()),
    (("!dsp",), "cortex-m0", ()),
    (("mve_f16",), "cortex-m55-dsp", ("mve_f16",)),
    (("f16_api",), "cortex-m4", ("f16_api",)),
    (("!f16_api",), "cortex-m4", ()),
    (("!autovectorize", "!single_rounding"), "cortex-m0", ()),
])
def test_unmet_features(requires, cpu: str, unmet) -> None:
    assert unmet_features(requires, cpu_features(cpu)) == unmet


@pytest.mark.parametrize("token", ["neon", "!neon", "", "!", "MVEI"])
def test_an_unmapped_token_is_refused(token: str) -> None:
    with pytest.raises(ValueError, match="not one helia-core-tester maps"):
        unmet_features([token], cpu_features("cortex-m55"))


def test_an_absent_contract_does_not_decide() -> None:
    gate = gate_kernel_symbols(ContractSet(status=STATUS_ABSENT, root=None, path=None), ["arm_k_s8"], "cortex-m4")
    assert not gate.decided and gate.passes


def test_symbols_outside_the_export_are_missing_and_not_checked_further() -> None:
    contracts = _contracts(_impl(impl_id="gone", entry="arm_gone_s8", requires=("mvei",)))
    gate = gate_kernel_symbols(contracts, ["arm_gone_s8", "arm_k_s8"], "cortex-m0")
    assert gate.missing_symbols == ("arm_gone_s8",) and gate.unmet == () and gate.contradictions == ()


def test_unmet_requirements_name_the_record() -> None:
    gate = gate_kernel_symbols(_contracts(_impl(requires=("mvei", "!autovectorize"))), ["arm_k_s8"], "cortex-m4")
    assert [u.as_dict() for u in gate.unmet] == [{"symbol": "arm_k_s8", "implementation": "k_s8", "unmet": ["mvei"]}]
    assert not gate.passes


def test_an_entry_that_declines_on_the_build_still_runs() -> None:
    """MVE-only entries return ARM_CMSIS_NN_NO_IMPL_ERROR elsewhere; a case expecting that decline must run."""
    contracts = _contracts(_impl(requires=("mvei", "!autovectorize"), declines_on_build=True))
    gate = gate_kernel_symbols(contracts, ["arm_k_s8"], "cortex-m0", expected_status=NO_IMPL)
    assert gate.passes and gate.unmet == ()
    assert [u.as_dict() for u in gate.declined] == [{"symbol": "arm_k_s8", "implementation": "k_s8", "unmet": ["mvei"]}]
    assert gate_kernel_symbols(contracts, ["arm_k_s8"], "cortex-m55", expected_status=NO_IMPL).declined == ()


@pytest.mark.parametrize("expected", ["ARM_CMSIS_NN_SUCCESS", "ARM_CMSIS_NN_ARG_ERROR"])
def test_a_case_expecting_more_than_the_decline_is_skipped_where_the_entry_declines(expected: str) -> None:
    contracts = _contracts(_impl(requires=("!mvei",), declines_on_build=True))
    gate = gate_kernel_symbols(contracts, ["arm_k_s8"], "cortex-m55", expected_status=expected)
    assert [u.unmet for u in gate.unmet] == [("!mvei",)] and gate.declined == () and not gate.passes
    assert gate_kernel_symbols(contracts, ["arm_k_s8"], "cortex-m4", expected_status=expected).passes


def test_every_record_of_a_symbol_must_hold() -> None:
    contracts = _contracts(_impl(impl_id="a", requires=("dsp",)), _impl(impl_id="b", requires=("mvei",)))
    gate = gate_kernel_symbols(contracts, ["arm_k_s8"], "cortex-m4")
    assert [u.implementation for u in gate.unmet] == ["b"]


def test_a_route_record_does_not_gate_a_direct_call_of_its_wrapper() -> None:
    contracts = _contracts(_impl(impl_id="to_conv", entry="arm_k_s8", wrapper="arm_k_s8", requires=("mvei",)))
    assert gate_kernel_symbols(contracts, ["arm_k_s8"], "cortex-m0").passes


def test_a_routed_kernel_is_gated_on_its_own_requires_only() -> None:
    contracts = _contracts(_impl(impl_id="routed", entry="arm_k_s8", wrapper="arm_k_wrapper_s8", requires=()))
    assert gate_kernel_symbols(contracts, ["arm_k_s8"], "cortex-m0").passes


@pytest.mark.parametrize("since, version", [("7.48.0", "7.47.0"), ("8.0.0", "7.99.99"), ("7.47.1", "7.47.0")])
def test_a_record_newer_than_the_checkout_is_a_contradiction(since: str, version: str) -> None:
    gate = gate_kernel_symbols(_contracts(_impl(since=since), library_version=version), ["arm_k_s8"], "cortex-m55")
    assert gate.contradictions and f"since {since}, after the checkout's library version {version}" in \
        gate.contradictions[0]
    assert not gate.passes


@pytest.mark.parametrize("since, version", [("7.47.0", "7.47.0"), ("7.9.0", "7.10.0"), ("0.0.0", "0.0.0")])
def test_a_record_at_or_before_the_checkout_holds(since: str, version: str) -> None:
    assert gate_kernel_symbols(_contracts(_impl(since=since), library_version=version), ["arm_k_s8"],
                               "cortex-m55").passes


@pytest.mark.parametrize("since, version", [("7.47", "7.47.0"), ("7.47.0", None), ("x.y.z", "7.47.0")])
def test_an_unreadable_version_is_a_contradiction(since: str, version) -> None:
    gate = gate_kernel_symbols(_contracts(_impl(since=since), library_version=version), ["arm_k_s8"], "cortex-m55")
    assert gate.contradictions and not gate.passes


def test_an_unmapped_token_in_a_record_is_a_contradiction_not_a_skip() -> None:
    gate = gate_kernel_symbols(_contracts(_impl(requires=("helium2",))), ["arm_k_s8"], "cortex-m55")
    assert gate.unmet == () and "helium2" in gate.contradictions[0]


def test_the_gate_reads_a_real_manifest(tmp_path: Path) -> None:
    root = write_checkout(tmp_path, [function("arm_k_s8"), function("arm_k_wrapper_s8"), function("arm_d_s8")],
                          implementations=[
        implementation("k_s8", "arm_k_s8", requires=["mvei", "!autovectorize"], since="7.37.0"),
        implementation("d_s8", "arm_d_s8", requires=["mvei"], declines=["build", "layer"]),
    ])
    contracts = load_contract_set(root)
    assert gate_kernel_symbols(contracts, ["arm_k_wrapper_s8", "arm_k_s8"], "cortex-m55").passes
    gate = gate_kernel_symbols(contracts, ["arm_k_wrapper_s8", "arm_k_s8"], "cortex-m55-dsp")
    assert [u.as_dict() for u in gate.unmet] == [{"symbol": "arm_k_s8", "implementation": "k_s8", "unmet": ["mvei"]}]
    declining = gate_kernel_symbols(contracts, ["arm_d_s8"], "cortex-m55-dsp", expected_status=NO_IMPL)
    assert declining.passes and [u.implementation for u in declining.declined] == ["d_s8"]


def test_the_shipped_mve_entries_decline_rather_than_skip() -> None:
    """The pilot manifest's MVE-only direct entries decline on other builds, so their decline cases keep running."""
    contracts = load_contract_set(FIXTURE_ROOT)
    for entry in ("arm_depthwise_conv_s8_opt_3x3", "arm_depthwise_conv_s8_opt_3x3_c64_s1",
                  "arm_depthwise_conv_s8_opt_planar", "arm_fully_connected_per_channel_packed_s8"):
        for cpu in ("cortex-m0", "cortex-m4", "cortex-m55-dsp"):
            gate = gate_kernel_symbols(contracts, [entry], cpu, expected_status=NO_IMPL)
            assert gate.passes and gate.declined, (entry, cpu)
            assert not gate_kernel_symbols(contracts, [entry], cpu).passes, (entry, cpu)
