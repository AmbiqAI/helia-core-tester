import pytest

from helia_core_tester.core.cpu_targets import (
    get_cpu_profile,
    known_capabilities,
    missing_required_capabilities,
    normalize_cpu,
    parse_cpu_list,
)


def test_normalize_cpu_aliases():
    assert normalize_cpu("m0") == "cortex-m0"
    assert normalize_cpu("m4") == "cortex-m4"
    assert normalize_cpu("m55") == "cortex-m55"


def test_parse_cpu_list_normalizes_and_deduplicates():
    assert parse_cpu_list("m0,cortex-m4,m55,m4") == ["cortex-m0", "cortex-m4", "cortex-m55"]


def test_get_cpu_profile_flags():
    assert get_cpu_profile("cortex-m0").has_mve is False
    assert get_cpu_profile("cortex-m4").has_dsp is True
    assert get_cpu_profile("cortex-m55").has_mve is True


def test_cpu_profiles_expose_float_capability_hooks():
    assert get_cpu_profile("cortex-m55").supports_execution_dtype("FP16") is True
    assert missing_required_capabilities("cortex-m4", ["fp32_execution"]) == []
    assert missing_required_capabilities("cortex-m55", ["fp16_execution"]) == []


def test_cortex_m0_runs_f32_but_not_f16():
    profile = get_cpu_profile("cortex-m0")
    assert profile.supports_execution_dtype("FP32") is True
    assert profile.supports_execution_dtype("FP16") is False
    assert missing_required_capabilities("cortex-m0", ["fp32_execution"]) == []
    assert missing_required_capabilities("cortex-m0", ["fp16_execution"]) == ["fp16_execution"]


def test_cortex_m0_claims_no_simd_capabilities():
    profile = get_cpu_profile("cortex-m0")
    assert profile.supports_capability("dsp") is False
    assert profile.supports_capability("mve") is False


def test_soft_float_is_a_cortex_m0_only_capability():
    """Kernel behaviour guarded on `#if !defined(__ARM_FP)` exists only on the
    soft-float leg; every hard-float target must capability-skip such a case."""
    assert get_cpu_profile("cortex-m0").supports_capability("soft_float") is True
    assert missing_required_capabilities("cortex-m0", ["soft_float"]) == []
    for cpu in ("cortex-m4", "cortex-m55", "cortex-m55-dsp"):
        assert get_cpu_profile(cpu).supports_capability("soft_float") is False
        assert missing_required_capabilities(cpu, ["soft_float"]) == ["soft_float"]


def test_an_undeclared_required_capability_is_a_generation_error():
    """A misspelled capability satisfies no profile, so without this it would skip
    the descriptor on every target and still report as covered."""
    for cpu in ("cortex-m0", "cortex-m55"):
        with pytest.raises(ValueError, match="Unknown required capability"):
            missing_required_capabilities(cpu, ["soft-float"])

    assert known_capabilities() >= {"dsp", "mve", "fp32_execution", "fp16_execution", "soft_float"}
