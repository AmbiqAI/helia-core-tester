from helia_core_tester.core.cpu_targets import (
    get_cpu_profile,
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
