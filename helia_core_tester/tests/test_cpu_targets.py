from helia_core_tester.core.cpu_targets import get_cpu_profile, normalize_cpu, parse_cpu_list


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