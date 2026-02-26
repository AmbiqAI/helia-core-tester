from helia_core_tester.reporting.gap_gate import ALLOWLIST_DESCRIPTOR_NAMES, ALLOWLIST_OPERATORS


def test_allowlist_removed_closed_ops():
    assert "batch_to_space_nd_basic_s8" not in ALLOWLIST_DESCRIPTOR_NAMES
    assert "space_to_depth_block2_s8" not in ALLOWLIST_DESCRIPTOR_NAMES
    assert "split_channels_pairs_s8" not in ALLOWLIST_DESCRIPTOR_NAMES
    assert "unpack_axis1_triple_s8" not in ALLOWLIST_DESCRIPTOR_NAMES
    assert "BatchToSpaceND" not in ALLOWLIST_OPERATORS
    assert "Split" not in ALLOWLIST_OPERATORS
