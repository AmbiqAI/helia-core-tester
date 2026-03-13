"""Compat hard-swish operator wrapper."""

from helia_core_tester.generation.ops._hard_swish_base import HardSwishFamilyBase


class OpHardSwishCompat(HardSwishFamilyBase):
    """Generate tests for `arm_hard_swish_compat_s8`."""

    VARIANT = "compat"
    OPERATOR_NAME = "HardSwishCompat"

