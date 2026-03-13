"""Precise hard-swish operator wrapper."""

from helia_core_tester.generation.ops._hard_swish_base import HardSwishFamilyBase


class OpHardSwishPrecise(HardSwishFamilyBase):
    """Generate tests for `arm_hard_swish_precise_*` kernels."""

    VARIANT = "precise"
    OPERATOR_NAME = "HardSwishPrecise"

