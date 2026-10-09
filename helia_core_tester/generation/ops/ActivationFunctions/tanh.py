"""Tanh operator (arm_tanh_s16)."""

from helia_core_tester.generation.ops._shared.tanh_logistic_base import TanhLogisticBase


class OpTanh(TanhLogisticBase):
    OPERATOR = "Tanh"
    ENTRY = "tanh"
