"""Logistic (sigmoid) operator (arm_logistic_s16)."""

from helia_core_tester.generation.ops._shared.tanh_logistic_base import TanhLogisticBase


class OpLogistic(TanhLogisticBase):
    OPERATOR = "Logistic"
    ENTRY = "logistic"
