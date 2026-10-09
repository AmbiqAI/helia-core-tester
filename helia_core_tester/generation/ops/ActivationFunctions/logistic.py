"""
Logistic (Sigmoid) operation implementation.
"""

from helia_core_tester.generation.ops._shared.lut_activation_s16 import LutActivationS16Base


class OpLogistic(LutActivationS16Base):
    """
    Logistic (Sigmoid) operation.
    """

    OPERATOR_NAME = "Logistic"
    STEM = "logistic"
    LOGISTIC = True
