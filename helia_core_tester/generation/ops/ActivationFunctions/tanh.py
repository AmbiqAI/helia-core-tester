"""
Tanh operation implementation.
"""

from helia_core_tester.generation.ops._shared.lut_activation_s16 import LutActivationS16Base


class OpTanh(LutActivationS16Base):
    """
    Tanh operation.
    """

    OPERATOR_NAME = "Tanh"
    STEM = "tanh"
    LOGISTIC = False
