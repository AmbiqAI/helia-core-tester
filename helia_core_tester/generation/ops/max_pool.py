"""Max pool operator wrapper."""

from helia_core_tester.generation.ops._pool_base import PoolFamilyBase


class OpMaxPool(PoolFamilyBase):
    """Generate tests for the CMSIS-NN max pool APIs."""

    POOL_KIND = "MAX"
    OPERATOR_NAME = "MaxPool"
    TEMPLATE_DIR = "max_pool"
    TEMPLATE_SUFFIX = "max_pool"

