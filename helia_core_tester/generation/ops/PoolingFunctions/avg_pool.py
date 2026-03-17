"""Average pool operator wrapper."""

from helia_core_tester.generation.ops._shared.pool_base import PoolFamilyBase


class OpAvgPool(PoolFamilyBase):
    """Generate tests for the CMSIS-NN average pool APIs."""

    POOL_KIND = "AVERAGE"
    OPERATOR_NAME = "AvgPool"
    TEMPLATE_DIR = "PoolingFunctions/avg_pool"
    TEMPLATE_SUFFIX = "avg_pool"
