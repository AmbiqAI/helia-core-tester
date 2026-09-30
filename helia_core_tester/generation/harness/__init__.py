"""The generic test harness: an operator describes the values its case carries as an
ArgumentPool, and one template pair renders any public kernel from the pool and the
kernel contract (iteration 1b)."""

from helia_core_tester.generation.harness.model import (
    ArgumentPool,
    ArrayLiteral,
    Declaration,
    FaultEdit,
    GuardedBuffer,
    OutputSlot,
    HarnessError,
    HarnessInput,
    Provider,
    RuleCheck,
    SizeQuery,
    render_declaration,
)
from helia_core_tester.generation.harness.plan import HarnessPlan, plan_harness

__all__ = [
    "ArgumentPool", "ArrayLiteral", "Declaration", "FaultEdit", "GuardedBuffer", "HarnessError", "HarnessInput", "HarnessPlan", "OutputSlot", "Provider", "RuleCheck", "SizeQuery",
    "plan_harness", "render_declaration",
]
