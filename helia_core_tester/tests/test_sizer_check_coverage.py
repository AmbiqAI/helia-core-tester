"""What keeps the #133 sizer checks honest across the templates that carry them.

Two assumptions hold the scope of that change together, and neither is visible in the
generated C. The first is that max pooling has no sizer, which is why its template still
carries the pre-#133 capacity comparison and why that is harmless. The second is that
every template which does call a sizer routes the answer through the checks rather than
past them. Both are pinned here, because a change elsewhere could quietly undo either.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


def _templates_dir() -> Path:
    from helia_core_tester.core.discovery import find_tester_templates_dir

    return Path(find_tester_templates_dir())


MAX_POOL_DTYPES = ["S8", "S16", "FP32", "FP16"]

# Templates that call a scratch sizer and must route the answer through the checks.
# max_pool is deliberately absent; see test_max_pool_still_has_no_sizer_to_check.
SIZER_TEMPLATES = [
    "common/harness/harness.c.j2",
    "SVDFunctions/svdf/svdf.c.j2",
    "SVDFunctions/svdf/svdf_f32.c.j2",
]


@pytest.mark.parametrize("activation_dtype", MAX_POOL_DTYPES)
def test_max_pool_still_has_no_sizer_to_check(activation_dtype: str) -> None:
    """Max pooling renders through the generic harness with no scratch buffer at all.

    Its generation-time capacity is zero, so its pool declares no scratch buffer, and the
    harness refuses a case that queries scratch without one. If anyone ever gives max
    pooling a sizer, this test fails first and points them at the capacity it needs.
    """
    from helia_core_tester.generation.ops.PoolingFunctions.max_pool import OpMaxPool

    desc = {
        "operator": "MaxPool",
        "name": f"probe_max_pool_{activation_dtype.lower()}",
        "activation_dtype": activation_dtype,
        "tensor_dtypes": {"input": activation_dtype, "output": activation_dtype},
    }
    op = OpMaxPool(desc, seed=0, target_cpu="cortex-m55")
    kernel_info = op._select_cmsis_pooling_kernel()
    assert kernel_info["kernel_get_buffer_size_fn"] is None, (
        f"max pooling {activation_dtype} now has a sizer "
        f"({kernel_info['kernel_get_buffer_size_fn']}); give max pooling a scratch capacity "
        f"(calculate_pooling_buffer_size_max) so its pool declares a scratch buffer, then "
        f"remove this assertion."
    )


@pytest.mark.parametrize("relpath", SIZER_TEMPLATES)
def test_a_template_that_calls_a_sizer_checks_the_answer(relpath: str) -> None:
    """No sizer-calling template may keep a bare capacity comparison.

    The pre-#133 shape is `<something>_buffer_size > <SOMETHING>_MAX`, silently accepting a
    negative answer because a negative number is not larger than anything. Finding that
    shape in a template that also calls a sizer means an answer is reaching a context size
    without going through the checks.
    """
    text = (_templates_dir() / relpath).read_text()
    assert "HELIA_VALIDATE_SIZER" in text, f"{relpath} is listed as a sizer template but checks nothing"
    bare_capacity = re.compile(r"if\s*\(\s*\w*buffer_size\w*\s*>\s*\{\{[^}]*\}\}[A-Z_]*MAX", re.IGNORECASE)
    assert not bare_capacity.search(text), (
        f"{relpath} still compares a sizer answer against a capacity directly. That shape "
        f"accepts the negative out-of-range sentinel; route it through "
        f"HELIA_VALIDATE_SIZER_FITS instead."
    )
