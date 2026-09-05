"""The dims-taking float call path for sub/add/mul (ns-cmsis-nn#415).

`hint: call_style: broadcast` selects arm_elementwise_{sub,add,mul}_broadcast_{f32,f16}
and the templates emit the dims-taking call; sub also takes it from two differing shapes
alone, since its flat float path has no dims to describe them. Everything else must
stay on the flat kernel byte for byte -- add/mul un-hinted mismatched shapes included,
because add_float_{channel,scalar}_broadcast_* pin the materialised-broadcast flat call.
"""

from __future__ import annotations

import re
from pathlib import Path

import jinja2
import numpy as np
import pytest

from helia_core_tester.core.discovery import find_descriptors_dir
from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.ops.BasicMathFunctions.add import OpAdd
from helia_core_tester.generation.ops.BasicMathFunctions.mul import OpMul
from helia_core_tester.generation.ops.BasicMathFunctions.sub import OpSub
from helia_core_tester.generation.utils.template_context import TemplateContextBuilder
import helia_core_tester.generation.test_ops as generation_module

_OPS = {"sub": (OpSub, "Sub"), "add": (OpAdd, "Add"), "mul": (OpMul, "Mul")}
_CPU = "cortex-m55"
_ARRAY_RE = re.compile(r"static const (\w+) (\w+)\[\] = \{([^}]*)\}")
_CAST_RE = re.compile(r"^\(\s*\w+\s*\)\s*")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _render(template_name: str, context: dict[str, object]) -> str:
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_repo_root() / "assets" / "templates")),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    render_context = TemplateContextBuilder.build_validation_context(template_name, context)
    return env.get_template(template_name).render(**render_context)


def _desc(op: str, name: str, shape_1, shape_2, dtype: str = "FP32", hint: dict | None = None) -> dict:
    desc = {
        "operator": _OPS[op][1],
        "name": name,
        "suite": "float",
        "tensor_dtypes": {"input": dtype, "output": dtype},
        "input_1_shape": list(shape_1),
        "input_2_shape": list(shape_2),
    }
    if hint is not None:
        desc["hint"] = hint
    return desc


def _select(op: str, desc: dict) -> dict:
    instance = _OPS[op][0](desc, 1, target_cpu=_CPU)
    return getattr(instance, f"_select_cmsis_{op}_kernel")()


@pytest.mark.parametrize("op", sorted(_OPS))
@pytest.mark.parametrize("dtype,suffix", [("FP32", "f32"), ("FP16", "f16")])
def test_hint_selects_the_broadcast_entry_point(op: str, dtype: str, suffix: str) -> None:
    info = _select(op, _desc(op, f"{op}_x", [1, 2, 3, 4], [1, 1, 1, 4], dtype, {"call_style": "broadcast"}))
    assert info["kernel_fn"] == f"arm_elementwise_{op}_broadcast_{suffix}"
    assert info["float_kernel"] and info["float_broadcast"]


@pytest.mark.parametrize("op", sorted(_OPS))
@pytest.mark.parametrize("dtype,suffix", [("FP32", "f32"), ("FP16", "f16")])
def test_equal_shapes_without_hint_keep_the_flat_kernel(op: str, dtype: str, suffix: str) -> None:
    info = _select(op, _desc(op, f"{op}_x", [1, 4, 4, 8], [1, 4, 4, 8], dtype))
    assert info["kernel_fn"] == f"arm_elementwise_{op}_{suffix}"
    assert not info["float_broadcast"]


def test_sub_takes_the_broadcast_kernel_from_two_shapes_alone() -> None:
    # The flat sub call has no dims, so two shapes have nowhere else to go.
    info = _select("sub", _desc("sub", "sub_x", [1, 2, 3, 4], [1, 1, 1, 4]))
    assert info["kernel_fn"] == "arm_elementwise_sub_broadcast_f32"


@pytest.mark.parametrize("op", ["add", "mul"])
def test_add_mul_unhinted_mismatch_keeps_the_materialised_flat_call(op: str) -> None:
    info = _select(op, _desc(op, f"{op}_x", [1, 3, 4, 5], [1, 1, 1, 5]))
    assert info["kernel_fn"] == f"arm_elementwise_{op}_f32"
    assert not info["float_broadcast"]


def test_legacy_fp16_add_refuses_the_broadcast_hint() -> None:
    desc = _desc("add", "add_x", [1, 1, 1, 9], [1, 1, 1, 9], "FP16", {"kernel_variant": "legacy_fp16", "call_style": "broadcast"})
    with pytest.raises(ValueError, match="legacy_fp16"):
        _select("add", desc)


@pytest.mark.parametrize("op", sorted(_OPS))
def test_templates_emit_the_dims_taking_call_only_when_asked(op: str) -> None:
    base = {
        "name": f"{op}_float_bcast_channel_f32",
        "input_dtype": "float",
        "output_dtype": "float",
        "float_kernel": True,
        "block_size": 24,
        "out_activation_min_literal": "-1.0e+30f",
        "out_activation_max_literal": "1.0e+30f",
        "output_dims": {"n": 1, "h": 2, "w": 3, "c": 4},
    }
    template = f"BasicMathFunctions/{op}/{op}.c.j2"
    flat = _render(template, {**base, "kernel_fn": f"arm_elementwise_{op}_f32"})
    broadcast = _render(
        template, {**base, "kernel_fn": f"arm_elementwise_{op}_broadcast_f32", "float_broadcast": True}
    )

    assert f"arm_elementwise_{op}_f32(" in flat
    assert "_input1_dims" not in flat and "24\n" in flat

    assert f"arm_elementwise_{op}_broadcast_f32(" in broadcast
    for role in ("input1", "input2", "output"):
        assert f"&{op}_float_bcast_channel_f32_{role}_dims" in broadcast
    call = broadcast[broadcast.index("kernel_status ="):broadcast.index(");")]
    assert "24" not in call, "the broadcast entry point takes no block_size"
    assert "-1.0e+30f" in call and "1.0e+30f" in call


def _parse_arrays(header: str) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for match in _ARRAY_RE.finditer(header):
        values = [float(_CAST_RE.sub("", tok.strip()).rstrip("fF")) for tok in match.group(3).split(",") if tok.strip()]
        arrays[match.group(2)] = np.array(values, dtype=np.float64)
    return arrays


@pytest.fixture(scope="module")
def checked_in_broadcast_cases() -> dict[str, dict]:
    descriptors = load_all_descriptors(str(find_descriptors_dir()))
    cases = {d["name"]: d for d in descriptors if "_float_bcast_" in d["name"]}
    # Eleven shapes x three ops x two precisions.
    assert len(cases) == 66, sorted(cases)
    return cases


def test_checked_in_broadcast_cases_are_gated_on_their_kernel_symbol(
    checked_in_broadcast_cases: dict[str, dict],
) -> None:
    # One tester pin serves ns-cmsis-nn main and the #415 branch alike: on a checkout
    # whose headers do not declare the broadcast entry point the case must skip as
    # skipped_kernel_symbol (hct#92) rather than fail to compile.
    for name, desc in checked_in_broadcast_cases.items():
        op, suffix = name.split("_")[0], name.rsplit("_", 1)[1]
        assert desc.get("required_kernel_symbols") == [f"arm_elementwise_{op}_broadcast_{suffix}"], name


@pytest.mark.parametrize(
    "case",
    [
        "sub_float_bcast_channel_left_f32",
        "sub_float_bcast_both_tail_f16",
        "add_float_bcast_scalar_f16",
        "add_float_bcast_batch_scalar_right_f32",
        "mul_float_bcast_height_f32",
        "mul_float_bcast_channel_scalar_right_f16",
        "sub_float_bcast_scalar_left_tail_f32",
        "add_float_bcast_channel_scalar_right_tail_f16",
    ],
)
def test_checked_in_broadcast_cases_generate_unmaterialised_operands(
    tmp_path: Path, checked_in_broadcast_cases: dict[str, dict], case: str
) -> None:
    desc = checked_in_broadcast_cases[case]
    op = case.split("_")[0]
    generation_module.generate_test(desc, str(tmp_path), cpu=_CPU)
    case_dir = tmp_path / desc["_family"] / case
    source = (case_dir / f"{case}_{op}.c").read_text()
    header = (case_dir / "includes" / f"{case}_{op}.h").read_text()

    suffix = "f16" if case.endswith("_f16") else "f32"
    assert f"arm_elementwise_{op}_broadcast_{suffix}(" in source
    assert f"&{case}_input1_dims" in source and f"&{case}_output_dims" in source

    arrays = _parse_arrays(header)
    shape_1, shape_2 = tuple(desc["input_1_shape"]), tuple(desc["input_2_shape"])
    input_1 = arrays[f"{case}_input1"]
    input_2 = arrays[f"{case}_input2"]
    # The operands are emitted at their own sizes: the kernel does the broadcast.
    assert input_1.size == int(np.prod(shape_1))
    assert input_2.size == int(np.prod(shape_2))

    combine = {"sub": np.subtract, "add": np.add, "mul": np.multiply}[op]
    dtype = np.float16 if suffix == "f16" else np.float32
    reference = combine(
        input_1.reshape(shape_1).astype(dtype).astype(np.float32),
        input_2.reshape(shape_2).astype(dtype).astype(np.float32),
    ).astype(dtype)
    golden = arrays[f"{case}_expected_output"]
    assert golden.size == reference.size
    # The f32 literal formatter keeps nine decimals, which is a last-bit print effect on
    # small-magnitude lanes; the FVP compare is tolerance-based for the same reason.
    np.testing.assert_allclose(golden, reference.astype(np.float64).ravel(), rtol=1e-6, atol=1e-8)
