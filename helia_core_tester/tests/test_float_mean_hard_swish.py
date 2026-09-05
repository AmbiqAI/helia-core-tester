"""Float Mean / HardSwish operator support and per-symbol codegen gating.

Covers the tester-side support for the ns-cmsis-nn float kernels
arm_nn_mean_f32 (#414), arm_nn_mean_f16 (#412), and
arm_hard_swish_f32/f16 (#413), plus the per-symbol codegen probe that
lets one tester pin serve checkouts that declare only a subset of those
kernels (each in-flight ns-cmsis-nn branch, and main with none).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

import helia_core_tester.generation.test_ops as generation_module
from helia_core_tester.generation.io.descriptors import load_descriptor
from helia_core_tester.generation.ops._shared.hard_swish_base import HardSwishFamilyBase
from helia_core_tester.generation.ops.ActivationFunctions.hard_swish_precise import OpHardSwishPrecise
from helia_core_tester.generation.ops.BasicMathFunctions.mean import OpMean
from helia_core_tester.generation.utils.temp_sizer_probe import missing_header_symbols

TESTER_ROOT = Path(__file__).resolve().parents[2]


def _fake_cmsis_nn_root(
    tmp_path: Path, header_text: str, source_symbols: tuple[str, ...] = ()
) -> Path:
    root = tmp_path / "ns-cmsis-nn-fake"
    include = root / "Include"
    include.mkdir(parents=True, exist_ok=True)
    (include / "arm_nnfunctions.h").write_text(header_text)
    for symbol in source_symbols:
        source_dir = root / "Source" / "BasicMathFunctions"
        source_dir.mkdir(parents=True, exist_ok=True)
        (source_dir / f"{symbol}.c").write_text(f"/* {symbol} kernel */\n")
    return root


def _parse_c_float_array(header_text: str, array_name: str) -> np.ndarray:
    match = re.search(
        rf"{re.escape(array_name)}\[\] = \{{(.*?)\}};",
        header_text,
        re.DOTALL,
    )
    assert match, f"array {array_name} not found"
    body = match.group(1).replace("(float16_t)", "")
    values = [float(tok.rstrip("f")) for tok in re.findall(r"[-+0-9.eE]+f", body)]
    assert values, f"array {array_name} is empty"
    return np.asarray(values, dtype=np.float64)


# ---------------------------------------------------------------------------
# Per-symbol probe
# ---------------------------------------------------------------------------


def test_missing_header_symbols_reports_only_undeclared_symbols(tmp_path: Path) -> None:
    root = _fake_cmsis_nn_root(
        tmp_path,
        """
        arm_cmsis_nn_status arm_nn_mean_f32(const float32_t *input_data,
                                            const cmsis_nn_dims *input_dims,
                                            const cmsis_nn_dims *axis_dims,
                                            float32_t *output_data,
                                            const cmsis_nn_dims *output_dims);
        /* doc mention only: arm_hard_swish_f32(input, output, size) */
        // line mention only: arm_nn_mean_f16(...)
        """,
    )

    assert missing_header_symbols(["arm_nn_mean_f32"], cmsis_nn_root=root) == []
    # Comment-form mentions must not look like declarations.
    assert missing_header_symbols(
        ["arm_hard_swish_f32", "arm_nn_mean_f16"], cmsis_nn_root=root
    ) == ["arm_hard_swish_f32", "arm_nn_mean_f16"]
    # Mixed: only the undeclared symbol is reported, input order preserved.
    assert missing_header_symbols(
        ["arm_hard_swish_f16", "arm_nn_mean_f32"], cmsis_nn_root=root
    ) == ["arm_hard_swish_f16"]


def test_missing_header_symbols_treats_unresolvable_checkout_as_all_missing(tmp_path: Path) -> None:
    assert missing_header_symbols(
        ["arm_nn_mean_f32"], cmsis_nn_root=tmp_path / "does-not-exist"
    ) == ["arm_nn_mean_f32"]


# ---------------------------------------------------------------------------
# Generation-loop gating
# ---------------------------------------------------------------------------


def _float_mean_descriptor() -> dict:
    return {
        "name": "mean_float_axis_c_f32",
        "operator": "Mean",
        "suite": "float",
        "tensor_dtypes": {"input": "FP32", "output": "FP32"},
        "resolved_tensor_dtypes": {"input": "FP32", "output": "FP32"},
        "input_shape": [1, 3, 4, 5],
        "axes": [3],
        "required_kernel_symbols": ["arm_nn_mean_f32"],
        "_family": "BasicMathFunctions",
        "_parity_kind": "cmsis",
        "_source_family": "BasicMathFunctions",
        "_source_stem": "mean_float",
        "_source_relpath": "BasicMathFunctions/mean_float.yaml",
        "_descriptor_suite": "float",
    }


def _filters(generated_tests_dir: Path) -> dict:
    return {
        "op": None,
        "dtype": None,
        "wtype": None,
        "name": None,
        "limit": None,
        "seed": 123,
        "cpu": "cortex-m55",
        "suite": "float",
        "float_precision": "both",
        "generated_tests_dir": str(generated_tests_dir),
    }


def _run_generation(tmp_path, monkeypatch, cmsis_nn_root: Path):
    repo_root = tmp_path / "repo"
    generated_tests_dir = repo_root / "artifacts" / "generated_tests" / "float" / "cortex-m55"

    monkeypatch.setattr(generation_module, "find_repo_root", lambda: repo_root)
    monkeypatch.setattr(
        generation_module, "find_descriptors_dir", lambda: repo_root / "assets" / "descriptors"
    )
    monkeypatch.setattr(
        generation_module, "load_all_descriptors", lambda _path: [_float_mean_descriptor()]
    )
    monkeypatch.setenv("CMSIS_NN_ROOT", str(cmsis_nn_root))

    generated: list[str] = []

    def _fake_generate_test(desc, out_dir, seed=None, cpu="cortex-m55", conversion_failures=None, generation_failures=None):
        generated.append(desc["name"])
        test_dir = Path(out_dir) / desc["_family"] / desc["name"]
        test_dir.mkdir(parents=True, exist_ok=True)
        (test_dir / f"{desc['name']}.tflite").write_bytes(b"\x01")
        (test_dir / f"{desc['name']}_mean.c").write_text("// fake\n")

    monkeypatch.setattr(generation_module, "generate_test", _fake_generate_test)
    generation_module.test_generation(_filters(generated_tests_dir))
    report_dir = repo_root / "artifacts" / "reports" / "generation" / "float" / "cortex-m55"
    summary = json.loads((report_dir / "generation_summary.json").read_text())
    manifest = json.loads((generated_tests_dir / "manifest.json").read_text())
    return generated, summary, manifest, generated_tests_dir


def test_generation_skips_descriptor_when_kernel_symbol_is_undeclared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _fake_cmsis_nn_root(
        tmp_path, "/* no float mean here */\n", source_symbols=("arm_mean_s8",)
    )
    generated, summary, manifest, generated_tests_dir = _run_generation(tmp_path, monkeypatch, root)

    assert generated == []
    assert summary["status"] == "skipped_only"
    assert summary["counts"]["generated"] == 0
    assert summary["counts"]["skipped_capability"] == 0
    assert summary["counts"]["skipped_kernel_symbol"] == 1
    assert manifest["generated_count"] == 0
    assert manifest["skipped_count"] == 1
    entry = manifest["skipped"][0]
    assert entry["status"] == "skipped_kernel_symbol"
    assert entry["missing_kernel_symbols"] == ["arm_nn_mean_f32"]
    assert entry["required_kernel_symbols"] == ["arm_nn_mean_f32"]
    # The skipped descriptor must not reach tests.cmake (nothing to build).
    assert "mean_float_axis_c_f32" not in (generated_tests_dir / "tests.cmake").read_text()


def test_generation_generates_descriptor_when_kernel_symbol_is_declared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _fake_cmsis_nn_root(
        tmp_path,
        "arm_cmsis_nn_status arm_nn_mean_f32(const float32_t *input_data,\n"
        "                                    const cmsis_nn_dims *input_dims,\n"
        "                                    const cmsis_nn_dims *axis_dims,\n"
        "                                    float32_t *output_data,\n"
        "                                    const cmsis_nn_dims *output_dims);\n",
    )
    generated, summary, manifest, _ = _run_generation(tmp_path, monkeypatch, root)

    assert generated == ["mean_float_axis_c_f32"]
    assert summary["counts"]["generated"] == 1
    assert summary["counts"]["skipped_kernel_symbol"] == 0
    assert manifest["skipped_count"] == 0


def test_generation_fails_loudly_when_undeclared_symbol_has_kernel_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """hct#92 review backstop: declaration missing from the probed public
    headers while Source/**/<symbol>.c ships means the probe missed a
    declaration (renamed symbol / unprobed header). That must fail generation
    with a nonzero outcome, never silently skip with green CI."""
    root = _fake_cmsis_nn_root(
        tmp_path,
        "/* declaration lives in some unprobed header */\n",
        source_symbols=("arm_nn_mean_f32",),
    )

    # With every filtered descriptor contradicted, the run has zero generated
    # models and zero skips, so the "no generated tests" guard fires first;
    # either way the generation step exits nonzero and the failure report
    # below carries the loud per-descriptor message.
    with pytest.raises(
        AssertionError,
        match="No TFLite models were generated|Generation failures occurred",
    ):
        _run_generation(tmp_path, monkeypatch, root)

    repo_root = tmp_path / "repo"
    report_dir = repo_root / "artifacts" / "reports" / "generation" / "float" / "cortex-m55"
    failures = json.loads((report_dir / "generation_failures.json").read_text())
    assert len(failures) == 1
    assert failures[0]["name"] == "mean_float_axis_c_f32"
    assert failures[0]["stage"] == "kernel_symbol_probe"
    assert "arm_nn_mean_f32" in failures[0]["exception"]
    # The contradicted descriptor is a failure, not a skip.
    summary = json.loads((report_dir / "generation_summary.json").read_text())
    assert summary["counts"]["skipped_kernel_symbol"] == 0
    assert summary["counts"]["generation_failures"] == 1


def test_kernel_source_exists_matches_only_shipped_sources(tmp_path: Path) -> None:
    from helia_core_tester.generation.utils.temp_sizer_probe import kernel_source_exists

    root = _fake_cmsis_nn_root(tmp_path, "", source_symbols=("arm_nn_mean_f32",))
    assert kernel_source_exists("arm_nn_mean_f32", cmsis_nn_root=root) is True
    assert kernel_source_exists("arm_hard_swish_f32", cmsis_nn_root=root) is False
    assert kernel_source_exists("arm_nn_mean_f32", cmsis_nn_root=tmp_path / "nope") is False


# ---------------------------------------------------------------------------
# Descriptor contracts
# ---------------------------------------------------------------------------


def test_float_mean_descriptors_require_their_kernel_symbol() -> None:
    path = TESTER_ROOT / "assets" / "descriptors" / "BasicMathFunctions" / "mean_float.yaml"
    descriptors = load_descriptor(str(path))
    assert len(descriptors) == 8
    for desc in descriptors:
        assert desc["operator"] == "Mean"
        assert desc["suite"] == "float"
        dtype = desc["resolved_tensor_dtypes"]["input"]
        expected_symbol = {"FP32": "arm_nn_mean_f32", "FP16": "arm_nn_mean_f16"}[dtype]
        assert desc["required_kernel_symbols"] == [expected_symbol]


def test_float_hard_swish_descriptors_require_their_kernel_symbol() -> None:
    path = TESTER_ROOT / "assets" / "descriptors" / "ActivationFunctions" / "hard_swish_float.yaml"
    descriptors = load_descriptor(str(path))
    assert len(descriptors) == 6
    for desc in descriptors:
        assert desc["operator"] == "HardSwishPrecise"
        assert desc["suite"] == "float"
        dtype = desc["resolved_tensor_dtypes"]["input"]
        expected_symbol = {"FP32": "arm_hard_swish_f32", "FP16": "arm_hard_swish_f16"}[dtype]
        assert desc["required_kernel_symbols"] == [expected_symbol]


# ---------------------------------------------------------------------------
# Float Mean codegen
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dtype", "expected_kernel", "np_dtype"),
    [("FP32", "arm_nn_mean_f32", np.float32), ("FP16", "arm_nn_mean_f16", np.float16)],
)
def test_mean_float_generates_reduce_sum_shaped_call_and_mean_golden(
    dtype: str, expected_kernel: str, np_dtype, tmp_path: Path
) -> None:
    name = f"mean_float_case_{dtype.lower()}"
    desc = {
        "operator": "Mean",
        "name": name,
        "tensor_dtypes": {"input": dtype, "output": dtype},
        "input_shape": [1, 3, 4, 5],
        "axes": [1, 2],
    }
    op = OpMean(desc, seed=7, target_cpu="cortex-m55")
    (tmp_path / f"{name}.tflite").write_bytes(b"\x01")

    op.generate_c_files(tmp_path)

    c_text = (tmp_path / f"{name}_mean.c").read_text()
    h_text = (tmp_path / "includes" / f"{name}_mean.h").read_text()

    assert expected_kernel in c_text
    assert "FLOAT" in c_text  # float validation mode, never TOLERANT_INT
    assert f"&{name}_axis_dims" in c_text
    # Axis mask marks h and w; output dims keep reduced axes at 1.
    assert ".n = 0, .h = 1,\n    .w = 1, .c = 0" in h_text
    assert ".n = 1, .h = 1,\n    .w = 1, .c = 5" in h_text

    input_values = _parse_c_float_array(h_text, f"{name}_input")
    expected_values = _parse_c_float_array(h_text, f"{name}_expected_output")
    assert input_values.size == 60
    assert expected_values.size == 5

    reshaped = input_values.reshape(1, 3, 4, 5).astype(np_dtype)
    golden = (
        np.sum(reshaped.astype(np.float32), axis=(1, 2), keepdims=True) / np.float32(12)
    ).astype(np_dtype)
    np.testing.assert_allclose(
        expected_values.reshape(golden.shape),
        golden.astype(np.float64),
        rtol=1e-6,
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# Float HardSwish codegen
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dtype", "expected_kernel", "np_dtype"),
    [("FP32", "arm_hard_swish_f32", np.float32), ("FP16", "arm_hard_swish_f16", np.float16)],
)
def test_hard_swish_float_generates_size_call_and_contract_golden(
    dtype: str, expected_kernel: str, np_dtype, tmp_path: Path
) -> None:
    name = f"hard_swish_float_case_{dtype.lower()}"
    desc = {
        "operator": "HardSwishPrecise",
        "name": name,
        "tensor_dtypes": {"input": dtype, "output": dtype},
        "input_shape": [1, 4, 4, 8],
    }
    op = OpHardSwishPrecise(desc, seed=11, target_cpu="cortex-m55")
    (tmp_path / f"{name}.tflite").write_bytes(b"\x01")

    op.generate_c_files(tmp_path)

    c_text = (tmp_path / f"{name}_hard_swish.c").read_text()
    h_text = (tmp_path / "includes" / f"{name}_hard_swish.h").read_text()

    assert expected_kernel in c_text
    assert "FLOAT" in c_text
    # (input, output, size) call shape: no quantization parameters.
    assert "input_offset" not in c_text
    assert "output_multiplier" not in c_text

    input_values = _parse_c_float_array(h_text, f"{name}_input").astype(np_dtype)
    expected_values = _parse_c_float_array(h_text, f"{name}_expected_output").astype(np_dtype)
    assert input_values.size == 128

    # #413 contract: identity region (x >= 3) bit-exact, zero region exact 0.
    identity = input_values >= np_dtype(3.0)
    zero = input_values <= np_dtype(-3.0)
    assert identity.any() and zero.any(), "input must exercise both saturated regions"
    np.testing.assert_array_equal(expected_values[identity], input_values[identity])
    assert np.all(expected_values[zero] == np_dtype(0.0))

    # Curved region: the f32 serializer's %.9f literals round-trip values in
    # [1e-4, 1) to ~7 significant digits, so recomputing from the parsed
    # input can land 1 ulp away from the generated golden. The on-device
    # comparison is unaffected (the device compares against the same
    # literal); assert closeness here, exactness in the saturated regions
    # above.
    reference = input_values.astype(np.float64)
    golden = (reference * np.clip(reference + 3.0, 0.0, 6.0) / 6.0).astype(np_dtype)
    np.testing.assert_allclose(
        expected_values.astype(np.float64),
        golden.astype(np.float64),
        rtol=1e-6,
        atol=1e-8,
    )


def test_hard_swish_compat_rejects_float_dtypes() -> None:
    from helia_core_tester.generation.ops.ActivationFunctions.hard_swish_compat import (
        OpHardSwishCompat,
    )

    op = OpHardSwishCompat(
        {
            "operator": "HardSwishCompat",
            "name": "hard_swish_compat_fp32",
            "tensor_dtypes": {"input": "FP32", "output": "FP32"},
            "input_shape": [1, 1, 1, 4],
        },
        seed=1,
        target_cpu="cortex-m55",
    )
    with pytest.raises(NotImplementedError, match="only supported for S8"):
        op._select_cmsis_hard_swish_kernel()
