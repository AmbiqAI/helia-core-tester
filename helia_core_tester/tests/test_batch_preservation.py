"""Declared batches must reach both the model and the emitted kernel call."""

from copy import deepcopy
from pathlib import Path
import re

from ai_edge_litert.interpreter import Interpreter
import numpy as np
import pytest
import yaml

from helia_core_tester.generation.test_ops import generate_test
from helia_core_tester.generation.io.dtypes import resolve_comparison
from helia_core_tester.perf_stream.generated_test_bridge import (
    GeneratedTestCase,
    UnsupportedGeneratedTestError,
    build_case_bundle_from_generated_test,
)

ROOT = Path(__file__).resolve().parents[2]
NAMES = {
    "fully_connected_float_vector8_f32",
    "fully_connected_float_vector8_f16",
    "convolve_kernel1x1_stride_xy_case_01_s8",
    "depthwise_conv_mult_batches_s8",
    "convolve_dilation_golden_s8",
    "batch_matmul_batched_s8",
    "batch_matmul_batched_s16",
    "batch_matmul_float_batched_f32",
    "batch_matmul_float_batched_f16",
}
CASES = [
    (family, desc)
    for family in ("FullyConnectedFunctions", "ConvolutionFunctions")
    for path in sorted((ROOT / "assets/descriptors" / family).glob("*.yaml"))
    for desc in yaml.safe_load_all(path.read_text())
    if desc and desc.get("name") in NAMES
]
assert {desc["name"] for _, desc in CASES} == NAMES


def _generated_case(family, case):
    descriptor = yaml.safe_load((case / "descriptor.yaml").read_text())
    descriptor["resolved_comparison"] = resolve_comparison(
        descriptor, descriptor.get("resolved_tensor_dtypes")
    )
    suite = "float" if descriptor["activation_dtype"] in ("FP32", "FP16") else "int"
    return GeneratedTestCase(case.name, "cortex-m55", family, case, descriptor, suite)


# Exercise the same dilation/bias lowering interaction for both integer widths
# without adding descriptors to the generated coverage corpus.
for desc in yaml.safe_load_all(
    (ROOT / "assets/descriptors/ConvolutionFunctions/depthwise_conv.yaml").read_text()
):
    if desc and desc.get("name") in {
        "depthwise_conv_dilation_s8",
        "depthwise_conv_dilation_s16",
    }:
        desc = deepcopy(desc)
        desc["input_shape"][0] = 2
        CASES.append(("ConvolutionFunctions", desc))


@pytest.mark.parametrize("family,desc", CASES, ids=[d["name"] for _, d in CASES])
def test_declared_batches_reach_emitted_data(tmp_path, family, desc):
    generate_test(desc, str(tmp_path), seed=500)
    case = tmp_path / family / desc["name"]
    interpreter = Interpreter(model_path=str(case / f'{desc["name"]}.tflite'))
    interpreter.allocate_tensors()
    shapes = (
        [desc["input_shape"]]
        if "input_shape" in desc
        else [desc["input_1_shape"], desc["input_2_shape"]]
    )
    assert [d["shape"].tolist() for d in interpreter.get_input_details()] == shapes
    batch = shapes[0][0]
    output_shape = interpreter.get_output_details()[0]["shape"].tolist()
    assert output_shape[0] == batch

    header = "\n".join(p.read_text() for p in (case / "includes").glob("*.h"))

    def dims(role):
        body = re.search(
            rf'\b{desc["name"]}_{role}_dims\s*=\s*\{{([^}}]+)', header
        ).group(1)
        return {
            axis: int(value)
            for axis, value in re.findall(r"\.([nhwc])\s*=\s*(\d+)", body)
        }

    def array(role):
        body = re.search(
            rf'\b{desc["name"]}_{role}\[[^]]*\]\s*=\s*\{{([^}}]+)', header
        ).group(1)
        return [value.strip() for value in body.split(",") if value.strip()]

    roles = ["input"] if len(shapes) == 1 else ["input_lhs", "input_rhs"]
    for role, shape in zip(roles + ["expected_output"], shapes + [output_shape]):
        values = array(role)
        emitted_dims = dims("output" if role == "expected_output" else role)
        assert (
            len(values)
            == int(np.prod(shape))
            == int(np.prod(list(emitted_dims.values())))
        )
        assert (
            emitted_dims["n"] * emitted_dims["h"]
            if len(shape) == 3
            else emitted_dims["n"]
        ) == batch
        rows = np.asarray(values).reshape(batch, -1)
        # Reusing the first batch must not reproduce any subsequent input/golden.
        assert all(not np.array_equal(rows[0], row) for row in rows[1:])

    if desc.get("dilation") and desc.get("use_bias", True):
        assert any(int(value) != 0 for value in array("biases"))

    if desc["name"] in {
        "convolve_kernel1x1_stride_xy_case_01_s8",
        "depthwise_conv_mult_batches_s8",
    }:
        generated = _generated_case(family, case)
        # The bridge intentionally rejects multi-batch Conv/Depthwise. Correct
        # headers must reach that guard, not silently serialize only batch zero.
        with pytest.raises(UnsupportedGeneratedTestError, match="batch size 2 > 1"):
            build_case_bundle_from_generated_test(
                ROOT,
                generated,
                output_root=tmp_path / "bridge",
                require_fvp_pass=False,
            )

    if desc["operator"] == "BatchMatMul":
        generated = _generated_case(family, case)
        if desc["activation_dtype"] in ("S8", "S16"):
            with pytest.raises(
                UnsupportedGeneratedTestError, match="quantized BatchMatMul.*batch"
            ):
                build_case_bundle_from_generated_test(
                    ROOT,
                    generated,
                    output_root=tmp_path / "bridge",
                    require_fvp_pass=False,
                )
            assert not (tmp_path / "bridge").exists()
        else:
            bundle = build_case_bundle_from_generated_test(
                ROOT, generated, output_root=tmp_path / "bridge", require_fvp_pass=False
            )
            assert len(bundle.blobs) == 3
            for blob, role in zip(
                bundle.blobs, ["input_lhs", "input_rhs", "expected_output"]
            ):
                emitted = dims("output" if role == "expected_output" else role)
                assert blob.dimensions == tuple(emitted[k] for k in "nhwc")
                dtype = np.float16 if desc["activation_dtype"] == "FP16" else np.float32
                values = np.asarray(
                    [
                        float(v.replace("(float16_t)", "").rstrip("f"))
                        for v in array(role)
                    ],
                    dtype=dtype,
                )
                assert blob.path.read_bytes() == values.tobytes()


@pytest.mark.parametrize("dtype", ["S8", "S16"])
@pytest.mark.parametrize("role", ["input_lhs", "input_rhs", "output"])
@pytest.mark.parametrize("axis", ["n", "h"])
def test_quantized_bmm_bridge_singleton_and_dimension_guards(
    tmp_path, dtype, role, axis
):
    family, original = next(
        (f, d)
        for f, d in CASES
        if d["operator"] == "BatchMatMul" and d["activation_dtype"] == dtype
    )
    desc = deepcopy(original)
    desc["input_1_shape"][0] = desc["input_2_shape"][0] = 1
    generate_test(desc, str(tmp_path), seed=500)
    case = tmp_path / family / desc["name"]
    generated = _generated_case(family, case)
    bundle = build_case_bundle_from_generated_test(
        ROOT, generated, output_root=tmp_path / "singleton", require_fvp_pass=False
    )
    assert len(bundle.blobs) == 3
    assert all(blob.dimensions[:2] == (1, 1) for blob in bundle.blobs)
    header_path = next((case / "includes").glob("*.h"))
    header = header_path.read_text()
    # Isolate each admission check in actual headers. Move w into n/h so
    # array counts stay valid. These are bridge probes, not kernel test cases.
    pattern = rf'(\b{desc["name"]}_{role}_dims\s*=\s*\{{)([^}}]+)'
    match = re.search(pattern, header)
    body = match.group(2)
    width = int(re.search(r"\.w\s*=\s*(\d+)", body).group(1))
    assert width > 1
    changed = re.sub(rf"\.{axis}\s*=\s*1\b", f".{axis} = {width}", body)
    changed = re.sub(r"\.w\s*=\s*\d+", ".w = 1", changed)
    header_path.write_text(header[: match.start(2)] + changed + header[match.end(2) :])
    output_root = tmp_path / "rejected"
    with pytest.raises(
        UnsupportedGeneratedTestError, match=rf"quantized BatchMatMul.*{role}.*batch"
    ):
        build_case_bundle_from_generated_test(
            ROOT, generated, output_root=output_root, require_fvp_pass=False
        )
    assert not output_root.exists()


@pytest.mark.parametrize("input_count", [1, 2])
def test_single_batch_retains_original_converter(monkeypatch, input_count):
    from helia_core_tester.generation.ops._shared import fixed_batch

    tf = fixed_batch.tf
    inputs = [tf.keras.Input(batch_shape=(1, 3)) for _ in range(input_count)]
    output = tf.keras.layers.Add()(inputs) if input_count > 1 else inputs[0] * 2
    model = tf.keras.Model(inputs, output)
    converter = object()
    seen = []
    monkeypatch.setattr(
        fixed_batch.tf.lite.TFLiteConverter,
        "from_keras_model",
        lambda value: seen.append(value) or converter,
    )
    assert (
        fixed_batch.converter_for_batched_model(model, [[1, 3]] * input_count)
        is converter
    )
    assert seen == [model]


@pytest.mark.parametrize("shape_count", [0, 1, 3])
def test_single_batch_rejects_mismatched_shape_count(monkeypatch, shape_count):
    from helia_core_tester.generation.ops._shared import fixed_batch

    tf = fixed_batch.tf
    inputs = [tf.keras.Input(batch_shape=(1, 3)) for _ in range(2)]
    model = tf.keras.Model(inputs, tf.keras.layers.Add()(inputs))
    seen = []
    monkeypatch.setattr(
        tf.lite.TFLiteConverter, "from_keras_model", lambda value: seen.append(value)
    )
    with pytest.raises(
        ValueError, match="Input shape count must match model input count"
    ):
        fixed_batch.converter_for_batched_model(model, [[1, 3]] * shape_count)
    assert seen == []


@pytest.mark.parametrize("input_count", [1, 2])
def test_batched_converter_accepts_legacy_zip(monkeypatch, input_count):
    from helia_core_tester.generation.ops._shared import fixed_batch

    # Python 3.8/3.9 zip accepts no keyword arguments.
    monkeypatch.setattr(fixed_batch, "zip", lambda *values: zip(*values), raising=False)
    tf = fixed_batch.tf
    inputs = [tf.keras.Input(batch_shape=(2, 3)) for _ in range(input_count)]
    output = tf.keras.layers.Add()(inputs) if input_count > 1 else inputs[0] * 2
    model = tf.keras.Model(inputs, output)
    converter = fixed_batch.converter_for_batched_model(model, [[2, 3]] * input_count)
    interpreter = Interpreter(model_content=converter.convert())
    interpreter.allocate_tensors()
    assert [item["shape"].tolist() for item in interpreter.get_input_details()] == [
        [2, 3]
    ] * input_count
    assert interpreter.get_output_details()[0]["shape"].tolist() == [2, 3]


@pytest.mark.parametrize("shape_count", [1, 3])
def test_batched_converter_rejects_mismatched_shape_count(shape_count):
    from helia_core_tester.generation.ops._shared import fixed_batch

    tf = fixed_batch.tf
    inputs = [tf.keras.Input(batch_shape=(2, 3)) for _ in range(2)]
    model = tf.keras.Model(inputs, tf.keras.layers.Add()(inputs))
    with pytest.raises(
        ValueError, match="Input shape count must match model input count"
    ):
        fixed_batch.converter_for_batched_model(model, [[2, 3]] * shape_count)
