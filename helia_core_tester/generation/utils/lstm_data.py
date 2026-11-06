"""
LSTM CMSIS test data generation helpers.
Ported from Tests/UnitTest/RefactoredTestGen/Lib/op_lstm.py and Lib/test.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import math
import os
import subprocess
import numpy as np

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - handled by caller
    tf = None

try:
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
except ImportError:  # pragma: no cover - handled by caller
    Interpreter = None
    OpResolverType = None

from helia_core_tester.generation.utils.tflite_utils import calculate_multiplier_shift


@dataclass
class LstmGeneratedData:
    params: Dict[str, int]
    tensors: Dict[str, np.ndarray]
    scales: Dict[str, float]
    effective_scales: Dict[str, float]


_MULTIPLIER_KEYS = [
    "forget_to_cell",
    "input_to_cell",
    "output",
    "output_gate_hidden",
    "cell_gate_hidden",
    "forget_gate_hidden",
    "input_gate_hidden",
    "output_gate_input",
    "cell_gate_input",
    "forget_gate_input",
    "input_gate_input",
]


def _get_shapes(batch_size: int, time_steps: int, input_size: int, hidden_size: int, time_major: bool) -> Dict[str, Any]:
    if time_major:
        input_tensor = (time_steps, batch_size, input_size)
    else:
        input_tensor = (batch_size, time_steps, input_size)

    return {
        "input_tensor": input_tensor,
        "input_weights": (input_size, hidden_size),
        "all_input_weights": (input_size, hidden_size * 4),
        "hidden_weights": (hidden_size, hidden_size),
        "all_hidden_weights": (hidden_size, hidden_size * 4),
        "bias": (1, hidden_size),
        "all_bias": (hidden_size * 4,),
        "representational_dataset": (batch_size, time_steps, input_size),
    }


def _np_dtype_for_c(c_type: str):
    if c_type == "int8_t":
        return np.int8
    if c_type == "int16_t":
        return np.int16
    if c_type == "int32_t":
        return np.int32
    if c_type == "int64_t":
        return np.int64
    raise ValueError(f"Unsupported c_type: {c_type}")


def _tf_dtype_for_c(c_type: str):
    if tf is None:
        raise ImportError("tensorflow is required for LSTM generation")
    if c_type == "int8_t":
        return tf.int8
    if c_type == "int16_t":
        return tf.int16
    if c_type == "int32_t":
        return tf.int32
    raise ValueError(f"Unsupported c_type: {c_type}")


def _generate_tf_tensor(rng: np.random.Generator, dims, minval, maxval, decimals=0, datatype=None):
    arr = minval + (maxval - minval) * rng.random(dims)
    arr = np.round(arr, decimals=decimals)
    if datatype is not None:
        return tf.convert_to_tensor(arr, dtype=datatype)
    return arr


def _convert_keras_to_tflite(
    out_path: Path,
    keras_model,
    rng: np.random.Generator,
    shape: Dict[str, Any],
    input_dtype: str,
    bias_dtype: str,
    output_dtype: str,
):
    if tf is None:
        raise ImportError("tensorflow is required for LSTM generation")

    keras_model.compile(loss=tf.keras.losses.categorical_crossentropy, metrics=["accuracy"])
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

    def representative_dataset():
        for _ in range(1):
            data = rng.random(shape["representational_dataset"])
            yield [data.astype(np.float32)]

    converter.representative_dataset = representative_dataset
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.inference_input_type = _tf_dtype_for_c(input_dtype)
    converter.inference_output_type = _tf_dtype_for_c(output_dtype)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    if bias_dtype == "int32_t":
        converter._experimental_full_integer_quantization_bias_type = tf.int32

    tflite_model = converter.convert()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tflite_model)


def _invoke_tflite(tflite_path: Path, input_tensor: np.ndarray) -> np.ndarray:
    if Interpreter is None:
        raise ImportError("ai_edge_litert is required for LSTM generation")
    interpreter = Interpreter(str(tflite_path), experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    input_index = input_detail["index"]
    expected_dtype = input_detail["dtype"]
    if input_tensor.dtype != expected_dtype:
        input_tensor = input_tensor.astype(expected_dtype)
    interpreter.set_tensor(input_index, input_tensor)
    interpreter.invoke()
    output_index = interpreter.get_output_details()[0]["index"]
    data = interpreter.get_tensor(output_index)
    return data.flatten()


def _calc_scale_from_details(details, idx, time_major_offset):
    scale = details[idx + time_major_offset]["quantization_parameters"]["scales"][0]
    return float(scale)


def _generate_data_tflite(tflite_path: Path, time_major: bool) -> LstmGeneratedData:
    if Interpreter is None:
        raise ImportError("ai_edge_litert is required for LSTM generation")
    interpreter = Interpreter(str(tflite_path), experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
    interpreter.allocate_tensors()
    tensor_details = interpreter.get_tensor_details()

    time_major_offset = 1 if time_major else 0

    scales = {}
    effective_scales = {}
    tensors = {}
    params = {}

    input_state = tensor_details[0]
    scales["input_scale"] = float(input_state["quantization_parameters"]["scales"][0])
    cell_state = tensor_details[14 + time_major_offset * 2]
    scales["cell_scale"] = float(cell_state["quantization_parameters"]["scales"][0])
    output_state = tensor_details[13 + time_major_offset * 2]
    scales["output_scale"] = float(output_state["quantization_parameters"]["scales"][0])

    tmp = math.log(scales["cell_scale"]) * (1 / math.log(2))
    params["cell_scale_power"] = int(round(tmp))

    effective_scales["forget_to_cell"] = pow(2, -15) * scales["cell_scale"] / scales["cell_scale"]
    effective_scales["input_to_cell"] = pow(2, -15) * pow(2, -15) / scales["cell_scale"]
    effective_scales["output"] = pow(2, -15) * pow(2, -15) / scales["output_scale"]

    def calc_scale(name, input_scale, tensor_index):
        detail = tensor_details[tensor_index + time_major_offset]
        tensors[name + "_weights"] = interpreter.get_tensor(detail["index"]).flatten()
        scales[name + "_scale"] = float(detail["quantization_parameters"]["scales"][0])
        effective_scales[name] = input_scale * scales[name + "_scale"] / pow(2, -12)

    calc_scale("output_gate_hidden", scales["output_scale"], 5)
    calc_scale("cell_gate_hidden", scales["output_scale"], 6)
    calc_scale("forget_gate_hidden", scales["output_scale"], 7)
    calc_scale("input_gate_hidden", scales["output_scale"], 8)
    calc_scale("output_gate_input", scales["input_scale"], 9)
    calc_scale("cell_gate_input", scales["input_scale"], 10)
    calc_scale("forget_gate_input", scales["input_scale"], 11)
    calc_scale("input_gate_input", scales["input_scale"], 12)

    tensors["output_gate_bias"] = interpreter.get_tensor(1 + time_major_offset).flatten()
    tensors["cell_gate_bias"] = interpreter.get_tensor(2 + time_major_offset).flatten()
    tensors["forget_gate_bias"] = interpreter.get_tensor(3 + time_major_offset).flatten()
    tensors["input_gate_bias"] = interpreter.get_tensor(4 + time_major_offset).flatten()

    params["input_zero_point"] = int(-input_state["quantization_parameters"]["zero_points"][0])
    params["output_zero_point"] = int(
        tensor_details[20 + time_major_offset * 2]["quantization_parameters"]["zero_points"][0]
    )
    params["cell_clip"] = 32767

    return LstmGeneratedData(params=params, tensors=tensors, scales=scales, effective_scales=effective_scales)


def _generate_data_json(
    rng: np.random.Generator,
    input_dtype: str,
    weights_dtype: str,
    batch_size: int,
    time_steps: int,
    input_size: int,
    hidden_size: int,
    time_major: bool,
    input_zero_point_override: int | None = None,
    output_zero_point_override: int | None = None,
) -> LstmGeneratedData:
    shapes = _get_shapes(batch_size, time_steps, input_size, hidden_size, time_major)
    tensors = {}
    scales = {}
    effective_scales = {}
    params = {}

    maxval = 0.001
    minval = 0.0001

    scales["input_scale"] = float(np.round(rng.random(1) * (maxval - minval) + minval, 6)[0])
    scales["cell_scale"] = float(np.round(rng.random(1) * (maxval - minval) + maxval, 6)[0])
    scales["output_scale"] = float(np.round(rng.random(1) * (maxval - minval) + minval, 6)[0])

    tmp = math.log(scales["cell_scale"]) * (1 / math.log(2))
    params["cell_scale_power"] = int(round(tmp))

    effective_scales["forget_to_cell"] = pow(2, -15) * scales["cell_scale"] / scales["cell_scale"]
    effective_scales["input_to_cell"] = pow(2, -15) * pow(2, -15) / scales["cell_scale"]
    effective_scales["output"] = pow(2, -15) * pow(2, -15) / scales["output_scale"]

    def create_scales(name, input_scale1):
        scales[name + "_scale"] = float(np.round(rng.random(1) * (maxval - minval) + minval, 6)[0])
        effective_scales[name] = input_scale1 * scales[name + "_scale"] / pow(2, -12)

    create_scales("output_gate_hidden", scales["output_scale"])
    create_scales("cell_gate_hidden", scales["output_scale"])
    create_scales("forget_gate_hidden", scales["output_scale"])
    create_scales("input_gate_hidden", scales["output_scale"])
    create_scales("output_gate_input", scales["input_scale"])
    create_scales("cell_gate_input", scales["input_scale"])
    create_scales("forget_gate_input", scales["input_scale"])
    create_scales("input_gate_input", scales["input_scale"])

    w_min = -128
    w_max = 127
    tensors["input_gate_hidden_weights"] = rng.integers(w_min, w_max + 1, size=shapes["hidden_weights"], dtype=np.int8)
    tensors["forget_gate_hidden_weights"] = rng.integers(w_min, w_max + 1, size=shapes["hidden_weights"], dtype=np.int8)
    tensors["cell_gate_hidden_weights"] = rng.integers(w_min, w_max + 1, size=shapes["hidden_weights"], dtype=np.int8)
    tensors["output_gate_hidden_weights"] = rng.integers(w_min, w_max + 1, size=shapes["hidden_weights"], dtype=np.int8)
    tensors["input_gate_input_weights"] = rng.integers(w_min, w_max + 1, size=shapes["input_weights"], dtype=np.int8)
    tensors["forget_gate_input_weights"] = rng.integers(w_min, w_max + 1, size=shapes["input_weights"], dtype=np.int8)
    tensors["cell_gate_input_weights"] = rng.integers(w_min, w_max + 1, size=shapes["input_weights"], dtype=np.int8)
    tensors["output_gate_input_weights"] = rng.integers(w_min, w_max + 1, size=shapes["input_weights"], dtype=np.int8)

    input_min = -32768 if input_dtype == "int16_t" else -128
    input_max = 32767 if input_dtype == "int16_t" else 127
    bias_dtype = np.float32
    tensors["input_gate_bias"] = np.zeros(shapes["bias"], dtype=bias_dtype)
    tensors["forget_gate_bias"] = np.zeros(shapes["bias"], dtype=bias_dtype)
    tensors["cell_gate_bias"] = np.zeros(shapes["bias"], dtype=bias_dtype)
    tensors["output_gate_bias"] = np.zeros(shapes["bias"], dtype=bias_dtype)

    params["output_zero_point"] = int(output_zero_point_override) if output_zero_point_override is not None else 0
    params["input_zero_point"] = int(input_zero_point_override) if input_zero_point_override is not None else 0
    params["cell_clip"] = 32767

    return LstmGeneratedData(params=params, tensors=tensors, scales=scales, effective_scales=effective_scales)


def _convert_json_to_tflite(json_template_fpath: Path, json_output_fpath: Path, tensors: Dict[str, np.ndarray], params: Dict[str, Any], schema_path: Path):
    json_output_fpath.parent.mkdir(parents=True, exist_ok=True)
    with json_template_fpath.open("r") as template:
        with json_output_fpath.open("w+") as output:
            for line in template:
                line_list = line.replace(",", "").split()
                replaced = False
                for key, val in params.items():
                    if key in line_list:
                        if isinstance(val, bool):
                            val = "true" if val else "false"
                        new_line = str(val).join(line.rsplit(key, 1))
                        output.write(new_line)
                        replaced = True
                        break

                for key in tensors:
                    if key in line:
                        dtype = "float32_t" if "bias" in key else "int8_t"
                        dtype_len = 4 if dtype == "float32_t" else 1
                        np_dtype = np.float32 if dtype == "float32_t" else np.int8

                        weights_in_bytes = []
                        for weight in tensors[key].flatten():
                            if dtype == "float32_t":
                                weights_in_bytes.extend([b for b in np.float32(weight).tobytes()])
                            else:
                                weights_in_bytes.extend(
                                    [b for b in int(np_dtype(weight)).to_bytes(dtype_len, "little", signed=True)]
                                )

                        for byte in weights_in_bytes[:-1]:
                            output.write(f"        {byte},\n")
                        output.write(f"        {weights_in_bytes[-1]}\n")

                        replaced = True
                        break

                if not replaced:
                    output.write(line)

    command = f"flatc -o {json_output_fpath.parent} -c -b {schema_path} {json_output_fpath}"
    command_list = command.split()
    process = subprocess.run(command_list, env={"PATH": os.getenv("PATH")})
    if process.returncode != 0:
        raise RuntimeError(f"flatc failed: {command = }")


def _parse_macro_value(line: str) -> int:
    parts = line.strip().split()
    if len(parts) < 3:
        raise ValueError(f"Bad macro line: {line}")
    value = parts[-1]
    if value.lower() in ("true", "false"):
        return 1 if value.lower() == "true" else 0
    return int(value)


def _parse_c_array(file_path: Path) -> np.ndarray:
    text = file_path.read_text()
    header = text.split("{", 1)[0]
    if "const" not in header:
        raise ValueError(f"Missing const declaration in {file_path}")
    dtype = header.split("const", 1)[1].strip().split()[0]
    data_part = text.split("{", 1)[1].rsplit("}", 1)[0]
    numbers = []
    for token in data_part.replace("\n", " ").split(","):
        token = token.strip()
        if not token:
            continue
        numbers.append(int(float(token)))
    np_dtype = _np_dtype_for_c(dtype)
    return np.asarray(numbers, dtype=np_dtype)


def _unit_test_data_root() -> Path:
    """Resolve the ns-cmsis-nn UnitTest/TestCases/TestData directory.

    Honors CMSIS_NN_ROOT (an ns-cmsis-nn checkout root, e.g. set via
    Config.cmsis_nn_root / --cmsis-nn-root) so this works when
    helia-core-tester is checked out standalone rather than nested under
    ns-cmsis-nn/Tests/. Note: this is distinct from CMSIS_NN_REPO_ROOT, which
    overrides discovery of the helia-core-tester repo root itself (see
    helia_core_tester.core.discovery.find_repo_root). Falls back to the
    historical parents[4]-relative layout
    (ns-cmsis-nn/Tests/helia-core-tester/helia_core_tester/...) for backward
    compatibility.
    """
    env_root = os.environ.get("CMSIS_NN_ROOT")
    if env_root:
        return Path(env_root).resolve() / "Tests" / "UnitTest" / "TestCases" / "TestData"
    return Path(__file__).resolve().parents[4] / "UnitTest" / "TestCases" / "TestData"


def _load_unit_test_data(dataset: str) -> LstmGeneratedData:
    base = _unit_test_data_root() / dataset
    config = base / "config_data.h"
    if not config.exists():
        raise FileNotFoundError(f"Missing UnitTest config_data.h for {dataset}")

    params: Dict[str, int] = {}
    for line in config.read_text().splitlines():
        if not line.startswith("#define"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        name = parts[1]
        value = _parse_macro_value(line)
        key = name.lower()
        params[key] = value

    # Map common fields from macros into generic params.
    def _pick_suffix(suffix: str) -> int | None:
        for k, v in params.items():
            if k.endswith(suffix):
                return v
        return None

    for suffix, key in [
        ("time_major", "time_major"),
        ("batch_size", "batch_size"),
        ("time_steps", "time_steps"),
        ("input_size", "input_size"),
        ("hidden_size", "hidden_size"),
        ("cell_scale_power", "cell_scale_power"),
        ("input_zero_point", "input_zero_point"),
        ("output_zero_point", "output_zero_point"),
        ("cell_clip", "cell_clip"),
    ]:
        val = _pick_suffix(suffix)
        if val is not None:
            params[key] = val

    tensors = {}
    for stem in [
        "input_tensor",
        "output",
        "input_gate_input_weights",
        "forget_gate_input_weights",
        "cell_gate_input_weights",
        "output_gate_input_weights",
        "input_gate_hidden_weights",
        "forget_gate_hidden_weights",
        "cell_gate_hidden_weights",
        "output_gate_hidden_weights",
        "input_gate_bias",
        "forget_gate_bias",
        "cell_gate_bias",
        "output_gate_bias",
    ]:
        fpath = base / f"{stem}.h"
        if not fpath.exists():
            raise FileNotFoundError(f"Missing UnitTest tensor: {fpath}")
        tensors[stem] = _parse_c_array(fpath)

    # Build mult/shift params from macros
    mult_shift = {}
    for key in _MULTIPLIER_KEYS:
        mult_key = f"{dataset}_{key}_multiplier".upper()
        shift_key = f"{dataset}_{key}_shift".upper()
        mult_val = params.get(mult_key.lower())
        shift_val = params.get(shift_key.lower())
        if mult_val is not None and shift_val is not None:
            mult_shift[key + "_multiplier"] = mult_val
            mult_shift[key + "_shift"] = shift_val

    params.update(mult_shift)
    return LstmGeneratedData(params=params, tensors=tensors, scales={}, effective_scales={})


def generate_lstm_data(
    *,
    rng: np.random.Generator,
    activation_dtype: str,
    batch_size: int,
    time_steps: int,
    input_size: int,
    hidden_size: int,
    time_major: bool,
    templates_dir: Path,
    schema_path: Path,
    work_dir: Path,
    dataset: str,
    input_zero_point_override: int | None = None,
    output_zero_point_override: int | None = None,
) -> LstmGeneratedData:
    shapes = _get_shapes(batch_size, time_steps, input_size, hidden_size, time_major)
    expected_sizes = {
        "input_tensor": int(np.prod(shapes["input_tensor"])),
        "output": int(batch_size * time_steps * hidden_size),
        "input_gate_input_weights": int(input_size * hidden_size),
        "forget_gate_input_weights": int(input_size * hidden_size),
        "cell_gate_input_weights": int(input_size * hidden_size),
        "output_gate_input_weights": int(input_size * hidden_size),
        "input_gate_hidden_weights": int(hidden_size * hidden_size),
        "forget_gate_hidden_weights": int(hidden_size * hidden_size),
        "cell_gate_hidden_weights": int(hidden_size * hidden_size),
        "output_gate_hidden_weights": int(hidden_size * hidden_size),
        "input_gate_bias": int(hidden_size),
        "forget_gate_bias": int(hidden_size),
        "cell_gate_bias": int(hidden_size),
        "output_gate_bias": int(hidden_size),
    }

    def _validate_data(data: LstmGeneratedData) -> bool:
        for key, expected in expected_sizes.items():
            arr = data.tensors.get(key)
            if arr is None:
                return False
            if int(np.asarray(arr).size) != expected:
                return False
        return True

    def _load_validated_unit_test_data() -> LstmGeneratedData:
        data = _load_unit_test_data(dataset=dataset)
        if not _validate_data(data):
            raise ValueError(f"UnitTest fallback data for {dataset} does not match descriptor sizes")
        return data

    if activation_dtype == "S8":
        if tf is None:
            return _load_validated_unit_test_data()
        input_dtype = "int8_t"
        output_dtype = "int8_t"
        bias_dtype = "int32_t"

        input_layer = tf.keras.layers.Input(shape=(time_steps, input_size), batch_size=batch_size, name="input")
        if time_major:
            input_layer_transposed = tf.keras.layers.Lambda(
                lambda x: tf.transpose(x, perm=[1, 0, 2]),
                name="time_major_transpose",
            )(input_layer)
            lstm_layer = tf.keras.layers.LSTM(units=hidden_size, return_sequences=True, unroll=True)(input_layer_transposed)
        else:
            lstm_layer = tf.keras.layers.LSTM(units=hidden_size, return_sequences=True, unroll=True)(input_layer)
        model = tf.keras.Model(input_layer, lstm_layer, name="LSTM")

        time_major_offset = 1 if time_major else 0
        input_weights = _generate_tf_tensor(rng, shapes["all_input_weights"], -1, 1, decimals=8, datatype=tf.float32)
        model.layers[1 + time_major_offset].weights[0].assign(input_weights)
        hidden_weights = _generate_tf_tensor(rng, shapes["all_hidden_weights"], -1, 1, decimals=8, datatype=tf.float32)
        model.layers[1 + time_major_offset].weights[1].assign(hidden_weights)
        biases = _generate_tf_tensor(rng, shapes["all_bias"], -1, 1, decimals=8, datatype=tf.float32) * 0
        model.layers[1 + time_major_offset].weights[2].assign(biases)

        tflite_path = work_dir / "lstm_s8.tflite"
        try:
            _convert_keras_to_tflite(tflite_path, model, rng, shapes, input_dtype, bias_dtype, output_dtype)
        except Exception:
            return _load_validated_unit_test_data()
        try:
            data = _generate_data_tflite(tflite_path, time_major)
        except Exception:
            return _load_validated_unit_test_data()
        if not _validate_data(data):
            return _load_validated_unit_test_data()

        # Generate input tensor and run model for output
        input_min, input_max = -128, 127
        input_tensor = rng.integers(input_min, input_max + 1, size=shapes["input_tensor"], dtype=np.int8)
        data.tensors["input_tensor"] = input_tensor
        try:
            data.tensors["output"] = _invoke_tflite(tflite_path, input_tensor).astype(np.int8)
        except Exception:
            return _load_validated_unit_test_data()
        if not _validate_data(data):
            return _load_validated_unit_test_data()

        return data

    if activation_dtype == "S16":
        data = _generate_data_json(
            rng,
            "int16_t",
            "int8_t",
            batch_size,
            time_steps,
            input_size,
            hidden_size,
            time_major,
            input_zero_point_override=input_zero_point_override,
            output_zero_point_override=output_zero_point_override,
        )

        json_template = "lstm_s16_tm.json" if time_major else "lstm_s16.json"
        json_template_fpath = templates_dir / json_template
        json_output_fpath = work_dir / "lstm_s16.json"
        params = {
            "batch_size": batch_size,
            "time_steps": time_steps,
            "input_size": input_size,
            "hidden_size": hidden_size,
            "time_major": time_major,
            "input_scale": data.scales["input_scale"],
            "output_scale": data.scales["output_scale"],
            "cell_scale": data.scales["cell_scale"],
            "input_zero_point": data.params["input_zero_point"],
            "output_zero_point": data.params["output_zero_point"],
            "cell_clip": data.params["cell_clip"],
            "cell_scale_power": data.params["cell_scale_power"],
            "output_gate_hidden_scale": data.scales["output_gate_hidden_scale"],
            "cell_gate_hidden_scale": data.scales["cell_gate_hidden_scale"],
            "forget_gate_hidden_scale": data.scales["forget_gate_hidden_scale"],
            "input_gate_hidden_scale": data.scales["input_gate_hidden_scale"],
            "output_gate_input_scale": data.scales["output_gate_input_scale"],
            "cell_gate_input_scale": data.scales["cell_gate_input_scale"],
            "forget_gate_input_scale": data.scales["forget_gate_input_scale"],
            "input_gate_input_scale": data.scales["input_gate_input_scale"],
        }
        try:
            _convert_json_to_tflite(json_template_fpath, json_output_fpath, data.tensors, params, schema_path)
        except Exception:
            return _load_validated_unit_test_data()

        input_min, input_max = -32768, 32767
        input_tensor = rng.integers(input_min, input_max + 1, size=shapes["input_tensor"], dtype=np.int16)
        data.tensors["input_tensor"] = input_tensor
        try:
            data.tensors["output"] = _invoke_tflite(json_output_fpath.with_suffix(".tflite"), input_tensor).astype(np.int16)
        except Exception:
            return _load_validated_unit_test_data()
        if not _validate_data(data):
            return _load_validated_unit_test_data()

        return data

    raise ValueError(f"Unsupported activation_dtype: {activation_dtype}")


def build_lstm_context(
    *,
    name: str,
    dataset: str,
    activation_dtype: str,
    batch_size: int,
    time_steps: int,
    input_size: int,
    hidden_size: int,
    time_major: bool,
    data: LstmGeneratedData,
) -> Dict[str, Any]:
    # Quantize scales to multipliers/shifts
    mult_shift = {}
    if data.effective_scales:
        for key, scale in data.effective_scales.items():
            mult, shift = calculate_multiplier_shift(scale)
            mult_shift[key + "_multiplier"] = int(mult)
            mult_shift[key + "_shift"] = int(shift)
    else:
        for key in _MULTIPLIER_KEYS:
            mult = data.params.get(key + "_multiplier")
            shift = data.params.get(key + "_shift")
            if mult is not None and shift is not None:
                mult_shift[key + "_multiplier"] = int(mult)
                mult_shift[key + "_shift"] = int(shift)

    macro_prefix = dataset.upper() + "_"
    data_prefix = dataset.lower() + "_"

    return {
        "name": name,
        "dataset": dataset,
        "macro_prefix": macro_prefix,
        "data_prefix": data_prefix,
        "dtype": "s16" if activation_dtype == "S16" else "s8",
        "output_dtype": "int16_t" if activation_dtype == "S16" else "int8_t",
        "time_major": int(bool(time_major)),
        "batch_size": int(batch_size),
        "time_steps": int(time_steps),
        "input_size": int(input_size),
        "hidden_size": int(hidden_size),
        "input_zero_point": int(data.params["input_zero_point"]),
        "output_zero_point": int(data.params["output_zero_point"]),
        "cell_scale_power": int(data.params["cell_scale_power"]),
        "cell_clip": int(data.params["cell_clip"]),
        "mult_shift": mult_shift,
        "tensors": data.tensors,
    }


# Explicit-quant s8 cases.
# TFLite gate order, input then recurrent.
LSTM_GATES = ("input", "forget", "cell", "output")
# Gate pre-activations are Q3.12.
_GATE_SCALE = 2.0**-12
# TFLM sigmoid_table_uint16, 256 entries.
_SIGMOID_LUT = np.array([
    32768, 33451, 34133, 34813, 35493, 36169, 36843, 37513, 38180, 38841, 39498, 40149, 40794, 41432, 42064, 42688,
    43304, 43912, 44511, 45102, 45683, 46255, 46817, 47369, 47911, 48443, 48964, 49475, 49975, 50464, 50942, 51409,
    51865, 52311, 52745, 53169, 53581, 53983, 54374, 54755, 55125, 55485, 55834, 56174, 56503, 56823, 57133, 57433,
    57724, 58007, 58280, 58544, 58800, 59048, 59288, 59519, 59743, 59959, 60168, 60370, 60565, 60753, 60935, 61110,
    61279, 61441, 61599, 61750, 61896, 62036, 62172, 62302, 62428, 62549, 62666, 62778, 62886, 62990, 63090, 63186,
    63279, 63368, 63454, 63536, 63615, 63691, 63765, 63835, 63903, 63968, 64030, 64090, 64148, 64204, 64257, 64308,
    64357, 64405, 64450, 64494, 64536, 64576, 64614, 64652, 64687, 64721, 64754, 64786, 64816, 64845, 64873, 64900,
    64926, 64950, 64974, 64997, 65019, 65039, 65060, 65079, 65097, 65115, 65132, 65149, 65164, 65179, 65194, 65208,
    65221, 65234, 65246, 65258, 65269, 65280, 65291, 65301, 65310, 65319, 65328, 65337, 65345, 65352, 65360, 65367,
    65374, 65381, 65387, 65393, 65399, 65404, 65410, 65415, 65420, 65425, 65429, 65433, 65438, 65442, 65445, 65449,
    65453, 65456, 65459, 65462, 65465, 65468, 65471, 65474, 65476, 65479, 65481, 65483, 65485, 65488, 65489, 65491,
    65493, 65495, 65497, 65498, 65500, 65501, 65503, 65504, 65505, 65507, 65508, 65509, 65510, 65511, 65512, 65513,
    65514, 65515, 65516, 65517, 65517, 65518, 65519, 65520, 65520, 65521, 65522, 65522, 65523, 65523, 65524, 65524,
    65525, 65525, 65526, 65526, 65526, 65527, 65527, 65528, 65528, 65528, 65529, 65529, 65529, 65529, 65530, 65530,
    65530, 65530, 65531, 65531, 65531, 65531, 65531, 65532, 65532, 65532, 65532, 65532, 65532, 65533, 65533, 65533,
    65533, 65533, 65533, 65533, 65533, 65534, 65534, 65534, 65534, 65534, 65534, 65534, 65534, 65534, 65534, 65535,
], dtype=np.int64)


@dataclass(frozen=True)
class LstmQuant:
    """Model-style s8 LSTM quant knobs."""

    input_scale: float
    input_zero_point: int
    output_scale: float
    output_zero_point: int
    cell_scale_power: int
    # Real-valued; 0 means no clip.
    cell_clip: float
    # Input i, f, c, o; then recurrent.
    weight_scales: tuple[float, ...]
    # Int8 weight spread before clipping.
    weight_std: float
    # Real bias range per gate.
    bias_ranges: tuple[tuple[float, float], ...]
    input_range: tuple[int, int]

    @classmethod
    def from_desc(cls, raw: Dict[str, Any]) -> "LstmQuant":
        scales = tuple(float(v) for v in raw["weight_scales"])
        biases = tuple((float(lo), float(hi)) for lo, hi in raw["bias_ranges"])
        if len(scales) != 8 or len(biases) != 4:
            raise ValueError("lstm_quant needs 8 weight scales, 4 bias ranges")
        lo, hi = (int(v) for v in raw.get("input_range", (-128, 127)))
        return cls(
            float(raw["input_scale"]), int(raw["input_zero_point"]), float(raw["output_scale"]),
            int(raw["output_zero_point"]), int(raw["cell_scale_power"]), float(raw.get("cell_clip", 0.0)),
            scales, float(raw.get("weight_std", 14.0)), biases, (lo, hi),
        )


def _draw_weights(rng: np.random.Generator, shape: tuple[int, int], std: float) -> np.ndarray:
    """Symmetric int8 weights; one hits 127."""
    w = np.clip(np.round(rng.normal(0.0, std, size=shape)), -127, 127).astype(np.int8)
    # Symmetric quant maps |max| to 127.
    w.flat[int(rng.integers(0, w.size))] = 127 if rng.random() < 0.5 else -127
    return w


def quantized_clip(quant: LstmQuant) -> int:
    """Kernel cell clip; 32767 means none."""
    if quant.cell_clip <= 0.0:
        return 32767
    # TFLM: double division, int16 cast.
    value = quant.cell_clip / float(np.float32(2.0**quant.cell_scale_power))
    return int(min(max(value, -32768.0), 32767.0))


def effective_scales(quant: LstmQuant) -> Dict[str, float]:
    """TFLM's effective scales, by multiplier key."""
    f32, scales = np.float32, {}
    for k, gate in enumerate(LSTM_GATES):
        # float32 product, then double.
        scales[f"{gate}_gate_input"] = float(f32(quant.weight_scales[k]) * f32(quant.input_scale)) / _GATE_SCALE
        scales[f"{gate}_gate_hidden"] = float(f32(quant.weight_scales[4 + k]) * f32(quant.output_scale)) / _GATE_SCALE
    scales["forget_to_cell"] = 2.0**-15
    scales["input_to_cell"] = 2.0 ** (-30 - quant.cell_scale_power)
    scales["output"] = 2.0**-30 / float(f32(quant.output_scale))
    return scales


def _requantize(x: np.ndarray, multiplier: int, shift: int) -> np.ndarray:
    """TFLM double-rounding MultiplyByQuantizedMultiplier."""
    x = np.asarray(x, dtype=np.int64) * (1 << max(shift, 0))
    product = x * multiplier
    nudge = np.where(product >= 0, 1 << 30, 1 - (1 << 30))
    total = product + nudge
    # C division truncates toward zero.
    high = np.where(total >= 0, total >> 31, -((-total) >> 31))
    right = max(-shift, 0)
    mask = (1 << right) - 1
    threshold = (mask >> 1) + (high < 0)
    return (high >> right) + ((high & mask) > threshold)


def _sigmoid_q15(x: np.ndarray) -> np.ndarray:
    """TFLM int16 Logistic on Q3.12."""
    value = np.asarray(x, dtype=np.int64) * 3
    magnitude = np.abs(value)
    index = np.minimum(magnitude >> 9, 254)
    low, high = _SIGMOID_LUT[index], _SIGMOID_LUT[index + 1]
    result = np.where(magnitude >> 9 >= 255, 0x7FFF << 10, (low << 9) + (magnitude & 0x1FF) * (high - low))
    result = np.where(value >= 0, result + (1 << 9), (1 << 25) - result + (1 << 9) - 1)
    return (result >> 10).astype(np.int16)


def _tanh_q15(x: np.ndarray, power: int) -> np.ndarray:
    """TFLM int16 Tanh; power is the input scale."""
    shift = 12 + power
    multiplier, shift = (3 << shift, 0) if shift >= 0 else (3, -shift)
    rounding = 1 << (shift - 1) if shift > 0 else 0
    value = (np.asarray(x, dtype=np.int64) * multiplier + rounding) >> shift
    magnitude = np.abs(value)
    index = np.minimum(magnitude >> 8, 254)
    low, high = _SIGMOID_LUT[index], _SIGMOID_LUT[index + 1]
    result = np.where(magnitude >> 8 >= 255, 0xFFFF << 8, (low << 8) + (magnitude & 0xFF) * (high - low))
    result = np.where(value >= 0, result - (1 << 23) + (1 << 7), -result + (1 << 23) + (1 << 7) - 1)
    return (result >> 8).astype(np.int16)


def _sat16(x: np.ndarray) -> np.ndarray:
    return np.clip(x, -32768, 32767)


def tflm_lstm_s8(
    inputs: np.ndarray, weights: list[np.ndarray], biases: list[np.ndarray], quant: LstmQuant,
    mult_shift: Dict[str, tuple[int, int]], time_major: bool, state: Optional[tuple[np.ndarray, np.ndarray]] = None,
) -> np.ndarray:
    """helia-rt's integer LSTM reference, in numpy.

    TFLite's gemmlowp sigmoid is an LSB off.
    """
    seq = inputs.astype(np.int64) if time_major else np.swapaxes(inputs.astype(np.int64), 0, 1)
    steps, batch, _ = seq.shape
    hidden_size = weights[0].shape[0]
    hidden = np.full((batch, hidden_size), quant.output_zero_point, dtype=np.int64)
    cell = np.zeros((batch, hidden_size), dtype=np.int64)
    if state is not None:
        hidden, cell = (np.asarray(v, dtype=np.int64).reshape(batch, hidden_size) for v in state)
    clip = quantized_clip(quant)
    out = np.zeros((steps, batch, hidden_size), dtype=np.int64)

    def gate(k: int, x: np.ndarray) -> np.ndarray:
        name = LSTM_GATES[k]
        fc_in = (x - quant.input_zero_point) @ weights[k].astype(np.int64).T + biases[k]
        fc_rec = (hidden - quant.output_zero_point) @ weights[4 + k].astype(np.int64).T
        total = _sat16(_sat16(_requantize(fc_in, *mult_shift[f"{name}_gate_input"]))
                       + _sat16(_requantize(fc_rec, *mult_shift[f"{name}_gate_hidden"])))
        return _tanh_q15(total, -12) if name == "cell" else _sigmoid_q15(total)

    for t in range(steps):
        x = seq[t]
        forget, input_gate, cell_gate = gate(1, x), gate(0, x), gate(2, x)
        cell = _sat16(_requantize(forget.astype(np.int64) * cell, *mult_shift["forget_to_cell"]))
        update = _sat16(_requantize(input_gate.astype(np.int64) * cell_gate, *mult_shift["input_to_cell"]))
        # The kernel always clamps.
        cell = np.clip(cell + update, -clip, clip)
        output_gate = gate(3, x)
        squashed = _tanh_q15(cell, quant.cell_scale_power).astype(np.int64)
        level = _requantize(squashed * output_gate, *mult_shift["output"]) + quant.output_zero_point
        hidden = np.clip(level, -128, 127)
        out[t] = hidden
    out = out if time_major else np.swapaxes(out, 0, 1)
    return out.reshape(-1).astype(np.int8)


def explicit_lstm_s8(
    *,
    rng: np.random.Generator,
    batch_size: int,
    time_steps: int,
    input_size: int,
    hidden_size: int,
    time_major: bool,
    quant: LstmQuant,
    stateful: bool = False,
) -> LstmGeneratedData:
    """Random model-style s8 LSTM; TFLM golden.

    Stateful cases start from random state.
    """
    shapes = _get_shapes(batch_size, time_steps, input_size, hidden_size, time_major)
    weights = [_draw_weights(rng, (hidden_size, input_size), quant.weight_std) for _ in LSTM_GATES]
    weights += [_draw_weights(rng, (hidden_size, hidden_size), quant.weight_std) for _ in LSTM_GATES]
    biases = []
    for k, (lo, hi) in enumerate(quant.bias_ranges):
        scale = float(np.float32(quant.input_scale) * np.float32(quant.weight_scales[k]))
        real = rng.uniform(lo, hi, size=hidden_size)
        biases.append(np.clip(np.round(real / scale), -(2**31), 2**31 - 1).astype(np.int32))
    lo, hi = quant.input_range
    inputs = rng.integers(lo, hi + 1, size=shapes["input_tensor"], dtype=np.int8)
    scales = effective_scales(quant)
    mult_shift = {key: calculate_multiplier_shift(value) for key, value in scales.items()}
    state = None
    if stateful:
        # Cell within half the clip.
        bound = quantized_clip(quant) // 2
        state = (rng.integers(-128, 128, size=(batch_size, hidden_size), dtype=np.int8),
                 rng.integers(-bound, bound + 1, size=(batch_size, hidden_size)).astype(np.int16))

    tensors: Dict[str, np.ndarray] = {
        "input_tensor": inputs, "output": tflm_lstm_s8(inputs, weights, biases, quant, mult_shift, time_major, state),
    }
    if state is not None:
        tensors["hidden_state"], tensors["cell_state"] = (v.flatten() for v in state)
    for k, gate in enumerate(LSTM_GATES):
        tensors[f"{gate}_gate_input_weights"] = weights[k].flatten()
        tensors[f"{gate}_gate_hidden_weights"] = weights[4 + k].flatten()
        tensors[f"{gate}_gate_bias"] = biases[k]
    params = {
        # Kernel takes the negated zero point.
        "input_zero_point": -quant.input_zero_point,
        "output_zero_point": quant.output_zero_point,
        "cell_scale_power": quant.cell_scale_power,
        "cell_clip": quantized_clip(quant),
    }
    return LstmGeneratedData(params=params, tensors=tensors, scales={}, effective_scales=scales)
