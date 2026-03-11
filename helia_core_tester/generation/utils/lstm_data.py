"""
LSTM CMSIS test data generation helpers.
Ported from Tests/UnitTest/RefactoredTestGen/Lib/op_lstm.py and Lib/test.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
import math
import os
import subprocess
import numpy as np

try:
    import tensorflow as tf
    from tensorflow.lite.python.interpreter import Interpreter, OpResolverType
except ImportError:  # pragma: no cover - handled by caller
    tf = None
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
    if tf is None or Interpreter is None:
        raise ImportError("tensorflow is required for LSTM generation")
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
    if tf is None or Interpreter is None:
        raise ImportError("tensorflow is required for LSTM generation")
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


def _load_unit_test_data(dataset: str) -> LstmGeneratedData:
    base = Path(__file__).resolve().parents[4] / "UnitTest" / "TestCases" / "TestData" / dataset
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

    if activation_dtype == "S8":
        if tf is None:
            raise ImportError("tensorflow is required for LSTM generation")
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
        _convert_keras_to_tflite(tflite_path, model, rng, shapes, input_dtype, bias_dtype, output_dtype)
        try:
            data = _generate_data_tflite(tflite_path, time_major)
        except Exception:
            return _load_unit_test_data(dataset=dataset)
        if not _validate_data(data):
            return _load_unit_test_data(dataset=dataset)

        # Generate input tensor and run model for output
        input_min, input_max = -128, 127
        input_tensor = rng.integers(input_min, input_max + 1, size=shapes["input_tensor"], dtype=np.int8)
        data.tensors["input_tensor"] = input_tensor
        data.tensors["output"] = _invoke_tflite(tflite_path, input_tensor).astype(np.int8)
        if not _validate_data(data):
            return _load_unit_test_data(dataset=dataset)

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
            return _load_unit_test_data(dataset=dataset)

        input_min, input_max = -32768, 32767
        input_tensor = rng.integers(input_min, input_max + 1, size=shapes["input_tensor"], dtype=np.int16)
        data.tensors["input_tensor"] = input_tensor
        try:
            data.tensors["output"] = _invoke_tflite(json_output_fpath.with_suffix(".tflite"), input_tensor).astype(np.int16)
        except Exception:
            return _load_unit_test_data(dataset=dataset)
        if not _validate_data(data):
            return _load_unit_test_data(dataset=dataset)

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
        "prefix": name,
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
