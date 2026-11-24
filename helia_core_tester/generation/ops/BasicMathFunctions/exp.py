"""Generate standalone exponential checks with float64 mathematical goldens."""

import math
from pathlib import Path

import numpy as np

from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.generation.utils.litert_builder import build_unary_same_shape_op


class OpExp(OperationBase):
    """Exercise finite arithmetic, saturation, NaN propagation and vector tails."""

    def needs_keras_model(self) -> bool:
        return False

    def build_keras_model(self):
        raise NotImplementedError("Exp builds a single LiteRT operator directly.")

    def validate_descriptor(self) -> None:
        dtype = self.tensor_dtype("input")
        if dtype not in ("FP32", "FP16") or dtype != self.tensor_dtype("output"):
            raise ValueError("Exp requires matching FP32 or FP16 input and output.")
        if math.prod(self.desc["input_shape"]) < 1:
            raise ValueError("Exp requires a nonempty input.")

    def allow_no_tflite(self) -> bool:
        return self.tensor_dtype("input") == "FP16"

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        if self.allow_no_tflite():
            raise NotImplementedError("The binary16 case exercises the native kernel.")
        data = build_unary_same_shape_op(
            op_name="EXP", input_shape=self.desc["input_shape"], dtype="float32"
        )
        self._write_tflite_bytes(out_path, data)

    def generate_c_files(self, output_dir: Path) -> None:
        n = math.prod(self.desc["input_shape"])
        half = self.tensor_dtype("input") == "FP16"
        storage = np.uint16 if half else np.uint32
        dtype = np.float16 if half else np.float32
        inputs = np.resize(
            np.array([-np.inf, -200, -1, -0.0, 0, 1, 100, np.inf, np.nan], dtype=dtype),
            n,
        )
        with np.errstate(over="ignore", under="ignore"):
            golden = np.exp(inputs.astype(np.float64)).astype(dtype)
        name = self.desc["name"]
        context = {
            "name": name,
            "count": n,
            "ctype": "float16_t" if half else "float32_t",
            "word": "uint16_t" if half else "uint32_t",
            "kernel_fn": "arm_nn_exp_f16" if half else "arm_nn_exp_f32",
            "inputs": inputs.view(storage).tolist(),
            "golden": golden.view(storage).tolist(),
            "max_ulp": np.where(np.abs(inputs) == 1, 2, 0).tolist(),
        }
        self._write_op_outputs(
            output_dir,
            "exp",
            "BasicMathFunctions/exp/exp.h.j2",
            "BasicMathFunctions/exp/exp.c.j2",
            context,
            {"name": name, "operator": "Exp", "operator_name": "exp"},
        )
