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
        if self.tensor_dtype("input") != "FP32" or self.tensor_dtype("output") != "FP32":
            raise ValueError("Exp requires FP32 input and output.")
        if math.prod(self.desc["input_shape"]) < 1:
            raise ValueError("Exp requires a nonempty input.")

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        data = build_unary_same_shape_op(
            op_name="EXP", input_shape=self.desc["input_shape"], dtype="float32"
        )
        self._write_tflite_bytes(out_path, data)

    def generate_c_files(self, output_dir: Path) -> None:
        n = math.prod(self.desc["input_shape"])
        inputs = np.resize(
            np.array([-np.inf, -200, -1, -0.0, 0, 1, 100, np.inf, np.nan], dtype=np.float32),
            n,
        )
        with np.errstate(over="ignore", under="ignore"):
            golden = np.exp(inputs.astype(np.float64)).astype(np.float32)
        name = self.desc["name"]
        context = {
            "name": name,
            "count": n,
            "inputs": inputs.view(np.uint32).tolist(),
            "golden": golden.view(np.uint32).tolist(),
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
