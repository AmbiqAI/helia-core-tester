"""Generate distances and radial basis function goldens for one reference matrix."""

from pathlib import Path

import numpy as np

from helia_core_tester.generation.ops._shared.base import OperationBase


class OpRbfKernel(OperationBase):
    """Check identical points, equal distances, a full four-row block and its tail."""

    def needs_keras_model(self) -> bool:
        return False

    def build_keras_model(self):
        raise NotImplementedError("RbfKernel consumes a native reference table.")

    def allow_no_tflite(self) -> bool:
        return True

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        raise NotImplementedError(
            "The reference-table contract exercises three native entry points."
        )

    def validate_descriptor(self) -> None:
        if (
            self.tensor_dtype("input") != "FP32"
            or self.tensor_dtype("output") != "FP32"
        ):
            raise ValueError("RbfKernel requires FP32.")

    def generate_c_files(self, output_dir: Path) -> None:
        row = np.array([1, -2, 3])
        references = np.array(
            [[1, -2, 3], [1, -1, 3], [1, -3, 3], [2, 1, 4], [0, 0, 0]]
        )
        differences = references - row
        squares = np.square(differences).sum(axis=1)
        name = self.desc["name"]
        context = {
            "name": name,
            "row": row.tolist(),
            "references": references.ravel().tolist(),
            "rows": len(references),
            "features": len(row),
            "squares": squares.tolist(),
            "manhattan": np.abs(differences).sum(axis=1).tolist(),
            "rbf_bits": np.exp(-0.25 * squares)
            .astype(np.float32)
            .view(np.uint32)
            .tolist(),
        }
        self._write_op_outputs(
            output_dir,
            "rbf_kernel",
            "BasicMathFunctions/rbf_kernel/rbf_kernel.h.j2",
            "BasicMathFunctions/rbf_kernel/rbf_kernel.c.j2",
            context,
            {"name": name, "operator": "RbfKernel", "operator_name": "rbf_kernel"},
        )
