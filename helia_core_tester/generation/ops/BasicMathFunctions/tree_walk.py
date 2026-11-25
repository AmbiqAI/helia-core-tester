"""Generate public packed-tree entry-point contract checks."""

from pathlib import Path

from helia_core_tester.generation.ops._shared.base import OperationBase


class OpTreeWalk(OperationBase):
    """Check leaf IDs, gathered values and class means against hand-evaluated stumps."""

    def needs_keras_model(self) -> bool:
        return False

    def build_keras_model(self):
        raise NotImplementedError("TreeWalk consumes a packed native tree table.")

    def allow_no_tflite(self) -> bool:
        return True

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        raise NotImplementedError(
            "The packed-tree contract uses native kernel entry points."
        )

    def validate_descriptor(self) -> None:
        if (
            self.tensor_dtype("input") != "FP32"
            or self.tensor_dtype("output") != "FP32"
        ):
            raise ValueError("Packed tree walk requires FP32.")

    def generate_c_files(self, output_dir: Path) -> None:
        name = self.desc["name"]
        self._write_op_outputs(
            output_dir,
            "tree_walk",
            "BasicMathFunctions/tree_walk/tree_walk.h.j2",
            "BasicMathFunctions/tree_walk/tree_walk.c.j2",
            {"name": name},
            {"name": name, "operator": "TreeWalk", "operator_name": "tree_walk"},
        )
