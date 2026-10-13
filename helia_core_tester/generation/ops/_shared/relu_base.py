"""Shared golden path of Relu and Relu6: TFLite's ReluQuantized on the C reference."""

from typing import Any, Dict, Tuple

import numpy as np

from helia_core_tester.generation.ops._shared.base import OperationBase

_KINDS = {"S8": "s8", "S16": "s16"}


class ReluFamilyBase(OperationBase):
    """Relu (ACT_MAX = inf) and Relu6 (ACT_MAX = 6)."""

    ACT_MAX = float("inf")
    OPERATOR = "Relu"
    # A one-signed draw leaves the clamp untested (and a small case degenerate).
    SIGN_SPAN_OPERANDS = ("input",)

    def uses_reference(self) -> bool:
        return True

    def _kind(self) -> str:
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        if activation_dtype not in _KINDS:
            raise NotImplementedError(f"Unsupported {self.OPERATOR} dtype: {activation_dtype}")
        return _KINDS[activation_dtype]

    def _balanced_draw(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Uniform magnitudes in [0, 1), half of the elements negative."""
        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        magnitudes = self.rng.uniform(0.0, 1.0, size=shape).astype(np.float32)
        signs = np.where(np.arange(magnitudes.size) < magnitudes.size // 2, -1.0, 1.0).astype(np.float32)
        self.rng.shuffle(signs)
        self.rng.__setstate__(rng_state)
        return magnitudes * signs.reshape(shape)

    def relu_reference(self) -> Dict[str, Any]:
        """Input draw, quantization, the prepared kernel parameters and the golden."""
        from helia_core_tester.generation.reference import policy
        from helia_core_tester.generation.reference import quant as ref_quant
        from helia_core_tester.generation.reference.bindings import get_bindings
        from helia_core_tester.generation.reference.call import ReferenceCall

        kind = self._kind()
        shape = tuple(int(d) for d in self.desc['input_shape'])
        if not shape or any(d < 1 for d in shape):
            raise ValueError(f"{self.desc['name']}: invalid input_shape {shape}")
        input_data = self._balanced_draw(shape)
        # Quantized over the [-1, 1] draw range (the range the converter used to calibrate
        # over), not the draw itself, so even a small draw keeps codes on both sides of zero.
        in_quant = self.activation_quant("input", (-1.0, 1.0), kind)
        out_quant = self.activation_quant("output", (0.0, min(1.0, self.ACT_MAX)), kind)
        input_q = policy.quantize(input_data, in_quant)
        (input_q,) = self._enforce_int_operand_sign_span((("input", input_q, in_quant.zero_point),),
                                                         steerable=("input",))
        params = get_bindings().prepare("relu_prepare", {
            "dtype": ref_quant.hct_dtype(kind.upper()), "input_scale": in_quant.scale,
            "input_zero_point": in_quant.zero_point, "output_scale": out_quant.scale,
            "output_zero_point": out_quant.zero_point, "act_max": self.ACT_MAX,
        })
        output = self.reference_golden(ReferenceCall(
            f"relu_{kind}", params, {"input": np.ascontiguousarray(input_q)}, {"output": shape},
            quant={"input": in_quant.to_json(), "output": out_quant.to_json(), "act_max": self.ACT_MAX},
        ))
        return {"shape": shape, "input_q": input_q, "params": params, "output": output}
