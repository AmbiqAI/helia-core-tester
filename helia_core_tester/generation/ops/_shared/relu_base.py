"""Shared golden path for Relu and Relu6: TFLite's quantized ReluX through the reference shim."""

from typing import Any, Dict, Tuple

import numpy as np

from helia_core_tester.generation.ops._shared.base import OperationBase

_RANGES = {"S8": ("s8", -128, 127), "S16": ("s16", -32768, 32767)}


class ReluFamilyBase(OperationBase):
    """Relu (ACT_MAX = inf) and Relu6 (ACT_MAX = 6)."""

    ACT_MAX = float("inf")
    KERNEL_PREFIX = "relu"
    # A one-signed draw leaves the clamp untested (and a small case degenerate).
    SIGN_SPAN_OPERANDS = ("input",)

    def uses_reference(self) -> bool:
        self._kind()
        return True

    def _kind(self) -> Tuple[str, int, int]:
        activation_dtype = self.desc.get('activation_dtype', 'S8')
        try:
            return _RANGES[activation_dtype]
        except KeyError as exc:
            raise NotImplementedError(f"Unsupported {self.KERNEL_PREFIX} dtype: {activation_dtype}") from exc

    def relu_reference(self) -> Dict[str, Any]:
        """Input draw, quantization, the prepared kernel parameters and the golden."""
        from helia_core_tester.generation.reference import bindings as b
        from helia_core_tester.generation.reference import policy
        from helia_core_tester.generation.reference.case import ReferenceCall

        kind, qmin, qmax = self._kind()
        shape = tuple(int(d) for d in self.desc['input_shape'])
        if not shape or any(d < 1 for d in shape):
            raise ValueError(f"{self.desc['name']}: invalid input_shape {shape}")

        rng_state = self.rng.__getstate__()
        self.rng = np.random.default_rng(self.seed)
        magnitudes = self.rng.uniform(0.0, 1.0, size=shape).astype(np.float32)
        # Half the elements negative, half positive: a small case drawn one-signed would
        # leave the golden degenerate and the clamp untested.
        signs = np.where(np.arange(magnitudes.size) < magnitudes.size // 2, -1.0, 1.0).astype(np.float32)
        self.rng.shuffle(signs)
        input_data = magnitudes * signs.reshape(shape)
        self.rng.__setstate__(rng_state)

        # Quantize over the [-1, 1] draw range (the converter's calibration range), not the
        # draw itself: a one-signed draw must still leave codes on both sides of zero.
        draw_range = np.array([-1.0, 1.0], dtype=np.float32)
        in_quant = self.activation_quant("input", draw_range, kind)
        out_quant = self.activation_quant("output", np.clip(draw_range, 0.0, self.ACT_MAX), kind)
        input_q = policy.quantize(input_data, in_quant)
        (input_q,) = self._enforce_int_operand_sign_span((("input", input_q, in_quant.zero_point),), steerable=("input",))
        params = b.struct_to_dict(b.get_bindings().relu_prepare(
            in_quant.scale, in_quant.zero_point, out_quant.scale, out_quant.zero_point, 0.0, self.ACT_MAX, qmin, qmax))
        call = ReferenceCall(
            f"relu_{kind}", params, {"input": input_q}, shape, input_q.dtype.name,
            quant={"input": in_quant.to_json(), "output": out_quant.to_json(), "act_max_real": self.ACT_MAX},
        )
        return {"shape": shape, "input_q": input_q, "params": params, "output": self.reference_golden(call)}
