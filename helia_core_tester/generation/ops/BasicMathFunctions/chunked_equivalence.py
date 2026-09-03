"""
Chunked-equivalence property cases for elementwise kernels (issue #81).

Every regular descriptor runs a kernel once, at one element count. A defect
that only appears at certain block sizes -- e.g. a packed 4-at-a-time DSP loop
that disagrees with its own scalar tail (ns-cmsis-nn#343) -- is invisible to
such cases no matter how many are added.

These cases assert a self-checking property instead: processing N elements in
ONE call must produce bit-identical results to processing the SAME data as a
sequence of calls over consecutive slices. The full-length call (a) is the
reference for the chunked calls (b); no golden/expected data is needed.

Chunk patterns are chosen so that (a) and (b) route the same elements through
different kernel paths:
  - all-singles ([1] * N) forces every element of (b) through the scalar tail
    while (a) runs the packed/vectorised loop for the first (N // stride)
    groups -- this is exactly the ns-cmsis-nn#343 discriminator;
  - mixed "offcut" chunks (sizes with mod-4 / mod-2 remainders) start packed
    loops on odd element boundaries and exercise every tail length.

Operand sign matters (the #343 trigger was sign-dependent: the packed loop
masked away the sign of value + input_offset). Generation therefore plants
strongly negative operands inside the packed region and fails the descriptor
if the data does not span negative and positive post-offset values there.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
from pathlib import Path

from helia_core_tester.generation.ops._shared.base import OperationBase

# Kernel selection: (kernel, activation_dtype) -> call info.
# "addsub" kernels take per-input requantization + left_shift;
# "mul" kernels only requantize the product.
_KERNEL_TABLE: Dict[Tuple[str, str], Dict[str, Any]] = {
    ("add", "S8"): {"kernel_fn": "arm_elementwise_add_s8", "call_style": "addsub", "c_type": "int8_t"},
    ("sub", "S8"): {"kernel_fn": "arm_elementwise_sub_s8", "call_style": "addsub", "c_type": "int8_t"},
    ("mul", "S8"): {"kernel_fn": "arm_elementwise_mul_s8", "call_style": "mul", "c_type": "int8_t"},
    ("add", "S16"): {"kernel_fn": "arm_elementwise_add_s16", "call_style": "addsub", "c_type": "int16_t"},
    ("sub", "S16"): {"kernel_fn": "arm_elementwise_sub_s16", "call_style": "addsub", "c_type": "int16_t"},
    ("mul", "S16"): {"kernel_fn": "arm_elementwise_mul_s16", "call_style": "mul", "c_type": "int16_t"},
}

# Fixed quantization parameters per (call_style, dtype). Any values work for
# the equivalence property (the full call is compared against the kernel
# itself), but these are chosen so the outputs use the full range without
# saturating everything, and -- for s8 add/sub -- they reuse the exact
# parameters of the ns-cmsis-nn#343 reproduction, which are proven to make
# the packed-loop sign defect visible.
# Note: the s16 add/sub kernels and both mul kernels ignore the input
# offsets (s16) or apply them before a single output requantization (mul).
_QUANT_TABLE: Dict[Tuple[str, str], Dict[str, int]] = {
    ("addsub", "S8"): {
        "input_1_offset": -3,
        "input_1_mult": 1 << 30,
        "input_1_shift": -1,
        "input_2_offset": 5,
        "input_2_mult": 1 << 30,
        "input_2_shift": -1,
        "left_shift": 0,
        "out_offset": 2,
        "out_mult": 1 << 30,
        "out_shift": -1,
        "out_activation_min": -128,
        "out_activation_max": 127,
    },
    ("addsub", "S16"): {
        # s16 add/sub ignore the input and output offsets (zero-point is 0).
        "input_1_offset": 0,
        "input_1_mult": 1 << 30,
        "input_1_shift": -1,
        "input_2_offset": 0,
        "input_2_mult": 1 << 30,
        "input_2_shift": -1,
        "left_shift": 0,
        "out_offset": 0,
        "out_mult": 1 << 30,
        "out_shift": -1,
        "out_activation_min": -32768,
        "out_activation_max": 32767,
    },
    ("mul", "S8"): {
        "input_1_offset": -3,
        "input_2_offset": 5,
        "out_offset": 2,
        "out_mult": 1 << 30,
        "out_shift": -8,
        "out_activation_min": -128,
        "out_activation_max": 127,
    },
    ("mul", "S16"): {
        # s16 mul ignores the offsets.
        "input_1_offset": 0,
        "input_2_offset": 0,
        "out_offset": 0,
        "out_mult": 1 << 30,
        "out_shift": -14,
        "out_activation_min": -32768,
        "out_activation_max": 32767,
    },
}

# Guaranteed sign-diverse operand heads planted at the start of both inputs
# (inside the packed region of any run with >= 4 elements). The first four
# values of each vector are the exact operands of the ns-cmsis-nn#343
# reproduction, whose packed results differed from the tail results.
_SIGN_SEED_INPUT_1 = (-100, -1, 5, 60, -128, 127, -3, 3)
_SIGN_SEED_INPUT_2 = (-7, 3, -80, 20, 127, -128, 5, -5)

# Vectorised strides whose packed regions must see negative post-offset
# operands: 4 (Armv7E-M DSP s8 quad loop), 2 (s16 dual-halfword loop). MVE
# tail-predicated loops have no separate tail path but are covered anyway.
_PACKED_STRIDES = (4, 2)


class OpChunkedEquivalence(OperationBase):
    """
    Block-size invariance ("chunked equivalence") case for elementwise
    add/sub/mul kernels: one full-length call is the reference for a sequence
    of chunked calls over the same data.
    """

    def needs_keras_model(self) -> bool:
        return False

    def allow_no_tflite(self) -> bool:
        return True

    def build_keras_model(self):
        raise NotImplementedError("ChunkedEquivalence is a kernel-property case; it has no model.")

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        raise NotImplementedError("ChunkedEquivalence needs no TFLite model: the full-length call is the reference.")

    # ---------------------------------------------------------------- config

    def _kernel_key(self) -> Tuple[str, str]:
        kernel = str(self.desc.get("kernel", "")).strip().lower()
        dtype = self.tensor_dtype("input", default=str(self.desc.get("activation_dtype", "S8"))).upper()
        if (kernel, dtype) not in _KERNEL_TABLE:
            supported = sorted({k for k, _ in _KERNEL_TABLE})
            raise ValueError(
                f"ChunkedEquivalence descriptor '{self.desc.get('name')}' has unsupported "
                f"kernel/dtype '{kernel}'/{dtype}; supported kernels: {supported} for S8/S16"
            )
        return kernel, dtype

    def _element_count_and_chunks(self) -> Tuple[int, List[int]]:
        element_count = int(self.desc["element_count"])
        chunk_sizes = [int(c) for c in self.desc["chunk_sizes"]]
        if element_count < 1:
            raise ValueError(f"'{self.desc.get('name')}': element_count must be >= 1")
        if any(c < 1 for c in chunk_sizes):
            raise ValueError(f"'{self.desc.get('name')}': every chunk size must be >= 1, got {chunk_sizes}")
        if sum(chunk_sizes) != element_count:
            raise ValueError(
                f"'{self.desc.get('name')}': chunk_sizes {chunk_sizes} sum to {sum(chunk_sizes)}, "
                f"expected element_count {element_count}"
            )
        if len(chunk_sizes) < 2:
            raise ValueError(
                f"'{self.desc.get('name')}': need at least 2 chunks, otherwise the chunked pass "
                f"is the same call as the full pass and the case cannot discriminate"
            )
        return element_count, chunk_sizes

    # ------------------------------------------------------------------ data

    def _generate_operands(self, element_count: int, dtype: str) -> Tuple[np.ndarray, np.ndarray]:
        rng = self._seeded_rng()
        if dtype == "S8":
            low, high, np_dtype = -128, 127, np.int8
            seed_scale = 1
        else:
            low, high, np_dtype = -32768, 32767, np.int16
            seed_scale = 199  # spread the planted s8-range seeds across the s16 range
        input_1 = rng.integers(low, high + 1, size=element_count).astype(np.int64)
        input_2 = rng.integers(low, high + 1, size=element_count).astype(np.int64)
        head = min(element_count, len(_SIGN_SEED_INPUT_1))
        input_1[:head] = np.clip(np.array(_SIGN_SEED_INPUT_1[:head]) * seed_scale, low, high)
        input_2[:head] = np.clip(np.array(_SIGN_SEED_INPUT_2[:head]) * seed_scale, low, high)
        return input_1.astype(np_dtype), input_2.astype(np_dtype)

    @staticmethod
    def _check_sign_coverage(
        name: str,
        operand_label: str,
        data: np.ndarray,
        offset: int,
        element_count: int,
    ) -> None:
        """
        Fail generation unless value + input_offset spans negative AND
        positive inside every vectorised packed region (issue #81 property 2:
        cases whose packed lanes never see a negative post-offset operand
        cannot discriminate sign-dependent packed-path defects like #343).
        """
        post = data.astype(np.int64) + int(offset)
        for stride in _PACKED_STRIDES:
            packed_len = (element_count // stride) * stride
            region = post[:packed_len] if packed_len else post
            scope = f"first {packed_len} elements (packed stride {stride})" if packed_len else "whole vector"
            if not (region < 0).any() or not (region > 0).any():
                raise ValueError(
                    f"'{name}': {operand_label} post-offset values do not span negative and "
                    f"positive within the {scope}; the case cannot catch sign-dependent "
                    f"packed-path defects (ns-cmsis-nn#343). Adjust the data seed/offsets."
                )

    # ------------------------------------------------------------- generation

    def generate_c_files(self, output_dir: Path) -> None:
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name = self.desc["name"]
        kernel, dtype = self._kernel_key()
        kernel_info = _KERNEL_TABLE[(kernel, dtype)]
        quant = dict(_QUANT_TABLE[(kernel_info["call_style"], dtype)])
        element_count, chunk_sizes = self._element_count_and_chunks()

        input_1, input_2 = self._generate_operands(element_count, dtype)
        self._check_sign_coverage(name, "input_1", input_1, quant["input_1_offset"], element_count)
        self._check_sign_coverage(name, "input_2", input_2, quant["input_2_offset"], element_count)

        builder = TemplateContextBuilder()
        context: Dict[str, Any] = {
            "name": name,
            "kernel_fn": kernel_info["kernel_fn"],
            "call_style": kernel_info["call_style"],
            "input_dtype": kernel_info["c_type"],
            "output_dtype": kernel_info["c_type"],
            "element_count": element_count,
            "chunk_count": len(chunk_sizes),
            "chunk_sizes_array": ", ".join(str(c) for c in chunk_sizes),
            "input1_data_array": builder.format_array_as_c_literal(input_1),
            "input2_data_array": builder.format_array_as_c_literal(input_2),
            "validation_mode": "exact_int",
            "validation_label": f"ChunkedEquivalence {kernel_info['kernel_fn']}",
            **quant,
        }

        cmake_context = {
            "name": name,
            "operator": self.desc.get("operator", "ChunkedEquivalence"),
            "operator_name": "chunked_equivalence",
        }
        self._write_op_outputs(
            output_dir,
            "chunked_equivalence",
            "BasicMathFunctions/chunked_equivalence/chunked_equivalence.h.j2",
            "BasicMathFunctions/chunked_equivalence/chunked_equivalence.c.j2",
            context,
            cmake_context,
        )
