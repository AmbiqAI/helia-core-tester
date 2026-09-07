"""
Chunked-equivalence property cases for sliceable int kernels (issue #81).

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

Kernels differ in how a slice is expressed. Three families are covered:
  - block_size kernels (add/sub/mul/squared_difference) take a plain element
    count, so a slice is a pointer pair plus a shorter count;
  - dims kernels (minimum/maximum) take cmsis_nn_dims, so a slice is
    identical [1,1,1,chunk] dims for both inputs and the output with the
    data pointers advanced -- identical dims are what keeps the walk in
    Include/Internal/arm_nn_broadcast_walk.h on its no-broadcast path, which
    is the one with the vector loop;
  - unary kernels (arm_requantize_s8_s8) have one operand and a `size`.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
from pathlib import Path

from helia_core_tester.generation.ops._shared.base import OperationBase

# Kernel selection: (kernel, activation_dtype) -> call info.
#   call_style "addsub": per-input requantization + left_shift + block_size
#   call_style "mul":    input offsets, one output requantization, block_size
#   call_style "minmax": cmsis_nn_context + cmsis_nn_dims, no quant params
#   call_style "requantize": single operand, `size`, scale mult/shift + zero points
# "vector_widths" are the lane counts of the kernel's vectorised loops as the
# reference checkout writes them, MVE and DSP together -- not a blanket list.
# The chunk pattern must leave a tail for each of them and the sign check must
# hold inside each packed region, so a width that is not really there would
# constrain the pattern for nothing and a missing one would let a vacuous
# pattern through.
_KERNEL_TABLE: Dict[Tuple[str, str], Dict[str, Any]] = {
    ("add", "S8"): {
        "kernel_fn": "arm_elementwise_add_s8",
        "call_style": "addsub",
        "c_type": "int8_t",
        "vector_widths": (4,),
    },
    ("sub", "S8"): {
        "kernel_fn": "arm_elementwise_sub_s8",
        "call_style": "addsub",
        "c_type": "int8_t",
        "vector_widths": (4,),
    },
    # The s8 mul MVE main loop widens to int16 and does 8 elements per
    # iteration, with a 4-lane predicated remainder; the DSP loop is 4-wide.
    ("mul", "S8"): {
        "kernel_fn": "arm_elementwise_mul_s8",
        "call_style": "mul",
        "c_type": "int8_t",
        "vector_widths": (8, 4),
    },
    ("add", "S16"): {
        "kernel_fn": "arm_elementwise_add_s16",
        "call_style": "addsub",
        "c_type": "int16_t",
        "vector_widths": (4, 2),
    },
    ("sub", "S16"): {
        "kernel_fn": "arm_elementwise_sub_s16",
        "call_style": "addsub",
        "c_type": "int16_t",
        "vector_widths": (4, 2),
    },
    ("mul", "S16"): {
        "kernel_fn": "arm_elementwise_mul_s16",
        "call_style": "mul",
        "c_type": "int16_t",
        "vector_widths": (4, 2),
    },
    ("squared_difference", "S8"): {
        "kernel_fn": "arm_elementwise_squared_difference_s8",
        "call_style": "addsub",
        "c_type": "int8_t",
        "vector_widths": (4,),
    },
    ("squared_difference", "S16"): {
        "kernel_fn": "arm_elementwise_squared_difference_s16",
        "call_style": "addsub",
        "c_type": "int16_t",
        "vector_widths": (4,),
    },
    # The minimum/maximum MVE inner loops are wlstp.8 / wlstp.16 over 128-bit
    # vectors, so the lane count is 16 for s8 and 8 for s16; the non-MVE build
    # is a plain scalar loop with no packed width at all.
    ("minimum", "S8"): {
        "kernel_fn": "arm_minimum_s8",
        "call_style": "minmax",
        "c_type": "int8_t",
        "vector_widths": (16,),
    },
    ("maximum", "S8"): {
        "kernel_fn": "arm_maximum_s8",
        "call_style": "minmax",
        "c_type": "int8_t",
        "vector_widths": (16,),
    },
    ("minimum", "S16"): {
        "kernel_fn": "arm_minimum_s16",
        "call_style": "minmax",
        "c_type": "int16_t",
        "vector_widths": (8,),
    },
    ("maximum", "S16"): {
        "kernel_fn": "arm_maximum_s16",
        "call_style": "minmax",
        "c_type": "int16_t",
        "vector_widths": (8,),
    },
    ("requantize", "S8"): {
        "kernel_fn": "arm_requantize_s8_s8",
        "call_style": "requantize",
        "c_type": "int8_t",
        "vector_widths": (4,),
    },
}

# Number of input operands per call style.
_OPERAND_COUNT: Dict[str, int] = {"addsub": 2, "mul": 2, "minmax": 2, "requantize": 1}

# Fixed quantization parameters per (call_style, dtype). Any values work for
# the equivalence property (the full call is compared against the kernel
# itself), but these are chosen so the outputs use the full range without
# saturating everything, and -- for s8 add/sub -- they reuse the exact
# parameters of the ns-cmsis-nn#343 reproduction, which are proven to make
# the packed-loop sign defect visible.
# Note: the s16 add/sub kernels and both mul kernels ignore the input
# offsets (s16) or apply them before a single output requantization (mul).
# minimum/maximum take no quantization parameters at all.
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
    ("minmax", "S8"): {},
    ("minmax", "S16"): {},
    ("requantize", "S8"): {
        # (value + 3) / 2 + 2 over the whole int8 domain: no output saturates,
        # so a lane that takes the wrong path shows up as a value difference
        # rather than being clamped into agreement.
        "effective_scale_multiplier": 1 << 30,
        "effective_scale_shift": 0,
        "input_zeropoint": -3,
        "output_zeropoint": 2,
    },
}

# Squared difference squares the requantized operand difference, so the output
# requantization has to absorb that square. Reusing the add/sub output
# parameters pins the great majority of lanes at out_activation_max -- for s16,
# every lane whose operands differ by more than a fraction of full scale -- and
# an output that is constant across the run cannot show a chunk-boundary
# disagreement.
_QUANT_TABLE[("addsub", "S8", "squared_difference")] = {
    **_QUANT_TABLE[("addsub", "S8")],
    "out_shift": -5,
    "out_offset": -30,
}
_QUANT_TABLE[("addsub", "S16", "squared_difference")] = {
    **_QUANT_TABLE[("addsub", "S16")],
    "out_shift": -13,
    "out_offset": -8000,
}

# Guaranteed sign-diverse operand heads planted at the start of both inputs
# (inside the packed region of any run with >= 4 elements). The first four
# values of each vector are the exact operands of the ns-cmsis-nn#343
# reproduction, whose packed results differed from the tail results.
_SIGN_SEED_INPUT_1 = (-100, -1, 5, 60, -128, 127, -3, 3)
_SIGN_SEED_INPUT_2 = (-7, 3, -80, 20, 127, -128, 5, -5)


class OpChunkedEquivalence(OperationBase):
    """
    Block-size invariance ("chunked equivalence") case for sliceable int
    kernels: one full-length call is the reference for a sequence of chunked
    calls over the same data.
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
            supported = sorted(_KERNEL_TABLE)
            raise ValueError(
                f"ChunkedEquivalence descriptor '{self.desc.get('name')}' has unsupported "
                f"kernel/dtype '{kernel}'/{dtype}; supported kernel/dtype pairs: {supported}"
            )
        return kernel, dtype

    @staticmethod
    def _quant_params(call_style: str, dtype: str, kernel: str) -> Dict[str, int]:
        specialised = _QUANT_TABLE.get((call_style, dtype, kernel))
        if specialised is not None:
            return dict(specialised)
        return dict(_QUANT_TABLE[(call_style, dtype)])

    @staticmethod
    def _operand_offsets(call_style: str, quant: Dict[str, int]) -> List[int]:
        """
        Offset added to each stored operand value before the kernel does any
        arithmetic on it. This is what the sign check has to look at: a lane
        that is negative in memory but non-negative after the offset never
        exercises a sign-dependent packed path.
        """
        if call_style in ("addsub", "mul"):
            return [int(quant["input_1_offset"]), int(quant["input_2_offset"])]
        if call_style == "minmax":
            return [0, 0]
        # arm_requantize_s8_s8 computes value - input_zeropoint.
        return [-int(quant["input_zeropoint"])]

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

    @staticmethod
    def _check_chunk_discrimination(
        name: str,
        element_count: int,
        chunk_sizes: List[int],
        vector_widths: Tuple[int, ...],
    ) -> None:
        """
        Fail generation unless, for every vectorised width the kernel uses,
        the full call leaves a tail AND at least one chunk boundary falls
        strictly inside a vector. Without both, pass (a) and pass (b) route
        every element through the same path and the case cannot discriminate.
        """
        boundaries = np.cumsum(chunk_sizes[:-1]).tolist()
        for width in vector_widths:
            if element_count % width == 0:
                raise ValueError(
                    f"'{name}': element_count {element_count} is a multiple of the kernel's "
                    f"vector width {width}, so the full-length call has no tail and the "
                    f"chunked pass cannot expose a packed-vs-tail disagreement"
                )
            if not any(b % width for b in boundaries):
                raise ValueError(
                    f"'{name}': every chunk boundary {boundaries} is aligned to the kernel's "
                    f"vector width {width}; the chunked pass runs the same vector lanes as "
                    f"the full call and the case cannot discriminate"
                )

    # ------------------------------------------------------------------ data

    def _generate_operands(self, element_count: int, dtype: str, operand_count: int = 2) -> List[np.ndarray]:
        rng = self._seeded_rng()
        if dtype == "S8":
            low, high, np_dtype = -128, 127, np.int8
            seed_scale = 1
        else:
            low, high, np_dtype = -32768, 32767, np.int16
            seed_scale = 199  # spread the planted s8-range seeds across the s16 range
        seeds = (_SIGN_SEED_INPUT_1, _SIGN_SEED_INPUT_2)
        head = min(element_count, len(_SIGN_SEED_INPUT_1))
        operands = []
        for index in range(operand_count):
            data = rng.integers(low, high + 1, size=element_count).astype(np.int64)
            data[:head] = np.clip(np.array(seeds[index][:head]) * seed_scale, low, high)
            operands.append(data.astype(np_dtype))
        return operands

    @staticmethod
    def _check_sign_coverage(
        name: str,
        operand_label: str,
        data: np.ndarray,
        offset: int,
        element_count: int,
        vector_widths: Tuple[int, ...],
    ) -> None:
        """
        Fail generation unless value + offset spans negative AND positive
        inside every vectorised packed region (issue #81 property 2: cases
        whose packed lanes never see a negative post-offset operand cannot
        discriminate sign-dependent packed-path defects like #343).
        """
        post = data.astype(np.int64) + int(offset)
        for width in vector_widths:
            packed_len = (element_count // width) * width
            region = post[:packed_len] if packed_len else post
            scope = f"first {packed_len} elements (packed stride {width})" if packed_len else "whole vector"
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
        call_style = kernel_info["call_style"]
        vector_widths = tuple(kernel_info["vector_widths"])
        operand_count = _OPERAND_COUNT[call_style]
        quant = self._quant_params(call_style, dtype, kernel)
        element_count, chunk_sizes = self._element_count_and_chunks()
        self._check_chunk_discrimination(name, element_count, chunk_sizes, vector_widths)

        operands = self._generate_operands(element_count, dtype, operand_count)
        offsets = self._operand_offsets(call_style, quant)
        for index, (data, offset) in enumerate(zip(operands, offsets), start=1):
            self._check_sign_coverage(name, f"input_{index}", data, offset, element_count, vector_widths)

        builder = TemplateContextBuilder()
        context: Dict[str, Any] = {
            "name": name,
            "kernel_fn": kernel_info["kernel_fn"],
            "call_style": call_style,
            "operand_count": operand_count,
            "input_dtype": kernel_info["c_type"],
            "output_dtype": kernel_info["c_type"],
            "element_count": element_count,
            "chunk_count": len(chunk_sizes),
            "chunk_sizes_array": ", ".join(str(c) for c in chunk_sizes),
            "input1_data_array": builder.format_array_as_c_literal(operands[0]),
            "validation_mode": "exact_int",
            "validation_label": f"ChunkedEquivalence {kernel_info['kernel_fn']}",
            **quant,
        }
        if operand_count == 2:
            context["input2_data_array"] = builder.format_array_as_c_literal(operands[1])

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
