"""
Mutant catalog v1 for mutation scoring (issue #76).

Every mutant re-introduces a bug class that has actually escaped -- or was
proven able to escape -- this suite. Each entry names the target file inside
an ns-cmsis-nn checkout, the exact source patch (literal or regex), how many
replacements the patch MUST make, and the case family that is expected to
detect it. A mutant whose patch does not match (source drifted) is an
APPLY_FAILED result and makes the run fail loudly; it is never silently
skipped.

Grounding of the v1 entries:

- drop_conv_bias:        tester#77. Measured against ns-cmsis-nn 18a89ff in
                         ghcr.io/ambiqai/ns-cmsis-nn-ci:latest on cortex-m4,
                         seed 500, over a 105-case baseline: 44 kill it (24
                         non-dilated s8 Convolve cases plus 20 s4 cases).
                         Every survivor is a survivor by construction:
                         * quantized dilated convs -- the TFLite converter
                           hoists the bias into a trailing Add and leaves a
                           zero placeholder on the CONV_2D op (tester#98);
                         * use_bias: false cases, which have no bias to drop;
                         * s16 convs, because no s16 conv kernel is mutated;
                         * the s4 cases that do not execute a mutated route,
                           i.e. convolve_1x1_fast_s4, convolve_1x1_stride_s4
                           and the even/odd rhs/lhs bias s4 cases (no s4
                           route through arm_nn_mat_mult_nt_t_s4 is patched);
                         * the FullyConnected cases in the baseline set --
                           the mutant patches conv kernels only.
                         Covers the s8_s16 kernel, the s4_s16 kernel, the
                         1xN/1x1 non-fast route (arm_nn_mat_mult_nt_t_s8),
                         the grouped row-offset kernel, and the
                         arm_convolve_s8 leftover loop.
- packed_sign_mask_343:  ns-cmsis-nn#343 (packed DSP loop masked the
                         sign-extended halfword with & 0x0FFFF, disagreeing
                         with its own scalar tail). The patch removes the
                         (int16_t) cast that today keeps the mask benign.
- tail_loop_off_by_one:  classic tail-path bound bug: the scalar tail after
                         the packed 4-at-a-time loop processes one element
                         too few whenever block_size % 4 != 0.
- sub_operand_swap:      swapped operands in the subtract kernel. Included
                         as a control: golden cases must kill it, while the
                         chunked-equivalence property (which compares the
                         kernel against itself) provably cannot.
- requantize_shift_off_by_one: requantize applies one extra right-shift
                         (halves every requantized value). The gross end of
                         the shift-off-by-one class; expected to be killed
                         almost everywhere.
- broadcast_row_reuse:   the broadcast-walk row-reuse class (the escaped
                         min/max broadcast bug family): the NHWC walk stops
                         advancing input 1 across rows and re-reads row 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class Edit:
    """One source patch inside the mutated ns-cmsis-nn tree.

    ``pattern`` is matched against the file content: literally when
    ``regex`` is False, as a Python regular expression otherwise.
    ``count`` is the exact number of replacements the patch must make;
    any other number means the source has drifted and the mutant must be
    reported as failed-to-apply.
    """

    relpath: str
    pattern: str
    replacement: str
    count: int
    regex: bool = False


@dataclass(frozen=True)
class Mutant:
    """A named, catalogued kernel mutation."""

    mutant_id: str
    description: str
    family: str
    edits: Tuple[Edit, ...]
    expected_detected_by: str
    refs: Tuple[str, ...] = field(default_factory=tuple)


MUTANTS_V1: Tuple[Mutant, ...] = (
    Mutant(
        mutant_id="drop_conv_bias",
        description="Convolution kernels ignore the bias tensor entirely",
        family="ConvolutionFunctions",
        edits=(
            Edit(
                relpath="Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16.c",
                pattern="const int32_t *bias = output_bias;",
                replacement="const int32_t *bias = 0; (void)output_bias; /* MUTANT drop_conv_bias */",
                count=1,
            ),
            Edit(
                relpath="Source/ConvolutionFunctions/arm_convolve_s8.c",
                pattern="sum = bias_data_ptr[i];",
                replacement="sum = 0; /* MUTANT drop_conv_bias */",
                count=1,
            ),
            Edit(
                relpath="Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s4_s16.c",
                pattern="const int32_t *bias = output_bias;",
                replacement="const int32_t *bias = 0; (void)output_bias; /* MUTANT drop_conv_bias */",
                count=1,
            ),
            Edit(
                relpath="Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_row_offset_s8_s16.c",
                pattern="const int32_t *bias = output_bias;",
                replacement="const int32_t *bias = 0; (void)output_bias; /* MUTANT drop_conv_bias */",
                count=1,
            ),
            Edit(
                relpath="Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8.c",
                pattern="    (void)lhs_offset;",
                replacement="    (void)lhs_offset;\n    bias = 0; /* MUTANT drop_conv_bias */",
                count=1,
            ),
        ),
        expected_detected_by=(
            "conv golden cases with a nonzero bias. Measured on cortex-m4 against "
            "ns-cmsis-nn 18a89ff (seed 500, 105 cases): 44 killers, 24 non-dilated s8 "
            "Convolve cases plus 20 s4 cases. Survivors are the quantized dilated convs "
            "(bias hoisted into a trailing Add by the converter, tester#98), the "
            "use_bias: false cases, the s16 convs and the s4 cases that do not execute a "
            "mutated route (no s16 conv kernel and no arm_nn_mat_mult_nt_t_s4 route is "
            "patched), and the FullyConnected cases (the mutant touches conv kernels "
            "only) (tester#77)"
        ),
        refs=("AmbiqAI/helia-core-tester#77",),
    ),
    Mutant(
        mutant_id="packed_sign_mask_343",
        description="Packed DSP add loop drops the sign of value + offset (& 0x0FFFF without cast)",
        family="BasicMathFunctions",
        edits=(
            Edit(
                relpath="Source/BasicMathFunctions/arm_elementwise_add_s8.c",
                pattern=r"\(int16_t\)\((?P<lane>[ab]_[12]) & 0x0FFFF\)",
                replacement=r"(\g<lane> & 0x0FFFF) /* MUTANT packed_sign_mask_343 */",
                count=4,
                regex=True,
            ),
        ),
        expected_detected_by="chunked-equivalence add s8 cases (hct#88); packed loop vs scalar tail disagree",
        refs=("AmbiqAI/ns-cmsis-nn#343", "AmbiqAI/helia-core-tester#88"),
    ),
    Mutant(
        mutant_id="tail_loop_off_by_one",
        description="Scalar tail after the packed add loop processes one element too few",
        family="BasicMathFunctions",
        edits=(
            Edit(
                relpath="Source/BasicMathFunctions/arm_elementwise_add_s8.c",
                pattern="loop_count = block_size & 0x3;",
                replacement="loop_count = (block_size & 0x3) - 1; /* MUTANT tail_loop_off_by_one */",
                count=1,
            ),
        ),
        expected_detected_by="add s8 cases whose element count is not a multiple of 4",
        refs=(),
    ),
    Mutant(
        mutant_id="sub_operand_swap",
        description="Subtract kernel computes input_2 - input_1",
        family="BasicMathFunctions",
        edits=(
            Edit(
                relpath="Source/BasicMathFunctions/arm_elementwise_sub_s8.c",
                pattern="diff = input_1 - input_2;",
                replacement="diff = input_2 - input_1; /* MUTANT sub_operand_swap */",
                count=5,
            ),
        ),
        expected_detected_by="sub s8 golden cases; chunked-equivalence cannot kill this by construction",
        refs=(),
    ),
    Mutant(
        mutant_id="requantize_shift_off_by_one",
        description="arm_nn_requantize applies one extra right shift (halves every result)",
        family="NNSupportFunctions",
        edits=(
            Edit(
                relpath="Include/arm_nnsupportfunctions.h",
                pattern=(
                    r"arm_nn_doubling_high_mult_no_sat\(val \* \(1 << LEFT_SHIFT\(shift\)\), multiplier\),\n"
                    r"(?P<indent>\s*)RIGHT_SHIFT\(shift\)\);"
                ),
                replacement=(
                    "arm_nn_doubling_high_mult_no_sat(val * (1 << LEFT_SHIFT(shift)), multiplier),\n"
                    "\\g<indent>RIGHT_SHIFT(shift) + 1); /* MUTANT requantize_shift_off_by_one */"
                ),
                count=1,
                regex=True,
            ),
        ),
        expected_detected_by="virtually every requantizing int case",
        refs=(),
    ),
    Mutant(
        mutant_id="broadcast_row_reuse",
        description="NHWC broadcast walk re-reads row 0 of input 1 for every output row",
        family="BroadcastWalk",
        edits=(
            Edit(
                relpath="Include/Internal/arm_nn_broadcast_walk.h",
                pattern=r"bw_r_1 = bw_p_1 \+ bw_h \* bw_h_stride_1;",
                replacement="bw_r_1 = bw_p_1 + 0 * bw_h_stride_1; /* MUTANT broadcast_row_reuse */",
                count=1,
                regex=True,
            ),
        ),
        expected_detected_by="broadcast cases where input 1 has more than one row and the walk descends to rows",
        refs=("ns-cmsis-nn#321",),
    ),
)


def get_mutants(ids=None):
    """Return the v1 catalog, optionally filtered to a list of mutant ids.

    Unknown ids raise KeyError -- a filter typo must not silently shrink a run.
    """
    if ids is None:
        return list(MUTANTS_V1)
    by_id = {m.mutant_id: m for m in MUTANTS_V1}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise KeyError(f"Unknown mutant id(s): {', '.join(missing)}")
    return [by_id[i] for i in ids]
