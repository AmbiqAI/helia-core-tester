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

- drop_conv_bias:        tester#77, tester#98. Measured against ns-cmsis-nn
                         18a89ff in ghcr.io/ambiqai/ns-cmsis-nn-ci:latest on
                         cortex-m4, seed 500, --ops Convolve,DepthwiseConv,
                         over a 123-case set: 51 kill it (24 non-dilated s8
                         Convolve cases, the 7 dilated s8 Convolve cases that
                         tester#98 gave an accumulator-scale bias, and 20 s4
                         cases). Every survivor is a survivor by construction:
                         * use_bias: false cases, which have no bias to drop;
                         * s16 convs, because no s16 conv kernel is mutated;
                         * every DepthwiseConv case -- the mutant patches
                           convolution kernels only;
                         * the five s4 cases that do not execute a mutated
                           route -- convolve_1x1_fast_s4,
                           convolve_1x1_stride_s4,
                           convolve_even_rhs50_lhs1_bias_s4,
                           convolve_odd_rhs3_lhs1_bias_s4 and
                           convolve_odd_rhs5_lhs1_bias_s4 (no s4 route
                           through arm_nn_mat_mult_nt_t_s4 is patched).
                         Covers the s8_s16 kernel, the s4_s16 kernel, the
                         1xN/1x1 non-fast route (arm_nn_mat_mult_nt_t_s8),
                         the grouped row-offset kernel, and the
                         arm_convolve_s8 leftover loop.
- drop_depthwise_bias:   tester#98. Measured against ns-cmsis-nn 18a89ff in
                         ghcr.io/ambiqai/ns-cmsis-nn-ci:latest on cortex-m4,
                         seed 500, --ops Convolve,DepthwiseConv, over the same
                         123-case set: 16 kill it, all of them DepthwiseConv
                         cases, the five dilated quantized ones included.
                         Survivors are the Convolve cases (the mutant patches
                         the depthwise entry points only), the 14 use_bias:
                         false depthwise cases, and the 18 bias-carrying
                         depthwise cases whose wrapper dispatches straight to
                         an optimized kernel rather than through one of the
                         three mutated entry points.
- drop_conv_s16_bias:    tester#98. Same checkout, image, cpu, seed and
                         123-case set: 9 kill it -- every s16 Convolve case
                         that carries a bias, the six dilated ones included.
                         The remaining nine s16 Convolve cases are use_bias:
                         false; every other survivor is an s8/s4 conv or a
                         DepthwiseConv case, which no edit here touches. The
                         arm_convolve_s16_fast_small_kernel.c edits sit inside
                         that file's ARM_MATH_MVEI branch, so they are inert on
                         cortex-m4 and the 9/123 figure is the non-MVE routes
                         only; an MVE run reaches more entry points and differs
                         by construction.
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
- squared_difference_tail_drop, minmax_no_broadcast_tail_drop,
  requantize_tail_drop: the same tail-drop class as tail_loop_off_by_one,
                         planted in the three families that gained chunked
                         cases in issue #81. Each is expressed in every
                         compiled variant of its kernel, scalar and vectorised
                         alike: the host scorer builds with ARM_MATH_DSP and no
                         MVEI, so an MVE-only edit would be dead code there and
                         score as vacuous, while a scalar-only edit would be
                         dead code on the FVP targets the suite actually runs.
                         Their chunked killers are gated on the mve capability,
                         which is why the CLI generates for cortex-m55.
- drop_conv_ctx_guard, drop_conv_group_guard, drop_conv_1xn_zero_stride_guard,
  drop_depthwise_channel_guard, drop_depthwise_ctx_guard,
  drop_transpose_conv_ctx_guard, drop_transpose_conv_dilation_guard,
  drop_transpose_conv_reverse_ctx_guard, drop_fc_s16_ctx_guard,
  drop_fc_s16_ctx_size_guard, drop_svdf_scratch_guard, drop_pool_batch_guard:
                         the argument-rejection class behind tester#72. Each
                         removes one ARM_CMSIS_NN_ARG_ERROR guard a fault
                         descriptor targets (the int-suite `<op>_fault_<kind>`
                         cases, which assert the returned status and never
                         validate output). Every edit here sits on a code path
                         the host build reaches: the DSP-only guards are live
                         under ARM_MATH_DSP, and where a kernel keeps one copy
                         of the check per MVE/non-MVE body both copies are
                         patched. The guards that exist only inside an
                         ARM_MATH_MVEI block -- the weight_sum_ctx checks in
                         arm_convolve_s8/1_x_n/depthwise_s8_opt, the ctx
                         checks in arm_fully_connected_s8/per_channel_s8,
                         arm_batch_matmul_s8 and arm_svdf_s8 -- are dead code
                         on the host, and their `required_capabilities: [mve]`
                         cases fail the host baseline (the kernel returns
                         SUCCESS) and are excluded before scoring. They are
                         deliberately not catalogued: a survivor there would
                         blame the suite for a path the scorer cannot compile.
                         The FVP cortex-m55 runs are the check for those.

- conv_sizer_negative, depthwise_sizer_negative, fc_sizer_negative,
  avgpool_sizer_negative, transpose_conv_sizer_negative,
  transpose_conv_reverse_sizer_negative, svdf_kernel_sum_sizer_negative:
                         the scratch-sizer sentinel class behind tester#133.
                         Each makes one *_get_buffer_size() the harness calls
                         return the documented negative out-of-range value with
                         a minimal literal edit. Before #133 that answer became
                         ctx.size unremarked -- a negative number is not larger
                         than the static the case allocated, so the capacity
                         comparison passed, and most kernels never read
                         ctx.size -- and every one of these mutants survived.
                         Where a kernel family exposes plain / _dsp / _mve
                         sizer entry points, all of them are patched by the one
                         edit, because the harness calls whichever the
                         generation CPU selected. Batch matmul has no entry of
                         its own: the harness sizes its int cases with
                         arm_fully_connected_{s8,s16}_get_buffer_size, so
                         fc_sizer_negative covers it. Transpose conv gets two
                         mutants rather than one because HELIA_VALIDATE_SIZER
                         returns on the first failure, so a forward-sizer
                         mutation would mask the reverse-sizer check.

                         Scope of the class, and why each boundary is where it
                         is:
                         * Each mutant patches the _s8 sizer file of its
                           family only. The s4 and s16 sizers live in sibling
                           arm_*_get_buffer_sizes_s4.c / _s16.c files and are
                           deliberately untouched, so s4 and s16 cases are
                           survivors by construction; each entry says so.
                         * The float sizers are untouched because host_build.py
                           excludes f16/f32 sources from the mutation library,
                           so a float edit would be dead code on the scorer and
                           would score as vacuous.
                         * Only average pooling has a mutant in the pooling
                           family, and only average pooling carries the checks.
                           pool_base.py sets kernel_get_buffer_size_fn for avg
                           pool alone, so max_pool.c.j2's sizer block never
                           renders and its capacity constant is zero, which
                           would reject any positive answer if it ever did.
                           Instrumenting it would be dead template text, so it
                           is left alone and there is no sizer for a mutant to
                           target.
                         * An under-reporting sizer -- one that answers a few
                           bytes short -- is NOT detectable by this work and is
                           deliberately not catalogued: the answer stays
                           non-negative, stays below the conservatively sized
                           static, and is read by nothing on that route, so
                           neither #133 check can fire. Catching it needs the
                           harness to compare the answer against an independent
                           bound, which is separate work.

A mutant may declare ``requires_capabilities``: the capabilities the generated
corpus needs before any case that could kill it exists at all. Scoring on a CPU
that lacks one reports NOT_APPLICABLE rather than SURVIVED, because a corpus
that cannot contain a killer says nothing about the suite's power against that
bug class, and --fail-on-survivor must not fire on it.
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
    # Capabilities the generation CPU must provide for the corpus to contain
    # any case that could kill this mutant. Empty means every CPU can score it.
    requires_capabilities: Tuple[str, ...] = field(default_factory=tuple)


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
            "ns-cmsis-nn 18a89ff (seed 500, --ops Convolve,DepthwiseConv, 123 cases): "
            "51 killers, 24 non-dilated s8 Convolve cases, 7 dilated s8 Convolve cases "
            "and 20 s4 cases. Survivors are the use_bias: false cases, the s16 convs, "
            "every DepthwiseConv case (the mutant touches conv kernels only), and the "
            "five s4 cases that do not execute a mutated route (convolve_1x1_fast_s4, "
            "convolve_1x1_stride_s4, convolve_even_rhs50_lhs1_bias_s4, "
            "convolve_odd_rhs3_lhs1_bias_s4 and convolve_odd_rhs5_lhs1_bias_s4 -- no s4 "
            "route through arm_nn_mat_mult_nt_t_s4 is patched) (tester#77, tester#98)"
        ),
        refs=("AmbiqAI/helia-core-tester#77", "AmbiqAI/helia-core-tester#98"),
    ),
    Mutant(
        mutant_id="drop_depthwise_bias",
        description="Depthwise convolution kernels ignore the bias tensor entirely",
        family="ConvolutionFunctions",
        edits=(
            Edit(
                relpath="Source/ConvolutionFunctions/arm_depthwise_conv_s8.c",
                pattern="    (void)bias_dims;",
                replacement="    (void)bias_dims;\n    bias = NULL; /* MUTANT drop_depthwise_bias */",
                count=1,
            ),
            Edit(
                relpath="Source/ConvolutionFunctions/arm_depthwise_conv_s16.c",
                pattern="    (void)bias_dims;",
                replacement="    (void)bias_dims;\n    bias = NULL; /* MUTANT drop_depthwise_bias */",
                count=1,
            ),
            Edit(
                relpath="Source/ConvolutionFunctions/arm_depthwise_conv_s4.c",
                pattern="    (void)bias_dims;",
                replacement="    (void)bias_dims;\n    bias = NULL; /* MUTANT drop_depthwise_bias */",
                count=1,
            ),
        ),
        expected_detected_by=(
            "depthwise golden cases with a nonzero bias that reach one of the three "
            "mutated entry points. Measured on cortex-m4 against ns-cmsis-nn 18a89ff "
            "(seed 500, --ops Convolve,DepthwiseConv, 123 cases): 16 killers, every one "
            "a DepthwiseConv case, including all five dilated ones the conv mutants "
            "cannot reach (depthwise_conv_dilation_s8, depthwise_conv_dilation_s16, "
            "depthwise_conv_buf_nonopt_dil2_s8, depthwise_conv_dil_2x1_bias_s16 and "
            "depthwise_conv_fast_false_dil2_bias_s16). Survivors are the Convolve "
            "cases, the 14 use_bias: false depthwise cases and the 18 bias-carrying "
            "depthwise cases the wrapper sends to an optimized kernel instead "
            "(tester#98)"
        ),
        refs=("AmbiqAI/helia-core-tester#98",),
    ),
    Mutant(
        mutant_id="drop_conv_s16_bias",
        description="s16 convolution kernels ignore the bias tensor entirely",
        family="ConvolutionFunctions",
        edits=(
            Edit(
                relpath="Source/ConvolutionFunctions/arm_convolve_s16.c",
                pattern="const int64_t *bias_s64 = (const int64_t *)bias_data_ptr->data;",
                replacement="const int64_t *bias_s64 = NULL; /* MUTANT drop_conv_s16_bias */",
                count=1,
            ),
            Edit(
                relpath="Source/ConvolutionFunctions/arm_convolve_s16.c",
                pattern="const int32_t *bias_s32 = (const int32_t *)bias_data_ptr->data;",
                replacement="const int32_t *bias_s32 = NULL; /* MUTANT drop_conv_s16_bias */",
                count=1,
            ),
            Edit(
                relpath="Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s16.c",
                pattern="const int64_t *bias_s64 = (const int64_t *)bias_data->data;",
                replacement="const int64_t *bias_s64 = NULL; /* MUTANT drop_conv_s16_bias */",
                count=1,
            ),
            Edit(
                relpath="Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s16.c",
                pattern="const int32_t *bias_s32 = (const int32_t *)bias_data->data;",
                replacement="const int32_t *bias_s32 = NULL; /* MUTANT drop_conv_s16_bias */",
                count=1,
            ),
            Edit(
                relpath="Source/ConvolutionFunctions/arm_convolve_s16_group_ch_mult_1.c",
                pattern="const bool has_bias = (bias_data != NULL) && (bias_data->data != NULL);",
                replacement="const bool has_bias = false; /* MUTANT drop_conv_s16_bias */",
                count=1,
            ),
            Edit(
                relpath="Source/ConvolutionFunctions/arm_convolve_s16_fast_small_kernel.c",
                pattern="const int64_t *bias_s64 = (const int64_t *)bias_data->data;",
                replacement="const int64_t *bias_s64 = NULL; /* MUTANT drop_conv_s16_bias */",
                count=1,
            ),
            Edit(
                relpath="Source/ConvolutionFunctions/arm_convolve_s16_fast_small_kernel.c",
                pattern="const int32_t *bias_s32 = (const int32_t *)bias_data->data;",
                replacement="const int32_t *bias_s32 = NULL; /* MUTANT drop_conv_s16_bias */",
                count=1,
            ),
        ),
        expected_detected_by=(
            "s16 conv golden cases with a nonzero bias. Measured on cortex-m4 against "
            "ns-cmsis-nn 18a89ff (seed 500, --ops Convolve,DepthwiseConv, 123 cases): "
            "9 killers, which is every bias-carrying s16 Convolve case in the set, "
            "including the six dilated ones drop_conv_bias cannot reach "
            "(convolve_int16xint8_dilation_case_01_s16 through _case_03_s16 and "
            "convolve_int16xint8xint32_case_04_s16 through _case_06_s16). Survivors "
            "are the nine use_bias: false s16 Convolve cases and everything that is "
            "not an s16 conv (tester#98)"
        ),
        refs=("AmbiqAI/helia-core-tester#98",),
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
    Mutant(
        mutant_id="squared_difference_tail_drop",
        description="Squared-difference kernels process only the 4-aligned prefix, never the tail",
        family="BasicMathFunctions",
        edits=(
            Edit(
                relpath="Source/BasicMathFunctions/arm_elementwise_squared_difference_s8.c",
                pattern="int32_t loop_count = block_size;",
                replacement="int32_t loop_count = block_size & ~3; /* MUTANT squared_difference_tail_drop */",
                count=2,
            ),
            Edit(
                relpath="Source/BasicMathFunctions/arm_elementwise_squared_difference_s16.c",
                pattern="int32_t loop_count = block_size;",
                replacement="int32_t loop_count = block_size & ~3; /* MUTANT squared_difference_tail_drop */",
                count=2,
            ),
        ),
        expected_detected_by=(
            "chunked-equivalence squared_difference s8/s16 cases (singles and offcut): the "
            "singles pass processes nothing at all while the full call processes its aligned "
            "prefix. Golden squared-difference cases kill it only when their flat element "
            "count is not a multiple of 4."
        ),
        refs=("AmbiqAI/helia-core-tester#81",),
    ),
    Mutant(
        mutant_id="minmax_no_broadcast_tail_drop",
        description="min/max no-broadcast inner loop stops one element short of the run",
        family="BasicMathFunctions",
        edits=(
            Edit(
                relpath="Source/BasicMathFunctions/arm_minimum_s8.c",
                pattern="    while (flat_size > 0)\n    {\n        int8_t in1 = *input_1++;",
                replacement=(
                    "    while (flat_size > 1) /* MUTANT minmax_no_broadcast_tail_drop */\n"
                    "    {\n        int8_t in1 = *input_1++;"
                ),
                count=1,
            ),
            # The MVE build compiles the wlstp.8/.16 asm instead of the
            # scalar loop above, so shortening only that loop would leave the
            # mutant dead on an MVE target. Trimming the loop-count operand is
            # the same defect expressed in the vectorised variant.
            Edit(
                relpath="Source/BasicMathFunctions/arm_minimum_s8.c",
                pattern=': [cnt] "r"(flat_size)',
                replacement=': [cnt] "r"(flat_size - 1) /* MUTANT minmax_no_broadcast_tail_drop */',
                count=1,
            ),
            Edit(
                relpath="Source/BasicMathFunctions/arm_maximum_s8.c",
                pattern="    while (flat_size > 0)\n    {\n        int8_t in1 = *input_1++;",
                replacement=(
                    "    while (flat_size > 1) /* MUTANT minmax_no_broadcast_tail_drop */\n"
                    "    {\n        int8_t in1 = *input_1++;"
                ),
                count=1,
            ),
            # The MVE build compiles the wlstp.8/.16 asm instead of the
            # scalar loop above, so shortening only that loop would leave the
            # mutant dead on an MVE target. Trimming the loop-count operand is
            # the same defect expressed in the vectorised variant.
            Edit(
                relpath="Source/BasicMathFunctions/arm_maximum_s8.c",
                pattern=': [cnt] "r"(flat_size)',
                replacement=': [cnt] "r"(flat_size - 1) /* MUTANT minmax_no_broadcast_tail_drop */',
                count=1,
            ),
            Edit(
                relpath="Source/BasicMathFunctions/arm_minimum_s16.c",
                pattern="    while (flat_size > 0)\n    {\n        int16_t in1 = *input_1++;",
                replacement=(
                    "    while (flat_size > 1) /* MUTANT minmax_no_broadcast_tail_drop */\n"
                    "    {\n        int16_t in1 = *input_1++;"
                ),
                count=1,
            ),
            # The MVE build compiles the wlstp.8/.16 asm instead of the
            # scalar loop above, so shortening only that loop would leave the
            # mutant dead on an MVE target. Trimming the loop-count operand is
            # the same defect expressed in the vectorised variant.
            Edit(
                relpath="Source/BasicMathFunctions/arm_minimum_s16.c",
                pattern=': [cnt] "r"(flat_size)',
                replacement=': [cnt] "r"(flat_size - 1) /* MUTANT minmax_no_broadcast_tail_drop */',
                count=1,
            ),
            Edit(
                relpath="Source/BasicMathFunctions/arm_maximum_s16.c",
                pattern="    while (flat_size > 0)\n    {\n        int16_t in1 = *input_1++;",
                replacement=(
                    "    while (flat_size > 1) /* MUTANT minmax_no_broadcast_tail_drop */\n"
                    "    {\n        int16_t in1 = *input_1++;"
                ),
                count=1,
            ),
            # The MVE build compiles the wlstp.8/.16 asm instead of the
            # scalar loop above, so shortening only that loop would leave the
            # mutant dead on an MVE target. Trimming the loop-count operand is
            # the same defect expressed in the vectorised variant.
            Edit(
                relpath="Source/BasicMathFunctions/arm_maximum_s16.c",
                pattern=': [cnt] "r"(flat_size)',
                replacement=': [cnt] "r"(flat_size - 1) /* MUTANT minmax_no_broadcast_tail_drop */',
                count=1,
            ),
        ),
        expected_detected_by=(
            "chunked-equivalence minimum/maximum s8/s16 cases: a size-1 slice writes nothing, "
            "so the singles pass disagrees with the full call on every element. Golden "
            "min/max cases kill it too (last element of every contiguous run is stale). "
            "Both the scalar loop and the MVE loop-count operand carry the edit, so the "
            "mutant is live in the DSP-only host build and on an MVE target alike."
        ),
        refs=("AmbiqAI/helia-core-tester#81",),
    ),
    Mutant(
        mutant_id="requantize_tail_drop",
        description="arm_requantize_s8_s8 drops the trailing size % 4 elements",
        family="QuantizationFunctions",
        edits=(
            Edit(
                relpath="Source/QuantizationFunctions/arm_quantize_s8_s8.c",
                pattern="int32_t count = (size + 3) / 4;",
                replacement="int32_t count = size / 4; /* MUTANT requantize_tail_drop */",
                count=1,
            ),
            Edit(
                relpath="Source/QuantizationFunctions/arm_quantize_s8_s8.c",
                pattern="for (int i = 0; i < size; i++)",
                replacement="for (int i = 0; i < (size & ~3); i++) /* MUTANT requantize_tail_drop */",
                count=1,
            ),
        ),
        expected_detected_by=(
            "chunked-equivalence requantize s8 cases only. The suite's one golden case for this "
            "kernel, requantize_default_s8, is 1x2x3x2 -- 12 elements, a multiple of 4 -- so it "
            "cannot see a size % 4 tail drop, and nothing else in the corpus calls it. The "
            "chunked cases require the mve capability, so scoring this mutant needs a "
            "generation CPU that has it."
        ),
        refs=("AmbiqAI/helia-core-tester#81",),
        requires_capabilities=("mve",),
    ),
    Mutant(
        mutant_id="drop_conv_ctx_guard",
        description="Convolution kernels accept a NULL ctx->buf instead of returning ARG_ERROR",
        family="ConvolutionFunctions",
        edits=(
            Edit(
                relpath="Source/ConvolutionFunctions/arm_convolve_s8.c",
                pattern="    if (ctx->buf == NULL)\n",
                replacement="    if (0) /* MUTANT drop_conv_ctx_guard */\n",
                count=1,
            ),
            Edit(
                relpath="Source/ConvolutionFunctions/arm_convolve_s16.c",
                pattern="    if (ctx->buf == NULL)\n",
                replacement="    if (0) /* MUTANT drop_conv_ctx_guard */\n",
                count=1,
            ),
            Edit(
                relpath="Source/ConvolutionFunctions/arm_convolve_s4.c",
                pattern="    if (ctx->buf == NULL)\n",
                replacement="    if (0) /* MUTANT drop_conv_ctx_guard */\n",
                count=1,
            ),
            Edit(
                relpath="Source/ConvolutionFunctions/arm_convolve_1_x_n_s8.c",
                pattern="conv_params->dilation.w != 1 || ctx->buf == NULL ||",
                replacement="conv_params->dilation.w != 1 || /* MUTANT drop_conv_ctx_guard */",
                count=1,
            ),
        ),
        expected_detected_by=(
            "the tester#72 status-only cases convolve_fault_null_ctx_buf_s8, "
            "convolve_fault_null_ctx_buf_1xn_s8, convolve_fault_null_ctx_buf_s16 and "
            "convolve_fault_null_ctx_buf_s4: each hands the kernel a NULL ctx->buf and "
            "asserts ARM_CMSIS_NN_ARG_ERROR, so a kernel that no longer diagnoses it either "
            "returns SUCCESS or faults on the NULL im2col buffer. Golden cases always pass a "
            "buffer and cannot see this."
        ),
        refs=("AmbiqAI/helia-core-tester#72",),
    ),
    Mutant(
        mutant_id="drop_conv_group_guard",
        description="Grouped convolution no longer rejects channel counts not divisible by groups",
        family="ConvolutionFunctions",
        edits=(
            Edit(
                relpath="Source/ConvolutionFunctions/arm_convolve_s8.c",
                pattern="    if (input_ch % groups != 0 || output_ch % groups != 0)\n",
                replacement="    if (0) /* MUTANT drop_conv_group_guard */\n",
                count=1,
            ),
            # Both the MVE and the non-MVE body of arm_convolve_s16 carry the
            # check, so the edit stays live on the host build and on the FVP.
            Edit(
                relpath="Source/ConvolutionFunctions/arm_convolve_s16.c",
                pattern="    if (input_ch % groups != 0 || output_ch % groups != 0)\n",
                replacement="    if (0) /* MUTANT drop_conv_group_guard */\n",
                count=2,
            ),
        ),
        expected_detected_by=(
            "convolve_fault_channel_group_mismatch_s8 and convolve_fault_channel_group_mismatch_s16 "
            "(tester#72): input_dims.c is set to 2 * filter_dims.c + 1 so that groups does not "
            "divide it, and the case asserts ARM_CMSIS_NN_ARG_ERROR."
        ),
        refs=("AmbiqAI/helia-core-tester#72",),
    ),
    Mutant(
        mutant_id="drop_conv_1xn_zero_stride_guard",
        description="arm_convolve_1_x_n_s8 no longer rejects stride.w == 0",
        family="ConvolutionFunctions",
        edits=(
            Edit(
                relpath="Source/ConvolutionFunctions/arm_convolve_1_x_n_s8.c",
                pattern="ctx->buf == NULL || conv_params->stride.w == 0 ||",
                replacement="ctx->buf == NULL || /* MUTANT drop_conv_1xn_zero_stride_guard */",
                count=1,
            ),
        ),
        expected_detected_by=(
            "convolve_fault_zero_stride_s8 (tester#72): a 1xN conv with stride.w forced to 0. "
            "0 * input_dims.c is a multiple of 4, so without the explicit check the kernel "
            "proceeds and returns SUCCESS instead of ARM_CMSIS_NN_ARG_ERROR."
        ),
        refs=("AmbiqAI/helia-core-tester#72",),
    ),
    Mutant(
        mutant_id="drop_depthwise_channel_guard",
        description="Optimized depthwise kernels no longer reject input_ch != output_ch",
        family="ConvolutionFunctions",
        edits=(
            Edit(
                relpath="Source/ConvolutionFunctions/arm_depthwise_conv_s8_opt.c",
                pattern="    if (input_ch != output_ch)\n",
                replacement="    if (0) /* MUTANT drop_depthwise_channel_guard */\n",
                count=1,
            ),
            Edit(
                relpath="Source/ConvolutionFunctions/arm_depthwise_conv_fast_s16.c",
                pattern="    if (input_ch != output_ch)\n",
                replacement="    if (0) /* MUTANT drop_depthwise_channel_guard */\n",
                count=1,
            ),
            Edit(
                relpath="Source/ConvolutionFunctions/arm_depthwise_conv_s4_opt.c",
                pattern="    if (input_ch != output_ch)\n",
                replacement="    if (0) /* MUTANT drop_depthwise_channel_guard */\n",
                count=1,
            ),
        ),
        expected_detected_by=(
            "depthwise_conv_fault_channel_mismatch_s8, _s16 and _s4 (tester#72): ch_mult is 1 "
            "and the wrapper dispatches to the optimized kernel, but output_dims.c is passed as "
            "input_dims.c + 1; the case asserts ARM_CMSIS_NN_ARG_ERROR."
        ),
        refs=("AmbiqAI/helia-core-tester#72",),
    ),
    Mutant(
        mutant_id="drop_depthwise_ctx_guard",
        description="Optimized depthwise kernels accept a NULL ctx->buf instead of returning ARG_ERROR",
        family="ConvolutionFunctions",
        edits=(
            Edit(
                relpath="Source/ConvolutionFunctions/arm_depthwise_conv_s8_opt.c",
                pattern="    if (ctx->buf == NULL && arm_depthwise_conv_s8_opt_get_buffer_size(input_dims, filter_dims) != 0)\n",
                replacement="    if (0) /* MUTANT drop_depthwise_ctx_guard */\n",
                count=1,
            ),
            Edit(
                relpath="Source/ConvolutionFunctions/arm_depthwise_conv_fast_s16.c",
                pattern="    if (ctx->buf == NULL && arm_depthwise_conv_fast_s16_get_buffer_size(input_dims, filter_dims) != 0)\n",
                replacement="    if (0) /* MUTANT drop_depthwise_ctx_guard */\n",
                count=1,
            ),
            Edit(
                relpath="Source/ConvolutionFunctions/arm_depthwise_conv_s4_opt.c",
                pattern="    if (ctx->buf == NULL)\n",
                replacement="    if (0) /* MUTANT drop_depthwise_ctx_guard */\n",
                count=1,
            ),
        ),
        expected_detected_by=(
            "depthwise_conv_fault_null_ctx_buf_s8, _s16 and _s4 (tester#72). The s8/s16 guards "
            "only fire where the optimized kernel needs a buffer, which is every DSP or MVE "
            "build, so those two cases carry required_capabilities: [dsp]; the host scorer "
            "builds with ARM_MATH_DSP and sees all three."
        ),
        refs=("AmbiqAI/helia-core-tester#72",),
    ),
    Mutant(
        mutant_id="drop_transpose_conv_ctx_guard",
        description="arm_transpose_conv_wrapper_s8 accepts a NULL ctx->buf",
        family="ConvolutionFunctions",
        edits=(
            Edit(
                relpath="Source/ConvolutionFunctions/arm_transpose_conv_wrapper_s8.c",
                pattern="    if (ctx->buf == NULL)\n",
                replacement="    if (0) /* MUTANT drop_transpose_conv_ctx_guard */\n",
                count=1,
            ),
        ),
        expected_detected_by=(
            "transpose_conv_fault_null_ctx_buf_s8 (tester#72): NULL ctx->buf with a stride-2 "
            "shape that takes the direct transpose-conv route; asserts ARM_CMSIS_NN_ARG_ERROR."
        ),
        refs=("AmbiqAI/helia-core-tester#72",),
    ),
    Mutant(
        mutant_id="drop_transpose_conv_dilation_guard",
        description="Transpose convolution silently ignores a non-unit dilation again (ns-cmsis-nn#261)",
        family="ConvolutionFunctions",
        edits=(
            Edit(
                relpath="Source/ConvolutionFunctions/arm_transpose_conv_wrapper_s8.c",
                pattern="    if (transpose_conv_params->dilation.w != 1 || transpose_conv_params->dilation.h != 1)\n",
                replacement="    if (0) /* MUTANT drop_transpose_conv_dilation_guard */\n",
                count=1,
            ),
            Edit(
                relpath="Source/ConvolutionFunctions/arm_transpose_conv_s8.c",
                pattern="    if (transpose_conv_params->dilation.w != 1 || transpose_conv_params->dilation.h != 1)\n",
                replacement="    if (0) /* MUTANT drop_transpose_conv_dilation_guard */\n",
                count=1,
            ),
        ),
        expected_detected_by=(
            "transpose_conv_fault_nonunit_dilation_s8 (tester#72): dilation.w = 2 through the "
            "wrapper; both the wrapper check and the arm_transpose_conv_s8 check are removed so "
            "neither route can diagnose it, and the case asserts ARM_CMSIS_NN_ARG_ERROR."
        ),
        refs=("AmbiqAI/helia-core-tester#72", "AmbiqAI/ns-cmsis-nn#261"),
    ),
    Mutant(
        mutant_id="drop_transpose_conv_reverse_ctx_guard",
        description="The reverse-conv transpose route accepts a NULL reverse_conv_ctx->buf",
        family="ConvolutionFunctions",
        edits=(
            Edit(
                relpath="Source/ConvolutionFunctions/arm_transpose_conv_wrapper_s8.c",
                pattern="        if (reverse_conv_ctx->buf == NULL)\n",
                replacement="        if (0) /* MUTANT drop_transpose_conv_reverse_ctx_guard */\n",
                count=1,
            ),
        ),
        expected_detected_by=(
            "transpose_conv_fault_null_reverse_conv_ctx_buf_s8 (tester#72): stride 1 with 20 "
            "input channels (above REVERSE_TCOL_EFFICIENT_THRESHOLD) selects the reverse-conv "
            "route, whose ctx is NULL; asserts ARM_CMSIS_NN_ARG_ERROR."
        ),
        refs=("AmbiqAI/helia-core-tester#72",),
    ),
    Mutant(
        mutant_id="drop_fc_s16_ctx_guard",
        description="arm_fully_connected_per_channel_s16 accepts a NULL kernel-sum ctx",
        family="FullyConnectedFunctions",
        edits=(
            Edit(
                relpath="Source/FullyConnectedFunctions/arm_fully_connected_per_channel_s16.c",
                pattern="    if ((ctx == NULL) || (ctx->buf == NULL) || (required_bytes > INT32_MAX))\n",
                replacement="    if (required_bytes > INT32_MAX) /* MUTANT drop_fc_s16_ctx_guard */\n",
                count=1,
            ),
        ),
        expected_detected_by=(
            "fully_connected_fault_null_ctx_buf_s16 (tester#72): per-channel s16 through the "
            "wrapper with ctx.buf NULL; asserts ARM_CMSIS_NN_ARG_ERROR."
        ),
        refs=("AmbiqAI/helia-core-tester#72",),
    ),
    Mutant(
        mutant_id="drop_fc_s16_ctx_size_guard",
        description="arm_fully_connected_per_channel_s16 no longer checks a declared ctx->size",
        family="FullyConnectedFunctions",
        edits=(
            Edit(
                relpath="Source/FullyConnectedFunctions/arm_fully_connected_per_channel_s16.c",
                pattern="    if ((ctx->size != 0) && (ctx->size < required_size))\n",
                replacement="    if (0) /* MUTANT drop_fc_s16_ctx_size_guard */\n",
                count=1,
            ),
        ),
        expected_detected_by=(
            "fully_connected_fault_small_ctx_size_s16 (tester#72): a valid buffer declared with "
            "ctx.size = 1, smaller than output_ch * sizeof(int32_t); asserts ARM_CMSIS_NN_ARG_ERROR."
        ),
        refs=("AmbiqAI/helia-core-tester#72",),
    ),
    Mutant(
        mutant_id="drop_svdf_scratch_guard",
        description="SVDF s8 kernels accept NULL input/output scratch contexts",
        family="SVDFunctions",
        edits=(
            Edit(
                relpath="Source/SVDFunctions/arm_svdf_s8.c",
                pattern="    if (input_ctx->buf == NULL)\n",
                replacement="    if (0) /* MUTANT drop_svdf_scratch_guard */\n",
                count=1,
            ),
            Edit(
                relpath="Source/SVDFunctions/arm_svdf_s8.c",
                pattern="    if (output_ctx->buf == NULL)\n",
                replacement="    if (0) /* MUTANT drop_svdf_scratch_guard */\n",
                count=1,
            ),
            Edit(
                relpath="Source/SVDFunctions/arm_svdf_state_s16_s8.c",
                pattern="    if (input_ctx->buf == NULL)\n",
                replacement="    if (0) /* MUTANT drop_svdf_scratch_guard */\n",
                count=1,
            ),
            Edit(
                relpath="Source/SVDFunctions/arm_svdf_state_s16_s8.c",
                pattern="    if (output_ctx->buf == NULL)\n",
                replacement="    if (0) /* MUTANT drop_svdf_scratch_guard */\n",
                count=1,
            ),
        ),
        expected_detected_by=(
            "svdf_fault_null_input_ctx_buf_s8, svdf_fault_null_output_ctx_buf_s8, "
            "svdf_fault_state_s16_null_input_ctx_buf_s8 and "
            "svdf_fault_state_s16_null_output_ctx_buf_s8 (tester#72): one of the two scratch "
            "contexts is NULL and the case asserts ARM_CMSIS_NN_ARG_ERROR; without the guard "
            "the kernel either returns SUCCESS or faults on the NULL scratch buffer."
        ),
        refs=("AmbiqAI/helia-core-tester#72",),
    ),
    Mutant(
        mutant_id="drop_pool_batch_guard",
        description="Pooling kernels accept input_dims->n < 1 and return SUCCESS having done nothing",
        family="PoolingFunctions",
        edits=(
            # Both the MVE and the non-MVE arm_avgpool_s8 body carry the check.
            Edit(
                relpath="Source/PoolingFunctions/arm_avgpool_s8.c",
                pattern="    if (batch_cnt < 1)\n",
                replacement="    if (0) /* MUTANT drop_pool_batch_guard */\n",
                count=2,
            ),
            Edit(
                relpath="Source/PoolingFunctions/arm_avgpool_s16.c",
                pattern="    if (batch_cnt < 1)\n",
                replacement="    if (0) /* MUTANT drop_pool_batch_guard */\n",
                count=1,
            ),
            Edit(
                relpath="Source/PoolingFunctions/arm_max_pool_s8.c",
                pattern="    if (batch_cnt < 1)\n",
                replacement="    if (0) /* MUTANT drop_pool_batch_guard */\n",
                count=1,
            ),
            Edit(
                relpath="Source/PoolingFunctions/arm_max_pool_s16.c",
                pattern="    if (batch_cnt < 1)\n",
                replacement="    if (0) /* MUTANT drop_pool_batch_guard */\n",
                count=1,
            ),
        ),
        expected_detected_by=(
            "avg_pool_fault_zero_dim_s8/_s16, avg_pool_fault_negative_dim_s8/_s16, "
            "max_pool_fault_zero_dim_s8/_s16 and max_pool_fault_negative_dim_s8/_s16 "
            "(tester#72): input_dims.n is 0 or -1, the batch loop never runs, and the "
            "kernel returns SUCCESS instead of the asserted ARM_CMSIS_NN_ARG_ERROR."
        ),
        refs=("AmbiqAI/helia-core-tester#72",),
    ),
    # ------------------------------------------------------------------------
    # Scratch-sizer sentinel mutants (tester#133). Each one makes a
    # *_get_buffer_size() the harness actually calls return the documented
    # negative out-of-range sentinel with a minimal literal edit. Before #133 a
    # negative answer flowed into ctx.size unremarked -- it is not larger than
    # the static the case allocated, so the capacity comparison passed, and most
    # kernels never read ctx.size -- and every one of these mutants survived.
    # They exist to prove HELIA_VALIDATE_SIZER catches something. See the module
    # docstring for the scope of the class and for the one sizer defect it
    # deliberately does not cover.
    # ------------------------------------------------------------------------
    Mutant(
        mutant_id="conv_sizer_negative",
        description="The s8 convolution wrapper sizer returns the -1 out-of-range sentinel",
        family="ConvolutionFunctions",
        edits=(
            # One literal, three replacements: the harness calls whichever of the
            # three wrapper entry points its generation CPU selected
            # (arm_convolve_wrapper_s8_get_buffer_size on a plain target, _dsp on
            # cortex-m4, _mve on cortex-m55), so all three bodies have to answer
            # -1 for the mutant to be reachable on every target. The dispatch
            # line is what makes the anchor exact: `(void)output_dims;` alone
            # matches five times in this file, and the two extra sites --
            # arm_convolve_s8_get_weights_sum_size and the non-MVE leg of
            # arm_convolve_1_x_n_s8_get_buffer_size -- are deliberately left alone.
            Edit(
                relpath="Source/ConvolutionFunctions/arm_convolve_get_buffer_sizes_s8.c",
                pattern=(
                    "    (void)output_dims;\n"
                    "    if (arm_nn_is_convolve_1x1(conv_params, input_dims, filter_dims))\n"
                ),
                replacement=(
                    "    (void)output_dims;\n"
                    "    return -1; /* MUTANT conv_sizer_negative */\n"
                    "    if (arm_nn_is_convolve_1x1(conv_params, input_dims, filter_dims))\n"
                ),
                count=3,
            ),
        ),
        expected_detected_by=(
            "every s8 Convolve case, through HELIA_VALIDATE_SIZER (tester#133). s4 and s16 "
            "Convolve cases survive by construction: their wrapper sizers live in "
            "arm_convolve_get_buffer_sizes_s4.c and _s16.c, which this mutant does not touch."
        ),
        refs=("AmbiqAI/helia-core-tester#133",),
    ),
    # The FITS half of the pair needs its own sabotage: a sizer that answers a
    # value larger than the static the case allocated. These two say the answer
    # is a gigabyte, which no generated case reserves, so HELIA_VALIDATE_SIZER
    # passes it and HELIA_VALIDATE_SIZER_FITS is the check that has to fire.
    # Without them the FITS macro would ship with no evidence that it detects
    # anything, since the negative-sentinel mutants above are all caught one
    # line earlier.
    Mutant(
        mutant_id="conv_sizer_over_capacity",
        description="The s8 convolution wrapper sizer answers a gigabyte, far above any case's static",
        family="ConvolutionFunctions",
        edits=(
            # Same anchor and the same three wrapper bodies as
            # conv_sizer_negative; only the answer differs, so a kill here is
            # attributable to the capacity check rather than the sign check.
            Edit(
                relpath="Source/ConvolutionFunctions/arm_convolve_get_buffer_sizes_s8.c",
                pattern=(
                    "    (void)output_dims;\n"
                    "    if (arm_nn_is_convolve_1x1(conv_params, input_dims, filter_dims))\n"
                ),
                replacement=(
                    "    (void)output_dims;\n"
                    "    return 0x40000000; /* MUTANT conv_sizer_over_capacity */\n"
                    "    if (arm_nn_is_convolve_1x1(conv_params, input_dims, filter_dims))\n"
                ),
                count=3,
            ),
        ),
        expected_detected_by=(
            "every s8 Convolve case, through HELIA_VALIDATE_SIZER_FITS (tester#133). The "
            "answer is positive, so the sign check passes it through and only the capacity "
            "comparison can catch it. Reported as sizer_capacity rather than sizer_contract: "
            "in production that pairing means our generation-time bound is too small, which "
            "is the tester's defect and not the kernel's."
        ),
        refs=("AmbiqAI/helia-core-tester#133",),
    ),
    Mutant(
        mutant_id="avgpool_sizer_over_capacity",
        description="The s8 average pooling sizer answers a gigabyte, far above any case's static",
        family="PoolingFunctions",
        edits=(
            # A second family, so a kill is not a property of the convolution
            # template alone. The insertion goes after the existing range guard
            # in each of the three bodies, which leaves that guard's own -1
            # behaviour intact and makes the gigabyte the only new answer.
            Edit(
                relpath="Source/PoolingFunctions/arm_avgpool_get_buffer_sizes_s8.c",
                pattern=(
                    "    if ((ch_src < 0) || (required_bytes > INT32_MAX))\n"
                    "    {\n"
                    "        return -1;\n"
                    "    }\n"
                ),
                replacement=(
                    "    if ((ch_src < 0) || (required_bytes > INT32_MAX))\n"
                    "    {\n"
                    "        return -1;\n"
                    "    }\n"
                    "    return 0x40000000; /* MUTANT avgpool_sizer_over_capacity */\n"
                ),
                count=3,
            ),
        ),
        expected_detected_by=(
            "every s8 AvgPool case, through HELIA_VALIDATE_SIZER_FITS (tester#133). s16 avg "
            "pool cases survive: their sizer is in arm_avgpool_get_buffer_sizes_s16.c."
        ),
        refs=("AmbiqAI/helia-core-tester#133",),
    ),
    Mutant(
        mutant_id="depthwise_sizer_negative",
        description="The s8 depthwise convolution wrapper sizer returns the -1 out-of-range sentinel",
        family="ConvolutionFunctions",
        edits=(
            # The same three-variant story as conv: plain, _dsp and _mve wrappers
            # each open with this declaration and each is the entry point for one
            # class of target. Seeding it at -1 and returning immediately keeps
            # `size` used, so the edit introduces no unused local.
            Edit(
                relpath="Source/ConvolutionFunctions/arm_depthwise_conv_get_buffer_sizes_s8.c",
                pattern="    int32_t size = 0;\n",
                replacement=(
                    "    int32_t size = -1;\n"
                    "    return size; /* MUTANT depthwise_sizer_negative */\n"
                ),
                count=3,
            ),
        ),
        expected_detected_by=(
            "every s8 DepthwiseConv case, through HELIA_VALIDATE_SIZER (tester#133). s4 and "
            "s16 depthwise cases survive by construction -- their wrapper sizers are in "
            "arm_depthwise_conv_get_buffer_sizes_s4.c and _s16.c."
        ),
        refs=("AmbiqAI/helia-core-tester#133",),
    ),
    Mutant(
        mutant_id="fc_sizer_negative",
        description="The s8 fully connected kernel-sum sizer returns the -1 out-of-range sentinel",
        family="FullyConnectedFunctions",
        edits=(
            # The dispatcher and both legs carry the identical range guard, so one
            # literal forces the kernel's own -1 sentinel out of whichever entry
            # point the generation CPU picked. This strengthens the guard to
            # always-true rather than removing it: the guard-removal class is a
            # separate concern and is catalogued separately.
            Edit(
                relpath="Source/FullyConnectedFunctions/arm_fully_connected_get_buffer_sizes_s8.c",
                pattern="    if ((filter_dims->c < 0) || (required_bytes > INT32_MAX))\n",
                replacement=(
                    "    if (1 || (filter_dims->c < 0) || (required_bytes > INT32_MAX)) "
                    "/* MUTANT fc_sizer_negative */\n"
                ),
                count=3,
            ),
        ),
        expected_detected_by=(
            "every s8 FullyConnected case and every s8 BatchMatMul case, through "
            "HELIA_VALIDATE_SIZER (tester#133). Batch matmul is covered here rather than by a "
            "mutant of its own: the kernel does ship arm_batch_matmul_s8_get_buffer_size, but "
            "the batch matmul harness does not call it -- resolve_batch_matmul_kernel() hands "
            "the template arm_fully_connected_s8_get_buffer_size for int8 and "
            "arm_fully_connected_s16_get_buffer_size for int16 -- so the s8 batch matmul route "
            "is this sizer. s16 cases of both families survive: their sizer is in "
            "arm_fully_connected_get_buffer_sizes_s16.c."
        ),
        refs=("AmbiqAI/helia-core-tester#133",),
    ),
    Mutant(
        mutant_id="avgpool_sizer_negative",
        description="The s8 average pooling sizer returns the -1 out-of-range sentinel",
        family="PoolingFunctions",
        edits=(
            # The avg pool harness always calls the undecorated
            # arm_avgpool_s8_get_buffer_size, which dispatches to the _dsp or _mve
            # leg; all three bodies carry this guard and all three are forced.
            Edit(
                relpath="Source/PoolingFunctions/arm_avgpool_get_buffer_sizes_s8.c",
                pattern="    if ((ch_src < 0) || (required_bytes > INT32_MAX))\n",
                replacement=(
                    "    if (1 || (ch_src < 0) || (required_bytes > INT32_MAX)) "
                    "/* MUTANT avgpool_sizer_negative */\n"
                ),
                count=3,
            ),
        ),
        expected_detected_by=(
            "every s8 AvgPool case, through HELIA_VALIDATE_SIZER (tester#133). This sizer "
            "legitimately answers 0 for many shapes, which is why a wrong answer here had to be "
            "distinguished by sign rather than by being falsy. s16 avg pool cases survive: "
            "their sizer is in arm_avgpool_get_buffer_sizes_s16.c. Max pool has no mutant in "
            "this class at all -- pool_base.py gives a kernel_get_buffer_size_fn to avg pool "
            "only, so max_pool.c.j2's sizer block never renders and is not instrumented."
        ),
        refs=("AmbiqAI/helia-core-tester#133",),
    ),
    Mutant(
        mutant_id="transpose_conv_sizer_negative",
        description="The s8 transpose convolution forward sizer returns the -1 out-of-range sentinel",
        family="ConvolutionFunctions",
        edits=(
            # arm_transpose_conv_s8_get_buffer_size() and its _mve leg each
            # propagate a negative rolling-buffer size through this guard; forcing
            # it true is the smallest edit that makes both answer -1. The harness
            # always calls the undecorated entry point, which is the MVE leg's
            # only caller on an MVE build.
            Edit(
                relpath="Source/ConvolutionFunctions/arm_transpose_conv_get_buffer_sizes_s8.c",
                pattern="    if (rolling_size < 0)\n",
                replacement=(
                    "    if (1 || (rolling_size < 0)) "
                    "/* MUTANT transpose_conv_sizer_negative */\n"
                ),
                count=2,
            ),
        ),
        expected_detected_by=(
            "every s8 TransposeConv case, through the first HELIA_VALIDATE_SIZER in the "
            "transpose conv template (tester#133)."
        ),
        refs=("AmbiqAI/helia-core-tester#133",),
    ),
    Mutant(
        mutant_id="transpose_conv_reverse_sizer_negative",
        description="arm_transpose_conv_s8_get_reverse_conv_buffer_size returns the -1 out-of-range sentinel",
        family="ConvolutionFunctions",
        edits=(
            # Kept separate from transpose_conv_sizer_negative on purpose:
            # HELIA_VALIDATE_SIZER returns ARM_CMSIS_NN_ARG_ERROR on the first
            # failure, so a forward-sizer mutation would fail the case before the
            # reverse check ever ran and the reverse check would go unproven.
            # This sizer has no -1 path of its own, so the sentinel is planted as
            # an unconditional early return at the top of the function.
            Edit(
                relpath="Source/ConvolutionFunctions/arm_transpose_conv_get_buffer_sizes_s8.c",
                pattern=(
                    "int32_t arm_transpose_conv_s8_get_reverse_conv_buffer_size("
                    "const cmsis_nn_transpose_conv_params *transpose_conv_params,\n"
                    "                                                           "
                    "const cmsis_nn_dims *input_dims,\n"
                    "                                                           "
                    "const cmsis_nn_dims *filter_dims)\n"
                    "{\n"
                ),
                replacement=(
                    "int32_t arm_transpose_conv_s8_get_reverse_conv_buffer_size("
                    "const cmsis_nn_transpose_conv_params *transpose_conv_params,\n"
                    "                                                           "
                    "const cmsis_nn_dims *input_dims,\n"
                    "                                                           "
                    "const cmsis_nn_dims *filter_dims)\n"
                    "{\n"
                    "    return -1; /* MUTANT transpose_conv_reverse_sizer_negative */\n"
                ),
                count=1,
            ),
        ),
        expected_detected_by=(
            "every s8 TransposeConv case, through the second HELIA_VALIDATE_SIZER in the "
            "transpose conv template (tester#133). This is the route the change was written "
            "for: the reverse sizer's answer is discarded in favour of a compile-time "
            "REVERSE_CONV_CTX_SIZE, so nothing downstream could ever have noticed the sentinel."
        ),
        refs=("AmbiqAI/helia-core-tester#133",),
    ),
    Mutant(
        mutant_id="svdf_kernel_sum_sizer_negative",
        description="arm_svdf_s8_get_buffer_size (the kernel-sum sizer) returns the -1 out-of-range sentinel",
        family="SVDFunctions",
        edits=(
            # The SVDF template calls the undecorated arm_svdf_s8_get_buffer_size;
            # the dispatcher and both legs carry the same guard, so one literal
            # covers every target.
            Edit(
                relpath="Source/SVDFunctions/arm_svdf_get_buffer_sizes_s8.c",
                pattern="    if ((weights_feature_dims->n < 0) || (required_bytes > INT32_MAX))\n",
                replacement=(
                    "    if (1 || (weights_feature_dims->n < 0) || (required_bytes > INT32_MAX)) "
                    "/* MUTANT svdf_kernel_sum_sizer_negative */\n"
                ),
                count=3,
            ),
        ),
        expected_detected_by=(
            "every SVDF case that allocates a kernel-sum context (the has_ctx branch of the "
            "svdf template), through HELIA_VALIDATE_SIZER (tester#133). The two staging "
            "contexts were already size-checked by tester#71; this query was not, and its "
            "negative answer became ctx.size unremarked."
        ),
        refs=("AmbiqAI/helia-core-tester#133",),
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
