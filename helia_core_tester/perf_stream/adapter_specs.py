"""Single source of truth for the perf-stream firmware's per-kernel dispatch C code.

Problem this solves: adding a new bridged kernel previously required writing the same
"how do I call this CMSIS-NN function" knowledge in two independent, hand-maintained
places that could silently drift out of sync:

  1. `helia_core_tester/perf_stream/generated_test_bridge.py` -- a Python builder that
     extracts real generated-test tensors/scalars and packages them into a CaseBundle.
  2. A hand-typed `run_xxx_once()` C function inside `cmake/perf_stream/benchmark_server_session.c`
     -- firmware code that reads those same blobs/scalars back out of the session struct
     and calls the real CMSIS-NN kernel.

This module collects (2) -- the actual C function bodies -- into ONE Python file per
adapter, alongside the exact list of session scalar fields each adapter depends on. A
generator script (`scripts/generate_perf_stream_adapters.py`) renders these bodies into a
clearly marked, auto-generated block inside `benchmark_server_session.c`, so there is
exactly one place a maintainer edits the calling convention for a given kernel, instead of
two files that must be kept in sync by hand. See `docs/performance-streaming-design.md` for
why the firmware still can't literally reuse the FVP-generated `.c.j2` template output
(that template bakes descriptor-specific tensors into flash per test case, which is exactly
what the streaming architecture -- one universal firmware, streamed per-case data -- exists
to avoid).

Each `FirmwareAdapterSpec.scalar_fields` list documents every `session->xxx` scalar this
adapter's C body reads (excluding the always-present `output_h/w/c` ground-truth dims,
`input_offset`/`output_offset`/`activation_min`/`activation_max`, which are reset for every
case in `handle_case_meta()` and shared by all adapters). This list exists so bridge authors
and reviewers can check a builder's `serialized_scalar_parameters` keys against the exact
set the firmware body actually consumes, catching a missing/renamed field at review time
instead of only at hardware-run time.
"""

from __future__ import annotations

from dataclasses import dataclass

GENERATED_BLOCK_BEGIN = (
    "/* >>> BEGIN GENERATED PERF-STREAM ADAPTERS -- see "
    "helia_core_tester/perf_stream/adapter_specs.py and "
    "scripts/generate_perf_stream_adapters.py. DO NOT EDIT THIS BLOCK BY HAND: rerun the "
    "generator after editing adapter_specs.py. >>> */"
)
GENERATED_BLOCK_END = "/* <<< END GENERATED PERF-STREAM ADAPTERS <<< */"


@dataclass(frozen=True)
class FirmwareAdapterSpec:
    """One bridged (family, operator) cluster's real firmware C dispatch code.

    `scalar_fields` is purely documentation/cross-check metadata (the C body below is the
    actual executable source of truth) -- see `generated_test_bridge_scalar_fields()` and
    its use in tests that assert a builder's manifest scalar keys are a subset of what the
    firmware body for that kernel actually reads.
    """

    label: str
    function_name: str
    guard: str | None
    scalar_fields: tuple[str, ...]
    c_body: str


_COMPUTE_CONVOLVE_OUTPUT_DIMS = '''\
static hctp_status_t compute_convolve_output_dims(const hct_server_session_t *session,
                                                  const hct_server_blob_t *input,
                                                  const hct_server_blob_t *weights,
                                                  cmsis_nn_dims *output_dims,
                                                  uint32_t *output_bytes)
{
    /* Use the ground-truth output dims sent explicitly by the host (see parse_scalar())
     * rather than re-deriving them from stride/padding here: a SAME/VALID formula-based
     * recomputation can silently diverge from the real generator's actual output size for
     * real generated test cases (e.g. asymmetric padding, rounding conventions), producing
     * a garbage output_length that corrupts the correctness-output stream. */
    (void)input;
    (void)weights;
    output_dims->n = 1;
    output_dims->h = session->output_h;
    output_dims->w = session->output_w;
    output_dims->c = session->output_c;

    if (output_dims->h <= 0 || output_dims->w <= 0 || output_dims->c <= 0)
    {
        return HCTP_STATUS_INVALID_ARGUMENT;
    }

    *output_bytes = (uint32_t)(output_dims->n * output_dims->h * output_dims->w * output_dims->c);
    if (*output_bytes > sizeof(session->output_buffer))
    {
        return HCTP_STATUS_INVALID_ARGUMENT;
    }
    return HCTP_STATUS_OK;
}'''

_RUN_CONVOLVE_ONCE = '''\
static arm_cmsis_nn_status run_convolve_once(hct_server_session_t *session)
{
    hct_server_blob_t *input = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_0);
    hct_server_blob_t *weights = find_blob_by_role(session, HCT_BLOB_ROLE_WEIGHTS);
    hct_server_blob_t *bias = find_blob_by_role(session, HCT_BLOB_ROLE_BIAS);
    hct_server_blob_t *multiplier = find_blob_by_role(session, HCT_BLOB_ROLE_MULTIPLIER);
    hct_server_blob_t *shift = find_blob_by_role(session, HCT_BLOB_ROLE_SHIFT);
    cmsis_nn_context ctx;
    cmsis_nn_context weight_sum_ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;
    hct_convolve_s8_request_t request;
    int32_t required_scratch;
    uint32_t weight_sum_offset;
    uint32_t weight_sum_bytes;

    if (input == NULL || weights == NULL || bias == NULL || multiplier == NULL || shift == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    input_dims.n = (int32_t)input->dimensions[0];
    input_dims.h = (int32_t)input->dimensions[1];
    input_dims.w = (int32_t)input->dimensions[2];
    input_dims.c = (int32_t)input->dimensions[3];
    filter_dims.h = (int32_t)weights->dimensions[0];
    filter_dims.w = (int32_t)weights->dimensions[1];
    filter_dims.c = (int32_t)weights->dimensions[2];
    filter_dims.n = (int32_t)weights->dimensions[3];
    bias_dims.n = 0;
    bias_dims.h = 0;
    bias_dims.w = 0;
    bias_dims.c = (int32_t)bias->dimensions[0];

    if (compute_convolve_output_dims(session, input, weights, &output_dims, &session->output_length) != HCTP_STATUS_OK)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    required_scratch = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    if (required_scratch < 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    if ((uint32_t)required_scratch > session->scratch_bytes)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    weight_sum_bytes = (uint32_t)output_dims.c * (uint32_t)sizeof(int32_t);
    weight_sum_offset = align_up(session->scratch_offset + session->scratch_bytes, 16u);
    if (weight_sum_offset + weight_sum_bytes > session->runtime_arena_capacity ||
        weight_sum_offset + weight_sum_bytes > sizeof(session->case_arena))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    ctx.buf = (session->scratch_bytes > 0u) ? &session->case_arena[session->scratch_offset] : NULL;
    ctx.size = session->scratch_bytes;
    weight_sum_ctx.buf = &session->case_arena[weight_sum_offset];
    weight_sum_ctx.size = (int32_t)weight_sum_bytes;
    conv_params.input_offset = session->input_offset;
    conv_params.output_offset = session->output_offset;
    conv_params.stride.w = (session->stride_w == 0) ? 1 : session->stride_w;
    conv_params.stride.h = (session->stride_h == 0) ? 1 : session->stride_h;
    /* Use the ground-truth "before" padding sent explicitly by the host (see
     * parse_scalar()) instead of re-deriving it here via a SAME-formula + symmetric
     * `/2` split, which assumes an even padding split that doesn't always hold and can
     * silently diverge from the real generator's exact padding convention. */
    conv_params.padding.w = session->pad_w;
    conv_params.padding.h = session->pad_h;
    conv_params.dilation.w = (session->dilation_w == 0) ? 1 : session->dilation_w;
    conv_params.dilation.h = (session->dilation_h == 0) ? 1 : session->dilation_h;
    conv_params.activation.min = session->activation_min;
    conv_params.activation.max = session->activation_max;
    quant_params.multiplier = (int32_t *)blob_ptr(session, multiplier);
    quant_params.shift = (int32_t *)blob_ptr(session, shift);
    if (arm_convolve_weight_sum((int32_t *)weight_sum_ctx.buf,
                                (const int8_t *)blob_ptr(session, weights),
                                &input_dims,
                                &filter_dims,
                                &output_dims,
                                session->input_offset,
                                (const int32_t *)blob_ptr(session, bias)) != ARM_CMSIS_NN_SUCCESS)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    if (weight_sum_offset + weight_sum_bytes > session->case_arena_used_bytes)
    {
        session->case_arena_used_bytes = weight_sum_offset + weight_sum_bytes;
    }

    request.ctx = &ctx;
    request.weight_sum_ctx = &weight_sum_ctx;
    request.conv_params = &conv_params;
    request.quant_params = &quant_params;
    request.input_dims = &input_dims;
    request.input_data = (const int8_t *)blob_ptr(session, input);
    request.filter_dims = &filter_dims;
    request.filter_data = (const int8_t *)blob_ptr(session, weights);
    request.bias_dims = &bias_dims;
    request.bias_data = (const int32_t *)blob_ptr(session, bias);
    request.upscale_dims = NULL;
    request.output_dims = &output_dims;
    request.output_data = (int8_t *)session->output_buffer;
    return hct_dispatch_convolve_s8(&request);
}'''

_RUN_DEPTHWISE_CONV_ONCE = '''\
/* arm_depthwise_conv_s8 is the low-level (non-wrapper) depthwise conv kernel: unlike
 * arm_convolve_s8 above, its `ctx` and `bias_dims` args are both unused internally (see
 * Source/ConvolutionFunctions/arm_depthwise_conv_s8.c's `(void)ctx;`/`(void)bias_dims;`),
 * so no scratch buffer or weight-sum precomputation is required here. filter_dims stay in
 * the generator's native (N=1, H, W, C_OUT) order -- no HWCN reordering like Convolve's
 * filter_dims (depthwise's cmsis_nn_dw_conv_params filter convention is already NHWC). */
static arm_cmsis_nn_status run_depthwise_conv_once(hct_server_session_t *session)
{
    hct_server_blob_t *input = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_0);
    hct_server_blob_t *weights = find_blob_by_role(session, HCT_BLOB_ROLE_WEIGHTS);
    hct_server_blob_t *bias = find_blob_by_role(session, HCT_BLOB_ROLE_BIAS);
    hct_server_blob_t *multiplier = find_blob_by_role(session, HCT_BLOB_ROLE_MULTIPLIER);
    hct_server_blob_t *shift = find_blob_by_role(session, HCT_BLOB_ROLE_SHIFT);
    cmsis_nn_context ctx = {NULL, 0};
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    if (input == NULL || weights == NULL || bias == NULL || multiplier == NULL || shift == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    input_dims.n = (int32_t)input->dimensions[0];
    input_dims.h = (int32_t)input->dimensions[1];
    input_dims.w = (int32_t)input->dimensions[2];
    input_dims.c = (int32_t)input->dimensions[3];
    filter_dims.n = (int32_t)weights->dimensions[0];
    filter_dims.h = (int32_t)weights->dimensions[1];
    filter_dims.w = (int32_t)weights->dimensions[2];
    filter_dims.c = (int32_t)weights->dimensions[3];
    bias_dims.n = 0;
    bias_dims.h = 0;
    bias_dims.w = 0;
    bias_dims.c = (int32_t)bias->dimensions[0];

    output_dims.n = 1;
    output_dims.h = session->output_h;
    output_dims.w = session->output_w;
    output_dims.c = session->output_c;
    if (output_dims.h <= 0 || output_dims.w <= 0 || output_dims.c <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    session->output_length = (uint32_t)(output_dims.n * output_dims.h * output_dims.w * output_dims.c);
    if (session->output_length > sizeof(session->output_buffer))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    dw_conv_params.input_offset = session->input_offset;
    dw_conv_params.output_offset = session->output_offset;
    dw_conv_params.ch_mult = session->ch_mult;
    dw_conv_params.stride.w = (session->stride_w == 0) ? 1 : session->stride_w;
    dw_conv_params.stride.h = (session->stride_h == 0) ? 1 : session->stride_h;
    dw_conv_params.padding.w = session->pad_w;
    dw_conv_params.padding.h = session->pad_h;
    dw_conv_params.dilation.w = (session->dilation_w == 0) ? 1 : session->dilation_w;
    dw_conv_params.dilation.h = (session->dilation_h == 0) ? 1 : session->dilation_h;
    dw_conv_params.activation.min = session->activation_min;
    dw_conv_params.activation.max = session->activation_max;
    quant_params.multiplier = (int32_t *)blob_ptr(session, multiplier);
    quant_params.shift = (int32_t *)blob_ptr(session, shift);

    return arm_depthwise_conv_s8(&ctx,
                                  &dw_conv_params,
                                  &quant_params,
                                  &input_dims,
                                  (const int8_t *)blob_ptr(session, input),
                                  &filter_dims,
                                  (const int8_t *)blob_ptr(session, weights),
                                  &bias_dims,
                                  (const int32_t *)blob_ptr(session, bias),
                                  &output_dims,
                                  (int8_t *)session->output_buffer);
}'''

_RUN_ELEMENTWISE_BINARY_ONCE = '''\
/* arm_add_s8/arm_sub_s8 share an identical signature/argument order; both are dispatched
 * from this one wrapper, branching only on session->expected_kernel_id (see
 * assets/kernel_registry.yaml for the kernel_id <-> operator mapping). Ground-truth output
 * dims (session->output_h/w/c) are sent explicitly by the host, same rationale as
 * compute_convolve_output_dims(): broadcasting output shape shouldn't be re-derived here. */
static arm_cmsis_nn_status run_elementwise_binary_once(hct_server_session_t *session)
{
    hct_server_blob_t *input1 = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_0);
    hct_server_blob_t *input2 = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_1);
    cmsis_nn_dims input1_dims;
    cmsis_nn_dims input2_dims;
    cmsis_nn_dims output_dims;

    if (input1 == NULL || input2 == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    input1_dims.n = (int32_t)input1->dimensions[0];
    input1_dims.h = (int32_t)input1->dimensions[1];
    input1_dims.w = (int32_t)input1->dimensions[2];
    input1_dims.c = (int32_t)input1->dimensions[3];
    input2_dims.n = (int32_t)input2->dimensions[0];
    input2_dims.h = (int32_t)input2->dimensions[1];
    input2_dims.w = (int32_t)input2->dimensions[2];
    input2_dims.c = (int32_t)input2->dimensions[3];

    output_dims.n = 1;
    output_dims.h = session->output_h;
    output_dims.w = session->output_w;
    output_dims.c = session->output_c;
    if (output_dims.h <= 0 || output_dims.w <= 0 || output_dims.c <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    session->output_length = (uint32_t)(output_dims.n * output_dims.h * output_dims.w * output_dims.c);
    if (session->output_length > sizeof(session->output_buffer))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int8_t *input1_data = (const int8_t *)blob_ptr(session, input1);
    const int8_t *input2_data = (const int8_t *)blob_ptr(session, input2);
    int8_t *output_data = (int8_t *)session->output_buffer;

    switch (session->expected_kernel_id)
    {
        case HCT_KERNEL_ID_ADD_S8:
            return arm_add_s8(input1_data, &input1_dims, input2_data, &input2_dims,
                              session->input1_offset, session->input1_mult, session->input1_shift,
                              session->input2_offset, session->input2_mult, session->input2_shift,
                              session->left_shift,
                              output_data, &output_dims,
                              session->output_offset, session->out_mult, session->out_shift,
                              session->activation_min, session->activation_max);
        case HCT_KERNEL_ID_SUB_S8:
            return arm_sub_s8(input1_data, &input1_dims, input2_data, &input2_dims,
                              session->input1_offset, session->input1_mult, session->input1_shift,
                              session->input2_offset, session->input2_mult, session->input2_shift,
                              session->left_shift,
                              output_data, &output_dims,
                              session->output_offset, session->out_mult, session->out_shift,
                              session->activation_min, session->activation_max);
        case HCT_KERNEL_ID_MUL_S8:
            /* arm_mul_s8 has no per-input mult/shift or left_shift -- it reuses only the
             * input1_offset/input2_offset scalar fields (shared with Add/Sub above). */
            return arm_mul_s8(input1_data, &input1_dims, input2_data, &input2_dims,
                              session->input1_offset, session->input2_offset,
                              output_data, &output_dims,
                              session->output_offset, session->out_mult, session->out_shift,
                              session->activation_min, session->activation_max);
        case HCT_KERNEL_ID_MAXIMUM_S8:
        case HCT_KERNEL_ID_MINIMUM_S8:
        {
            /* Maximum/Minimum have no quant scalars at all -- just a scratch context, always
             * {NULL, 0} per the generated tests (no buffer-sizing helper exists for these ops). */
            cmsis_nn_context ctx = {NULL, 0};
            if (session->expected_kernel_id == HCT_KERNEL_ID_MAXIMUM_S8)
            {
                return arm_maximum_s8(&ctx, input1_data, &input1_dims, input2_data, &input2_dims,
                                      output_data, &output_dims);
            }
            return arm_minimum_s8(&ctx, input1_data, &input1_dims, input2_data, &input2_dims,
                                  output_data, &output_dims);
        }
        default:
            return ARM_CMSIS_NN_ARG_ERROR;
    }
}'''


# Ordered list -- rendering emits function bodies in this order. `compute_convolve_output_dims`
# is a private helper of `run_convolve_once` (not a dispatched adapter itself, no scalar_fields
# of its own beyond what run_convolve_once already declares) but lives in this same generated
# block since it's only ever called from there.
FIRMWARE_ADAPTERS: tuple[FirmwareAdapterSpec, ...] = (
    FirmwareAdapterSpec(
        label="ConvolutionFunctions/Convolve (helper)",
        function_name="compute_convolve_output_dims",
        guard="HCT_HOST_ABS_ONLY",
        scalar_fields=(),
        c_body=_COMPUTE_CONVOLVE_OUTPUT_DIMS,
    ),
    FirmwareAdapterSpec(
        label="ConvolutionFunctions/Convolve",
        function_name="run_convolve_once",
        guard="HCT_HOST_ABS_ONLY",
        scalar_fields=(
            "stride_h", "stride_w", "pad_h", "pad_w", "dilation_h", "dilation_w",
            "output_h", "output_w", "output_c", "input_offset", "output_offset",
            "activation_min", "activation_max",
        ),
        c_body=_RUN_CONVOLVE_ONCE,
    ),
    FirmwareAdapterSpec(
        label="ConvolutionFunctions/DepthwiseConv",
        function_name="run_depthwise_conv_once",
        guard="HCT_HOST_ABS_ONLY",
        scalar_fields=(
            "stride_h", "stride_w", "pad_h", "pad_w", "dilation_h", "dilation_w",
            "output_h", "output_w", "output_c", "input_offset", "output_offset",
            "activation_min", "activation_max", "ch_mult",
        ),
        c_body=_RUN_DEPTHWISE_CONV_ONCE,
    ),
    FirmwareAdapterSpec(
        label="BasicMathFunctions/Add,Sub,Mul,Maximum,Minimum",
        function_name="run_elementwise_binary_once",
        guard="HCT_HOST_ABS_ONLY",
        scalar_fields=(
            "output_h", "output_w", "output_c",
            "input1_offset", "input1_mult", "input1_shift",
            "input2_offset", "input2_mult", "input2_shift",
            "left_shift", "out_mult", "out_shift",
            "activation_min", "activation_max",
        ),
        c_body=_RUN_ELEMENTWISE_BINARY_ONCE,
    ),
)


def generated_test_bridge_scalar_fields(function_name: str) -> tuple[str, ...]:
    """Look up the documented scalar-field list for one adapter's C body, by function name.
    Used by regression tests to cross-check a Python builder's `serialized_scalar_parameters`
    keys against what the firmware body for that same kernel actually consumes.
    """
    for adapter in FIRMWARE_ADAPTERS:
        if adapter.function_name == function_name:
            return adapter.scalar_fields
    raise KeyError(f"No FirmwareAdapterSpec registered with function_name={function_name!r}")


def render_generated_adapters_block() -> str:
    """Render the full auto-generated block (marker comments + every adapter's C body,
    each wrapped in its `#ifndef {guard}` guard when one is set) to be spliced into
    `cmake/perf_stream/benchmark_server_session.c` by `scripts/generate_perf_stream_adapters.py`.
    """
    pieces: list[str] = [GENERATED_BLOCK_BEGIN]
    open_guard: str | None = None
    for adapter in FIRMWARE_ADAPTERS:
        if adapter.guard != open_guard:
            if open_guard is not None:
                pieces.append("#endif")
            if adapter.guard is not None:
                pieces.append(f"#ifndef {adapter.guard}")
            open_guard = adapter.guard
        pieces.append("")
        pieces.append(adapter.c_body)
    if open_guard is not None:
        pieces.append("#endif")
    pieces.append(GENERATED_BLOCK_END)
    return "\n".join(pieces) + "\n"
