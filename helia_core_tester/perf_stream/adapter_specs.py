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

    if (!hct_checked_dims_bytes(output_dims, 1u, session->output_capacity_bytes, output_bytes))
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
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    /* multiplier/shift (per-channel requantization params) only exist for quantized
     * (S8/S16) Convolve cases -- the F16/F32 float bundles built by
     * generated_test_bridge.py never emit those blobs, so requiring them
     * unconditionally here would reject every float Convolve case before the
     * float-specific dispatch below is ever reached. */
    const bool is_float_kernel = (session->expected_kernel_id == HCT_KERNEL_ID_CONVOLVE_F32 ||
                                  session->expected_kernel_id == HCT_KERNEL_ID_CONVOLVE_F16);
    if (input == NULL || weights == NULL || bias == NULL || (!is_float_kernel && (multiplier == NULL || shift == NULL)))
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

    if (session->expected_kernel_id == HCT_KERNEL_ID_CONVOLVE_F32 ||
        session->expected_kernel_id == HCT_KERNEL_ID_CONVOLVE_F16)
    {
        const bool is_f16 = (session->expected_kernel_id == HCT_KERNEL_ID_CONVOLVE_F16);
        const uint32_t element_size = is_f16 ? (uint32_t)sizeof(float16_t) : (uint32_t)sizeof(float);
        const float activation_min = quant_scale_from_bits(session->float_activation_min_bits);
        const float activation_max = quant_scale_from_bits(session->float_activation_max_bits);
        int32_t required_scratch;
        uint32_t operand_bytes_needed;

        if (!hct_checked_dims_bytes(&input_dims, element_size, input->byte_length, &operand_bytes_needed) ||
            !hct_checked_dims_bytes(&filter_dims, element_size, weights->byte_length, &operand_bytes_needed) ||
            (bias != NULL && !hct_checked_count_bytes(bias_dims.c, element_size, bias->byte_length, &operand_bytes_needed)))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }

        if (session->output_length > session->output_capacity_bytes / element_size)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        session->output_length *= element_size;

        if (is_f16)
        {
            cmsis_nn_conv_params_f16 params = {
                .stride = { .w = (session->stride_w == 0) ? 1 : session->stride_w, .h = (session->stride_h == 0) ? 1 : session->stride_h },
                .padding = { .w = session->pad_w, .h = session->pad_h },
                .dilation = { .w = (session->dilation_w == 0) ? 1 : session->dilation_w, .h = (session->dilation_h == 0) ? 1 : session->dilation_h },
                .activation = { .min = (float16_t)activation_min, .max = (float16_t)activation_max },
                .weight_format = (session->weight_format_is_packed != 0) ? ARM_NN_WEIGHT_FORMAT_NT_N_PACKED : ARM_NN_WEIGHT_FORMAT_STANDARD,
            };
            required_scratch = arm_convolve_f16_get_buffer_size(&params, &input_dims, &filter_dims, &output_dims, ARM_NN_LAYOUT_NHWC);
            if (required_scratch < 0 || (uint32_t)required_scratch > session->scratch_bytes)
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            ctx.buf = (required_scratch > 0) ? &session->workspace[session->scratch_offset] : NULL;
            ctx.size = required_scratch;
            return arm_convolve_f16(&ctx,
                                    &params,
                                    &input_dims,
                                    (const float16_t *)blob_ptr(session, input),
                                    &filter_dims,
                                    (const float16_t *)blob_ptr(session, weights),
                                    &bias_dims,
                                    (bias != NULL) ? (const float16_t *)blob_ptr(session, bias) : NULL,
                                    &output_dims,
                                    (float16_t *)hct_output_ptr(session),
                                    ARM_NN_LAYOUT_NHWC);
        }

        {
            cmsis_nn_conv_params_f32 params = {
                .stride = { .w = (session->stride_w == 0) ? 1 : session->stride_w, .h = (session->stride_h == 0) ? 1 : session->stride_h },
                .padding = { .w = session->pad_w, .h = session->pad_h },
                .dilation = { .w = (session->dilation_w == 0) ? 1 : session->dilation_w, .h = (session->dilation_h == 0) ? 1 : session->dilation_h },
                .activation = { .min = activation_min, .max = activation_max },
                .weight_format = (session->weight_format_is_packed != 0) ? ARM_NN_WEIGHT_FORMAT_NT_N_PACKED : ARM_NN_WEIGHT_FORMAT_STANDARD,
            };
            required_scratch = arm_convolve_f32_get_buffer_size(&params, &input_dims, &filter_dims, &output_dims, ARM_NN_LAYOUT_NHWC);
            if (required_scratch < 0 || (uint32_t)required_scratch > session->scratch_bytes)
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            ctx.buf = (required_scratch > 0) ? &session->workspace[session->scratch_offset] : NULL;
            ctx.size = required_scratch;
            return arm_convolve_f32(&ctx,
                                    &params,
                                    &input_dims,
                                    (const float *)blob_ptr(session, input),
                                    &filter_dims,
                                    (const float *)blob_ptr(session, weights),
                                    &bias_dims,
                                    (bias != NULL) ? (const float *)blob_ptr(session, bias) : NULL,
                                    &output_dims,
                                    (float *)hct_output_ptr(session),
                                    ARM_NN_LAYOUT_NHWC);
        }
    }

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

    {
        /* multiplier/shift are always required (checked above) once we're past the float
         * dispatch, so their received byte_length can be validated here for every
         * quantized path below. */
        uint32_t operand_bytes_needed;
        if (!hct_checked_count_bytes(output_dims.c, sizeof(int32_t), multiplier->byte_length, &operand_bytes_needed) ||
            !hct_checked_count_bytes(output_dims.c, sizeof(int32_t), shift->byte_length, &operand_bytes_needed))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
    }

    if (session->expected_kernel_id == HCT_KERNEL_ID_CONVOLVE_S16)
    {
        /* arm_convolve_wrapper_s16 needs no weight-sum precompute (unlike S8's
         * arm_convolve_s8, whose bias must be pre-adjusted via arm_convolve_weight_sum())
         * and its bias is a cmsis_nn_bias_data struct wrapping a plain int64_t payload
         * (see arm_nnfunctions.h), not S8's raw int32_t* bias pointer. */
        cmsis_nn_bias_data bias_data = {blob_ptr(session, bias), false};
        int32_t required_scratch = arm_convolve_wrapper_s16_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
        uint32_t operand_bytes_needed;
        if (required_scratch < 0 || (uint32_t)required_scratch > session->scratch_bytes)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        if (!hct_checked_dims_bytes(&input_dims, sizeof(int16_t), input->byte_length, &operand_bytes_needed) ||
            !hct_checked_dims_bytes(&filter_dims, sizeof(int8_t), weights->byte_length, &operand_bytes_needed) ||
            !hct_checked_count_bytes(bias_dims.c, sizeof(int64_t), bias->byte_length, &operand_bytes_needed))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        ctx.buf = (session->scratch_bytes > 0u) ? &session->workspace[session->scratch_offset] : NULL;
        ctx.size = session->scratch_bytes;

        /* session->output_length is transmitted to the host as a raw byte count (see the
         * RTT send loop below) -- compute_convolve_output_dims() above computed it in
         * elements, so rescale to bytes for 2-byte-per-element S16 output. */
        if (!hct_checked_dims_bytes(&output_dims, sizeof(int16_t),
                                    session->output_capacity_bytes, &session->output_length))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }

        return arm_convolve_wrapper_s16(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims,
                                        (const int16_t *)blob_ptr(session, input),
                                        &filter_dims,
                                        (const int8_t *)blob_ptr(session, weights),
                                        &bias_dims,
                                        &bias_data,
                                        &output_dims,
                                        (int16_t *)hct_output_ptr(session));
    }

    if (session->expected_kernel_id == HCT_KERNEL_ID_CONVOLVE_S4)
    {
        int32_t required_scratch = arm_convolve_wrapper_s4_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
        uint32_t operand_bytes_needed;
        const int32_t filter_shape[4] = {filter_dims.n, filter_dims.h, filter_dims.w, filter_dims.c};
        if (required_scratch < 0 || (uint32_t)required_scratch > session->scratch_bytes)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        if (!hct_checked_dims_bytes(&input_dims, sizeof(int8_t), input->byte_length, &operand_bytes_needed) ||
            !hct_checked_packed4_bytes(filter_shape, 4, weights->byte_length, &operand_bytes_needed) ||
            !hct_checked_count_bytes(bias_dims.c, sizeof(int32_t), bias->byte_length, &operand_bytes_needed))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        ctx.buf = (session->scratch_bytes > 0u) ? &session->workspace[session->scratch_offset] : NULL;
        ctx.size = session->scratch_bytes;

        return arm_convolve_wrapper_s4(&ctx,
                                       &conv_params,
                                       &quant_params,
                                       &input_dims,
                                       (const int8_t *)blob_ptr(session, input),
                                       &filter_dims,
                                       (const int8_t *)blob_ptr(session, weights),
                                       &bias_dims,
                                       (const int32_t *)blob_ptr(session, bias),
                                       &output_dims,
                                       (int8_t *)hct_output_ptr(session));
    }

    {
        /* S8 path: arm_convolve_s8 requires a precomputed per-output-channel weight sum
         * (via arm_convolve_weight_sum()) placed in its own scratch region, distinct from
         * the general im2col-style `ctx` scratch above. */
        cmsis_nn_context weight_sum_ctx;
        hct_convolve_s8_request_t request;
        int32_t required_scratch;
        uint32_t weight_sum_relative_offset;
        uint32_t weight_sum_bytes;
        uint32_t weight_sum_end;
        uint32_t operand_bytes_needed;

        if (!hct_checked_dims_bytes(&input_dims, sizeof(int8_t), input->byte_length, &operand_bytes_needed) ||
            !hct_checked_dims_bytes(&filter_dims, sizeof(int8_t), weights->byte_length, &operand_bytes_needed) ||
            !hct_checked_count_bytes(bias_dims.c, sizeof(int32_t), bias->byte_length, &operand_bytes_needed))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }

        required_scratch = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
        if (required_scratch < 0 || (uint32_t)required_scratch > session->scratch_bytes)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        {
            const int32_t weight_sum_shape[1] = {output_dims.c};
            if (!hct_checked_shape_bytes(weight_sum_shape, 1, sizeof(int32_t),
                                         session->scratch_bytes, &weight_sum_bytes) ||
                !hct_checked_aligned_range((uint32_t)required_scratch,
                                           16u,
                                           weight_sum_bytes,
                                           session->scratch_bytes,
                                           &weight_sum_relative_offset,
                                           &weight_sum_end))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
        }

        ctx.buf = (required_scratch > 0) ? &session->workspace[session->scratch_offset] : NULL;
        ctx.size = required_scratch;
        weight_sum_ctx.buf = &session->workspace[session->scratch_offset + weight_sum_relative_offset];
        weight_sum_ctx.size = (int32_t)weight_sum_bytes;
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
        request.output_data = (int8_t *)hct_output_ptr(session);
        return hct_dispatch_convolve_s8(&request);
    }
}'''

_RUN_DEPTHWISE_CONV_ONCE = '''\
static arm_cmsis_nn_status run_depthwise_conv_once(hct_server_session_t *session)
{
    hct_server_blob_t *input = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_0);
    hct_server_blob_t *weights = find_blob_by_role(session, HCT_BLOB_ROLE_WEIGHTS);
    hct_server_blob_t *bias = find_blob_by_role(session, HCT_BLOB_ROLE_BIAS);
    hct_server_blob_t *multiplier = find_blob_by_role(session, HCT_BLOB_ROLE_MULTIPLIER);
    hct_server_blob_t *shift = find_blob_by_role(session, HCT_BLOB_ROLE_SHIFT);
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    if (input == NULL || weights == NULL || bias == NULL)
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

    output_dims.n = (session->output_n > 0) ? session->output_n : 1;
    output_dims.h = session->output_h;
    output_dims.w = session->output_w;
    output_dims.c = session->output_c;
    if (output_dims.h <= 0 || output_dims.w <= 0 || output_dims.c <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    if (!hct_checked_dims_bytes(&output_dims, 1u, session->output_capacity_bytes, &session->output_length))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    if (session->expected_kernel_id == HCT_KERNEL_ID_DEPTHWISE_CONV_F32 ||
        session->expected_kernel_id == HCT_KERNEL_ID_DEPTHWISE_CONV_F16)
    {
        const bool is_f16 = (session->expected_kernel_id == HCT_KERNEL_ID_DEPTHWISE_CONV_F16);
        const uint32_t element_size = is_f16 ? (uint32_t)sizeof(float16_t) : (uint32_t)sizeof(float);
        const float activation_min = quant_scale_from_bits(session->float_activation_min_bits);
        const float activation_max = quant_scale_from_bits(session->float_activation_max_bits);
        cmsis_nn_context ctx;
        int32_t required_scratch;
        uint32_t operand_bytes_needed;

        if (!hct_checked_dims_bytes(&input_dims, element_size, input->byte_length, &operand_bytes_needed) ||
            !hct_checked_dims_bytes(&filter_dims, element_size, weights->byte_length, &operand_bytes_needed) ||
            !hct_checked_count_bytes(bias_dims.c, element_size, bias->byte_length, &operand_bytes_needed))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }

        if (session->output_length > session->output_capacity_bytes / element_size)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        session->output_length *= element_size;

        if (is_f16)
        {
            cmsis_nn_dw_conv_params_f16 params = {
                .ch_mult = session->ch_mult,
                .stride = { .w = (session->stride_w == 0) ? 1 : session->stride_w, .h = (session->stride_h == 0) ? 1 : session->stride_h },
                .padding = { .w = session->pad_w, .h = session->pad_h },
                .dilation = { .w = (session->dilation_w == 0) ? 1 : session->dilation_w, .h = (session->dilation_h == 0) ? 1 : session->dilation_h },
                .activation = { .min = (float16_t)activation_min, .max = (float16_t)activation_max },
            };
            required_scratch = arm_depthwise_conv_f16_get_buffer_size(&params, &input_dims, &filter_dims, &output_dims, ARM_NN_LAYOUT_NHWC);
            if (required_scratch < 0 || (uint32_t)required_scratch > session->scratch_bytes)
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            ctx.buf = (required_scratch > 0) ? &session->workspace[session->scratch_offset] : NULL;
            ctx.size = required_scratch;
            return arm_depthwise_conv_f16(&ctx,
                                          &params,
                                          &input_dims,
                                          (const float16_t *)blob_ptr(session, input),
                                          &filter_dims,
                                          (const float16_t *)blob_ptr(session, weights),
                                          &bias_dims,
                                          (const float16_t *)blob_ptr(session, bias),
                                          &output_dims,
                                          (float16_t *)hct_output_ptr(session),
                                          ARM_NN_LAYOUT_NHWC);
        }
        else
        {
            cmsis_nn_dw_conv_params_f32 params = {
                .ch_mult = session->ch_mult,
                .stride = { .w = (session->stride_w == 0) ? 1 : session->stride_w, .h = (session->stride_h == 0) ? 1 : session->stride_h },
                .padding = { .w = session->pad_w, .h = session->pad_h },
                .dilation = { .w = (session->dilation_w == 0) ? 1 : session->dilation_w, .h = (session->dilation_h == 0) ? 1 : session->dilation_h },
                .activation = { .min = activation_min, .max = activation_max },
            };
            required_scratch = arm_depthwise_conv_f32_get_buffer_size(&params, &input_dims, &filter_dims, &output_dims, ARM_NN_LAYOUT_NHWC);
            if (required_scratch < 0 || (uint32_t)required_scratch > session->scratch_bytes)
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            ctx.buf = (required_scratch > 0) ? &session->workspace[session->scratch_offset] : NULL;
            ctx.size = required_scratch;
            return arm_depthwise_conv_f32(&ctx,
                                          &params,
                                          &input_dims,
                                          (const float *)blob_ptr(session, input),
                                          &filter_dims,
                                          (const float *)blob_ptr(session, weights),
                                          &bias_dims,
                                          (const float *)blob_ptr(session, bias),
                                          &output_dims,
                                          (float *)hct_output_ptr(session),
                                          ARM_NN_LAYOUT_NHWC);
        }
    }

    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    if (multiplier == NULL || shift == NULL)
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
    {
        uint32_t operand_bytes_needed;
        if (!hct_checked_count_bytes(output_dims.c, sizeof(int32_t), multiplier->byte_length, &operand_bytes_needed) ||
            !hct_checked_count_bytes(output_dims.c, sizeof(int32_t), shift->byte_length, &operand_bytes_needed))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
    }

    if (session->expected_kernel_id == HCT_KERNEL_ID_DEPTHWISE_CONV_S16)
    {
        cmsis_nn_context ctx;
        int32_t required_scratch = arm_depthwise_conv_wrapper_s16_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims);
        uint32_t operand_bytes_needed;
        if (required_scratch < 0 || (uint32_t)required_scratch > session->scratch_bytes)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        if (!hct_checked_dims_bytes(&input_dims, sizeof(int16_t), input->byte_length, &operand_bytes_needed) ||
            !hct_checked_dims_bytes(&filter_dims, sizeof(int8_t), weights->byte_length, &operand_bytes_needed) ||
            !hct_checked_count_bytes(bias_dims.c, sizeof(int64_t), bias->byte_length, &operand_bytes_needed))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        ctx.buf = (session->scratch_bytes > 0u) ? &session->workspace[session->scratch_offset] : NULL;
        ctx.size = session->scratch_bytes;

        if (!hct_checked_dims_bytes(&output_dims, sizeof(int16_t),
                                    session->output_capacity_bytes, &session->output_length))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }

        return arm_depthwise_conv_wrapper_s16(&ctx,
                                              &dw_conv_params,
                                              &quant_params,
                                              &input_dims,
                                              (const int16_t *)blob_ptr(session, input),
                                              &filter_dims,
                                              (const int8_t *)blob_ptr(session, weights),
                                              &bias_dims,
                                              (const int64_t *)blob_ptr(session, bias),
                                              &output_dims,
                                              (int16_t *)hct_output_ptr(session));
    }

    if (session->expected_kernel_id == HCT_KERNEL_ID_DEPTHWISE_CONV_S4)
    {
        cmsis_nn_context ctx;
        int32_t required_scratch = arm_depthwise_conv_wrapper_s4_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims);
        uint32_t operand_bytes_needed;
        const int32_t filter_shape[4] = {filter_dims.n, filter_dims.h, filter_dims.w, filter_dims.c};
        if (required_scratch < 0 || (uint32_t)required_scratch > session->scratch_bytes)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        if (!hct_checked_dims_bytes(&input_dims, sizeof(int8_t), input->byte_length, &operand_bytes_needed) ||
            !hct_checked_packed4_bytes(filter_shape, 4, weights->byte_length, &operand_bytes_needed) ||
            !hct_checked_count_bytes(bias_dims.c, sizeof(int32_t), bias->byte_length, &operand_bytes_needed))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        ctx.buf = (session->scratch_bytes > 0u) ? &session->workspace[session->scratch_offset] : NULL;
        ctx.size = session->scratch_bytes;

        return arm_depthwise_conv_wrapper_s4(&ctx,
                                             &dw_conv_params,
                                             &quant_params,
                                             &input_dims,
                                             (const int8_t *)blob_ptr(session, input),
                                             &filter_dims,
                                             (const int8_t *)blob_ptr(session, weights),
                                             &bias_dims,
                                             (const int32_t *)blob_ptr(session, bias),
                                             &output_dims,
                                             (int8_t *)hct_output_ptr(session));
    }

    if (session->output_length > session->output_capacity_bytes)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    {
        cmsis_nn_context ctx = {NULL, 0};
        uint32_t operand_bytes_needed;
        if (!hct_checked_dims_bytes(&input_dims, sizeof(int8_t), input->byte_length, &operand_bytes_needed) ||
            !hct_checked_dims_bytes(&filter_dims, sizeof(int8_t), weights->byte_length, &operand_bytes_needed) ||
            !hct_checked_count_bytes(bias_dims.c, sizeof(int32_t), bias->byte_length, &operand_bytes_needed))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
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
                                     (int8_t *)hct_output_ptr(session));
    }
}
'''

_RUN_POOLING_ONCE = '''\
/* arm_avgpool_s8/arm_max_pool_s8 (and their S16/F32 counterparts) all share the same
 * pool_params-based signature -- unlike Convolve/DepthwiseConv, pooling has no
 * input_offset/output_offset (see cmsis_nn_pool_params's definition in arm_nn_types.h:
 * only stride, padding, activation) and no weights/bias blobs at all, since a pool window
 * has no learned parameters -- just its size (session->pool_h/w, sent explicitly by the
 * host since there is no weights blob to read filter dims off of, unlike Convolve's).
 * MaxPool never needs a scratch buffer for any dtype; F32 AvgPool never does either (its
 * generated harness always passes ctx.buf=NULL). Int8/int16 AvgPool needs one sized via
 * arm_avgpool_{s8,s16}_get_buffer_size(output_w, input_c) -- zero for many small cases,
 * so scratch_bytes may legitimately be 0 (the workspace pointer is never dereferenced when
 * ctx.size is 0). F32's activation.min/max are real floats (e.g. +/-1e30), not int32
 * clamps, so they're sent bit-cast through float_activation_min_bits/max_bits (same
 * bit-cast-through-int32 wire convention as scale_bits -- see quant_scale_from_bits()). */
static arm_cmsis_nn_status run_pooling_once(hct_server_session_t *session)
{
    hct_server_blob_t *input = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_0);
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;
    cmsis_nn_context ctx;

    if (input == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    input_dims.n = (int32_t)input->dimensions[0];
    input_dims.h = (int32_t)input->dimensions[1];
    input_dims.w = (int32_t)input->dimensions[2];
    input_dims.c = (int32_t)input->dimensions[3];
    filter_dims.n = 1;
    filter_dims.h = session->pool_h;
    filter_dims.w = session->pool_w;
    filter_dims.c = 1;

    output_dims.n = (session->output_n > 0) ? session->output_n : 1;
    output_dims.h = session->output_h;
    output_dims.w = session->output_w;
    output_dims.c = session->output_c;
    if (output_dims.h <= 0 || output_dims.w <= 0 || output_dims.c <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    if (session->expected_kernel_id == HCT_KERNEL_ID_AVGPOOL_F32 || session->expected_kernel_id == HCT_KERNEL_ID_MAXPOOL_F32 ||
        session->expected_kernel_id == HCT_KERNEL_ID_AVGPOOL_F16 || session->expected_kernel_id == HCT_KERNEL_ID_MAXPOOL_F16)
    {
        bool is_f16 = (session->expected_kernel_id == HCT_KERNEL_ID_AVGPOOL_F16 || session->expected_kernel_id == HCT_KERNEL_ID_MAXPOOL_F16);
        uint32_t operand_bytes_needed;
        ctx.buf = NULL;
        ctx.size = 0;
        if (!hct_checked_dims_bytes(&input_dims, is_f16 ? sizeof(float16_t) : sizeof(float),
                                    input->byte_length, &operand_bytes_needed))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        if (!hct_checked_dims_bytes(&output_dims,
                                    is_f16 ? sizeof(float16_t) : sizeof(float),
                                    session->output_capacity_bytes,
                                    &session->output_length))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        if (is_f16)
        {
            cmsis_nn_pool_params_f16 pool_params_f16;
            pool_params_f16.stride.w = (session->stride_w == 0) ? 1 : session->stride_w;
            pool_params_f16.stride.h = (session->stride_h == 0) ? 1 : session->stride_h;
            pool_params_f16.padding.w = session->pad_w;
            pool_params_f16.padding.h = session->pad_h;
            pool_params_f16.activation.min = (float16_t)quant_scale_from_bits(session->float_activation_min_bits);
            pool_params_f16.activation.max = (float16_t)quant_scale_from_bits(session->float_activation_max_bits);
            if (session->expected_kernel_id == HCT_KERNEL_ID_AVGPOOL_F16)
            {
                return arm_avg_pool_f16(&ctx, &pool_params_f16, &input_dims, (const float16_t *)blob_ptr(session, input),
                                        &filter_dims, &output_dims, (float16_t *)hct_output_ptr(session));
            }
            return arm_max_pool_f16(&ctx, &pool_params_f16, &input_dims, (const float16_t *)blob_ptr(session, input),
                                    &filter_dims, &output_dims, (float16_t *)hct_output_ptr(session));
        }
        {
            cmsis_nn_pool_params_f32 pool_params_f32;
            pool_params_f32.stride.w = (session->stride_w == 0) ? 1 : session->stride_w;
            pool_params_f32.stride.h = (session->stride_h == 0) ? 1 : session->stride_h;
            pool_params_f32.padding.w = session->pad_w;
            pool_params_f32.padding.h = session->pad_h;
            pool_params_f32.activation.min = quant_scale_from_bits(session->float_activation_min_bits);
            pool_params_f32.activation.max = quant_scale_from_bits(session->float_activation_max_bits);
            if (session->expected_kernel_id == HCT_KERNEL_ID_AVGPOOL_F32)
            {
                return arm_avg_pool_f32(&ctx, &pool_params_f32, &input_dims, (const float *)blob_ptr(session, input),
                                        &filter_dims, &output_dims, (float *)hct_output_ptr(session));
            }
            return arm_max_pool_f32(&ctx, &pool_params_f32, &input_dims, (const float *)blob_ptr(session, input),
                                    &filter_dims, &output_dims, (float *)hct_output_ptr(session));
        }
    }

    {
        cmsis_nn_pool_params pool_params;
        pool_params.stride.w = (session->stride_w == 0) ? 1 : session->stride_w;
        pool_params.stride.h = (session->stride_h == 0) ? 1 : session->stride_h;
        pool_params.padding.w = session->pad_w;
        pool_params.padding.h = session->pad_h;
        pool_params.activation.min = session->activation_min;
        pool_params.activation.max = session->activation_max;

        if (session->expected_kernel_id == HCT_KERNEL_ID_MAXPOOL_S8 || session->expected_kernel_id == HCT_KERNEL_ID_MAXPOOL_S16)
        {
            ctx.buf = NULL;
            ctx.size = 0;
        }
        else
        {
            int32_t required_scratch = (session->expected_kernel_id == HCT_KERNEL_ID_AVGPOOL_S16)
                ? arm_avgpool_s16_get_buffer_size((int)output_dims.w, (int)input_dims.c)
                : arm_avgpool_s8_get_buffer_size((int)output_dims.w, (int)input_dims.c);
            if (required_scratch < 0 || (uint32_t)required_scratch > session->scratch_bytes)
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            ctx.buf = (session->scratch_bytes > 0u) ? &session->workspace[session->scratch_offset] : NULL;
            ctx.size = session->scratch_bytes;
        }

        if (session->expected_kernel_id == HCT_KERNEL_ID_AVGPOOL_S16 || session->expected_kernel_id == HCT_KERNEL_ID_MAXPOOL_S16)
        {
            uint32_t operand_bytes_needed;
            if (!hct_checked_dims_bytes(&input_dims, sizeof(int16_t), input->byte_length, &operand_bytes_needed))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            if (!hct_checked_dims_bytes(&output_dims, sizeof(int16_t),
                                        session->output_capacity_bytes, &session->output_length))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_AVGPOOL_S16)
            {
                return arm_avgpool_s16(&ctx, &pool_params, &input_dims, (const int16_t *)blob_ptr(session, input),
                                       &filter_dims, &output_dims, (int16_t *)hct_output_ptr(session));
            }
            return arm_max_pool_s16(&ctx, &pool_params, &input_dims, (const int16_t *)blob_ptr(session, input),
                                    &filter_dims, &output_dims, (int16_t *)hct_output_ptr(session));
        }

        {
            uint32_t operand_bytes_needed;
            if (!hct_checked_dims_bytes(&input_dims, sizeof(int8_t), input->byte_length, &operand_bytes_needed))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
        }
        if (!hct_checked_dims_bytes(&output_dims, sizeof(int8_t),
                                    session->output_capacity_bytes, &session->output_length))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        if (session->expected_kernel_id == HCT_KERNEL_ID_AVGPOOL_S8)
        {
            return arm_avgpool_s8(&ctx, &pool_params, &input_dims, (const int8_t *)blob_ptr(session, input),
                                  &filter_dims, &output_dims, (int8_t *)hct_output_ptr(session));
        }
        return arm_max_pool_s8(&ctx, &pool_params, &input_dims, (const int8_t *)blob_ptr(session, input),
                               &filter_dims, &output_dims, (int8_t *)hct_output_ptr(session));
    }
}'''

_RUN_ACTIVATION_ONCE = '''\
/* ActivationFunctions unary ops (Relu/Relu6/Clamp/LeakyRelu/Logistic/Tanh/HardSwish*) all
 * take a single input tensor and produce a same-shape output with no weights/bias blob and
 * no scratch buffer -- unlike Convolve/Pooling, their generated headers have no named
 * scalar-params struct either (see generated_test_bridge.py's _build_activation_case,
 * which reads the scalars positionally out of the generated .c file's call site, mirroring
 * how BasicMathFunctions elementwise ops are bridged). session->output_h/w/c holds the
 * (single) output shape; element count is used directly as each kernel's `size` argument. */
static arm_cmsis_nn_status run_activation_once(hct_server_session_t *session)
{
    hct_server_blob_t *input = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_0);
    int32_t size;
    bool is_s16;

    if (input == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    if (session->output_h <= 0 || session->output_w <= 0 || session->output_c <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    size = session->output_h * session->output_w * session->output_c;

    is_s16 = (session->expected_kernel_id == HCT_KERNEL_ID_RELU_S16 ||
              session->expected_kernel_id == HCT_KERNEL_ID_RELU6_S16 ||
              session->expected_kernel_id == HCT_KERNEL_ID_CLAMP_S16 ||
              session->expected_kernel_id == HCT_KERNEL_ID_LEAKY_RELU_S16 ||
              session->expected_kernel_id == HCT_KERNEL_ID_LOGISTIC_S16 ||
              session->expected_kernel_id == HCT_KERNEL_ID_TANH_S16 ||
              session->expected_kernel_id == HCT_KERNEL_ID_HARD_SWISH_PRECISE_S16);
    session->output_length = (uint32_t)(size * (is_s16 ? (int32_t)sizeof(int16_t) : (int32_t)sizeof(int8_t)));
    if (session->output_length > session->output_capacity_bytes)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    {
        uint32_t operand_bytes_needed;
        if (!hct_checked_count_bytes(size, is_s16 ? sizeof(int16_t) : sizeof(int8_t),
                                     input->byte_length, &operand_bytes_needed))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
    }

    if (is_s16)
    {
        const int16_t *in16 = (const int16_t *)blob_ptr(session, input);
        int16_t *out16 = (int16_t *)hct_output_ptr(session);
        switch (session->expected_kernel_id)
        {
            case HCT_KERNEL_ID_RELU_S16:
                return arm_relu_s16(in16, session->input_offset, session->output_offset,
                                    session->out_mult, session->out_shift, out16, size);
            case HCT_KERNEL_ID_RELU6_S16:
                return arm_relu_generic_s16(in16, session->input_offset, session->output_offset,
                                             session->out_mult, session->out_shift,
                                             session->activation_min, session->activation_max, out16, size);
            case HCT_KERNEL_ID_CLAMP_S16:
                return arm_clamp_s16(in16, (int16_t)session->activation_min, (int16_t)session->activation_max,
                                      out16, size);
            case HCT_KERNEL_ID_LEAKY_RELU_S16:
                return arm_leaky_relu_s16(in16, session->input_offset, session->output_offset,
                                          session->out_mult_alpha, session->out_shift_alpha,
                                          session->out_mult, session->out_shift, out16, size);
            case HCT_KERNEL_ID_LOGISTIC_S16:
                return arm_logistic_s16(in16, out16, size, session->input_mult, session->input_left_shift);
            case HCT_KERNEL_ID_TANH_S16:
                return arm_tanh_s16(in16, out16, size, session->input_mult, session->input_left_shift);
            case HCT_KERNEL_ID_HARD_SWISH_PRECISE_S16:
                return arm_hard_swish_precise_s16(in16, session->input_offset, session->output_offset,
                                                  session->out_mult, session->out_shift,
                                                  session->relu_q3, session->relu_q6, session->prescale,
                                                  out16, size);
            default:
                return ARM_CMSIS_NN_ARG_ERROR;
        }
    }
    else
    {
        const int8_t *in8 = (const int8_t *)blob_ptr(session, input);
        int8_t *out8 = (int8_t *)hct_output_ptr(session);
        switch (session->expected_kernel_id)
        {
            case HCT_KERNEL_ID_RELU_S8:
                return arm_relu_s8(in8, session->input_offset, session->output_offset,
                                   session->out_mult, session->out_shift, out8, size);
            case HCT_KERNEL_ID_RELU6_S8:
                return arm_relu_generic_s8(in8, session->input_offset, session->output_offset,
                                           session->out_mult, session->out_shift,
                                           session->activation_min, session->activation_max, out8, size);
            case HCT_KERNEL_ID_CLAMP_S8:
                return arm_clamp_s8(in8, (int8_t)session->activation_min, (int8_t)session->activation_max,
                                    out8, size);
            case HCT_KERNEL_ID_LEAKY_RELU_S8:
                return arm_leaky_relu_s8(in8, session->input_offset, session->output_offset,
                                         session->out_mult_alpha, session->out_shift_alpha,
                                         session->out_mult, session->out_shift, out8, size);
            case HCT_KERNEL_ID_HARD_SWISH_COMPAT_S8:
                return arm_hard_swish_compat_s8(in8, session->input_offset, session->output_offset,
                                                session->out_mult_fp, session->out_mult_exp,
                                                session->relu_mult_fp, session->relu_mult_exp, out8, size);
            case HCT_KERNEL_ID_HARD_SWISH_PRECISE_S8:
                return arm_hard_swish_precise_s8(in8, session->input_offset, session->output_offset,
                                                 session->out_mult, session->out_shift,
                                                 session->relu_q3, session->relu_q6, session->prescale,
                                                 out8, size);
            default:
                return ARM_CMSIS_NN_ARG_ERROR;
        }
    }
}'''

_RUN_PRELU_ONCE = '''\
/* PReLU (arm_prelu_s8/s16 -- alpha broadcastable per cmsis_nn_dims semantics, same style as
 * elementwise binary Add/Sub) and PReLUScalar (arm_prelu_scalar_s8/s16 -- a direct
 * flat-vector API used when one side is a true per-pixel scalar; see arm_nnfunctions.h).
 * Both share input_offset/output_offset (own tensor's zero points, reused from the unary
 * activations above), out_mult/out_shift (the "identity" branch), and out_mult_alpha/
 * out_shift_alpha (reused from LeakyRelu's alpha branch -- identical semantics). alpha_offset
 * is the one new quantized scalar. PReLUScalar's `scalar_is_input` argument is always `true`
 * in every real generated test (see prelu_scalar.c.j2's hardcoded call), so it's hardcoded
 * here rather than added as a session field. */
static arm_cmsis_nn_status run_prelu_once(hct_server_session_t *session)
{
    hct_server_blob_t *input = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_0);
    hct_server_blob_t *alpha = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_1);

    if (input == NULL || alpha == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    if (session->expected_kernel_id == HCT_KERNEL_ID_PRELU_SCALAR_S8 ||
        session->expected_kernel_id == HCT_KERNEL_ID_PRELU_SCALAR_S16)
    {
        bool is_s16 = (session->expected_kernel_id == HCT_KERNEL_ID_PRELU_SCALAR_S16);
        int32_t block_size = session->block_size;
        int32_t element_size = is_s16 ? (int32_t)sizeof(int16_t) : (int32_t)sizeof(int8_t);
        int32_t num_pixels;
        if (block_size <= 0)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        if ((input->byte_length % (uint32_t)element_size) != 0u || (alpha->byte_length % (uint32_t)element_size) != 0u)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        num_pixels = (int32_t)(input->byte_length / (uint32_t)element_size);
        if (num_pixels <= 0 || alpha->byte_length != (uint32_t)(num_pixels * block_size * element_size))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        session->output_length = (uint32_t)(num_pixels * block_size * element_size);
        if (session->output_length > session->output_capacity_bytes)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        if (is_s16)
        {
            const int16_t *input_data = (const int16_t *)blob_ptr(session, input);
            const int16_t *alpha_data = (const int16_t *)blob_ptr(session, alpha);
            int16_t *output_data = (int16_t *)hct_output_ptr(session);
            for (int32_t pixel = 0; pixel < num_pixels; ++pixel)
            {
                arm_cmsis_nn_status status = arm_prelu_scalar_s16(input_data + pixel,
                                                                  alpha_data + pixel * block_size,
                                                                  true,
                                                                  session->input_offset, session->alpha_offset, session->output_offset,
                                                                  session->out_mult, session->out_shift,
                                                                  session->out_mult_alpha, session->out_shift_alpha,
                                                                  output_data + pixel * block_size, block_size);
                if (status != ARM_CMSIS_NN_SUCCESS)
                {
                    return status;
                }
            }
            return ARM_CMSIS_NN_SUCCESS;
        }
        {
            const int8_t *input_data = (const int8_t *)blob_ptr(session, input);
            const int8_t *alpha_data = (const int8_t *)blob_ptr(session, alpha);
            int8_t *output_data = (int8_t *)hct_output_ptr(session);
            for (int32_t pixel = 0; pixel < num_pixels; ++pixel)
            {
                arm_cmsis_nn_status status = arm_prelu_scalar_s8(input_data + pixel,
                                                                 alpha_data + pixel * block_size,
                                                                 true,
                                                                 session->input_offset, session->alpha_offset, session->output_offset,
                                                                 session->out_mult, session->out_shift,
                                                                 session->out_mult_alpha, session->out_shift_alpha,
                                                                 output_data + pixel * block_size, block_size);
                if (status != ARM_CMSIS_NN_SUCCESS)
                {
                    return status;
                }
            }
            return ARM_CMSIS_NN_SUCCESS;
        }
    }

    cmsis_nn_dims input_dims;
    cmsis_nn_dims alpha_dims;
    cmsis_nn_dims output_dims;
    input_dims.n = (int32_t)input->dimensions[0];
    input_dims.h = (int32_t)input->dimensions[1];
    input_dims.w = (int32_t)input->dimensions[2];
    input_dims.c = (int32_t)input->dimensions[3];
    alpha_dims.n = (int32_t)alpha->dimensions[0];
    alpha_dims.h = (int32_t)alpha->dimensions[1];
    alpha_dims.w = (int32_t)alpha->dimensions[2];
    alpha_dims.c = (int32_t)alpha->dimensions[3];
    output_dims.n = (session->output_n > 0) ? session->output_n : 1;
    output_dims.h = session->output_h;
    output_dims.w = session->output_w;
    output_dims.c = session->output_c;
    if (output_dims.h <= 0 || output_dims.w <= 0 || output_dims.c <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    bool out_is_s16 = (session->expected_kernel_id == HCT_KERNEL_ID_PRELU_S16);
    bool out_is_f16 = (session->expected_kernel_id == HCT_KERNEL_ID_PRELU_F16);
    bool out_is_f32 = (session->expected_kernel_id == HCT_KERNEL_ID_PRELU_F32);
    if (!hct_checked_dims_bytes(&output_dims,
                                out_is_f32 ? sizeof(float) :
                                out_is_f16 ? sizeof(float16_t) :
                                out_is_s16 ? sizeof(int16_t) : sizeof(int8_t),
                                session->output_capacity_bytes,
                                &session->output_length))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    {
        const uint32_t operand_element_size = out_is_f32 ? sizeof(float) :
                                              out_is_f16 ? sizeof(float16_t) :
                                              out_is_s16 ? sizeof(int16_t) : sizeof(int8_t);
        uint32_t operand_bytes_needed;
        if (!hct_checked_dims_bytes(&input_dims, operand_element_size, input->byte_length, &operand_bytes_needed) ||
            !hct_checked_dims_bytes(&alpha_dims, operand_element_size, alpha->byte_length, &operand_bytes_needed))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
    }
    if (out_is_f32)
    {
        return arm_prelu_f32(&input_dims, (const float *)blob_ptr(session, input),
                             &alpha_dims, (const float *)blob_ptr(session, alpha),
                             &output_dims, (float *)hct_output_ptr(session));
    }
    if (out_is_f16)
    {
        return arm_prelu_f16(&input_dims, (const float16_t *)blob_ptr(session, input),
                             &alpha_dims, (const float16_t *)blob_ptr(session, alpha),
                             &output_dims, (float16_t *)hct_output_ptr(session));
    }
    if (out_is_s16)
    {
        return arm_prelu_s16(&input_dims, (const int16_t *)blob_ptr(session, input),
                             &alpha_dims, (const int16_t *)blob_ptr(session, alpha),
                             session->input_offset, session->alpha_offset, session->output_offset,
                             session->out_mult, session->out_shift,
                             session->out_mult_alpha, session->out_shift_alpha,
                             &output_dims, (int16_t *)hct_output_ptr(session));
    }
    return arm_prelu_s8(&input_dims, (const int8_t *)blob_ptr(session, input),
                        &alpha_dims, (const int8_t *)blob_ptr(session, alpha),
                        session->input_offset, session->alpha_offset, session->output_offset,
                        session->out_mult, session->out_shift,
                        session->out_mult_alpha, session->out_shift_alpha,
                        &output_dims, (int8_t *)hct_output_ptr(session));
}'''

_RUN_NN_ACTIVATION_FLOAT_ONCE = '''\
/* ActivationFunctions NNActivationFloat shares one kernel_id per float dtype; the concrete
 * activation variant is chosen by the streamed activation_kind enum plus a float param
 * bit-cast through scale_bits (used only by LeakyRelu, 0.0f otherwise). */
static arm_cmsis_nn_status run_nn_activation_float_once(hct_server_session_t *session)
{
    hct_server_blob_t *input = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_0);
    int32_t block_size;

    if (input == NULL || session->block_size <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    block_size = session->block_size;

    if (session->expected_kernel_id == HCT_KERNEL_ID_NN_ACTIVATION_FLOAT_F32)
    {
        session->output_length = (uint32_t)(block_size * (int32_t)sizeof(float));
        if (session->output_length > session->output_capacity_bytes)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        {
            float param = quant_scale_from_bits(session->scale_bits);
            uint32_t operand_bytes_needed;
            if (!hct_checked_count_bytes(block_size, sizeof(float), input->byte_length, &operand_bytes_needed))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            return arm_nn_activation_f32((const float *)blob_ptr(session, input),
                                     (float *)hct_output_ptr(session),
                                     block_size,
                                     (arm_nn_activation_type_flt)session->activation_kind,
                                     param);
        }
    }
    if (session->expected_kernel_id == HCT_KERNEL_ID_NN_ACTIVATION_FLOAT_F16)
    {
        session->output_length = (uint32_t)(block_size * (int32_t)sizeof(float16_t));
        if (session->output_length > session->output_capacity_bytes)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        {
            float param = quant_scale_from_bits(session->scale_bits);
            uint32_t operand_bytes_needed;
            if (!hct_checked_count_bytes(block_size, sizeof(float16_t), input->byte_length, &operand_bytes_needed))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            return arm_nn_activation_f16((const float16_t *)blob_ptr(session, input),
                                     (float16_t *)hct_output_ptr(session),
                                     block_size,
                                     (arm_nn_activation_type_flt)session->activation_kind,
                                     param);
        }
    }
    return ARM_CMSIS_NN_ARG_ERROR;
}'''

_RUN_REDUCE_SUM_ONCE = '''\
/* BasicMathFunctions ReduceSum uses the same axis_dims encoding already emitted in the
 * generated headers: each of n/h/w/c is either 0 (not reduced) or 1 (reduce this axis). */
static arm_cmsis_nn_status run_reduce_sum_once(hct_server_session_t *session)
{
    hct_server_blob_t *input = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_0);
    cmsis_nn_dims input_dims;
    cmsis_nn_dims axis_dims;
    cmsis_nn_dims output_dims;

    if (input == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    input_dims.n = (int32_t)input->dimensions[0];
    input_dims.h = (int32_t)input->dimensions[1];
    input_dims.w = (int32_t)input->dimensions[2];
    input_dims.c = (int32_t)input->dimensions[3];
    axis_dims.n = session->axis_n;
    axis_dims.h = session->axis_h;
    axis_dims.w = session->axis_w;
    axis_dims.c = session->axis_c;
    output_dims.n = (session->output_n > 0) ? session->output_n : 1;
    output_dims.h = session->output_h;
    output_dims.w = session->output_w;
    output_dims.c = session->output_c;

    if (output_dims.n <= 0 || output_dims.h <= 0 || output_dims.w <= 0 || output_dims.c <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    if (session->expected_kernel_id == HCT_KERNEL_ID_REDUCE_SUM_F32)
    {
        uint32_t operand_bytes_needed;
        if (!hct_checked_dims_bytes(&output_dims, sizeof(float),
                                    session->output_capacity_bytes, &session->output_length))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        if (!hct_checked_dims_bytes(&input_dims, sizeof(float), input->byte_length, &operand_bytes_needed))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        return arm_reduce_sum_f32((const float *)blob_ptr(session, input),
                                  &input_dims, &axis_dims,
                                  (float *)hct_output_ptr(session),
                                  &output_dims);
    }

    if (session->expected_kernel_id == HCT_KERNEL_ID_REDUCE_SUM_F16)
    {
        uint32_t operand_bytes_needed;
        if (!hct_checked_dims_bytes(&output_dims, sizeof(float16_t),
                                    session->output_capacity_bytes, &session->output_length))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        if (!hct_checked_dims_bytes(&input_dims, sizeof(float16_t), input->byte_length, &operand_bytes_needed))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        return arm_reduce_sum_f16((const float16_t *)blob_ptr(session, input),
                                  &input_dims, &axis_dims,
                                  (float16_t *)hct_output_ptr(session),
                                  &output_dims);
    }
    return ARM_CMSIS_NN_ARG_ERROR;
}'''

_RUN_BATCH_NORM_ONCE = '''\
static arm_cmsis_nn_status run_batch_norm_once(hct_server_session_t *session)
{
    hct_server_blob_t *input = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_0);
    hct_server_blob_t *scale = find_blob_by_role(session, HCT_BLOB_ROLE_MULTIPLIER);
    hct_server_blob_t *bias = find_blob_by_role(session, HCT_BLOB_ROLE_BIAS);
    cmsis_nn_dims input_dims;
    int32_t element_count;

    if (input == NULL || scale == NULL || bias == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    input_dims.n = (int32_t)input->dimensions[0];
    input_dims.h = (int32_t)input->dimensions[1];
    input_dims.w = (int32_t)input->dimensions[2];
    input_dims.c = (int32_t)input->dimensions[3];
    element_count = input_dims.n * input_dims.h * input_dims.w * input_dims.c;
    if (element_count <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    if (session->expected_kernel_id == HCT_KERNEL_ID_BATCH_NORM_F32)
    {
        uint32_t operand_bytes_needed;
        session->output_length = (uint32_t)(element_count * (int32_t)sizeof(float));
        if (session->output_length > session->output_capacity_bytes)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        if (!hct_checked_count_bytes(element_count, sizeof(float), input->byte_length, &operand_bytes_needed) ||
            !hct_checked_count_bytes(input_dims.c, sizeof(float), scale->byte_length, &operand_bytes_needed) ||
            !hct_checked_count_bytes(input_dims.c, sizeof(float), bias->byte_length, &operand_bytes_needed))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        return arm_batch_norm_f32((const float *)blob_ptr(session, input),
                                  (float *)hct_output_ptr(session),
                                  (const float *)blob_ptr(session, scale),
                                  (const float *)blob_ptr(session, bias),
                                  &input_dims,
                                  session->activation_kind);
    }
    if (session->expected_kernel_id == HCT_KERNEL_ID_BATCH_NORM_F16)
    {
        uint32_t operand_bytes_needed;
        session->output_length = (uint32_t)(element_count * (int32_t)sizeof(float16_t));
        if (session->output_length > session->output_capacity_bytes)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        if (!hct_checked_count_bytes(element_count, sizeof(float16_t), input->byte_length, &operand_bytes_needed) ||
            !hct_checked_count_bytes(input_dims.c, sizeof(float16_t), scale->byte_length, &operand_bytes_needed) ||
            !hct_checked_count_bytes(input_dims.c, sizeof(float16_t), bias->byte_length, &operand_bytes_needed))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        return arm_batch_norm_f16((const float16_t *)blob_ptr(session, input),
                                  (float16_t *)hct_output_ptr(session),
                                  (const float16_t *)blob_ptr(session, scale),
                                  (const float16_t *)blob_ptr(session, bias),
                                  &input_dims,
                                  session->activation_kind);
    }
    return ARM_CMSIS_NN_ARG_ERROR;
}'''

_RUN_QUANTIZE_ONCE = '''\
/* Reinterprets session->scale_bits (transmitted as a raw int32) back into the float scale
 * value it was bit-cast from on the host -- see the scale_bits field comment in
 * benchmark_server_session.h. */
static float quant_scale_from_bits(int32_t bits)
{
    float scale;
    memcpy(&scale, &bits, sizeof(scale));
    return scale;
}

/* arm_quantize_f32_{s8,s16} (float input -> quantized output). The generated test's ReLU/
 * ReLU6 activation (if any) is applied to the float input BEFORE this kernel call in the
 * real TFLite-parity template, entirely in float space -- so the host-side bridge folds it
 * into the input blob it sends (see _build_quantize_case() in generated_test_bridge.py),
 * and this wrapper only ever needs to invoke the kernel itself. */
static arm_cmsis_nn_status run_quantize_once(hct_server_session_t *session)
{
    hct_server_blob_t *input = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_0);
    int32_t size;
    float scale;

    if (input == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    if (session->output_h <= 0 || session->output_w <= 0 || session->output_c <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    size = session->output_h * session->output_w * session->output_c;
    scale = quant_scale_from_bits(session->scale_bits);
    {
        uint32_t operand_bytes_needed;
        if (!hct_checked_count_bytes(size, sizeof(float), input->byte_length, &operand_bytes_needed))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
    }

    if (session->expected_kernel_id == HCT_KERNEL_ID_QUANTIZE_S16)
    {
        session->output_length = (uint32_t)(size * (int32_t)sizeof(int16_t));
        if (session->output_length > session->output_capacity_bytes)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        return arm_quantize_f32_s16((const float *)blob_ptr(session, input),
                                    (int16_t *)hct_output_ptr(session),
                                    size, session->output_offset, scale);
    }
    session->output_length = (uint32_t)size;
    if (session->output_length > session->output_capacity_bytes)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    return arm_quantize_f32_s8((const float *)blob_ptr(session, input),
                              (int8_t *)hct_output_ptr(session),
                              size, session->output_offset, scale);
}'''

_RUN_DEQUANTIZE_ONCE = '''\
/* arm_dequantize_{s8,s16}_f32 (quantized input -> float output). Unlike Quantize, the
 * generated test's ReLU/ReLU6 activation (if any) is applied AFTER this kernel call, to
 * the dequantized float output -- so, unlike Quantize, it must be replicated here to match
 * the golden output (see activation_kind field comment in benchmark_server_session.h). Reuses
 * quant_scale_from_bits() defined above by the Quantize adapter. */
static arm_cmsis_nn_status run_dequantize_once(hct_server_session_t *session)
{
    hct_server_blob_t *input = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_0);
    int32_t size;
    float scale;
    float *out;
    arm_cmsis_nn_status status;

    if (input == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    if (session->output_h <= 0 || session->output_w <= 0 || session->output_c <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    size = session->output_h * session->output_w * session->output_c;
    session->output_length = (uint32_t)(size * (int32_t)sizeof(float));
    if (session->output_length > session->output_capacity_bytes)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    scale = quant_scale_from_bits(session->scale_bits);
    out = (float *)hct_output_ptr(session);
    {
        const bool dequantize_is_s16 = (session->expected_kernel_id == HCT_KERNEL_ID_DEQUANTIZE_S16);
        uint32_t operand_bytes_needed;
        if (!hct_checked_count_bytes(size, dequantize_is_s16 ? sizeof(int16_t) : sizeof(int8_t),
                                     input->byte_length, &operand_bytes_needed))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
    }

    if (session->expected_kernel_id == HCT_KERNEL_ID_DEQUANTIZE_S16)
    {
        status = arm_dequantize_s16_f32((const int16_t *)blob_ptr(session, input),
                                        out, size, session->input_offset, scale);
    }
    else
    {
        status = arm_dequantize_s8_f32((const int8_t *)blob_ptr(session, input),
                                       out, size, session->input_offset, scale);
    }
    if (status != ARM_CMSIS_NN_SUCCESS)
    {
        return status;
    }

    if (session->activation_kind == 1) /* RELU */
    {
        int32_t i;
        for (i = 0; i < size; ++i)
        {
            if (out[i] < 0.0f) out[i] = 0.0f;
        }
    }
    else if (session->activation_kind == 2) /* RELU6 */
    {
        int32_t i;
        for (i = 0; i < size; ++i)
        {
            if (out[i] < 0.0f) out[i] = 0.0f;
            if (out[i] > 6.0f) out[i] = 6.0f;
        }
    }
    return ARM_CMSIS_NN_SUCCESS;
}'''

_RUN_REQUANTIZE_ONCE = '''\
static arm_cmsis_nn_status run_requantize_once(hct_server_session_t *session)
{
    hct_server_blob_t *input = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_0);

    if (input == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    session->output_length = input->byte_length;
    if (session->output_length > session->output_capacity_bytes)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    if (session->expected_kernel_id == HCT_KERNEL_ID_REQUANTIZE_S16)
    {
        return arm_requantize_s16_s16((const int16_t *)blob_ptr(session, input),
                                      (int16_t *)hct_output_ptr(session),
                                      (int32_t)(input->byte_length / sizeof(int16_t)),
                                      session->out_mult,
                                      session->out_shift,
                                      session->input_offset,
                                      session->output_offset);
    }
    return arm_requantize_s8_s8((const int8_t *)blob_ptr(session, input),
                                (int8_t *)hct_output_ptr(session),
                                (int32_t)input->byte_length,
                                session->out_mult,
                                session->out_shift,
                                session->input_offset,
                                session->output_offset);
}'''

_RUN_COMPARISON_ONCE = '''\
static arm_cmsis_nn_status run_comparison_once(hct_server_session_t *session)
{
    hct_server_blob_t *input_1 = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_0);
    hct_server_blob_t *input_2 = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_1);
    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    if (input_1 == NULL || input_2 == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    input_1_dims.n = (int32_t)input_1->dimensions[0];
    input_1_dims.h = (int32_t)input_1->dimensions[1];
    input_1_dims.w = (int32_t)input_1->dimensions[2];
    input_1_dims.c = (int32_t)input_1->dimensions[3];

    input_2_dims.n = (int32_t)input_2->dimensions[0];
    input_2_dims.h = (int32_t)input_2->dimensions[1];
    input_2_dims.w = (int32_t)input_2->dimensions[2];
    input_2_dims.c = (int32_t)input_2->dimensions[3];

    output_dims.n = session->output_n;
    output_dims.h = session->output_h;
    output_dims.w = session->output_w;
    output_dims.c = session->output_c;
    if (!hct_checked_dims_bytes(&output_dims, sizeof(bool),
                                session->output_capacity_bytes, &session->output_length))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    {
        const bool comparison_is_s16 = (session->expected_kernel_id == HCT_KERNEL_ID_EQUAL_S16 ||
                                        session->expected_kernel_id == HCT_KERNEL_ID_NOT_EQUAL_S16 ||
                                        session->expected_kernel_id == HCT_KERNEL_ID_GREATER_S16 ||
                                        session->expected_kernel_id == HCT_KERNEL_ID_GREATER_EQUAL_S16 ||
                                        session->expected_kernel_id == HCT_KERNEL_ID_LESS_S16 ||
                                        session->expected_kernel_id == HCT_KERNEL_ID_LESS_EQUAL_S16);
        const uint32_t comparison_element_size = comparison_is_s16 ? sizeof(int16_t) : sizeof(int8_t);
        uint32_t operand_bytes_needed;
        if (!hct_checked_dims_bytes(&input_1_dims, comparison_element_size, input_1->byte_length, &operand_bytes_needed) ||
            !hct_checked_dims_bytes(&input_2_dims, comparison_element_size, input_2->byte_length, &operand_bytes_needed))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
    }
    switch (session->expected_kernel_id)
    {
        case HCT_KERNEL_ID_EQUAL_S8:
            return arm_equal_s8(NULL,
                                (const int8_t *)blob_ptr(session, input_1),
                                &input_1_dims,
                                (const int8_t *)blob_ptr(session, input_2),
                                &input_2_dims,
                                (bool *)hct_output_ptr(session),
                                &output_dims,
                                session->input1_offset,
                                session->input1_mult,
                                session->input1_shift,
                                session->input2_offset,
                                session->input2_mult,
                                session->input2_shift,
                                session->left_shift);
        case HCT_KERNEL_ID_NOT_EQUAL_S8:
            return arm_not_equal_s8(NULL,
                                    (const int8_t *)blob_ptr(session, input_1),
                                    &input_1_dims,
                                    (const int8_t *)blob_ptr(session, input_2),
                                    &input_2_dims,
                                    (bool *)hct_output_ptr(session),
                                    &output_dims,
                                    session->input1_offset,
                                    session->input1_mult,
                                    session->input1_shift,
                                    session->input2_offset,
                                    session->input2_mult,
                                    session->input2_shift,
                                    session->left_shift);
        case HCT_KERNEL_ID_GREATER_S8:
            return arm_greater_s8(NULL,
                                  (const int8_t *)blob_ptr(session, input_1),
                                  &input_1_dims,
                                  (const int8_t *)blob_ptr(session, input_2),
                                  &input_2_dims,
                                  (bool *)hct_output_ptr(session),
                                  &output_dims,
                                  session->input1_offset,
                                  session->input1_mult,
                                  session->input1_shift,
                                  session->input2_offset,
                                  session->input2_mult,
                                  session->input2_shift,
                                  session->left_shift);
        case HCT_KERNEL_ID_GREATER_EQUAL_S8:
            return arm_greater_equal_s8(NULL,
                                        (const int8_t *)blob_ptr(session, input_1),
                                        &input_1_dims,
                                        (const int8_t *)blob_ptr(session, input_2),
                                        &input_2_dims,
                                        (bool *)hct_output_ptr(session),
                                        &output_dims,
                                        session->input1_offset,
                                        session->input1_mult,
                                        session->input1_shift,
                                        session->input2_offset,
                                        session->input2_mult,
                                        session->input2_shift,
                                        session->left_shift);
        case HCT_KERNEL_ID_LESS_S8:
            return arm_less_s8(NULL,
                               (const int8_t *)blob_ptr(session, input_1),
                               &input_1_dims,
                               (const int8_t *)blob_ptr(session, input_2),
                               &input_2_dims,
                               (bool *)hct_output_ptr(session),
                               &output_dims,
                               session->input1_offset,
                               session->input1_mult,
                               session->input1_shift,
                               session->input2_offset,
                               session->input2_mult,
                               session->input2_shift,
                               session->left_shift);
        case HCT_KERNEL_ID_LESS_EQUAL_S8:
            return arm_less_equal_s8(NULL,
                                     (const int8_t *)blob_ptr(session, input_1),
                                     &input_1_dims,
                                     (const int8_t *)blob_ptr(session, input_2),
                                     &input_2_dims,
                                     (bool *)hct_output_ptr(session),
                                     &output_dims,
                                     session->input1_offset,
                                     session->input1_mult,
                                     session->input1_shift,
                                     session->input2_offset,
                                     session->input2_mult,
                                     session->input2_shift,
                                     session->left_shift);
        case HCT_KERNEL_ID_EQUAL_S16:
            return arm_equal_s16(NULL,
                                 (const int16_t *)blob_ptr(session, input_1),
                                 &input_1_dims,
                                 (const int16_t *)blob_ptr(session, input_2),
                                 &input_2_dims,
                                 (bool *)hct_output_ptr(session),
                                 &output_dims,
                                 session->input1_offset,
                                 session->input1_mult,
                                 session->input1_shift,
                                 session->input2_offset,
                                 session->input2_mult,
                                 session->input2_shift,
                                 session->left_shift);
        case HCT_KERNEL_ID_NOT_EQUAL_S16:
            return arm_not_equal_s16(NULL,
                                     (const int16_t *)blob_ptr(session, input_1),
                                     &input_1_dims,
                                     (const int16_t *)blob_ptr(session, input_2),
                                     &input_2_dims,
                                     (bool *)hct_output_ptr(session),
                                     &output_dims,
                                     session->input1_offset,
                                     session->input1_mult,
                                     session->input1_shift,
                                     session->input2_offset,
                                     session->input2_mult,
                                     session->input2_shift,
                                     session->left_shift);
        case HCT_KERNEL_ID_GREATER_S16:
            return arm_greater_s16(NULL,
                                   (const int16_t *)blob_ptr(session, input_1),
                                   &input_1_dims,
                                   (const int16_t *)blob_ptr(session, input_2),
                                   &input_2_dims,
                                   (bool *)hct_output_ptr(session),
                                   &output_dims,
                                   session->input1_offset,
                                   session->input1_mult,
                                   session->input1_shift,
                                   session->input2_offset,
                                   session->input2_mult,
                                   session->input2_shift,
                                   session->left_shift);
        case HCT_KERNEL_ID_GREATER_EQUAL_S16:
            return arm_greater_equal_s16(NULL,
                                         (const int16_t *)blob_ptr(session, input_1),
                                         &input_1_dims,
                                         (const int16_t *)blob_ptr(session, input_2),
                                         &input_2_dims,
                                         (bool *)hct_output_ptr(session),
                                         &output_dims,
                                         session->input1_offset,
                                         session->input1_mult,
                                         session->input1_shift,
                                         session->input2_offset,
                                         session->input2_mult,
                                         session->input2_shift,
                                         session->left_shift);
        case HCT_KERNEL_ID_LESS_S16:
            return arm_less_s16(NULL,
                                (const int16_t *)blob_ptr(session, input_1),
                                &input_1_dims,
                                (const int16_t *)blob_ptr(session, input_2),
                                &input_2_dims,
                                (bool *)hct_output_ptr(session),
                                &output_dims,
                                session->input1_offset,
                                session->input1_mult,
                                session->input1_shift,
                                session->input2_offset,
                                session->input2_mult,
                                session->input2_shift,
                                session->left_shift);
        case HCT_KERNEL_ID_LESS_EQUAL_S16:
            return arm_less_equal_s16(NULL,
                                      (const int16_t *)blob_ptr(session, input_1),
                                      &input_1_dims,
                                      (const int16_t *)blob_ptr(session, input_2),
                                      &input_2_dims,
                                      (bool *)hct_output_ptr(session),
                                      &output_dims,
                                      session->input1_offset,
                                      session->input1_mult,
                                      session->input1_shift,
                                      session->input2_offset,
                                      session->input2_mult,
                                      session->input2_shift,
                                      session->left_shift);
        default:
            return ARM_CMSIS_NN_ARG_ERROR;
    }
}'''

_RUN_TRANSPOSE_CONV_ONCE = '''\
static arm_cmsis_nn_status run_transpose_conv_once(hct_server_session_t *session)
{
    hct_server_blob_t *input = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_0);
    hct_server_blob_t *weights = find_blob_by_role(session, HCT_BLOB_ROLE_WEIGHTS);
    hct_server_blob_t *bias = find_blob_by_role(session, HCT_BLOB_ROLE_BIAS);
    cmsis_nn_context ctx;
    cmsis_nn_context output_ctx;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;
    uint32_t local_offset = 0u;
    uint32_t ctx_offset;
    uint32_t output_ctx_offset;
    int32_t required_ctx;
    int32_t required_output_ctx;

    if (input == NULL || weights == NULL)
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
    output_dims.n = (session->output_n > 0) ? session->output_n : 1;
    output_dims.h = session->output_h;
    output_dims.w = session->output_w;
    output_dims.c = session->output_c;
    if (input_dims.n != 1 || output_dims.n != 1 || output_dims.h <= 0 || output_dims.w <= 0 || output_dims.c <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    bias_dims.n = 1;
    bias_dims.h = 1;
    bias_dims.w = 1;
    bias_dims.c = output_dims.c;
    if (bias != NULL && bias->rank > 0u && bias->dimensions[0] > 0u)
    {
        bias_dims.c = (int32_t)bias->dimensions[0];
    }

    if (session->expected_kernel_id == HCT_KERNEL_ID_TRANSPOSE_CONV_F32 ||
        session->expected_kernel_id == HCT_KERNEL_ID_TRANSPOSE_CONV_F16)
    {
        const bool is_f16 = (session->expected_kernel_id == HCT_KERNEL_ID_TRANSPOSE_CONV_F16);
        const uint32_t element_size = is_f16 ? (uint32_t)sizeof(float16_t) : (uint32_t)sizeof(float);
        const float activation_min = quant_scale_from_bits(session->float_activation_min_bits);
        const float activation_max = quant_scale_from_bits(session->float_activation_max_bits);
        cmsis_nn_transpose_conv_params_f32 params_f32;
        cmsis_nn_transpose_conv_params_f16 params_f16;

        if (!hct_checked_dims_bytes(&output_dims, element_size,
                                    session->output_capacity_bytes, &session->output_length))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        {
            uint32_t operand_bytes_needed;
            if (!hct_checked_dims_bytes(&input_dims, element_size, input->byte_length, &operand_bytes_needed) ||
                !hct_checked_dims_bytes(&filter_dims, element_size, weights->byte_length, &operand_bytes_needed) ||
                (bias != NULL && !hct_checked_count_bytes(bias_dims.c, element_size, bias->byte_length, &operand_bytes_needed)))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
        }

        if (is_f16)
        {
            params_f16.stride.w = session->stride_w;
            params_f16.stride.h = session->stride_h;
            params_f16.dilation.w = session->dilation_w;
            params_f16.dilation.h = session->dilation_h;
            params_f16.padding.w = session->pad_w;
            params_f16.padding.h = session->pad_h;
            params_f16.padding_offsets.w = session->pad_offset_w;
            params_f16.padding_offsets.h = session->pad_offset_h;
            params_f16.activation.min = (float16_t)activation_min;
            params_f16.activation.max = (float16_t)activation_max;
            required_ctx = arm_transpose_conv_f16_get_buffer_size(&params_f16, &input_dims, &filter_dims, &output_dims);
            required_output_ctx = arm_transpose_conv_f16_get_reverse_conv_buffer_size(&params_f16, &input_dims, &filter_dims);
        }
        else
        {
            params_f32.stride.w = session->stride_w;
            params_f32.stride.h = session->stride_h;
            params_f32.dilation.w = session->dilation_w;
            params_f32.dilation.h = session->dilation_h;
            params_f32.padding.w = session->pad_w;
            params_f32.padding.h = session->pad_h;
            params_f32.padding_offsets.w = session->pad_offset_w;
            params_f32.padding_offsets.h = session->pad_offset_h;
            params_f32.activation.min = activation_min;
            params_f32.activation.max = activation_max;
            required_ctx = arm_transpose_conv_f32_get_buffer_size(&params_f32, &input_dims, &filter_dims, &output_dims);
            required_output_ctx = arm_transpose_conv_f32_get_reverse_conv_buffer_size(&params_f32, &input_dims, &filter_dims);
        }
        if (required_ctx < 0 || required_output_ctx < 0)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }

        if (!hct_checked_aligned_range(local_offset, 16u, (uint32_t)required_ctx,
                                       session->scratch_bytes, &ctx_offset, &local_offset) ||
            !hct_checked_aligned_range(local_offset, 16u, (uint32_t)required_output_ctx,
                                       session->scratch_bytes, &output_ctx_offset, &local_offset))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }

        ctx.buf = (required_ctx > 0) ? &session->workspace[session->scratch_offset + ctx_offset] : NULL;
        ctx.size = required_ctx;
        output_ctx.buf = (required_output_ctx > 0) ? &session->workspace[session->scratch_offset + output_ctx_offset] : NULL;
        output_ctx.size = required_output_ctx;

        if (is_f16)
        {
            return arm_transpose_conv_f16(&ctx,
                                          &output_ctx,
                                          &params_f16,
                                          &input_dims,
                                          (const float16_t *)blob_ptr(session, input),
                                          &filter_dims,
                                          (const float16_t *)blob_ptr(session, weights),
                                          &bias_dims,
                                          (bias != NULL) ? (const float16_t *)blob_ptr(session, bias) : NULL,
                                          &output_dims,
                                          (float16_t *)hct_output_ptr(session),
                                          ARM_NN_LAYOUT_NHWC);
        }
        return arm_transpose_conv_f32(&ctx,
                                      &output_ctx,
                                      &params_f32,
                                      &input_dims,
                                      (const float *)blob_ptr(session, input),
                                      &filter_dims,
                                      (const float *)blob_ptr(session, weights),
                                      &bias_dims,
                                      (bias != NULL) ? (const float *)blob_ptr(session, bias) : NULL,
                                      &output_dims,
                                      (float *)hct_output_ptr(session),
                                      ARM_NN_LAYOUT_NHWC);
    }

    {
        hct_server_blob_t *multiplier = find_blob_by_role(session, HCT_BLOB_ROLE_MULTIPLIER);
        hct_server_blob_t *shift = find_blob_by_role(session, HCT_BLOB_ROLE_SHIFT);
        cmsis_nn_context weight_sum_ctx;
        cmsis_nn_transpose_conv_params params;
        cmsis_nn_per_channel_quant_params quant_params;
        uint32_t weight_sum_offset;
        uint32_t weight_sum_bytes;
        uint32_t weight_sum_end;
        const int32_t *bias_data = NULL;

        if (multiplier == NULL || shift == NULL)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }

        if (!hct_checked_dims_bytes(&output_dims, sizeof(int8_t),
                                    session->output_capacity_bytes, &session->output_length))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }

        params.input_offset = session->input_offset;
        params.output_offset = session->output_offset;
        params.stride.w = session->stride_w;
        params.stride.h = session->stride_h;
        params.dilation.w = session->dilation_w;
        params.dilation.h = session->dilation_h;
        params.padding.w = session->pad_w;
        params.padding.h = session->pad_h;
        params.padding_offsets.w = session->pad_offset_w;
        params.padding_offsets.h = session->pad_offset_h;
        params.activation.min = session->activation_min;
        params.activation.max = session->activation_max;

        quant_params.multiplier = (int32_t *)blob_ptr(session, multiplier);
        quant_params.shift = (int32_t *)blob_ptr(session, shift);
        if (bias != NULL)
        {
            bias_data = (const int32_t *)blob_ptr(session, bias);
        }

        {
            uint32_t operand_bytes_needed;
            if (!hct_checked_dims_bytes(&input_dims, sizeof(int8_t), input->byte_length, &operand_bytes_needed) ||
                !hct_checked_dims_bytes(&filter_dims, sizeof(int8_t), weights->byte_length, &operand_bytes_needed) ||
                (bias != NULL && !hct_checked_count_bytes(bias_dims.c, sizeof(int32_t), bias->byte_length, &operand_bytes_needed)) ||
                !hct_checked_count_bytes(output_dims.c, sizeof(int32_t), multiplier->byte_length, &operand_bytes_needed) ||
                !hct_checked_count_bytes(output_dims.c, sizeof(int32_t), shift->byte_length, &operand_bytes_needed))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
        }

        required_ctx = arm_transpose_conv_s8_get_buffer_size(&params, &input_dims, &filter_dims, &output_dims);
        required_output_ctx = arm_transpose_conv_s8_get_reverse_conv_buffer_size(&params, &input_dims, &filter_dims);
        if (required_ctx < 0 || required_output_ctx < 0)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }

        {
            const int32_t weight_sum_shape[1] = {output_dims.c};
            if (!hct_checked_shape_bytes(weight_sum_shape, 1, sizeof(int32_t),
                                         session->scratch_bytes, &weight_sum_bytes) ||
                !hct_checked_aligned_range(local_offset, 16u, (uint32_t)required_ctx,
                                           session->scratch_bytes, &ctx_offset, &local_offset) ||
                !hct_checked_aligned_range(local_offset, 16u, (uint32_t)required_output_ctx,
                                           session->scratch_bytes, &output_ctx_offset, &local_offset) ||
                !hct_checked_aligned_range(local_offset, 16u, weight_sum_bytes,
                                           session->scratch_bytes, &weight_sum_offset, &weight_sum_end))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
        }
        if (weight_sum_end > session->scratch_bytes)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }

        ctx.buf = (required_ctx > 0) ? &session->workspace[session->scratch_offset + ctx_offset] : NULL;
        ctx.size = required_ctx;
        output_ctx.buf = (required_output_ctx > 0) ? &session->workspace[session->scratch_offset + output_ctx_offset] : NULL;
        output_ctx.size = required_output_ctx;
        weight_sum_ctx.buf = (weight_sum_bytes > 0u) ? &session->workspace[session->scratch_offset + weight_sum_offset] : NULL;
        weight_sum_ctx.size = (int32_t)weight_sum_bytes;

        if (arm_convolve_weight_sum((int32_t *)weight_sum_ctx.buf,
                                    (const int8_t *)blob_ptr(session, weights),
                                    &input_dims,
                                    &filter_dims,
                                    &output_dims,
                                    params.input_offset,
                                    bias_data) != ARM_CMSIS_NN_SUCCESS)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }

        return arm_transpose_conv_wrapper_s8(&ctx,
                                             &weight_sum_ctx,
                                             &output_ctx,
                                             &params,
                                             &quant_params,
                                             &input_dims,
                                             (const int8_t *)blob_ptr(session, input),
                                             &filter_dims,
                                             (const int8_t *)blob_ptr(session, weights),
                                             &bias_dims,
                                             bias_data,
                                             &output_dims,
                                             (int8_t *)hct_output_ptr(session));
    }
}'''



_RUN_SOFTMAX_ONCE = '''\
/* Fixed CMSIS-NN reference lookup tables required by arm_softmax_s16() -- identical bit
 * patterns are used by every generated S16 softmax test case (see
 * Tests/helia-core-tester/assets/templates/SoftmaxFunctions/softmax/softmax.h.j2), so they
 * are embedded once as firmware constants rather than transmitted per case. */
static const int16_t hct_softmax_exp_lut[513] = {

    2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
    2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     3,     3,     3,     3,     3,
    3,     3,     3,     3,     3,     3,     3,     3,     3,     3,     3,     3,     4,     4,     4,     4,
    4,     4,     4,     4,     4,     4,     4,     4,     4,     5,     5,     5,     5,     5,     5,     5,
    5,     5,     5,     6,     6,     6,     6,     6,     6,     6,     6,     7,     7,     7,     7,     7,
    7,     7,     7,     8,     8,     8,     8,     8,     8,     9,     9,     9,     9,     9,     9,     10,
    10,    10,    10,    10,    11,    11,    11,    11,    11,    12,    12,    12,    12,    13,    13,    13,
    13,    14,    14,    14,    14,    15,    15,    15,    16,    16,    16,    17,    17,    17,    18,    18,
    18,    19,    19,    19,    20,    20,    21,    21,    21,    22,    22,    23,    23,    24,    24,    25,
    25,    26,    26,    27,    27,    28,    28,    29,    29,    30,    30,    31,    32,    32,    33,    34,
    34,    35,    36,    36,    37,    37,    38,    39,    40,    40,    42,    42,    43,    44,    45,    45,
    46,    47,    48,    49,    50,    51,    52,    53,    54,    55,    56,    57,    59,    60,    60,    62,
    63,    65,    65,    67,    68,    69,    71,    73,    74,    75,    77,    78,    80,    81,    83,    85,
    86,    88,    90,    92,    93,    95,    97,    99,    101,   103,   105,   107,   109,   112,   114,   116,
    118,   121,   123,   126,   128,   131,   133,   135,   139,   141,   144,   147,   149,   152,   155,   158,
    162,   165,   168,   171,   174,   178,   181,   185,   189,   192,   196,   200,   204,   208,   212,   217,
    221,   225,   230,   234,   239,   243,   248,   253,   258,   263,   268,   273,   279,   284,   290,   296,
    302,   308,   314,   320,   327,   333,   340,   346,   353,   360,   366,   374,   381,   389,   397,   404,
    413,   421,   429,   437,   446,   455,   464,   473,   482,   492,   501,   511,   522,   532,   543,   553,
    564,   575,   586,   598,   610,   622,   634,   646,   659,   672,   685,   699,   713,   727,   741,   756,
    771,   786,   801,   817,   833,   850,   866,   884,   901,   919,   937,   955,   974,   993,   1013,  1033,
    1053,  1074,  1095,  1117,  1139,  1161,  1184,  1207,  1232,  1256,  1281,  1306,  1332,  1358,  1385,  1412,
    1440,  1468,  1497,  1527,  1557,  1587,  1619,  1651,  1683,  1716,  1750,  1785,  1820,  1856,  1892,  1930,
    1968,  2006,  2046,  2087,  2128,  2170,  2212,  2256,  2300,  2346,  2392,  2439,  2488,  2537,  2587,  2638,
    2690,  2743,  2796,  2852,  2908,  2966,  3024,  3084,  3145,  3207,  3270,  3334,  3400,  3467,  3535,  3605,
    3677,  3749,  3822,  3898,  3975,  4053,  4133,  4214,  4297,  4383,  4469,  4557,  4647,  4739,  4833,  4927,
    5024,  5124,  5225,  5328,  5433,  5541,  5649,  5761,  5875,  5991,  6109,  6230,  6352,  6477,  6605,  6736,
    6868,  7004,  7141,  7282,  7427,  7572,  7722,  7874,  8030,  8188,  8350,  8514,  8683,  8854,  9028,  9206,
    9387,  9572,  9762,  9954,  10151, 10351, 10555, 10763, 10976, 11191, 11412, 11637, 11867, 12102, 12341, 12583,
    12831, 13085, 13342, 13606, 13874, 14148, 14427, 14711, 15002, 15297, 15599, 15907, 16221, 16541, 16867, 17199,
    17539, 17884, 18237, 18597, 18964, 19338, 19719, 20108, 20505, 20909, 21322, 21742, 22171, 22608, 23054, 23509,
    23973, 24445, 24928, 25419, 25921, 26432, 26953, 27485, 28027, 28580, 29143, 29718, 30304, 30902, 31512, 32133,
    32767
};
static const int16_t hct_softmax_one_by_one_lut[513] = {

    32767, 32704, 32640, 32578, 32514, 32451, 32388, 32326, 32264, 32202, 32141, 32079, 32018, 31957, 31896, 31835,
    31775, 31715, 31655, 31596, 31537, 31476, 31418, 31359, 31301, 31242, 31184, 31127, 31069, 31011, 30954, 30897,
    30840, 30784, 30727, 30671, 30615, 30560, 30504, 30449, 30394, 30339, 30283, 30229, 30175, 30121, 30067, 30013,
    29960, 29906, 29853, 29800, 29746, 29694, 29642, 29589, 29537, 29486, 29434, 29382, 29331, 29280, 29229, 29177,
    29127, 29076, 29026, 28976, 28926, 28877, 28827, 28777, 28728, 28679, 28630, 28581, 28532, 28484, 28436, 28388,
    28340, 28292, 28244, 28197, 28150, 28103, 28056, 28008, 27962, 27915, 27869, 27823, 27777, 27731, 27685, 27640,
    27594, 27549, 27504, 27459, 27413, 27369, 27324, 27280, 27236, 27192, 27148, 27104, 27060, 27016, 26973, 26930,
    26887, 26844, 26801, 26758, 26715, 26673, 26630, 26588, 26546, 26504, 26463, 26421, 26380, 26338, 26297, 26255,
    26214, 26174, 26132, 26092, 26051, 26011, 25971, 25931, 25891, 25851, 25811, 25772, 25732, 25693, 25653, 25614,
    25575, 25536, 25497, 25458, 25420, 25381, 25343, 25305, 25267, 25229, 25191, 25153, 25116, 25078, 25041, 25003,
    24966, 24928, 24892, 24855, 24818, 24781, 24745, 24709, 24672, 24636, 24600, 24564, 24528, 24492, 24457, 24421,
    24385, 24350, 24315, 24280, 24245, 24210, 24175, 24140, 24105, 24070, 24036, 24002, 23967, 23933, 23899, 23865,
    23831, 23798, 23764, 23730, 23697, 23664, 23630, 23597, 23564, 23530, 23498, 23465, 23432, 23399, 23366, 23334,
    23302, 23269, 23237, 23205, 23173, 23141, 23109, 23077, 23046, 23014, 22982, 22951, 22920, 22888, 22857, 22826,
    22795, 22764, 22733, 22703, 22672, 22641, 22611, 22580, 22550, 22520, 22490, 22459, 22429, 22400, 22370, 22340,
    22310, 22281, 22251, 22221, 22192, 22163, 22134, 22104, 22075, 22046, 22017, 21988, 21959, 21931, 21902, 21874,
    21845, 21817, 21788, 21760, 21732, 21704, 21676, 21648, 21620, 21592, 21565, 21537, 21509, 21482, 21455, 21427,
    21400, 21372, 21345, 21318, 21291, 21264, 21237, 21210, 21183, 21157, 21130, 21103, 21077, 21050, 21024, 20998,
    20971, 20945, 20919, 20893, 20867, 20841, 20816, 20790, 20764, 20738, 20713, 20687, 20662, 20636, 20611, 20586,
    20560, 20535, 20510, 20485, 20460, 20435, 20410, 20385, 20360, 20336, 20311, 20287, 20262, 20238, 20213, 20189,
    20165, 20141, 20117, 20092, 20068, 20044, 20021, 19997, 19973, 19949, 19926, 19902, 19878, 19855, 19832, 19808,
    19784, 19762, 19738, 19715, 19692, 19668, 19645, 19622, 19600, 19577, 19553, 19531, 19508, 19485, 19463, 19440,
    19418, 19395, 19373, 19351, 19328, 19306, 19284, 19262, 19240, 19218, 19196, 19174, 19152, 19130, 19109, 19087,
    19065, 19044, 19022, 19000, 18979, 18958, 18936, 18915, 18893, 18872, 18851, 18830, 18809, 18787, 18766, 18745,
    18725, 18704, 18682, 18662, 18641, 18620, 18600, 18579, 18559, 18538, 18518, 18497, 18477, 18457, 18436, 18416,
    18396, 18376, 18356, 18336, 18316, 18296, 18276, 18256, 18236, 18216, 18197, 18177, 18157, 18138, 18118, 18099,
    18079, 18059, 18040, 18021, 18001, 17982, 17963, 17944, 17924, 17905, 17886, 17867, 17848, 17829, 17810, 17791,
    17772, 17754, 17735, 17716, 17697, 17679, 17660, 17641, 17623, 17604, 17586, 17568, 17549, 17531, 17513, 17494,
    17476, 17458, 17440, 17422, 17404, 17386, 17368, 17350, 17332, 17314, 17296, 17278, 17261, 17243, 17225, 17208,
    17190, 17172, 17155, 17137, 17120, 17102, 17085, 17067, 17050, 17033, 17015, 16999, 16981, 16964, 16947, 16930,
    16913, 16895, 16878, 16862, 16845, 16828, 16810, 16794, 16777, 16760, 16743, 16727, 16710, 16693, 16677, 16660,
    16644, 16627, 16611, 16594, 16578, 16562, 16545, 16529, 16513, 16497, 16480, 16464, 16448, 16432, 16416, 16400,
    16384
};
static const cmsis_nn_softmax_lut_s16 hct_softmax_lut_s16 = {
    .exp_lut = hct_softmax_exp_lut,
    .one_by_one_lut = hct_softmax_one_by_one_lut
};

/* arm_softmax_s8/arm_softmax_s16/arm_softmax_s8_s16 -- all three share the same
 * (num_rows, row_size, mult, shift[, diff_min]) requantization scheme; softmax always
 * operates over the last dimension of a flattened 2D (num_rows, row_size) view, sent
 * explicitly as scalar params by the host (see _build_softmax_case() in
 * generated_test_bridge.py). out_mult/out_shift are reused for mult/shift; diff_min is only
 * meaningful for the two int8-input kernels (arm_softmax_s8, arm_softmax_s8_s16) and is
 * left unused (0) for the pure-S16 kernel. */
static arm_cmsis_nn_status run_softmax_once(hct_server_session_t *session)
{
    hct_server_blob_t *input = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_0);
    int32_t size;

    if (input == NULL || session->num_rows <= 0 || session->row_size <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    size = session->num_rows * session->row_size;

    if (session->expected_kernel_id == HCT_KERNEL_ID_SOFTMAX_F32)
    {
        uint32_t operand_bytes_needed;
        session->output_length = (uint32_t)(size * (int32_t)sizeof(float));
        if (session->output_length > session->output_capacity_bytes)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        if (!hct_checked_count_bytes(size, sizeof(float), input->byte_length, &operand_bytes_needed))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        return arm_softmax_f32((const float *)blob_ptr(session, input),
                               session->num_rows, session->row_size,
                               (float *)hct_output_ptr(session));
    }
    if (session->expected_kernel_id == HCT_KERNEL_ID_SOFTMAX_F16)
    {
        uint32_t operand_bytes_needed;
        session->output_length = (uint32_t)(size * (int32_t)sizeof(float16_t));
        if (session->output_length > session->output_capacity_bytes)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        if (!hct_checked_count_bytes(size, sizeof(float16_t), input->byte_length, &operand_bytes_needed))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        return arm_softmax_f16((const float16_t *)blob_ptr(session, input),
                               session->num_rows, session->row_size,
                               (float16_t *)hct_output_ptr(session));
    }
    if (session->expected_kernel_id == HCT_KERNEL_ID_SOFTMAX_S16)
    {
        uint32_t operand_bytes_needed;
        session->output_length = (uint32_t)(size * (int32_t)sizeof(int16_t));
        if (session->output_length > session->output_capacity_bytes)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        if (!hct_checked_count_bytes(size, sizeof(int16_t), input->byte_length, &operand_bytes_needed))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        return arm_softmax_s16((const int16_t *)blob_ptr(session, input),
                               session->num_rows, session->row_size,
                               session->out_mult, session->out_shift,
                               &hct_softmax_lut_s16,
                               (int16_t *)hct_output_ptr(session));
    }
    if (session->expected_kernel_id == HCT_KERNEL_ID_SOFTMAX_S8_S16)
    {
        uint32_t operand_bytes_needed;
        session->output_length = (uint32_t)(size * (int32_t)sizeof(int16_t));
        if (session->output_length > session->output_capacity_bytes)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        if (!hct_checked_count_bytes(size, sizeof(int8_t), input->byte_length, &operand_bytes_needed))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        arm_softmax_s8_s16((const int8_t *)blob_ptr(session, input),
                           session->num_rows, session->row_size,
                           session->out_mult, session->out_shift, session->diff_min,
                           (int16_t *)hct_output_ptr(session));
        return ARM_CMSIS_NN_SUCCESS;
    }
    session->output_length = (uint32_t)size;
    if (session->output_length > session->output_capacity_bytes)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    {
        uint32_t operand_bytes_needed;
        if (!hct_checked_count_bytes(size, sizeof(int8_t), input->byte_length, &operand_bytes_needed))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
    }
    arm_softmax_s8((const int8_t *)blob_ptr(session, input),
                   session->num_rows, session->row_size,
                   session->out_mult, session->out_shift, session->diff_min,
                   (int8_t *)hct_output_ptr(session));
    return ARM_CMSIS_NN_SUCCESS;
}'''

_RUN_FULLY_CONNECTED_ONCE = '''\
/* FullyConnectedFunctions FullyConnected -- int variants retain the existing S8/S16/S4
 * dispatch behavior and float variants mirror the generated CMSIS-NN harness directly:
 * no quant arrays, no weight-sum precompute, and optional plain float/float16 bias blobs. */
static arm_cmsis_nn_status run_fully_connected_once(hct_server_session_t *session)
{
    hct_server_blob_t *input = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_0);
    hct_server_blob_t *weights = find_blob_by_role(session, HCT_BLOB_ROLE_WEIGHTS);
    hct_server_blob_t *bias = find_blob_by_role(session, HCT_BLOB_ROLE_BIAS); /* optional */
    cmsis_nn_context ctx;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    if (input == NULL || weights == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    input_dims.n = (int32_t)input->dimensions[0];
    input_dims.h = 1;
    input_dims.w = 1;
    input_dims.c = (int32_t)input->dimensions[1];
    filter_dims.c = (int32_t)weights->dimensions[0];
    filter_dims.n = (int32_t)weights->dimensions[1];
    filter_dims.h = 1;
    filter_dims.w = 1;
    bias_dims.n = 0;
    bias_dims.h = 0;
    bias_dims.w = 0;
    bias_dims.c = filter_dims.c;
    output_dims.n = input_dims.n;
    output_dims.h = 1;
    output_dims.w = 1;
    output_dims.c = filter_dims.c;

    if (session->expected_kernel_id == HCT_KERNEL_ID_FULLY_CONNECTED_F32 ||
        session->expected_kernel_id == HCT_KERNEL_ID_FULLY_CONNECTED_F16)
    {
        const bool is_f16 = (session->expected_kernel_id == HCT_KERNEL_ID_FULLY_CONNECTED_F16);
        const uint32_t element_size = is_f16 ? (uint32_t)sizeof(float16_t) : (uint32_t)sizeof(float);
        const float activation_min = quant_scale_from_bits(session->float_activation_min_bits);
        const float activation_max = quant_scale_from_bits(session->float_activation_max_bits);
        int32_t required_scratch;

        if (!hct_checked_dims_bytes(&output_dims, element_size,
                                    session->output_capacity_bytes, &session->output_length))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        {
            uint32_t operand_bytes_needed;
            if (!hct_checked_dims_bytes(&input_dims, element_size, input->byte_length, &operand_bytes_needed) ||
                !hct_checked_dims_bytes(&filter_dims, element_size, weights->byte_length, &operand_bytes_needed) ||
                (bias != NULL && !hct_checked_count_bytes(bias_dims.c, element_size, bias->byte_length, &operand_bytes_needed)))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
        }

        if (is_f16)
        {
            cmsis_nn_fc_params_f16 fc_params_f16;
            fc_params_f16.activation.min = (float16_t)activation_min;
            fc_params_f16.activation.max = (float16_t)activation_max;
            fc_params_f16.weight_format = ARM_NN_WEIGHT_FORMAT_STANDARD;
            required_scratch = arm_fully_connected_f16_get_buffer_size(&fc_params_f16, &input_dims, &filter_dims, &output_dims, ARM_NN_LAYOUT_NHWC);
            if (required_scratch < 0 || (uint32_t)required_scratch > session->scratch_bytes)
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            ctx.buf = (required_scratch > 0) ? &session->workspace[session->scratch_offset] : NULL;
            ctx.size = required_scratch;
            return arm_fully_connected_f16(&ctx,
                                           &fc_params_f16,
                                           &input_dims,
                                           (const float16_t *)blob_ptr(session, input),
                                           &filter_dims,
                                           (const float16_t *)blob_ptr(session, weights),
                                           &bias_dims,
                                           (bias != NULL) ? (const float16_t *)blob_ptr(session, bias) : NULL,
                                           &output_dims,
                                           (float16_t *)hct_output_ptr(session),
                                           ARM_NN_LAYOUT_NHWC);
        }

        {
            cmsis_nn_fc_params_f32 fc_params_f32;
            fc_params_f32.activation.min = activation_min;
            fc_params_f32.activation.max = activation_max;
            fc_params_f32.weight_format = ARM_NN_WEIGHT_FORMAT_STANDARD;
            required_scratch = arm_fully_connected_f32_get_buffer_size(&fc_params_f32, &input_dims, &filter_dims, &output_dims, ARM_NN_LAYOUT_NHWC);
            if (required_scratch < 0 || (uint32_t)required_scratch > session->scratch_bytes)
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            ctx.buf = (required_scratch > 0) ? &session->workspace[session->scratch_offset] : NULL;
            ctx.size = required_scratch;
            return arm_fully_connected_f32(&ctx,
                                           &fc_params_f32,
                                           &input_dims,
                                           (const float *)blob_ptr(session, input),
                                           &filter_dims,
                                           (const float *)blob_ptr(session, weights),
                                           &bias_dims,
                                           (bias != NULL) ? (const float *)blob_ptr(session, bias) : NULL,
                                           &output_dims,
                                           (float *)hct_output_ptr(session),
                                           ARM_NN_LAYOUT_NHWC);
        }
    }

    {
        hct_server_blob_t *multiplier = find_blob_by_role(session, HCT_BLOB_ROLE_MULTIPLIER);
        hct_server_blob_t *shift = find_blob_by_role(session, HCT_BLOB_ROLE_SHIFT);
        cmsis_nn_fc_params fc_params;
        cmsis_nn_quant_params quant_params;
        cmsis_nn_per_tensor_quant_params quant_params_s4;
        uint32_t required_scratch;

        if (multiplier == NULL || shift == NULL)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        {
            uint32_t operand_bytes_needed;
            if (!hct_checked_count_bytes(output_dims.c, sizeof(int32_t), multiplier->byte_length, &operand_bytes_needed) ||
                !hct_checked_count_bytes(output_dims.c, sizeof(int32_t), shift->byte_length, &operand_bytes_needed))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
        }

        fc_params.input_offset = session->input_offset;
        fc_params.filter_offset = session->filter_offset;
        fc_params.output_offset = session->output_offset;
        fc_params.activation.min = session->activation_min;
        fc_params.activation.max = session->activation_max;
        required_scratch = (session->expected_kernel_id == HCT_KERNEL_ID_FULLY_CONNECTED_S4)
            ? 0u
            : (uint32_t)filter_dims.c * (uint32_t)sizeof(int32_t);
        if (required_scratch > session->scratch_bytes)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        ctx.buf = (session->scratch_bytes > 0u) ? &session->workspace[session->scratch_offset] : NULL;
        ctx.size = (int32_t)session->scratch_bytes;

        if (session->expected_kernel_id == HCT_KERNEL_ID_FULLY_CONNECTED_S16)
        {
            quant_params.multiplier = (int32_t *)blob_ptr(session, multiplier);
            quant_params.shift = (int32_t *)blob_ptr(session, shift);
            quant_params.is_per_channel = 1;
            const int64_t *bias_i64 = (bias != NULL) ? (const int64_t *)blob_ptr(session, bias) : NULL;
            uint32_t operand_bytes_needed;

            if (!hct_checked_dims_bytes(&input_dims, sizeof(int16_t), input->byte_length, &operand_bytes_needed) ||
                !hct_checked_dims_bytes(&filter_dims, sizeof(int8_t), weights->byte_length, &operand_bytes_needed) ||
                (bias != NULL && !hct_checked_count_bytes(bias_dims.c, sizeof(int64_t), bias->byte_length, &operand_bytes_needed)))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }

            if (!hct_checked_dims_bytes(&output_dims, sizeof(int16_t),
                                        session->output_capacity_bytes, &session->output_length))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            return arm_fully_connected_wrapper_s16(&ctx,
                                                   &fc_params,
                                                   &quant_params,
                                                   &input_dims,
                                                   (const int16_t *)blob_ptr(session, input),
                                                   &filter_dims,
                                                   (const int8_t *)blob_ptr(session, weights),
                                                   &bias_dims,
                                                   bias_i64,
                                                   &output_dims,
                                                   (int16_t *)hct_output_ptr(session));
        }

        if (session->expected_kernel_id == HCT_KERNEL_ID_FULLY_CONNECTED_S4)
        {
            const int32_t *bias_i32 = (bias != NULL) ? (const int32_t *)blob_ptr(session, bias) : NULL;
            uint32_t operand_bytes_needed;
            const int32_t filter_shape[4] = {filter_dims.n, filter_dims.h, filter_dims.w, filter_dims.c};
            quant_params_s4.multiplier = *(const int32_t *)blob_ptr(session, multiplier);
            quant_params_s4.shift = *(const int32_t *)blob_ptr(session, shift);

            if (!hct_checked_dims_bytes(&input_dims, sizeof(int8_t), input->byte_length, &operand_bytes_needed) ||
                !hct_checked_packed4_bytes(filter_shape, 4, weights->byte_length, &operand_bytes_needed) ||
                (bias != NULL && !hct_checked_count_bytes(bias_dims.c, sizeof(int32_t), bias->byte_length, &operand_bytes_needed)))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }

            if (!hct_checked_dims_bytes(&output_dims, sizeof(int8_t),
                                        session->output_capacity_bytes, &session->output_length))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            return arm_fully_connected_s4(&ctx,
                                          &fc_params,
                                          &quant_params_s4,
                                          &input_dims,
                                          (const int8_t *)blob_ptr(session, input),
                                          &filter_dims,
                                          (const int8_t *)blob_ptr(session, weights),
                                          &bias_dims,
                                          bias_i32,
                                          &output_dims,
                                          (int8_t *)hct_output_ptr(session));
        }

        {
            const int32_t *bias_i32 = (bias != NULL) ? (const int32_t *)blob_ptr(session, bias) : NULL;
            uint32_t operand_bytes_needed;
            quant_params.multiplier = (int32_t *)blob_ptr(session, multiplier);
            quant_params.shift = (int32_t *)blob_ptr(session, shift);
            quant_params.is_per_channel = 1;

            if (!hct_checked_dims_bytes(&input_dims, sizeof(int8_t), input->byte_length, &operand_bytes_needed) ||
                !hct_checked_dims_bytes(&filter_dims, sizeof(int8_t), weights->byte_length, &operand_bytes_needed) ||
                (bias != NULL && !hct_checked_count_bytes(bias_dims.c, sizeof(int32_t), bias->byte_length, &operand_bytes_needed)))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }

            if (arm_vector_sum_s8((int32_t *)ctx.buf,
                                  filter_dims.n,
                                  filter_dims.c,
                                  (const int8_t *)blob_ptr(session, weights),
                                  session->input_offset,
                                  session->filter_offset,
                                  bias_i32) != ARM_CMSIS_NN_SUCCESS)
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }

            if (!hct_checked_dims_bytes(&output_dims, sizeof(int8_t),
                                        session->output_capacity_bytes, &session->output_length))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            return arm_fully_connected_wrapper_s8(&ctx,
                                                  &fc_params,
                                                  &quant_params,
                                                  &input_dims,
                                                  (const int8_t *)blob_ptr(session, input),
                                                  &filter_dims,
                                                  (const int8_t *)blob_ptr(session, weights),
                                                  &bias_dims,
                                                  NULL,
                                                  &output_dims,
                                                  (int8_t *)hct_output_ptr(session));
        }
    }
}'''


_RUN_BATCH_MATMUL_ONCE = '''\
static arm_cmsis_nn_status run_batch_matmul_once(hct_server_session_t *session)
{
    hct_server_blob_t *input_lhs = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_0);
    hct_server_blob_t *input_rhs = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_1);
    cmsis_nn_context ctx;
    cmsis_nn_dims input_lhs_dims;
    cmsis_nn_dims input_rhs_dims;
    cmsis_nn_dims output_dims;

    if (input_lhs == NULL || input_rhs == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    input_lhs_dims.n = (int32_t)((input_lhs->rank > 0u) ? input_lhs->dimensions[0] : 1u);
    input_lhs_dims.h = (int32_t)((input_lhs->rank > 1u) ? input_lhs->dimensions[1] : 1u);
    input_lhs_dims.w = (int32_t)((input_lhs->rank > 2u) ? input_lhs->dimensions[2] : input_lhs->dimensions[0]);
    input_lhs_dims.c = (int32_t)((input_lhs->rank > 3u) ? input_lhs->dimensions[3] : input_lhs->dimensions[1]);
    input_rhs_dims.n = (int32_t)((input_rhs->rank > 0u) ? input_rhs->dimensions[0] : 1u);
    input_rhs_dims.h = (int32_t)((input_rhs->rank > 1u) ? input_rhs->dimensions[1] : 1u);
    input_rhs_dims.w = (int32_t)((input_rhs->rank > 2u) ? input_rhs->dimensions[2] : input_rhs->dimensions[0]);
    input_rhs_dims.c = (int32_t)((input_rhs->rank > 3u) ? input_rhs->dimensions[3] : input_rhs->dimensions[1]);
    output_dims.n = (session->output_n > 0) ? session->output_n : 1;
    output_dims.h = (session->output_h > 0) ? session->output_h : 1;
    output_dims.w = session->output_w;
    output_dims.c = session->output_c;
    if (output_dims.w <= 0 || output_dims.c <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    if (session->expected_kernel_id == HCT_KERNEL_ID_BATCH_MATMUL_F32 ||
        session->expected_kernel_id == HCT_KERNEL_ID_BATCH_MATMUL_F16)
    {
        const bool is_f16 = (session->expected_kernel_id == HCT_KERNEL_ID_BATCH_MATMUL_F16);
        const uint32_t element_size = is_f16 ? (uint32_t)sizeof(float16_t) : (uint32_t)sizeof(float);
        const bool adj_x = (session->adj_x != 0);
        const bool adj_y = (session->adj_y != 0);
        const float activation_min = quant_scale_from_bits(session->float_activation_min_bits);
        const float activation_max = quant_scale_from_bits(session->float_activation_max_bits);
        int32_t required_scratch;

        if (!hct_checked_dims_bytes(&output_dims, element_size,
                                    session->output_capacity_bytes, &session->output_length))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        {
            uint32_t operand_bytes_needed;
            if (!hct_checked_dims_bytes(&input_lhs_dims, element_size, input_lhs->byte_length, &operand_bytes_needed) ||
                !hct_checked_dims_bytes(&input_rhs_dims, element_size, input_rhs->byte_length, &operand_bytes_needed))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
        }

        if (is_f16)
        {
            const cmsis_nn_bmm_params_f16 bmm_params = {
                .adj_x = adj_x,
                .adj_y = adj_y,
                .activation = { .min = (float16_t)activation_min, .max = (float16_t)activation_max },
                .rhs_format = ARM_NN_WEIGHT_FORMAT_STANDARD,
            };
            required_scratch = arm_batch_matmul_f16_get_buffer_size(&bmm_params, &input_lhs_dims, &input_rhs_dims, &output_dims);
            if (required_scratch < 0 || (uint32_t)required_scratch > session->scratch_bytes)
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            ctx.buf = (required_scratch > 0) ? &session->workspace[session->scratch_offset] : NULL;
            ctx.size = required_scratch;
            return arm_batch_matmul_f16(&ctx,
                                        &bmm_params,
                                        &input_lhs_dims,
                                        (const float16_t *)blob_ptr(session, input_lhs),
                                        &input_rhs_dims,
                                        (const float16_t *)blob_ptr(session, input_rhs),
                                        &output_dims,
                                        (float16_t *)hct_output_ptr(session));
        }
        else
        {
            const cmsis_nn_bmm_params_f32 bmm_params = {
                .adj_x = adj_x,
                .adj_y = adj_y,
                .activation = { .min = activation_min, .max = activation_max },
                .rhs_format = ARM_NN_WEIGHT_FORMAT_STANDARD,
            };
            required_scratch = arm_batch_matmul_f32_get_buffer_size(&bmm_params, &input_lhs_dims, &input_rhs_dims, &output_dims);
            if (required_scratch < 0 || (uint32_t)required_scratch > session->scratch_bytes)
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            ctx.buf = (required_scratch > 0) ? &session->workspace[session->scratch_offset] : NULL;
            ctx.size = required_scratch;
            return arm_batch_matmul_f32(&ctx,
                                        &bmm_params,
                                        &input_lhs_dims,
                                        (const float *)blob_ptr(session, input_lhs),
                                        &input_rhs_dims,
                                        (const float *)blob_ptr(session, input_rhs),
                                        &output_dims,
                                        (float *)hct_output_ptr(session));
        }
    }

    {
        cmsis_nn_per_tensor_quant_params quant_params;
        /* cmsis_nn_bmm_params's adj_x/adj_y are `const bool` fields, which makes the whole
         * struct non-assignable after declaration in C -- must be fully initialized here via
         * a designated initializer instead of member-by-member assignment. */
        const cmsis_nn_bmm_params bmm_params = {
            .adj_x = false,
            .adj_y = false,
            .fc_params = {
                .input_offset = session->input_offset,
                .filter_offset = session->filter_offset,
                .output_offset = session->output_offset,
                .activation = { .min = session->activation_min, .max = session->activation_max }
            }
        };
        quant_params.multiplier = session->out_mult;
        quant_params.shift = session->out_shift;

        if (session->expected_kernel_id == HCT_KERNEL_ID_BATCH_MATMUL_S16)
        {
            const int32_t output_shape[2] = {output_dims.w, output_dims.c};
            uint32_t operand_bytes_needed;
            if (!hct_checked_shape_bytes(output_shape, 2, sizeof(int16_t),
                                         session->output_capacity_bytes, &session->output_length))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            if (!hct_checked_dims_bytes(&input_lhs_dims, sizeof(int16_t), input_lhs->byte_length, &operand_bytes_needed) ||
                !hct_checked_dims_bytes(&input_rhs_dims, sizeof(int16_t), input_rhs->byte_length, &operand_bytes_needed))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            ctx.buf = NULL;
            ctx.size = 0;
            return arm_batch_matmul_s16(&ctx,
                                        &bmm_params,
                                        &quant_params,
                                        &input_lhs_dims,
                                        (const int16_t *)blob_ptr(session, input_lhs),
                                        &input_rhs_dims,
                                        (const int16_t *)blob_ptr(session, input_rhs),
                                        &output_dims,
                                        (int16_t *)hct_output_ptr(session));
        }

        {
            const int32_t scratch_shape[1] = {input_rhs_dims.w};
            const int32_t output_shape[2] = {output_dims.w, output_dims.c};
            uint32_t required_scratch;
            if (!hct_checked_shape_bytes(scratch_shape, 1, sizeof(int32_t),
                                         session->scratch_bytes, &required_scratch))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            ctx.buf = (session->scratch_bytes > 0u) ? &session->workspace[session->scratch_offset] : NULL;
            ctx.size = (int32_t)session->scratch_bytes;

            if (!hct_checked_shape_bytes(output_shape, 2, sizeof(int8_t),
                                         session->output_capacity_bytes, &session->output_length))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            {
                uint32_t operand_bytes_needed;
                if (!hct_checked_dims_bytes(&input_lhs_dims, sizeof(int8_t), input_lhs->byte_length, &operand_bytes_needed) ||
                    !hct_checked_dims_bytes(&input_rhs_dims, sizeof(int8_t), input_rhs->byte_length, &operand_bytes_needed))
                {
                    return ARM_CMSIS_NN_ARG_ERROR;
                }
            }
            return arm_batch_matmul_s8(&ctx,
                                       &bmm_params,
                                       &quant_params,
                                       &input_lhs_dims,
                                       (const int8_t *)blob_ptr(session, input_lhs),
                                       &input_rhs_dims,
                                       (const int8_t *)blob_ptr(session, input_rhs),
                                       &output_dims,
                                       (int8_t *)hct_output_ptr(session));
        }
    }
}
'''

_RUN_BASIC_MATH_REDUCTION_ONCE = '''\
static arm_cmsis_nn_status run_basic_math_reduction_once(hct_server_session_t *session)
{
    hct_server_blob_t *input = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_0);
    cmsis_nn_dims input_dims;
    cmsis_nn_dims axis_dims;
    cmsis_nn_dims output_dims;
    uint32_t output_elements;

    if (input == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    input_dims.n = (int32_t)input->dimensions[0];
    input_dims.h = (int32_t)input->dimensions[1];
    input_dims.w = (int32_t)input->dimensions[2];
    input_dims.c = (int32_t)input->dimensions[3];

    axis_dims.n = session->axis_n;
    axis_dims.h = session->axis_h;
    axis_dims.w = session->axis_w;
    axis_dims.c = session->axis_c;

    output_dims.n = (session->output_n > 0) ? session->output_n : 1;
    output_dims.h = session->output_h;
    output_dims.w = session->output_w;
    output_dims.c = session->output_c;
    if (output_dims.n <= 0 || output_dims.h <= 0 || output_dims.w <= 0 || output_dims.c <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    if (!hct_checked_dims_bytes(&output_dims, 1u, UINT32_MAX, &output_elements))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    {
        const bool reduction_is_s16 = (session->expected_kernel_id == HCT_KERNEL_ID_ARGMAX_S16 ||
                                       session->expected_kernel_id == HCT_KERNEL_ID_ARGMIN_S16 ||
                                       session->expected_kernel_id == HCT_KERNEL_ID_MEAN_S16 ||
                                       session->expected_kernel_id == HCT_KERNEL_ID_REDUCE_MAX_S16 ||
                                       session->expected_kernel_id == HCT_KERNEL_ID_REDUCE_MIN_S16);
        uint32_t operand_bytes_needed;
        if (!hct_checked_dims_bytes(&input_dims, reduction_is_s16 ? sizeof(int16_t) : sizeof(int8_t),
                                    input->byte_length, &operand_bytes_needed))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
    }

    switch (session->expected_kernel_id)
    {
        case HCT_KERNEL_ID_ARGMAX_S8:
        case HCT_KERNEL_ID_ARGMIN_S8:
        case HCT_KERNEL_ID_ARGMAX_S16:
        case HCT_KERNEL_ID_ARGMIN_S16:
            if (!hct_checked_dims_bytes(&output_dims, sizeof(int32_t),
                                        session->output_capacity_bytes, &session->output_length) ||
                session->axis < 0 || session->axis > 3)
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_ARGMAX_S8)
            {
                return arm_argmax_s8((const int8_t *)blob_ptr(session, input), &input_dims, session->axis, (int32_t *)hct_output_ptr(session));
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_ARGMIN_S8)
            {
                return arm_argmin_s8((const int8_t *)blob_ptr(session, input), &input_dims, session->axis, (int32_t *)hct_output_ptr(session));
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_ARGMAX_S16)
            {
                return arm_argmax_s16((const int16_t *)blob_ptr(session, input), &input_dims, session->axis, (int32_t *)hct_output_ptr(session));
            }
            return arm_argmin_s16((const int16_t *)blob_ptr(session, input), &input_dims, session->axis, (int32_t *)hct_output_ptr(session));

        case HCT_KERNEL_ID_MEAN_S8:
        case HCT_KERNEL_ID_REDUCE_MAX_S8:
        case HCT_KERNEL_ID_REDUCE_MIN_S8:
            session->output_length = output_elements;
            if (session->output_length > session->output_capacity_bytes)
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_MEAN_S8)
            {
                return arm_mean_s8((const int8_t *)blob_ptr(session, input), &input_dims, session->input_offset,
                                   &axis_dims, (int8_t *)hct_output_ptr(session), &output_dims,
                                   session->output_offset, session->out_mult, session->out_shift);
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_REDUCE_MAX_S8)
            {
                return arm_reduce_max_s8((const int8_t *)blob_ptr(session, input), &input_dims, &axis_dims,
                                         (int8_t *)hct_output_ptr(session), &output_dims);
            }
            return arm_reduce_min_s8((const int8_t *)blob_ptr(session, input), &input_dims, &axis_dims,
                                     (int8_t *)hct_output_ptr(session), &output_dims);

        case HCT_KERNEL_ID_MEAN_S16:
        case HCT_KERNEL_ID_REDUCE_MAX_S16:
        case HCT_KERNEL_ID_REDUCE_MIN_S16:
            if (!hct_checked_dims_bytes(&output_dims, sizeof(int16_t),
                                        session->output_capacity_bytes, &session->output_length))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_MEAN_S16)
            {
                return arm_mean_s16((const int16_t *)blob_ptr(session, input), &input_dims, session->input_offset,
                                    &axis_dims, (int16_t *)hct_output_ptr(session), &output_dims,
                                    session->output_offset, session->out_mult, session->out_shift);
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_REDUCE_MAX_S16)
            {
                return arm_reduce_max_s16((const int16_t *)blob_ptr(session, input), &input_dims, &axis_dims,
                                          (int16_t *)hct_output_ptr(session), &output_dims);
            }
            return arm_reduce_min_s16((const int16_t *)blob_ptr(session, input), &input_dims, &axis_dims,
                                      (int16_t *)hct_output_ptr(session), &output_dims);

        default:
            return ARM_CMSIS_NN_ARG_ERROR;
    }
}'''

_RUN_BASIC_MATH_LUT_ONCE = '''\
static arm_cmsis_nn_status run_basic_math_lut_once(hct_server_session_t *session)
{
    hct_server_blob_t *input = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_0);
    hct_server_blob_t *lut = find_blob_by_role(session, HCT_BLOB_ROLE_WEIGHTS);
    cmsis_nn_dims input_dims;
    int32_t block_size;

    if (input == NULL || lut == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    input_dims.n = (int32_t)input->dimensions[0];
    input_dims.h = (int32_t)input->dimensions[1];
    input_dims.w = (int32_t)input->dimensions[2];
    input_dims.c = (int32_t)input->dimensions[3];
    block_size = input_dims.n * input_dims.h * input_dims.w * input_dims.c;
    session->output_length = input->byte_length;
    if (session->output_length > session->output_capacity_bytes)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    {
        const bool lut_is_s16 = (session->expected_kernel_id != HCT_KERNEL_ID_SQRT_S8);
        uint32_t operand_bytes_needed;
        if (!hct_checked_count_bytes(block_size, lut_is_s16 ? sizeof(int16_t) : sizeof(int8_t),
                                     input->byte_length, &operand_bytes_needed))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
    }

    switch (session->expected_kernel_id)
    {
        case HCT_KERNEL_ID_SQRT_S8:
            return arm_sqrt_s8((const int8_t *)blob_ptr(session, input), &input_dims,
                               (int8_t *)hct_output_ptr(session), (int8_t *)blob_ptr(session, lut));
        case HCT_KERNEL_ID_SQRT_S16:
            return arm_sqrt_s16((const int16_t *)blob_ptr(session, input), &input_dims,
                                (int16_t *)hct_output_ptr(session), (const int16_t *)blob_ptr(session, lut));
        case HCT_KERNEL_ID_RSQRT_S16_PER_OP:
            return arm_rsqrt_s16_per_op((const int16_t *)blob_ptr(session, input), session->input_offset,
                                        (int16_t *)hct_output_ptr(session), session->output_offset,
                                        session->activation_min, session->activation_max, block_size,
                                        (const int16_t *)blob_ptr(session, lut));
        case HCT_KERNEL_ID_RSQRT_S16_UNIVERSAL:
            return arm_rsqrt_s16_universal((const int16_t *)blob_ptr(session, input), session->input_offset,
                                           (int16_t *)hct_output_ptr(session), session->output_offset,
                                           session->out_mult, session->out_shift, session->needs_rescale != 0,
                                           session->activation_min, session->activation_max, block_size,
                                           (const int32_t *)blob_ptr(session, lut));
        default:
            return ARM_CMSIS_NN_ARG_ERROR;
    }
}'''

_RUN_ELEMENTWISE_BINARY_ONCE = '''\
/* Int Add/Sub/Mul/Maximum/Minimum/SquaredDifference and float Add/Sub/Mul/Maximum/Minimum
 * share the same tensor-plumbing wrapper: blob lookup, dims extraction, explicit host-sent
 * output dims, then branch on kernel_id for the exact CMSIS-NN entrypoint/signature. */
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

    output_dims.n = (session->output_n > 0) ? session->output_n : 1;
    output_dims.h = session->output_h;
    output_dims.w = session->output_w;
    output_dims.c = session->output_c;
    if (output_dims.h <= 0 || output_dims.w <= 0 || output_dims.c <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    if (!hct_checked_dims_bytes(&output_dims, 1u, session->output_capacity_bytes, &session->output_length))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    switch (session->expected_kernel_id)
    {
        case HCT_KERNEL_ID_ADD_S8:
        case HCT_KERNEL_ID_SUB_S8:
        case HCT_KERNEL_ID_MUL_S8:
        case HCT_KERNEL_ID_MAXIMUM_S8:
        case HCT_KERNEL_ID_MINIMUM_S8:
        case HCT_KERNEL_ID_SQUARED_DIFFERENCE_S8:
        {
            uint32_t operand_bytes_needed;
            if (session->output_length > session->output_capacity_bytes)
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            if (!hct_checked_dims_bytes(&input1_dims, sizeof(int8_t), input1->byte_length, &operand_bytes_needed) ||
                !hct_checked_dims_bytes(&input2_dims, sizeof(int8_t), input2->byte_length, &operand_bytes_needed))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            const int8_t *input1_data = (const int8_t *)blob_ptr(session, input1);
            const int8_t *input2_data = (const int8_t *)blob_ptr(session, input2);
            int8_t *output_data = (int8_t *)hct_output_ptr(session);
            if (session->expected_kernel_id == HCT_KERNEL_ID_ADD_S8)
            {
                return arm_add_s8(input1_data, &input1_dims, input2_data, &input2_dims,
                                  session->input1_offset, session->input1_mult, session->input1_shift,
                                  session->input2_offset, session->input2_mult, session->input2_shift,
                                  session->left_shift,
                                  output_data, &output_dims,
                                  session->output_offset, session->out_mult, session->out_shift,
                                  session->activation_min, session->activation_max);
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_SUB_S8)
            {
                return arm_sub_s8(input1_data, &input1_dims, input2_data, &input2_dims,
                                  session->input1_offset, session->input1_mult, session->input1_shift,
                                  session->input2_offset, session->input2_mult, session->input2_shift,
                                  session->left_shift,
                                  output_data, &output_dims,
                                  session->output_offset, session->out_mult, session->out_shift,
                                  session->activation_min, session->activation_max);
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_MUL_S8)
            {
                /* arm_mul_s8 has no per-input mult/shift or left_shift -- it reuses only the
                 * input1_offset/input2_offset scalar fields (shared with Add/Sub above). */
                return arm_mul_s8(input1_data, &input1_dims, input2_data, &input2_dims,
                                  session->input1_offset, session->input2_offset,
                                  output_data, &output_dims,
                                  session->output_offset, session->out_mult, session->out_shift,
                                  session->activation_min, session->activation_max);
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_SQUARED_DIFFERENCE_S8)
            {
                return arm_squared_difference_s8(input1_data, &input1_dims, input2_data, &input2_dims,
                                                 session->input1_offset, session->input1_mult, session->input1_shift,
                                                 session->input2_offset, session->input2_mult, session->input2_shift,
                                                 session->left_shift,
                                                 output_data, &output_dims,
                                                 session->output_offset, session->out_mult, session->out_shift,
                                                 session->activation_min, session->activation_max);
            }
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
        case HCT_KERNEL_ID_ADD_S16:
        case HCT_KERNEL_ID_SUB_S16:
        case HCT_KERNEL_ID_MUL_S16:
        case HCT_KERNEL_ID_MAXIMUM_S16:
        case HCT_KERNEL_ID_MINIMUM_S16:
        case HCT_KERNEL_ID_SQUARED_DIFFERENCE_S16:
        {
            if (!hct_checked_dims_bytes(&output_dims, sizeof(int16_t),
                                        session->output_capacity_bytes, &session->output_length))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            /* session->output_length is transmitted to the host as a raw byte count (see
             * the RTT send loop in benchmark_server_session.c), so it must be rescaled from
             * elements to bytes for 2-byte-per-element S16 output -- otherwise only the
             * first half of the output buffer is sent back over the wire. */
            uint32_t operand_bytes_needed;
            if (!hct_checked_dims_bytes(&input1_dims, sizeof(int16_t), input1->byte_length, &operand_bytes_needed) ||
                !hct_checked_dims_bytes(&input2_dims, sizeof(int16_t), input2->byte_length, &operand_bytes_needed))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            const int16_t *input1_data = (const int16_t *)blob_ptr(session, input1);
            const int16_t *input2_data = (const int16_t *)blob_ptr(session, input2);
            int16_t *output_data = (int16_t *)hct_output_ptr(session);
            if (session->expected_kernel_id == HCT_KERNEL_ID_ADD_S16)
            {
                return arm_add_s16(input1_data, &input1_dims, input2_data, &input2_dims,
                                   session->input1_offset, session->input1_mult, session->input1_shift,
                                   session->input2_offset, session->input2_mult, session->input2_shift,
                                   session->left_shift,
                                   output_data, &output_dims,
                                   session->output_offset, session->out_mult, session->out_shift,
                                   session->activation_min, session->activation_max);
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_SUB_S16)
            {
                return arm_sub_s16(input1_data, &input1_dims, input2_data, &input2_dims,
                                   session->input1_offset, session->input1_mult, session->input1_shift,
                                   session->input2_offset, session->input2_mult, session->input2_shift,
                                   session->left_shift,
                                   output_data, &output_dims,
                                   session->output_offset, session->out_mult, session->out_shift,
                                   session->activation_min, session->activation_max);
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_MUL_S16)
            {
                /* arm_mul_s16 mirrors arm_mul_s8's (shorter) signature: no per-input mult/shift. */
                return arm_mul_s16(input1_data, &input1_dims, input2_data, &input2_dims,
                                   session->input1_offset, session->input2_offset,
                                   output_data, &output_dims,
                                   session->output_offset, session->out_mult, session->out_shift,
                                   session->activation_min, session->activation_max);
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_SQUARED_DIFFERENCE_S16)
            {
                return arm_squared_difference_s16(input1_data, &input1_dims, input2_data, &input2_dims,
                                                  session->input1_offset, session->input1_mult, session->input1_shift,
                                                  session->input2_offset, session->input2_mult, session->input2_shift,
                                                  session->left_shift,
                                                  output_data, &output_dims,
                                                  session->output_offset, session->out_mult, session->out_shift,
                                                  session->activation_min, session->activation_max);
            }
            /* Maximum/Minimum S16 mirror the S8 variants: scratch ctx only, no quant scalars. */
            cmsis_nn_context ctx16 = {NULL, 0};
            if (session->expected_kernel_id == HCT_KERNEL_ID_MAXIMUM_S16)
            {
                return arm_maximum_s16(&ctx16, input1_data, &input1_dims, input2_data, &input2_dims,
                                       output_data, &output_dims);
            }
            return arm_minimum_s16(&ctx16, input1_data, &input1_dims, input2_data, &input2_dims,
                                   output_data, &output_dims);
        }
        case HCT_KERNEL_ID_ADD_F32:
        case HCT_KERNEL_ID_SUB_F32:
        case HCT_KERNEL_ID_MUL_F32:
        case HCT_KERNEL_ID_MAXIMUM_F32:
        case HCT_KERNEL_ID_MINIMUM_F32:
        {
            if (!hct_checked_dims_bytes(&output_dims, sizeof(float),
                                        session->output_capacity_bytes, &session->output_length))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            {
                /* Add/Sub/Mul read `session->block_size` flat elements (no broadcasting,
                 * no dims param); Maximum/Minimum read via input1_dims/input2_dims instead
                 * (see the dims-based check below) -- validate whichever this op actually
                 * uses to size its read. */
                const bool uses_block_size_path = (session->expected_kernel_id == HCT_KERNEL_ID_ADD_F32 ||
                                                   session->expected_kernel_id == HCT_KERNEL_ID_SUB_F32 ||
                                                   session->expected_kernel_id == HCT_KERNEL_ID_MUL_F32);
                uint32_t operand_bytes_needed;
                if (uses_block_size_path &&
                    (!hct_checked_count_bytes(session->block_size, sizeof(float), input1->byte_length, &operand_bytes_needed) ||
                     !hct_checked_count_bytes(session->block_size, sizeof(float), input2->byte_length, &operand_bytes_needed)))
                {
                    return ARM_CMSIS_NN_ARG_ERROR;
                }
                if (!uses_block_size_path &&
                    (!hct_checked_dims_bytes(&input1_dims, sizeof(float), input1->byte_length, &operand_bytes_needed) ||
                     !hct_checked_dims_bytes(&input2_dims, sizeof(float), input2->byte_length, &operand_bytes_needed)))
                {
                    return ARM_CMSIS_NN_ARG_ERROR;
                }
            }
            const float *input1_data = (const float *)blob_ptr(session, input1);
            const float *input2_data = (const float *)blob_ptr(session, input2);
            float *output_data = (float *)hct_output_ptr(session);
            const float activation_min = quant_scale_from_bits(session->float_activation_min_bits);
            const float activation_max = quant_scale_from_bits(session->float_activation_max_bits);
            if (session->expected_kernel_id == HCT_KERNEL_ID_ADD_F32)
            {
                return arm_elementwise_add_f32(input1_data, input2_data, output_data,
                                               activation_min, activation_max, session->block_size);
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_SUB_F32)
            {
                return arm_elementwise_sub_f32(input1_data, input2_data, output_data,
                                               activation_min, activation_max, session->block_size);
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_MUL_F32)
            {
                return arm_elementwise_mul_f32(input1_data, input2_data, output_data,
                                               activation_min, activation_max, session->block_size);
            }
            cmsis_nn_context ctxf32 = {NULL, 0};
            if (session->expected_kernel_id == HCT_KERNEL_ID_MAXIMUM_F32)
            {
                return arm_maximum_f32(&ctxf32, input1_data, &input1_dims, input2_data, &input2_dims,
                                       output_data, &output_dims);
            }
            return arm_minimum_f32(&ctxf32, input1_data, &input1_dims, input2_data, &input2_dims,
                                   output_data, &output_dims);
        }
        case HCT_KERNEL_ID_ADD_F16:
        case HCT_KERNEL_ID_SUB_F16:
        case HCT_KERNEL_ID_MUL_F16:
        case HCT_KERNEL_ID_MAXIMUM_F16:
        case HCT_KERNEL_ID_MINIMUM_F16:
        {
            if (!hct_checked_dims_bytes(&output_dims, sizeof(float16_t),
                                        session->output_capacity_bytes, &session->output_length))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            {
                const bool uses_block_size_path = (session->expected_kernel_id == HCT_KERNEL_ID_ADD_F16 ||
                                                   session->expected_kernel_id == HCT_KERNEL_ID_SUB_F16 ||
                                                   session->expected_kernel_id == HCT_KERNEL_ID_MUL_F16);
                uint32_t operand_bytes_needed;
                if (uses_block_size_path &&
                    (!hct_checked_count_bytes(session->block_size, sizeof(float16_t), input1->byte_length, &operand_bytes_needed) ||
                     !hct_checked_count_bytes(session->block_size, sizeof(float16_t), input2->byte_length, &operand_bytes_needed)))
                {
                    return ARM_CMSIS_NN_ARG_ERROR;
                }
                if (!uses_block_size_path &&
                    (!hct_checked_dims_bytes(&input1_dims, sizeof(float16_t), input1->byte_length, &operand_bytes_needed) ||
                     !hct_checked_dims_bytes(&input2_dims, sizeof(float16_t), input2->byte_length, &operand_bytes_needed)))
                {
                    return ARM_CMSIS_NN_ARG_ERROR;
                }
            }
            const float16_t *input1_data = (const float16_t *)blob_ptr(session, input1);
            const float16_t *input2_data = (const float16_t *)blob_ptr(session, input2);
            float16_t *output_data = (float16_t *)hct_output_ptr(session);
            const float activation_min = quant_scale_from_bits(session->float_activation_min_bits);
            const float activation_max = quant_scale_from_bits(session->float_activation_max_bits);
            if (session->expected_kernel_id == HCT_KERNEL_ID_ADD_F16)
            {
                return arm_elementwise_add_f16(input1_data, input2_data, output_data,
                                               activation_min, activation_max, session->block_size);
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_SUB_F16)
            {
                return arm_elementwise_sub_f16(input1_data, input2_data, output_data,
                                               activation_min, activation_max, session->block_size);
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_MUL_F16)
            {
                return arm_elementwise_mul_f16(input1_data, input2_data, output_data,
                                               activation_min, activation_max, session->block_size);
            }
            cmsis_nn_context ctxf16 = {NULL, 0};
            if (session->expected_kernel_id == HCT_KERNEL_ID_MAXIMUM_F16)
            {
                return arm_maximum_f16(&ctxf16, input1_data, &input1_dims, input2_data, &input2_dims,
                                       output_data, &output_dims);
            }
            return arm_minimum_f16(&ctxf16, input1_data, &input1_dims, input2_data, &input2_dims,
                                   output_data, &output_dims);
        }
        default:
            return ARM_CMSIS_NN_ARG_ERROR;
    }
}'''


_RUN_DATA_MOVEMENT_ONCE = '''\
static uint32_t hct_dtype_size_bytes(uint8_t dtype)
{
    switch (dtype)
    {
        case HCT_DTYPE_S8: return sizeof(int8_t);
        case HCT_DTYPE_S16: return sizeof(int16_t);
        case HCT_DTYPE_S32: return sizeof(int32_t);
        case HCT_DTYPE_S64: return sizeof(int64_t);
        case HCT_DTYPE_BOOL: return sizeof(bool);
        case HCT_DTYPE_F32: return sizeof(float);
        case HCT_DTYPE_F16: return sizeof(float16_t);
        default: return 0u;
    }
}

static uint32_t hct_blob_element_count(const hct_server_blob_t *blob)
{
    const uint32_t element_size = hct_dtype_size_bytes(blob->dtype);
    return (element_size == 0u) ? 0u : (blob->byte_length / element_size);
}

static void hct_fill_shape_from_blob(const hct_server_blob_t *blob, int32_t rank, int32_t *shape, bool right_align)
{
    int32_t offset = 0;
    int32_t i;
    for (i = 0; i < rank && i < 4; ++i)
    {
        shape[i] = 1;
    }
    if (rank <= 0)
    {
        return;
    }
    if (right_align && (int32_t)blob->rank < rank)
    {
        offset = rank - (int32_t)blob->rank;
    }
    for (i = 0; i < (int32_t)blob->rank && i < rank && (offset + i) < 4; ++i)
    {
        shape[offset + i] = (int32_t)blob->dimensions[i];
    }
}

static void hct_fill_dims_from_blob(const hct_server_blob_t *blob, cmsis_nn_dims *dims)
{
    int32_t shape[4] = {1, 1, 1, 1};
    hct_fill_shape_from_blob(blob, (int32_t)((blob->rank == 0u) ? 4u : blob->rank), shape, false);
    dims->n = shape[0];
    dims->h = shape[1];
    dims->w = shape[2];
    dims->c = shape[3];
}

static void hct_fill_output_shape_from_session(const hct_server_session_t *session, int32_t rank, int32_t *shape)
{
    const int32_t base[4] = {
        (session->output_n > 0) ? session->output_n : 1,
        (session->output_h > 0) ? session->output_h : 1,
        (session->output_w > 0) ? session->output_w : 1,
        (session->output_c > 0) ? session->output_c : 1,
    };
    int32_t i;
    for (i = 0; i < rank && i < 4; ++i)
    {
        shape[i] = base[i];
    }
}

static void hct_fill_output_dims_from_session(const hct_server_session_t *session, cmsis_nn_dims *dims)
{
    dims->n = (session->output_n > 0) ? session->output_n : 1;
    dims->h = (session->output_h > 0) ? session->output_h : 1;
    dims->w = (session->output_w > 0) ? session->output_w : 1;
    dims->c = (session->output_c > 0) ? session->output_c : 1;
}

static bool hct_checked_dims_bytes(const cmsis_nn_dims *dims,
                                   uint32_t element_size,
                                   uint32_t capacity,
                                   uint32_t *output_bytes)
{
    const int32_t shape[4] = {dims->n, dims->h, dims->w, dims->c};
    return hct_checked_shape_bytes(shape, 4, element_size, capacity, output_bytes);
}

/* F007: validates the META_0 blob's rank/axis/byte-count *before* any data-movement
 * kernel is dispatched, since the dispatcher below casts the blob straight to
 * `const int32_t *` and indexes it at kernel-specific fixed offsets with no proof the
 * byte length actually covers them. `required_ints` is the number of leading int32_t
 * elements the selected kernel's case in `run_data_movement_once()` reads; `rank`/`axis`
 * are the kernel's declared tensor rank and (if applicable) concatenation/split axis --
 * pass -1 for either when the kernel doesn't use that concept so it's skipped. */
static bool hct_validate_meta0_blob(const hct_server_blob_t *meta_blob,
                                    size_t required_ints,
                                    int32_t rank,
                                    int32_t axis)
{
    if (meta_blob == NULL || meta_blob->dtype != HCT_DTYPE_S32 ||
        meta_blob->alignment < sizeof(int32_t) ||
        (meta_blob->byte_length % sizeof(int32_t)) != 0u)
    {
        return false;
    }
    if ((meta_blob->arena_offset % sizeof(int32_t)) != 0u)
    {
        /* 4-byte alignment requirement: the dispatcher reinterprets this blob's raw
         * bytes as `const int32_t *`, which is undefined behavior on a misaligned
         * pointer. */
        return false;
    }
    if ((uint64_t)required_ints * sizeof(int32_t) > (uint64_t)meta_blob->byte_length)
    {
        return false;
    }
    if (rank >= 0 && (rank < 1 || rank > 4))
    {
        return false;
    }
    if (axis >= 0 && rank >= 0 && axis >= rank)
    {
        return false;
    }
    return true;
}

static void hct_row_major_strides(const int32_t *shape, int32_t rank, int32_t *strides)
{
    int32_t stride = 1;
    int32_t i;
    for (i = rank - 1; i >= 0; --i)
    {
        strides[i] = stride;
        stride *= shape[i];
    }
}

static void hct_broadcast_strides(const hct_server_blob_t *blob,
                                  const int32_t *output_shape,
                                  int32_t rank,
                                  int32_t *strides)
{
    int32_t input_shape[4] = {1, 1, 1, 1};
    int32_t base_strides[4] = {0, 0, 0, 0};
    int32_t i;
    hct_fill_shape_from_blob(blob, rank, input_shape, true);
    hct_row_major_strides(input_shape, rank, base_strides);
    for (i = 0; i < rank; ++i)
    {
        strides[i] = (input_shape[i] == 1 && output_shape[i] != 1) ? 0 : base_strides[i];
    }
}

static arm_cmsis_nn_status run_data_movement_once(hct_server_session_t *session)
{
    hct_server_blob_t *input0 = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_0);
    hct_server_blob_t *input1 = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_1);
    hct_server_blob_t *input2 = find_blob_by_role(session, HCT_BLOB_ROLE_INPUT_2);
    hct_server_blob_t *meta_blob = find_blob_by_role(session, HCT_BLOB_ROLE_META_0);
    const int32_t *meta = (meta_blob != NULL) ? (const int32_t *)blob_ptr(session, meta_blob) : NULL;

    switch (session->expected_kernel_id)
    {
        case HCT_KERNEL_ID_RESHAPE_S8:
        case HCT_KERNEL_ID_SQUEEZE_S8:
        {
            if (input0 == NULL)
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            session->output_length = input0->byte_length;
            if (session->output_length > session->output_capacity_bytes)
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            arm_reshape_s8((const int8_t *)blob_ptr(session, input0),
                           (int8_t *)hct_output_ptr(session),
                           hct_blob_element_count(input0));
            return ARM_CMSIS_NN_SUCCESS;
        }

        case HCT_KERNEL_ID_RESHAPE_F32:
        case HCT_KERNEL_ID_RESHAPE_F16:
        {
            if (input0 == NULL)
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            session->output_length = input0->byte_length;
            if (session->output_length > session->output_capacity_bytes)
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_RESHAPE_F16)
            {
                arm_reshape_f16((const float16_t *)blob_ptr(session, input0),
                                (float16_t *)hct_output_ptr(session),
                                hct_blob_element_count(input0));
            }
            else
            {
                arm_reshape_f32((const float *)blob_ptr(session, input0),
                                (float *)hct_output_ptr(session),
                                hct_blob_element_count(input0));
            }
            return ARM_CMSIS_NN_SUCCESS;
        }

        case HCT_KERNEL_ID_TRANSPOSE_S8:
        case HCT_KERNEL_ID_TRANSPOSE_S16:
        {
            cmsis_nn_dims input_dims;
            cmsis_nn_dims output_dims;
            int32_t output_shape[4] = {1, 1, 1, 1};
            uint32_t element_size = (session->expected_kernel_id == HCT_KERNEL_ID_TRANSPOSE_S16) ? sizeof(int16_t) : sizeof(int8_t);
            uint32_t permutations[4] = {0u, 1u, 2u, 3u};
            int32_t rank;
            int32_t i;
            if (input0 == NULL || !hct_validate_meta0_blob(meta_blob, 1u, -1, -1))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            rank = meta[0];
            /* F007: now that rank is known, re-validate that the blob actually covers
             * the rank int32_t plus its permutation entries before reading them, and
             * reject an out-of-range rank up front. */
            if (rank < 1 || rank > 4 || !hct_validate_meta0_blob(meta_blob, (size_t)(1 + rank), rank, -1))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            for (i = 0; i < rank; ++i)
            {
                permutations[i] = (uint32_t)meta[1 + i];
                /* F007: reject any permutation entry outside [0, rank) -- an invalid
                 * axis here would let arm_transpose_s8/s16 index the input dims array
                 * out of bounds. */
                if (permutations[i] >= (uint32_t)rank)
                {
                    return ARM_CMSIS_NN_ARG_ERROR;
                }
            }
            hct_fill_dims_from_blob(input0, &input_dims);
            hct_fill_output_dims_from_session(session, &output_dims);
            hct_fill_output_shape_from_session(session, rank, output_shape);
            if (!hct_checked_shape_bytes(output_shape, rank, element_size,
                                         session->output_capacity_bytes, &session->output_length))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            {
                uint32_t operand_bytes_needed;
                if (!hct_checked_dims_bytes(&input_dims, element_size, input0->byte_length, &operand_bytes_needed))
                {
                    return ARM_CMSIS_NN_ARG_ERROR;
                }
            }
            cmsis_nn_transpose_params params = {rank, permutations};
            if (session->expected_kernel_id == HCT_KERNEL_ID_TRANSPOSE_S16)
            {
                return arm_transpose_s16((const int16_t *)blob_ptr(session, input0),
                                         (int16_t *)hct_output_ptr(session),
                                         &input_dims,
                                         &output_dims,
                                         &params);
            }
            return arm_transpose_s8((const int8_t *)blob_ptr(session, input0),
                                    (int8_t *)hct_output_ptr(session),
                                    &input_dims,
                                    &output_dims,
                                    &params);
        }

        case HCT_KERNEL_ID_TRANSPOSE_F32:
        case HCT_KERNEL_ID_TRANSPOSE_F16:
        {
            cmsis_nn_context ctx = {NULL, 0};
            cmsis_nn_dims input_dims;
            cmsis_nn_dims output_dims;
            int32_t output_shape[4] = {1, 1, 1, 1};
            uint32_t element_size = (session->expected_kernel_id == HCT_KERNEL_ID_TRANSPOSE_F16) ? sizeof(float16_t) : sizeof(float);
            int32_t perm[4] = {0, 1, 2, 3};
            int32_t rank;
            int32_t i;
            if (input0 == NULL || !hct_validate_meta0_blob(meta_blob, 1u, -1, -1))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            rank = meta[0];
            if (rank < 1 || rank > 4 || !hct_validate_meta0_blob(meta_blob, (size_t)(1 + rank), rank, -1))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            for (i = 0; i < rank; ++i)
            {
                perm[i] = meta[1 + i];
                if (perm[i] < 0 || perm[i] >= rank)
                {
                    return ARM_CMSIS_NN_ARG_ERROR;
                }
            }
            hct_fill_dims_from_blob(input0, &input_dims);
            hct_fill_output_dims_from_session(session, &output_dims);
            hct_fill_output_shape_from_session(session, rank, output_shape);
            if (!hct_checked_shape_bytes(output_shape, rank, element_size,
                                         session->output_capacity_bytes, &session->output_length))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            {
                uint32_t operand_bytes_needed;
                if (!hct_checked_dims_bytes(&input_dims, element_size, input0->byte_length, &operand_bytes_needed))
                {
                    return ARM_CMSIS_NN_ARG_ERROR;
                }
            }
            cmsis_nn_transpose_params_f32 params_f32;
            cmsis_nn_transpose_params_f16 params_f16;
            params_f32.num_dims = rank;
            params_f32.layout = ARM_NN_LAYOUT_NHWC;
            params_f16.num_dims = rank;
            params_f16.layout = ARM_NN_LAYOUT_NHWC;
            for (i = 0; i < 4; ++i)
            {
                params_f32.perm[i] = perm[i];
                params_f16.perm[i] = perm[i];
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_TRANSPOSE_F16)
            {
                return arm_transpose_f16(&ctx,
                                        &params_f16,
                                        &input_dims,
                                        (const float16_t *)blob_ptr(session, input0),
                                        &output_dims,
                                        (float16_t *)hct_output_ptr(session));
            }
            return arm_transpose_f32(&ctx,
                                     &params_f32,
                                     &input_dims,
                                     (const float *)blob_ptr(session, input0),
                                     &output_dims,
                                     (float *)hct_output_ptr(session));
        }

        case HCT_KERNEL_ID_PAD_S8:
        case HCT_KERNEL_ID_PAD_S16:
        {
            cmsis_nn_dims input_dims;
            cmsis_nn_dims pre_pad;
            cmsis_nn_dims post_pad;
            cmsis_nn_dims pad_output_dims;
            uint32_t element_size = (session->expected_kernel_id == HCT_KERNEL_ID_PAD_S16) ? sizeof(int16_t) : sizeof(int8_t);
            uint32_t operand_bytes_needed;
            /* F007: meta[0..8] is a fixed 9-int32_t layout (pad value + 4 pre-pad + 4
             * post-pad dims); prove the blob covers all 9 before reading any of them. */
            if (input0 == NULL || !hct_validate_meta0_blob(meta_blob, 9u, -1, -1))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            hct_fill_dims_from_blob(input0, &input_dims);
            pre_pad.n = meta[1];
            pre_pad.h = meta[2];
            pre_pad.w = meta[3];
            pre_pad.c = meta[4];
            post_pad.n = meta[5];
            post_pad.h = meta[6];
            post_pad.w = meta[7];
            post_pad.c = meta[8];
            /* Sized via hct_checked_dims_bytes() (64-bit-safe product), not a raw 32-bit
             * multiply -- oversized session->output_n/h/w/c would otherwise wrap mod 2^32
             * and silently pass the capacity check below. */
            pad_output_dims.n = (session->output_n > 0) ? session->output_n : 1;
            pad_output_dims.h = (session->output_h > 0) ? session->output_h : 1;
            pad_output_dims.w = (session->output_w > 0) ? session->output_w : 1;
            pad_output_dims.c = (session->output_c > 0) ? session->output_c : 1;
            if (!hct_checked_dims_bytes(&pad_output_dims, element_size,
                                        session->output_capacity_bytes, &session->output_length))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            if (!hct_checked_dims_bytes(&input_dims, element_size, input0->byte_length, &operand_bytes_needed))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_PAD_S16)
            {
                return arm_pad_s16((const int16_t *)blob_ptr(session, input0),
                                   (int16_t *)hct_output_ptr(session),
                                   (int16_t)meta[0],
                                   &input_dims,
                                   &pre_pad,
                                   &post_pad);
            }
            return arm_pad_s8((const int8_t *)blob_ptr(session, input0),
                              (int8_t *)hct_output_ptr(session),
                              (int8_t)meta[0],
                              &input_dims,
                              &pre_pad,
                              &post_pad);
        }

        case HCT_KERNEL_ID_PAD_F32:
        case HCT_KERNEL_ID_PAD_F16:
        {
            cmsis_nn_dims input_dims;
            cmsis_nn_dims pre_pad;
            cmsis_nn_dims post_pad;
            cmsis_nn_dims pad_output_dims;
            union { int32_t i; float f; } pad_value_bits;
            uint32_t element_size = (session->expected_kernel_id == HCT_KERNEL_ID_PAD_F16) ? sizeof(float16_t) : sizeof(float);
            uint32_t operand_bytes_needed;
            /* pad_value is bit-cast into meta[0] (same slot the S8/S16 path uses),
             * same wire-encoding convention as the pooling FP32 activation clamp. */
            if (input0 == NULL || !hct_validate_meta0_blob(meta_blob, 9u, -1, -1))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            hct_fill_dims_from_blob(input0, &input_dims);
            pre_pad.n = meta[1];
            pre_pad.h = meta[2];
            pre_pad.w = meta[3];
            pre_pad.c = meta[4];
            post_pad.n = meta[5];
            post_pad.h = meta[6];
            post_pad.w = meta[7];
            post_pad.c = meta[8];
            /* Sized via hct_checked_dims_bytes() (64-bit-safe product), not a raw 32-bit
             * multiply -- oversized session->output_n/h/w/c would otherwise wrap mod 2^32
             * and silently pass the capacity check below. */
            pad_output_dims.n = (session->output_n > 0) ? session->output_n : 1;
            pad_output_dims.h = (session->output_h > 0) ? session->output_h : 1;
            pad_output_dims.w = (session->output_w > 0) ? session->output_w : 1;
            pad_output_dims.c = (session->output_c > 0) ? session->output_c : 1;
            if (!hct_checked_dims_bytes(&pad_output_dims, element_size,
                                        session->output_capacity_bytes, &session->output_length))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            if (!hct_checked_dims_bytes(&input_dims, element_size, input0->byte_length, &operand_bytes_needed))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            pad_value_bits.i = meta[0];
            if (session->expected_kernel_id == HCT_KERNEL_ID_PAD_F16)
            {
                return arm_pad_f16((const float16_t *)blob_ptr(session, input0),
                                   (float16_t *)hct_output_ptr(session),
                                   (float16_t)pad_value_bits.f,
                                   &input_dims,
                                   &pre_pad,
                                   &post_pad);
            }
            return arm_pad_f32((const float *)blob_ptr(session, input0),
                               (float *)hct_output_ptr(session),
                               pad_value_bits.f,
                               &input_dims,
                               &pre_pad,
                               &post_pad);
        }

        case HCT_KERNEL_ID_MIRROR_PAD_S8:
        case HCT_KERNEL_ID_MIRROR_PAD_S16:
        {
            cmsis_nn_mirror_pad_params params;
            int32_t input_shape[4] = {1, 1, 1, 1};
            int32_t output_shape[4] = {1, 1, 1, 1};
            int32_t pad_before[4] = {0, 0, 0, 0};
            uint32_t element_size = (session->expected_kernel_id == HCT_KERNEL_ID_MIRROR_PAD_S16) ? sizeof(int16_t) : sizeof(int8_t);
            int32_t rank;
            int32_t i;
            if (input0 == NULL || !hct_validate_meta0_blob(meta_blob, 2u, -1, -1))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            rank = meta[0];
            if (rank < 1 || rank > 4 || !hct_validate_meta0_blob(meta_blob, (size_t)(2 + rank), rank, -1))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            hct_fill_shape_from_blob(input0, rank, input_shape, false);
            hct_fill_output_shape_from_session(session, rank, output_shape);
            for (i = 0; i < rank; ++i)
            {
                pad_before[i] = meta[2 + i];
            }
            if (!hct_checked_shape_bytes(output_shape, rank, element_size,
                                         session->output_capacity_bytes, &session->output_length))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            {
                uint32_t operand_bytes_needed;
                if (!hct_checked_shape_bytes(input_shape, rank, element_size, input0->byte_length, &operand_bytes_needed))
                {
                    return ARM_CMSIS_NN_ARG_ERROR;
                }
            }
            params.rank = rank;
            params.input_shape = input_shape;
            params.output_shape = output_shape;
            params.pad_before = pad_before;
            params.mode = meta[1];
            if (session->expected_kernel_id == HCT_KERNEL_ID_MIRROR_PAD_S16)
            {
                return arm_mirror_pad_s16((const int16_t *)blob_ptr(session, input0), &params, (int16_t *)hct_output_ptr(session));
            }
            return arm_mirror_pad_s8((const int8_t *)blob_ptr(session, input0), &params, (int8_t *)hct_output_ptr(session));
        }

        case HCT_KERNEL_ID_CONCATENATION_S8:
        case HCT_KERNEL_ID_CONCATENATION_S16:
        case HCT_KERNEL_ID_CONCATENATION_S32:
        {
            int32_t output_shape[4] = {1, 1, 1, 1};
            int32_t input_shape0[4] = {1, 1, 1, 1};
            int32_t input_shape1[4] = {1, 1, 1, 1};
            int32_t input_concat_dims[2];
            const void *input_ptrs[2];
            uint32_t style_code;
            int32_t rank;
            int32_t axis;
            int32_t inputs_count;
            if (input0 == NULL || input1 == NULL || !hct_validate_meta0_blob(meta_blob, 4u, -1, -1))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            style_code = (uint32_t)meta[0];
            rank = meta[1];
            axis = meta[2];
            inputs_count = meta[3];
            /* F007: axis must be a valid index into a rank-sized shape array -- meta[2]
             * previously indexed input_shape0[axis]/input_shape1[axis] with no bound,
             * so a wire-controlled out-of-range axis could read/write past the 4-element
             * stack arrays below. */
            if (rank < 1 || rank > 4 || inputs_count != 2 || axis < 0 || axis >= rank)
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            hct_fill_output_shape_from_session(session, rank, output_shape);
            hct_fill_shape_from_blob(input0, rank, input_shape0, false);
            hct_fill_shape_from_blob(input1, rank, input_shape1, false);
            input_concat_dims[0] = input_shape0[axis];
            input_concat_dims[1] = input_shape1[axis];
            input_ptrs[0] = blob_ptr(session, input0);
            input_ptrs[1] = blob_ptr(session, input1);
            if (!hct_checked_shape_bytes(output_shape, rank, hct_dtype_size_bytes(input0->dtype),
                                         session->output_capacity_bytes, &session->output_length))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            {
                uint32_t operand_bytes_needed;
                if (!hct_checked_shape_bytes(input_shape0, rank, hct_dtype_size_bytes(input0->dtype),
                                             input0->byte_length, &operand_bytes_needed) ||
                    !hct_checked_shape_bytes(input_shape1, rank, hct_dtype_size_bytes(input1->dtype),
                                             input1->byte_length, &operand_bytes_needed))
                {
                    return ARM_CMSIS_NN_ARG_ERROR;
                }
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_CONCATENATION_S8 && style_code != 0u)
            {
                uint32_t offset = 0u;
                const int32_t *shapes[2] = {input_shape0, input_shape1};
                int32_t idx;
                for (idx = 0; idx < 2; ++idx)
                {
                    const int32_t *shape = shapes[idx];
                    const int8_t *input_data = (idx == 0) ? (const int8_t *)blob_ptr(session, input0) : (const int8_t *)blob_ptr(session, input1);
                    switch (style_code)
                    {
                        case 1u:
                            arm_concatenation_s8_x(input_data,
                                                   (uint16_t)shape[2],
                                                   (uint16_t)shape[1],
                                                   (uint16_t)shape[3],
                                                   (uint16_t)shape[0],
                                                   (int8_t *)hct_output_ptr(session),
                                                   (uint16_t)output_shape[2],
                                                   offset);
                            offset += (uint32_t)shape[2];
                            break;
                        case 2u:
                            arm_concatenation_s8_y(input_data,
                                                   (uint16_t)shape[2],
                                                   (uint16_t)shape[1],
                                                   (uint16_t)shape[3],
                                                   (uint16_t)shape[0],
                                                   (int8_t *)hct_output_ptr(session),
                                                   (uint16_t)output_shape[1],
                                                   offset);
                            offset += (uint32_t)shape[1];
                            break;
                        case 3u:
                            arm_concatenation_s8_z(input_data,
                                                   (uint16_t)shape[2],
                                                   (uint16_t)shape[1],
                                                   (uint16_t)shape[3],
                                                   (uint16_t)shape[0],
                                                   (int8_t *)hct_output_ptr(session),
                                                   (uint16_t)output_shape[3],
                                                   offset);
                            offset += (uint32_t)shape[3];
                            break;
                        case 4u:
                            arm_concatenation_s8_w(input_data,
                                                   (uint16_t)shape[2],
                                                   (uint16_t)shape[1],
                                                   (uint16_t)shape[3],
                                                   (uint16_t)shape[0],
                                                   (int8_t *)hct_output_ptr(session),
                                                   offset);
                            offset += (uint32_t)shape[0];
                            break;
                        default:
                            return ARM_CMSIS_NN_ARG_ERROR;
                    }
                }
                return ARM_CMSIS_NN_SUCCESS;
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_CONCATENATION_S16)
            {
                return arm_concatenation_s16((const int16_t *const *)input_ptrs,
                                             inputs_count,
                                             input_concat_dims,
                                             axis,
                                             (int16_t *)hct_output_ptr(session),
                                             rank,
                                             output_shape);
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_CONCATENATION_S32)
            {
                return arm_concatenation_s32((const int32_t *const *)input_ptrs,
                                             inputs_count,
                                             input_concat_dims,
                                             axis,
                                             (int32_t *)hct_output_ptr(session),
                                             rank,
                                             output_shape);
            }
            return arm_concatenation_s8((const int8_t *const *)input_ptrs,
                                        inputs_count,
                                        input_concat_dims,
                                        axis,
                                        (int8_t *)hct_output_ptr(session),
                                        rank,
                                        output_shape);
        }

        case HCT_KERNEL_ID_CONCATENATION_F32:
        case HCT_KERNEL_ID_CONCATENATION_F16:
        {
            /* Float concatenation only ships the 4 style-suffixed entrypoints
             * (arm_concatenation_f32/f16_w/_x/_y/_z) -- there is no generic
             * axis-based arm_concatenation_f32/f16, so style_code must be nonzero. */
            int32_t output_shape[4] = {1, 1, 1, 1};
            int32_t input_shapes[3][4];
            uint32_t style_code;
            int32_t rank;
            int32_t axis;
            int32_t inputs_count;
            uint32_t offset;
            int32_t idx;
            bool is_f16 = (session->expected_kernel_id == HCT_KERNEL_ID_CONCATENATION_F16);
            uint32_t element_size = is_f16 ? sizeof(float16_t) : sizeof(float);
            if (input0 == NULL || !hct_validate_meta0_blob(meta_blob, 4u, -1, -1))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            style_code = (uint32_t)meta[0];
            rank = meta[1];
            axis = meta[2];
            inputs_count = meta[3];
            if (rank < 1 || rank > 4 || inputs_count < 1 || inputs_count > 3 || axis < 0 || axis >= rank || style_code == 0u || style_code > 4u)
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            hct_fill_output_shape_from_session(session, rank, output_shape);
            if (!hct_checked_shape_bytes(output_shape, rank, element_size,
                                         session->output_capacity_bytes, &session->output_length))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            offset = 0u;
            for (idx = 0; idx < inputs_count; ++idx)
            {
                hct_server_blob_t *input_blob = (idx == 0) ? input0 : ((idx == 1) ? input1 : input2);
                const int32_t *shape = input_shapes[idx];
                const void *input_data;
                if (input_blob == NULL)
                {
                    return ARM_CMSIS_NN_ARG_ERROR;
                }
                hct_fill_shape_from_blob(input_blob, rank, input_shapes[idx], false);
                {
                    uint32_t operand_bytes_needed;
                    if (!hct_checked_shape_bytes(input_shapes[idx], rank, element_size,
                                                 input_blob->byte_length, &operand_bytes_needed))
                    {
                        return ARM_CMSIS_NN_ARG_ERROR;
                    }
                }
                input_data = blob_ptr(session, input_blob);
                switch (style_code)
                {
                    case 1u:
                        if (is_f16)
                        {
                            arm_concatenation_f16_x((const float16_t *)input_data, (uint16_t)shape[2], (uint16_t)shape[1], (uint16_t)shape[3], (uint16_t)shape[0], (float16_t *)hct_output_ptr(session), (uint16_t)output_shape[2], offset);
                        }
                        else
                        {
                            arm_concatenation_f32_x((const float *)input_data, (uint16_t)shape[2], (uint16_t)shape[1], (uint16_t)shape[3], (uint16_t)shape[0], (float *)hct_output_ptr(session), (uint16_t)output_shape[2], offset);
                        }
                        offset += (uint32_t)shape[2];
                        break;
                    case 2u:
                        if (is_f16)
                        {
                            arm_concatenation_f16_y((const float16_t *)input_data, (uint16_t)shape[2], (uint16_t)shape[1], (uint16_t)shape[3], (uint16_t)shape[0], (float16_t *)hct_output_ptr(session), (uint16_t)output_shape[1], offset);
                        }
                        else
                        {
                            arm_concatenation_f32_y((const float *)input_data, (uint16_t)shape[2], (uint16_t)shape[1], (uint16_t)shape[3], (uint16_t)shape[0], (float *)hct_output_ptr(session), (uint16_t)output_shape[1], offset);
                        }
                        offset += (uint32_t)shape[1];
                        break;
                    case 3u:
                        if (is_f16)
                        {
                            arm_concatenation_f16_z((const float16_t *)input_data, (uint16_t)shape[2], (uint16_t)shape[1], (uint16_t)shape[3], (uint16_t)shape[0], (float16_t *)hct_output_ptr(session), (uint16_t)output_shape[3], offset);
                        }
                        else
                        {
                            arm_concatenation_f32_z((const float *)input_data, (uint16_t)shape[2], (uint16_t)shape[1], (uint16_t)shape[3], (uint16_t)shape[0], (float *)hct_output_ptr(session), (uint16_t)output_shape[3], offset);
                        }
                        offset += (uint32_t)shape[3];
                        break;
                    default:
                        if (is_f16)
                        {
                            arm_concatenation_f16_w((const float16_t *)input_data, (uint16_t)shape[2], (uint16_t)shape[1], (uint16_t)shape[3], (uint16_t)shape[0], (float16_t *)hct_output_ptr(session), offset);
                        }
                        else
                        {
                            arm_concatenation_f32_w((const float *)input_data, (uint16_t)shape[2], (uint16_t)shape[1], (uint16_t)shape[3], (uint16_t)shape[0], (float *)hct_output_ptr(session), offset);
                        }
                        offset += (uint32_t)shape[0];
                        break;
                }
            }
            return ARM_CMSIS_NN_SUCCESS;
        }

        case HCT_KERNEL_ID_SPLIT_S8:
        case HCT_KERNEL_ID_SPLIT_S16:
        {
            int32_t rank;
            int32_t axis;
            int32_t num_splits;
            int32_t input_shape[4] = {1, 1, 1, 1};
            uint32_t offset_bytes = 0u;
            if (input0 == NULL || !hct_validate_meta0_blob(meta_blob, 3u, -1, -1))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            rank = meta[0];
            axis = meta[1];
            num_splits = meta[2];
            /* F007: axis must be a valid index into the rank-sized shape array (it's
             * used to index output_shape[axis] below), and the required-ints check must
             * be widened once num_splits is known, since the split sizes at meta[3..]
             * come after the fixed 3-int header. */
            if (rank < 1 || rank > 4 || num_splits < 1 || num_splits > 4 || axis < 0 || axis >= rank ||
                !hct_validate_meta0_blob(meta_blob, (size_t)(3 + num_splits), rank, axis))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            hct_fill_shape_from_blob(input0, rank, input_shape, false);
            {
                uint32_t operand_bytes_needed;
                const uint32_t split_element_size = (session->expected_kernel_id == HCT_KERNEL_ID_SPLIT_S16) ? sizeof(int16_t) : sizeof(int8_t);
                if (!hct_checked_shape_bytes(input_shape, rank, split_element_size, input0->byte_length, &operand_bytes_needed))
                {
                    return ARM_CMSIS_NN_ARG_ERROR;
                }
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_SPLIT_S16)
            {
                int16_t *output_ptrs[4] = {NULL, NULL, NULL, NULL};
                int32_t split_index;
                for (split_index = 0; split_index < num_splits; ++split_index)
                {
                    int32_t output_shape[4] = {input_shape[0], input_shape[1], input_shape[2], input_shape[3]};
                    uint32_t bytes;
                    output_shape[axis] = meta[3 + split_index];
                    if (!hct_checked_shape_bytes(output_shape, rank, sizeof(int16_t),
                                                 session->output_capacity_bytes - offset_bytes, &bytes))
                    {
                        return ARM_CMSIS_NN_ARG_ERROR;
                    }
                    if (offset_bytes + bytes > session->output_capacity_bytes)
                    {
                        return ARM_CMSIS_NN_ARG_ERROR;
                    }
                    output_ptrs[split_index] = (int16_t *)&hct_output_ptr(session)[offset_bytes];
                    offset_bytes += bytes;
                }
                session->output_length = offset_bytes;
                return arm_split_s16((const int16_t *)blob_ptr(session, input0),
                                     rank,
                                     input_shape,
                                     axis,
                                     num_splits,
                                     &meta[3],
                                     output_ptrs);
            }
            else
            {
                int8_t *output_ptrs[4] = {NULL, NULL, NULL, NULL};
                int32_t split_index;
                for (split_index = 0; split_index < num_splits; ++split_index)
                {
                    int32_t output_shape[4] = {input_shape[0], input_shape[1], input_shape[2], input_shape[3]};
                    uint32_t bytes;
                    output_shape[axis] = meta[3 + split_index];
                    if (!hct_checked_shape_bytes(output_shape, rank, sizeof(int8_t),
                                                 session->output_capacity_bytes - offset_bytes, &bytes))
                    {
                        return ARM_CMSIS_NN_ARG_ERROR;
                    }
                    if (offset_bytes + bytes > session->output_capacity_bytes)
                    {
                        return ARM_CMSIS_NN_ARG_ERROR;
                    }
                    output_ptrs[split_index] = (int8_t *)&hct_output_ptr(session)[offset_bytes];
                    offset_bytes += bytes;
                }
                session->output_length = offset_bytes;
                return arm_split_s8((const int8_t *)blob_ptr(session, input0),
                                    rank,
                                    input_shape,
                                    axis,
                                    num_splits,
                                    &meta[3],
                                    output_ptrs);
            }
        }

        case HCT_KERNEL_ID_SPLIT_F16:
        {
            int32_t rank;
            int32_t axis;
            int32_t num_splits;
            int32_t input_shape[4] = {1, 1, 1, 1};
            uint32_t offset_bytes = 0u;
            float16_t *output_ptrs[4] = {NULL, NULL, NULL, NULL};
            int32_t split_index;
            if (input0 == NULL || !hct_validate_meta0_blob(meta_blob, 3u, -1, -1))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            rank = meta[0];
            axis = meta[1];
            num_splits = meta[2];
            if (rank < 1 || rank > 4 || num_splits < 1 || num_splits > 4 || axis < 0 || axis >= rank ||
                !hct_validate_meta0_blob(meta_blob, (size_t)(3 + num_splits), rank, axis))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            hct_fill_shape_from_blob(input0, rank, input_shape, false);
            {
                uint32_t operand_bytes_needed;
                if (!hct_checked_shape_bytes(input_shape, rank, sizeof(float16_t), input0->byte_length, &operand_bytes_needed))
                {
                    return ARM_CMSIS_NN_ARG_ERROR;
                }
            }
            for (split_index = 0; split_index < num_splits; ++split_index)
            {
                int32_t output_shape[4] = {input_shape[0], input_shape[1], input_shape[2], input_shape[3]};
                uint32_t bytes;
                output_shape[axis] = meta[3 + split_index];
                if (!hct_checked_shape_bytes(output_shape, rank, sizeof(float16_t),
                                             session->output_capacity_bytes - offset_bytes, &bytes))
                {
                    return ARM_CMSIS_NN_ARG_ERROR;
                }
                if (offset_bytes + bytes > session->output_capacity_bytes)
                {
                    return ARM_CMSIS_NN_ARG_ERROR;
                }
                output_ptrs[split_index] = (float16_t *)&hct_output_ptr(session)[offset_bytes];
                offset_bytes += bytes;
            }
            session->output_length = offset_bytes;
            return arm_split_f16((const float16_t *)blob_ptr(session, input0),
                                 rank,
                                 input_shape,
                                 axis,
                                 num_splits,
                                 &meta[3],
                                 output_ptrs);
        }

        case HCT_KERNEL_ID_BATCH_TO_SPACE_ND_S8:
        case HCT_KERNEL_ID_BATCH_TO_SPACE_ND_S16:
        case HCT_KERNEL_ID_SPACE_TO_BATCH_ND_S8:
        case HCT_KERNEL_ID_SPACE_TO_BATCH_ND_S16:
        {
            cmsis_nn_dims input_dims;
            cmsis_nn_dims output_dims;
            cmsis_nn_dims extra_dims;
            cmsis_nn_tile block_shape;
            uint32_t element_size = (input0 != NULL) ? hct_dtype_size_bytes(input0->dtype) : 0u;
            const bool is_batch_to_space =
                session->expected_kernel_id == HCT_KERNEL_ID_BATCH_TO_SPACE_ND_S8 ||
                session->expected_kernel_id == HCT_KERNEL_ID_BATCH_TO_SPACE_ND_S16;
            const size_t required_meta_ints = is_batch_to_space ? 6u : 7u;
            if (input0 == NULL || !hct_validate_meta0_blob(meta_blob, required_meta_ints, -1, -1))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            hct_fill_dims_from_blob(input0, &input_dims);
            hct_fill_output_dims_from_session(session, &output_dims);
            block_shape.h = meta[0];
            block_shape.w = meta[1];
            extra_dims.n = meta[2];
            extra_dims.h = meta[3];
            extra_dims.w = meta[4];
            extra_dims.c = meta[5];
            if (!hct_checked_dims_bytes(&output_dims, element_size,
                                        session->output_capacity_bytes, &session->output_length))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            {
                uint32_t operand_bytes_needed;
                if (!hct_checked_dims_bytes(&input_dims, element_size, input0->byte_length, &operand_bytes_needed))
                {
                    return ARM_CMSIS_NN_ARG_ERROR;
                }
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_BATCH_TO_SPACE_ND_S16)
            {
                return arm_batch_to_space_nd_s16((const int16_t *)blob_ptr(session, input0), &input_dims, &block_shape, &extra_dims, (int16_t *)hct_output_ptr(session), &output_dims);
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_BATCH_TO_SPACE_ND_S8)
            {
                return arm_batch_to_space_nd_s8((const int8_t *)blob_ptr(session, input0), &input_dims, &block_shape, &extra_dims, (int8_t *)hct_output_ptr(session), &output_dims);
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_SPACE_TO_BATCH_ND_S16)
            {
                return arm_space_to_batch_nd_s16((const int16_t *)blob_ptr(session, input0), &input_dims, &block_shape, &extra_dims, (int16_t *)hct_output_ptr(session), &output_dims, meta[6]);
            }
            return arm_space_to_batch_nd_s8((const int8_t *)blob_ptr(session, input0), &input_dims, &block_shape, &extra_dims, (int8_t *)hct_output_ptr(session), &output_dims, meta[6]);
        }

        case HCT_KERNEL_ID_SPACE_TO_DEPTH_S8:
        case HCT_KERNEL_ID_SPACE_TO_DEPTH_S16:
        case HCT_KERNEL_ID_DEPTH_TO_SPACE_S8:
        case HCT_KERNEL_ID_DEPTH_TO_SPACE_S16:
        {
            cmsis_nn_dims input_dims;
            cmsis_nn_dims output_dims;
            uint32_t element_size = (input0 != NULL) ? hct_dtype_size_bytes(input0->dtype) : 0u;
            if (input0 == NULL || !hct_validate_meta0_blob(meta_blob, 1u, -1, -1))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            hct_fill_dims_from_blob(input0, &input_dims);
            hct_fill_output_dims_from_session(session, &output_dims);
            if (!hct_checked_dims_bytes(&output_dims, element_size,
                                        session->output_capacity_bytes, &session->output_length))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            {
                uint32_t operand_bytes_needed;
                if (!hct_checked_dims_bytes(&input_dims, element_size, input0->byte_length, &operand_bytes_needed))
                {
                    return ARM_CMSIS_NN_ARG_ERROR;
                }
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_SPACE_TO_DEPTH_S16)
            {
                return arm_space_to_depth_s16((const int16_t *)blob_ptr(session, input0), &input_dims, meta[0], (int16_t *)hct_output_ptr(session), &output_dims);
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_SPACE_TO_DEPTH_S8)
            {
                return arm_space_to_depth_s8((const int8_t *)blob_ptr(session, input0), &input_dims, meta[0], (int8_t *)hct_output_ptr(session), &output_dims);
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_DEPTH_TO_SPACE_S16)
            {
                return arm_depth_to_space_s16((const int16_t *)blob_ptr(session, input0), &input_dims, meta[0], (int16_t *)hct_output_ptr(session), &output_dims);
            }
            return arm_depth_to_space_s8((const int8_t *)blob_ptr(session, input0), &input_dims, meta[0], (int8_t *)hct_output_ptr(session), &output_dims);
        }

        case HCT_KERNEL_ID_RESIZE_NEAREST_NEIGHBOR_S8:
        case HCT_KERNEL_ID_RESIZE_NEAREST_NEIGHBOR_S16:
        {
            cmsis_nn_context ctx;
            cmsis_nn_resize_params params;
            cmsis_nn_dims input_dims;
            cmsis_nn_dims output_dims;
            cmsis_nn_dims output_size_dims;
            int32_t output_size_data[2];
            uint32_t required_ctx_bytes;
            uint32_t element_size = (input0 != NULL) ? hct_dtype_size_bytes(input0->dtype) : 0u;
            if (input0 == NULL || !hct_validate_meta0_blob(meta_blob, 2u, -1, -1))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            hct_fill_dims_from_blob(input0, &input_dims);
            hct_fill_output_dims_from_session(session, &output_dims);
            if (output_dims.h <= 0 || output_dims.w <= 0 ||
                ((uint64_t)(uint32_t)output_dims.h + (uint32_t)output_dims.w) * sizeof(int32_t) > session->scratch_bytes)
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            required_ctx_bytes = ((uint32_t)output_dims.h + (uint32_t)output_dims.w) * sizeof(int32_t);
            if (!hct_checked_dims_bytes(&output_dims, element_size,
                                        session->output_capacity_bytes, &session->output_length))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            {
                uint32_t operand_bytes_needed;
                if (!hct_checked_dims_bytes(&input_dims, element_size, input0->byte_length, &operand_bytes_needed))
                {
                    return ARM_CMSIS_NN_ARG_ERROR;
                }
            }
            ctx.buf = (required_ctx_bytes > 0u) ? &session->workspace[session->scratch_offset] : NULL;
            ctx.size = (int32_t)required_ctx_bytes;
            params.align_corners = meta[0];
            params.half_pixel_centers = meta[1];
            output_size_dims.n = 1;
            output_size_dims.h = 1;
            output_size_dims.w = 1;
            output_size_dims.c = 2;
            output_size_data[0] = output_dims.h;
            output_size_data[1] = output_dims.w;
            if (session->expected_kernel_id == HCT_KERNEL_ID_RESIZE_NEAREST_NEIGHBOR_S16)
            {
                return arm_resize_nearest_neighbor_s16(&ctx,
                                                       &params,
                                                       &input_dims,
                                                       (const int16_t *)blob_ptr(session, input0),
                                                       &output_size_dims,
                                                       output_size_data,
                                                       &output_dims,
                                                       (int16_t *)hct_output_ptr(session));
            }
            return arm_resize_nearest_neighbor_s8(&ctx,
                                                  &params,
                                                  &input_dims,
                                                  (const int8_t *)blob_ptr(session, input0),
                                                  &output_size_dims,
                                                  output_size_data,
                                                  &output_dims,
                                                  (int8_t *)hct_output_ptr(session));
        }

        case HCT_KERNEL_ID_TILE_S8:
        case HCT_KERNEL_ID_TILE_S16:
        {
            cmsis_nn_tile_params params;
            int32_t input_shape[4] = {1, 1, 1, 1};
            int32_t output_shape[4] = {1, 1, 1, 1};
            int32_t multiples[4] = {1, 1, 1, 1};
            int32_t rank;
            int32_t i;
            if (input0 == NULL || !hct_validate_meta0_blob(meta_blob, 1u, -1, -1))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            rank = meta[0];
            if (rank < 1 || rank > 4)
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            if (!hct_validate_meta0_blob(meta_blob, (size_t)(1 + rank), rank, -1))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            hct_fill_shape_from_blob(input0, rank, input_shape, false);
            hct_fill_output_shape_from_session(session, rank, output_shape);
            for (i = 0; i < rank; ++i)
            {
                multiples[i] = meta[1 + i];
            }
            if (!hct_checked_shape_bytes(output_shape, rank, hct_dtype_size_bytes(input0->dtype),
                                         session->output_capacity_bytes, &session->output_length))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            {
                uint32_t operand_bytes_needed;
                if (!hct_checked_shape_bytes(input_shape, rank, hct_dtype_size_bytes(input0->dtype),
                                             input0->byte_length, &operand_bytes_needed))
                {
                    return ARM_CMSIS_NN_ARG_ERROR;
                }
            }
            params.rank = rank;
            params.input_shape = input_shape;
            params.multiples = multiples;
            if (session->expected_kernel_id == HCT_KERNEL_ID_TILE_S16)
            {
                return arm_tile_s16((const int16_t *)blob_ptr(session, input0), &params, (int16_t *)hct_output_ptr(session));
            }
            return arm_tile_s8((const int8_t *)blob_ptr(session, input0), &params, (int8_t *)hct_output_ptr(session));
        }

        case HCT_KERNEL_ID_GATHER_S8:
        case HCT_KERNEL_ID_GATHER_S16:
        {
            cmsis_nn_dims input_dims;
            cmsis_nn_dims indices_dims;
            cmsis_nn_dims output_dims;
            cmsis_nn_gather_params params;
            if (input0 == NULL || input1 == NULL || !hct_validate_meta0_blob(meta_blob, 4u, -1, -1))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            hct_fill_dims_from_blob(input0, &input_dims);
            hct_fill_dims_from_blob(input1, &indices_dims);
            hct_fill_output_dims_from_session(session, &output_dims);
            params.axis = meta[0];
            params.batch_dims = meta[1];
            params.input_rank = meta[2];
            params.coords_rank = meta[3];
            if (!hct_checked_dims_bytes(&output_dims, hct_dtype_size_bytes(input0->dtype),
                                        session->output_capacity_bytes, &session->output_length))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            {
                uint32_t operand_bytes_needed;
                if (!hct_checked_dims_bytes(&input_dims, hct_dtype_size_bytes(input0->dtype), input0->byte_length, &operand_bytes_needed) ||
                    !hct_checked_dims_bytes(&indices_dims, sizeof(int32_t), input1->byte_length, &operand_bytes_needed))
                {
                    return ARM_CMSIS_NN_ARG_ERROR;
                }
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_GATHER_S16)
            {
                return arm_gather_s16((const int16_t *)blob_ptr(session, input0),
                                      &input_dims,
                                      (const int32_t *)blob_ptr(session, input1),
                                      &indices_dims,
                                      &params,
                                      (int16_t *)hct_output_ptr(session),
                                      &output_dims);
            }
            return arm_gather_s8((const int8_t *)blob_ptr(session, input0),
                                 &input_dims,
                                 (const int32_t *)blob_ptr(session, input1),
                                 &indices_dims,
                                 &params,
                                 (int8_t *)hct_output_ptr(session),
                                 &output_dims);
        }

        case HCT_KERNEL_ID_GATHER_ND_S8:
        case HCT_KERNEL_ID_GATHER_ND_S16:
        {
            cmsis_nn_dims params_dims;
            cmsis_nn_dims indices_dims;
            cmsis_nn_dims output_dims;
            cmsis_nn_gather_nd_params params;
            if (input0 == NULL || input1 == NULL || !hct_validate_meta0_blob(meta_blob, 3u, -1, -1))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            hct_fill_dims_from_blob(input0, &params_dims);
            hct_fill_dims_from_blob(input1, &indices_dims);
            hct_fill_output_dims_from_session(session, &output_dims);
            params.params_rank = meta[0];
            params.indices_rank = meta[1];
            params.batch_dims = meta[2];
            if (!hct_checked_dims_bytes(&output_dims, hct_dtype_size_bytes(input0->dtype),
                                        session->output_capacity_bytes, &session->output_length))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            {
                uint32_t operand_bytes_needed;
                if (!hct_checked_dims_bytes(&params_dims, hct_dtype_size_bytes(input0->dtype), input0->byte_length, &operand_bytes_needed) ||
                    !hct_checked_dims_bytes(&indices_dims, sizeof(int32_t), input1->byte_length, &operand_bytes_needed))
                {
                    return ARM_CMSIS_NN_ARG_ERROR;
                }
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_GATHER_ND_S16)
            {
                return arm_gather_nd_s16((const int16_t *)blob_ptr(session, input0),
                                         &params_dims,
                                         (const int32_t *)blob_ptr(session, input1),
                                         &indices_dims,
                                         &params,
                                         (int16_t *)hct_output_ptr(session),
                                         &output_dims);
            }
            return arm_gather_nd_s8((const int8_t *)blob_ptr(session, input0),
                                    &params_dims,
                                    (const int32_t *)blob_ptr(session, input1),
                                    &indices_dims,
                                    &params,
                                    (int8_t *)hct_output_ptr(session),
                                    &output_dims);
        }

        case HCT_KERNEL_ID_WHERE_S8:
        case HCT_KERNEL_ID_WHERE_S16:
        {
            cmsis_nn_where_params params;
            int32_t shape[4] = {1, 1, 1, 1};
            int32_t num_true = 0;
            int32_t rank;
            uint32_t max_output_bytes;
            arm_cmsis_nn_status status;
            if (input0 == NULL || !hct_validate_meta0_blob(meta_blob, 1u, -1, -1))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            rank = meta[0];
            if (rank < 1 || rank > 4)
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            hct_fill_shape_from_blob(input0, rank, shape, false);
            {
                uint32_t operand_bytes_needed;
                if (!hct_checked_shape_bytes(shape, rank, hct_dtype_size_bytes(input0->dtype),
                                             input0->byte_length, &operand_bytes_needed))
                {
                    return ARM_CMSIS_NN_ARG_ERROR;
                }
            }
            max_output_bytes = hct_blob_element_count(input0) * (uint32_t)rank * sizeof(int64_t);
            if (max_output_bytes > session->output_capacity_bytes)
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            params.rank = rank;
            params.shape = shape;
            if (session->expected_kernel_id == HCT_KERNEL_ID_WHERE_S16)
            {
                status = arm_where_s16((const int16_t *)blob_ptr(session, input0), &params, (int64_t *)hct_output_ptr(session), &num_true);
            }
            else
            {
                status = arm_where_s8((const int8_t *)blob_ptr(session, input0), &params, (int64_t *)hct_output_ptr(session), &num_true);
            }
            session->output_length = (uint32_t)num_true * (uint32_t)rank * sizeof(int64_t);
            return status;
        }

        case HCT_KERNEL_ID_SELECT_V2_S8:
        case HCT_KERNEL_ID_SELECT_V2_S16:
        {
            cmsis_nn_select_v2_params params;
            int32_t output_shape[4] = {1, 1, 1, 1};
            int32_t cond_strides[4] = {0, 0, 0, 0};
            int32_t x_strides[4] = {0, 0, 0, 0};
            int32_t y_strides[4] = {0, 0, 0, 0};
            int32_t rank;
            if (input0 == NULL || input1 == NULL || input2 == NULL || !hct_validate_meta0_blob(meta_blob, 1u, -1, -1))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            rank = meta[0];
            if (rank < 1 || rank > 4)
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            hct_fill_output_shape_from_session(session, rank, output_shape);
            hct_broadcast_strides(input0, output_shape, rank, cond_strides);
            hct_broadcast_strides(input1, output_shape, rank, x_strides);
            hct_broadcast_strides(input2, output_shape, rank, y_strides);
            if (!hct_checked_shape_bytes(output_shape, rank, hct_dtype_size_bytes(input1->dtype),
                                         session->output_capacity_bytes, &session->output_length))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            {
                /* Each operand's own shape (not output_shape) bounds the actual read: a
                 * broadcast dim has stride 0 (see hct_broadcast_strides()), so the
                 * highest index touched in that blob is bounded by its own declared
                 * per-axis extent, not by the broadcast output's. */
                int32_t cond_shape[4] = {1, 1, 1, 1};
                int32_t x_shape[4] = {1, 1, 1, 1};
                int32_t y_shape[4] = {1, 1, 1, 1};
                uint32_t operand_bytes_needed;
                hct_fill_shape_from_blob(input0, rank, cond_shape, true);
                hct_fill_shape_from_blob(input1, rank, x_shape, true);
                hct_fill_shape_from_blob(input2, rank, y_shape, true);
                if (!hct_checked_shape_bytes(cond_shape, rank, sizeof(bool), input0->byte_length, &operand_bytes_needed) ||
                    !hct_checked_shape_bytes(x_shape, rank, hct_dtype_size_bytes(input1->dtype), input1->byte_length, &operand_bytes_needed) ||
                    !hct_checked_shape_bytes(y_shape, rank, hct_dtype_size_bytes(input2->dtype), input2->byte_length, &operand_bytes_needed))
                {
                    return ARM_CMSIS_NN_ARG_ERROR;
                }
            }
            params.rank = rank;
            params.output_shape = output_shape;
            params.cond_strides = cond_strides;
            params.x_strides = x_strides;
            params.y_strides = y_strides;
            if (session->expected_kernel_id == HCT_KERNEL_ID_SELECT_V2_S16)
            {
                return arm_select_v2_s16((const bool *)blob_ptr(session, input0),
                                         (const int16_t *)blob_ptr(session, input1),
                                         (const int16_t *)blob_ptr(session, input2),
                                         &params,
                                         (int16_t *)hct_output_ptr(session));
            }
            return arm_select_v2_s8((const bool *)blob_ptr(session, input0),
                                    (const int8_t *)blob_ptr(session, input1),
                                    (const int8_t *)blob_ptr(session, input2),
                                    &params,
                                    (int8_t *)hct_output_ptr(session));
        }

        case HCT_KERNEL_ID_REVERSE_SEQUENCE_S8:
        case HCT_KERNEL_ID_REVERSE_SEQUENCE_S16:
        {
            cmsis_nn_reverse_sequence_params params;
            int32_t shape[4] = {1, 1, 1, 1};
            int32_t rank;
            if (input0 == NULL || input1 == NULL || !hct_validate_meta0_blob(meta_blob, 3u, -1, -1))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            rank = meta[0];
            if (rank < 1 || rank > 4)
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            hct_fill_shape_from_blob(input0, rank, shape, false);
            params.rank = rank;
            params.shape = shape;
            params.seq_dim = meta[1];
            params.batch_dim = meta[2];
            session->output_length = input0->byte_length;
            if (session->output_length > session->output_capacity_bytes)
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            {
                /* seq_lengths (input1) sizing is batch_dim-relative and not independently
                 * confirmed here -- only the primary operand (input0) is bounds-checked. */
                uint32_t operand_bytes_needed;
                if (!hct_checked_shape_bytes(shape, rank, hct_dtype_size_bytes(input0->dtype),
                                             input0->byte_length, &operand_bytes_needed))
                {
                    return ARM_CMSIS_NN_ARG_ERROR;
                }
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_REVERSE_SEQUENCE_S16)
            {
                return arm_reverse_sequence_s16((const int16_t *)blob_ptr(session, input0),
                                                (const int32_t *)blob_ptr(session, input1),
                                                &params,
                                                (int16_t *)hct_output_ptr(session));
            }
            return arm_reverse_sequence_s8((const int8_t *)blob_ptr(session, input0),
                                           (const int32_t *)blob_ptr(session, input1),
                                           &params,
                                           (int8_t *)hct_output_ptr(session));
        }

        case HCT_KERNEL_ID_SCATTER_ND_S8:
        case HCT_KERNEL_ID_SCATTER_ND_S16:
        {
            cmsis_nn_scatter_nd_params params;
            if (input0 == NULL || input1 == NULL || !hct_validate_meta0_blob(meta_blob, 4u, -1, -1))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            params.num_updates = meta[0];
            params.index_depth = meta[1];
            params.slice_size = meta[2];
            params.output_size = meta[3];
            if (params.index_depth < 0 || params.index_depth > 4 ||
                !hct_validate_meta0_blob(meta_blob, (size_t)(4 + params.index_depth), -1, -1))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            params.output_strides = &meta[4];
            {
                const int32_t output_shape[1] = {params.output_size};
                if (!hct_checked_shape_bytes(output_shape, 1, hct_dtype_size_bytes(input1->dtype),
                                             session->output_capacity_bytes, &session->output_length))
                {
                    return ARM_CMSIS_NN_ARG_ERROR;
                }
            }
            if (session->output_length != session->output_capacity_bytes)
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            {
                const int32_t indices_shape[2] = {params.num_updates, params.index_depth};
                const int32_t updates_shape[2] = {params.num_updates, params.slice_size};
                uint32_t operand_bytes_needed;
                if (!hct_checked_shape_bytes(indices_shape, 2, sizeof(int32_t), input0->byte_length, &operand_bytes_needed) ||
                    !hct_checked_shape_bytes(updates_shape, 2, hct_dtype_size_bytes(input1->dtype),
                                             input1->byte_length, &operand_bytes_needed))
                {
                    return ARM_CMSIS_NN_ARG_ERROR;
                }
            }
            memset(hct_output_ptr(session), 0, session->output_length);
            if (session->expected_kernel_id == HCT_KERNEL_ID_SCATTER_ND_S16)
            {
                return arm_scatter_nd_s16((const int32_t *)blob_ptr(session, input0),
                                          (const int16_t *)blob_ptr(session, input1),
                                          &params,
                                          (int16_t *)hct_output_ptr(session));
            }
            return arm_scatter_nd_s8((const int32_t *)blob_ptr(session, input0),
                                     (const int8_t *)blob_ptr(session, input1),
                                     &params,
                                     (int8_t *)hct_output_ptr(session));
        }

        case HCT_KERNEL_ID_BROADCAST_TO_S8:
        case HCT_KERNEL_ID_BROADCAST_TO_S16:
        {
            cmsis_nn_broadcast_to_params params;
            int32_t input_shape[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
            int32_t output_shape[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
            int32_t rank;
            const cmsis_nn_broadcast_to_params *params_ptr;
            if (input0 == NULL || !hct_validate_meta0_blob(meta_blob, 1u, -1, -1))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            rank = meta[0];
            if (!expects_exact_status(session) && (rank < 1 || rank > 4))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            hct_fill_shape_from_blob(input0, rank, input_shape, true);
            hct_fill_output_shape_from_session(session, rank, output_shape);
            if (!expects_exact_status(session))
            {
                if (!hct_checked_shape_bytes(output_shape, rank, hct_dtype_size_bytes(input0->dtype),
                                             session->output_capacity_bytes, &session->output_length))
                {
                    return ARM_CMSIS_NN_ARG_ERROR;
                }
                if (!null_arg_requested(session, HCT_NULL_ARG_INPUT0_BIT))
                {
                    uint32_t operand_bytes_needed;
                    if (!hct_checked_shape_bytes(input_shape, rank, hct_dtype_size_bytes(input0->dtype),
                                                 input0->byte_length, &operand_bytes_needed))
                    {
                        return ARM_CMSIS_NN_ARG_ERROR;
                    }
                }
            }
            params.rank = rank;
            params.input_shape = input_shape;
            params.output_shape = output_shape;
            params_ptr = null_arg_requested(session, HCT_NULL_ARG_PARAMS_BIT) ? NULL : &params;
            if (session->expected_kernel_id == HCT_KERNEL_ID_BROADCAST_TO_S16)
            {
                return arm_broadcast_to_s16(
                    null_arg_requested(session, HCT_NULL_ARG_INPUT0_BIT) ? NULL : (const int16_t *)blob_ptr(session, input0),
                    params_ptr,
                    null_arg_requested(session, HCT_NULL_ARG_OUTPUT_BIT) ? NULL : (int16_t *)hct_output_ptr(session)
                );
            }
            return arm_broadcast_to_s8(
                null_arg_requested(session, HCT_NULL_ARG_INPUT0_BIT) ? NULL : (const int8_t *)blob_ptr(session, input0),
                params_ptr,
                null_arg_requested(session, HCT_NULL_ARG_OUTPUT_BIT) ? NULL : (int8_t *)hct_output_ptr(session)
            );
        }

        case HCT_KERNEL_ID_DYNAMIC_UPDATE_SLICE_S8:
        case HCT_KERNEL_ID_DYNAMIC_UPDATE_SLICE_S16:
        {
            cmsis_nn_dynamic_update_slice_params params;
            int32_t operand_shape[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
            int32_t update_shape[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
            int32_t operand_strides[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
            int32_t rank;
            const cmsis_nn_dynamic_update_slice_params *params_ptr;
            if (input0 == NULL || input1 == NULL || input2 == NULL || !hct_validate_meta0_blob(meta_blob, 1u, -1, -1))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            rank = meta[0];
            if (!expects_exact_status(session) && (rank < 1 || rank > 4))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            hct_fill_shape_from_blob(input0, rank, operand_shape, false);
            hct_fill_shape_from_blob(input1, rank, update_shape, true);
            hct_row_major_strides(operand_shape, rank, operand_strides);
            params.rank = rank;
            params.operand_shape = operand_shape;
            params.update_shape = update_shape;
            params.operand_size = (int32_t)hct_blob_element_count(input0);
            params.update_size = (int32_t)hct_blob_element_count(input1);
            params.operand_strides = operand_strides;
            if (!expects_exact_status(session))
            {
                session->output_length = input0->byte_length;
                if (session->output_length > session->output_capacity_bytes)
                {
                    return ARM_CMSIS_NN_ARG_ERROR;
                }
                if (!null_arg_requested(session, HCT_NULL_ARG_INPUT0_BIT) &&
                    !null_arg_requested(session, HCT_NULL_ARG_INPUT1_BIT) &&
                    !null_arg_requested(session, HCT_NULL_ARG_INPUT2_BIT))
                {
                    uint32_t operand_bytes_needed;
                    if (!hct_checked_shape_bytes(operand_shape, rank, hct_dtype_size_bytes(input0->dtype),
                                                 input0->byte_length, &operand_bytes_needed) ||
                        !hct_checked_shape_bytes(update_shape, rank, hct_dtype_size_bytes(input1->dtype),
                                                 input1->byte_length, &operand_bytes_needed) ||
                        !hct_checked_count_bytes(rank, sizeof(int32_t), input2->byte_length, &operand_bytes_needed))
                    {
                        return ARM_CMSIS_NN_ARG_ERROR;
                    }
                }
            }
            params_ptr = null_arg_requested(session, HCT_NULL_ARG_PARAMS_BIT) ? NULL : &params;
            if (session->expected_kernel_id == HCT_KERNEL_ID_DYNAMIC_UPDATE_SLICE_S16)
            {
                return arm_dynamic_update_slice_s16(
                    null_arg_requested(session, HCT_NULL_ARG_INPUT0_BIT) ? NULL : (const int16_t *)blob_ptr(session, input0),
                    null_arg_requested(session, HCT_NULL_ARG_INPUT1_BIT) ? NULL : (const int16_t *)blob_ptr(session, input1),
                    null_arg_requested(session, HCT_NULL_ARG_INPUT2_BIT) ? NULL : (const int32_t *)blob_ptr(session, input2),
                    params_ptr,
                    null_arg_requested(session, HCT_NULL_ARG_OUTPUT_BIT) ? NULL : (int16_t *)hct_output_ptr(session)
                );
            }
            return arm_dynamic_update_slice_s8(
                null_arg_requested(session, HCT_NULL_ARG_INPUT0_BIT) ? NULL : (const int8_t *)blob_ptr(session, input0),
                null_arg_requested(session, HCT_NULL_ARG_INPUT1_BIT) ? NULL : (const int8_t *)blob_ptr(session, input1),
                null_arg_requested(session, HCT_NULL_ARG_INPUT2_BIT) ? NULL : (const int32_t *)blob_ptr(session, input2),
                params_ptr,
                null_arg_requested(session, HCT_NULL_ARG_OUTPUT_BIT) ? NULL : (int8_t *)hct_output_ptr(session)
            );
        }

        case HCT_KERNEL_ID_STRIDED_SLICE_S8:
        case HCT_KERNEL_ID_STRIDED_SLICE_S16:
        case HCT_KERNEL_ID_STRIDED_SLICE_S32:
        {
            cmsis_nn_dims input_dims;
            cmsis_nn_dims output_dims;
            cmsis_nn_dims begin_dims;
            cmsis_nn_dims stride_dims;
            uint32_t element_size;
            if (input0 == NULL || !hct_validate_meta0_blob(meta_blob, 8u, -1, -1))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            hct_fill_dims_from_blob(input0, &input_dims);
            hct_fill_output_dims_from_session(session, &output_dims);
            begin_dims.n = meta[0];
            begin_dims.h = meta[1];
            begin_dims.w = meta[2];
            begin_dims.c = meta[3];
            stride_dims.n = meta[4];
            stride_dims.h = meta[5];
            stride_dims.w = meta[6];
            stride_dims.c = meta[7];
            element_size = hct_dtype_size_bytes(input0->dtype);
            if (!hct_checked_dims_bytes(&output_dims, element_size,
                                        session->output_capacity_bytes, &session->output_length))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            {
                uint32_t operand_bytes_needed;
                if (!hct_checked_dims_bytes(&input_dims, element_size, input0->byte_length, &operand_bytes_needed))
                {
                    return ARM_CMSIS_NN_ARG_ERROR;
                }
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_STRIDED_SLICE_S16)
            {
                return arm_strided_slice_s16((const int16_t *)blob_ptr(session, input0),
                                             (int16_t *)hct_output_ptr(session),
                                             &input_dims,
                                             &begin_dims,
                                             &stride_dims,
                                             &output_dims);
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_STRIDED_SLICE_S32)
            {
                return arm_strided_slice_s32((const int32_t *)blob_ptr(session, input0),
                                             (int32_t *)hct_output_ptr(session),
                                             &input_dims,
                                             &begin_dims,
                                             &stride_dims,
                                             &output_dims);
            }
            return arm_strided_slice_s8((const int8_t *)blob_ptr(session, input0),
                                        (int8_t *)hct_output_ptr(session),
                                        &input_dims,
                                        &begin_dims,
                                        &stride_dims,
                                        &output_dims);
        }

        case HCT_KERNEL_ID_STRIDED_SLICE_F32:
        case HCT_KERNEL_ID_STRIDED_SLICE_F16:
        {
            cmsis_nn_dims input_dims;
            cmsis_nn_dims output_dims;
            cmsis_nn_dims begin_dims;
            cmsis_nn_dims stride_dims;
            uint32_t element_size;
            if (input0 == NULL || !hct_validate_meta0_blob(meta_blob, 8u, -1, -1))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            hct_fill_dims_from_blob(input0, &input_dims);
            hct_fill_output_dims_from_session(session, &output_dims);
            begin_dims.n = meta[0];
            begin_dims.h = meta[1];
            begin_dims.w = meta[2];
            begin_dims.c = meta[3];
            stride_dims.n = meta[4];
            stride_dims.h = meta[5];
            stride_dims.w = meta[6];
            stride_dims.c = meta[7];
            element_size = hct_dtype_size_bytes(input0->dtype);
            if (!hct_checked_dims_bytes(&output_dims, element_size,
                                        session->output_capacity_bytes, &session->output_length))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            {
                uint32_t operand_bytes_needed;
                if (!hct_checked_dims_bytes(&input_dims, element_size, input0->byte_length, &operand_bytes_needed))
                {
                    return ARM_CMSIS_NN_ARG_ERROR;
                }
            }
            if (session->expected_kernel_id == HCT_KERNEL_ID_STRIDED_SLICE_F16)
            {
                return arm_strided_slice_f16((const float16_t *)blob_ptr(session, input0),
                                            (float16_t *)hct_output_ptr(session),
                                            &input_dims,
                                            &begin_dims,
                                            &stride_dims,
                                            &output_dims);
            }
            return arm_strided_slice_f32((const float *)blob_ptr(session, input0),
                                        (float *)hct_output_ptr(session),
                                        &input_dims,
                                        &begin_dims,
                                        &stride_dims,
                                        &output_dims);
        }

        default:
            return ARM_CMSIS_NN_ARG_ERROR;
    }
}
'''


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
            "activation_min", "activation_max", "float_activation_min_bits", "float_activation_max_bits", "ch_mult",
        ),
        c_body=_RUN_DEPTHWISE_CONV_ONCE,
    ),
    FirmwareAdapterSpec(
        label="PoolingFunctions/AvgPool,MaxPool",
        function_name="run_pooling_once",
        guard="HCT_HOST_ABS_ONLY",
        scalar_fields=(
            "stride_h", "stride_w", "pad_h", "pad_w",
            "output_h", "output_w", "output_c",
            "activation_min", "activation_max", "pool_h", "pool_w",
            "float_activation_min_bits", "float_activation_max_bits",
        ),
        c_body=_RUN_POOLING_ONCE,
    ),
    FirmwareAdapterSpec(
        label="ActivationFunctions/Relu,Relu6,Clamp,LeakyRelu,Logistic,Tanh,HardSwishCompat,HardSwishPrecise",
        function_name="run_activation_once",
        guard="HCT_HOST_ABS_ONLY",
        scalar_fields=(
            "output_h", "output_w", "output_c", "input_offset", "output_offset",
            "activation_min", "activation_max", "out_mult", "out_shift",
            "out_mult_alpha", "out_shift_alpha", "out_mult_fp", "out_mult_exp",
            "relu_mult_fp", "relu_mult_exp", "relu_q3", "relu_q6", "prescale",
            "input_mult", "input_left_shift",
        ),
        c_body=_RUN_ACTIVATION_ONCE,
    ),
    FirmwareAdapterSpec(
        label="ActivationFunctions/PReLU,PReLUScalar",
        function_name="run_prelu_once",
        guard="HCT_HOST_ABS_ONLY",
        scalar_fields=(
            "output_h", "output_w", "output_c", "input_offset", "output_offset",
            "out_mult", "out_shift", "out_mult_alpha", "out_shift_alpha",
            "alpha_offset", "block_size",
        ),
        c_body=_RUN_PRELU_ONCE,
    ),
    FirmwareAdapterSpec(
        label="ActivationFunctions/NNActivationFloat",
        function_name="run_nn_activation_float_once",
        guard="HCT_HOST_ABS_ONLY",
        scalar_fields=("block_size", "activation_kind", "scale_bits"),
        c_body=_RUN_NN_ACTIVATION_FLOAT_ONCE,
    ),
    FirmwareAdapterSpec(
        label="QuantizationFunctions/Quantize",
        function_name="run_quantize_once",
        guard="HCT_HOST_ABS_ONLY",
        scalar_fields=("output_h", "output_w", "output_c", "output_offset", "scale_bits"),
        c_body=_RUN_QUANTIZE_ONCE,
    ),
    FirmwareAdapterSpec(
        label="QuantizationFunctions/Dequantize",
        function_name="run_dequantize_once",
        guard="HCT_HOST_ABS_ONLY",
        scalar_fields=("output_h", "output_w", "output_c", "input_offset", "scale_bits", "activation_kind"),
        c_body=_RUN_DEQUANTIZE_ONCE,
    ),
    FirmwareAdapterSpec(
        label="NNSupportFunctions/Requantize",
        function_name="run_requantize_once",
        guard="HCT_HOST_ABS_ONLY",
        scalar_fields=("input_offset", "output_offset", "out_mult", "out_shift"),
        c_body=_RUN_REQUANTIZE_ONCE,
    ),
    FirmwareAdapterSpec(
        label="ComparisonFunctions/Equal,NotEqual,Greater,GreaterEqual,Less,LessEqual",
        function_name="run_comparison_once",
        guard="HCT_HOST_ABS_ONLY",
        scalar_fields=(
            "output_n", "output_h", "output_w", "output_c",
            "input1_offset", "input1_mult", "input1_shift",
            "input2_offset", "input2_mult", "input2_shift",
            "left_shift",
        ),
        c_body=_RUN_COMPARISON_ONCE,
    ),
    FirmwareAdapterSpec(
        label="ConvolutionFunctions/TransposeConv",
        function_name="run_transpose_conv_once",
        guard="HCT_HOST_ABS_ONLY",
        scalar_fields=(
            "stride_h", "stride_w", "pad_h", "pad_w", "pad_offset_h", "pad_offset_w",
            "output_n", "output_h", "output_w", "output_c",
            "dilation_h", "dilation_w",
            "input_offset", "output_offset", "activation_min", "activation_max",
            "float_activation_min_bits", "float_activation_max_bits",
        ),
        c_body=_RUN_TRANSPOSE_CONV_ONCE,
    ),
    FirmwareAdapterSpec(
        label="SoftmaxFunctions/Softmax,SoftmaxS8S16",
        function_name="run_softmax_once",
        guard="HCT_HOST_ABS_ONLY",
        scalar_fields=("num_rows", "row_size", "out_mult", "out_shift", "diff_min"),
        c_body=_RUN_SOFTMAX_ONCE,
    ),
    FirmwareAdapterSpec(
        label="FullyConnectedFunctions/FullyConnected",
        function_name="run_fully_connected_once",
        guard="HCT_HOST_ABS_ONLY",
        scalar_fields=("input_offset", "filter_offset", "output_offset", "activation_min", "activation_max", "float_activation_min_bits", "float_activation_max_bits"),
        c_body=_RUN_FULLY_CONNECTED_ONCE,
    ),
    FirmwareAdapterSpec(
        label="FullyConnectedFunctions/BatchMatMul",
        function_name="run_batch_matmul_once",
        guard="HCT_HOST_ABS_ONLY",
        scalar_fields=("input_offset", "filter_offset", "output_offset", "activation_min", "activation_max", "float_activation_min_bits", "float_activation_max_bits", "out_mult", "out_shift", "adj_x", "adj_y", "output_n", "output_h", "output_w", "output_c"),
        c_body=_RUN_BATCH_MATMUL_ONCE,
    ),
    FirmwareAdapterSpec(
        label="BasicMathFunctions/ArgMax,ArgMin,Mean,ReduceMax,ReduceMin",
        function_name="run_basic_math_reduction_once",
        guard="HCT_HOST_ABS_ONLY",
        scalar_fields=(
            "output_n", "output_h", "output_w", "output_c",
            "axis_n", "axis_h", "axis_w", "axis_c", "axis",
            "input_offset", "output_offset", "out_mult", "out_shift",
        ),
        c_body=_RUN_BASIC_MATH_REDUCTION_ONCE,
    ),
    FirmwareAdapterSpec(
        label="BasicMathFunctions/ReduceSum",
        function_name="run_reduce_sum_once",
        guard="HCT_HOST_ABS_ONLY",
        scalar_fields=("output_n", "output_h", "output_w", "output_c", "axis_n", "axis_h", "axis_w", "axis_c"),
        c_body=_RUN_REDUCE_SUM_ONCE,
    ),
    FirmwareAdapterSpec(
        label="BasicMathFunctions/Sqrt,Rsqrt",
        function_name="run_basic_math_lut_once",
        guard="HCT_HOST_ABS_ONLY",
        scalar_fields=(
            "input_offset", "output_offset", "out_mult", "out_shift",
            "needs_rescale", "activation_min", "activation_max",
        ),
        c_body=_RUN_BASIC_MATH_LUT_ONCE,
    ),
    FirmwareAdapterSpec(
        label="NNSupportFunctions/BatchNorm",
        function_name="run_batch_norm_once",
        guard="HCT_HOST_ABS_ONLY",
        scalar_fields=("activation_kind",),
        c_body=_RUN_BATCH_NORM_ONCE,
    ),
    FirmwareAdapterSpec(
        label="BasicMathFunctions/Add,Sub,Mul,Maximum,Minimum,SquaredDifference",
        function_name="run_elementwise_binary_once",
        guard="HCT_HOST_ABS_ONLY",
        scalar_fields=(
            "output_n", "output_h", "output_w", "output_c",
            "input1_offset", "input1_mult", "input1_shift",
            "input2_offset", "input2_mult", "input2_shift",
            "left_shift", "output_offset", "out_mult", "out_shift",
            "activation_min", "activation_max",
        ),
        c_body=_RUN_ELEMENTWISE_BINARY_ONCE,
    ),
    FirmwareAdapterSpec(
        label="Phase3e data movement/indexing families",
        function_name="run_data_movement_once",
        guard="HCT_HOST_ABS_ONLY",
        scalar_fields=("output_n", "output_h", "output_w", "output_c", "null_arg_mask"),
        c_body=_RUN_DATA_MOVEMENT_ONCE,
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
