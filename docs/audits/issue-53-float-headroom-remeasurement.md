# Issue #53, item 1: post-#54 float tolerance headroom re-measurement

## Methodology

Full FVP float suite run on `ghcr.io/ambiqai/ns-cmsis-nn-ci:latest`
(Corstone-300), at `main` `a9c4233` (post-#78, which added the
`max_diff`/`max_tolerance_fraction` measurement primitive this report
depends on):

```
helia_core_tester full --cpu cortex-m55 --suite float --float-precision both --report-formats json
helia_core_tester full --cpu cortex-m4  --suite float --float-precision f32   --report-formats json
```

(cortex-m4 does not support FP16 execution, so it is run f32-only and
cortex-m55 covers both f16 and f32; together this is every float case in
both configurations that can run it.) All 579 (cpu, case) invocations
**passed**. `max_tolerance_fraction` is the worst-case ratio of measured
error to the case's configured `atol`/`rtol` budget across all validated
tensors (1.0 = right at the boundary, values above 1.0 would have failed).

This directly answers item 1 of the re-scoped issue: of the 387 float
cases, 369 inherit the typed default comparison (no hand-set
`comparison` block) and only began performing a *real* float comparison
after #54 closed. All 369 now have a measured headroom number below.
The remaining 18 carry explicit overrides and already had measured
justifications from prior audit passes.

## Results

- Total (cpu, case) results: 579
- With a real float comparison (max_diff/max_tolerance_fraction populated): 548
- No float comparison performed (status-only / error-injection cases, no output to compare): 31

## Distribution of tolerance-fraction used (0 = exact match, 1 = right at the tolerance boundary)

- min=0.0000 median=0.0117 p90=0.3575 max=0.9756

## Cases using >50% of their configured tolerance (33 of 548)

| cpu | case | frac | atol | rtol |
|---|---|---|---|---|
| cortex-m55 | depthwise_conv_float_wrapper_f16 | 0.976 | 0.001 | 0.001 |
| cortex-m55 | convolve_float_1xn_direct_k5_valid_oc4_f16 | 0.968 | 0.001 | 0.001 |
| cortex-m55 | convolve_float_1x1_patch_gemm_packed_f16 | 0.910 | 0.001 | 0.001 |
| cortex-m55 | convolve_float_1x1_standard_large_f16 | 0.910 | 0.001 | 0.001 |
| cortex-m55 | depthwise_conv_float_3x3_valid_stride1_f16 | 0.895 | 0.001 | 0.001 |
| cortex-m55 | depthwise_conv_float_single_input_channel_f16 | 0.766 | 0.001 | 0.001 |
| cortex-m55 | avg_pool_float_rect_kernel_f16 | 0.756 | 0.001 | 0.001 |
| cortex-m55 | convolve_float_stride2_valid_f16 | 0.739 | 0.001 | 0.001 |
| cortex-m55 | convolve_float_1x1_patch_gemm_packed_no_bias_f16 | 0.721 | 0.001 | 0.001 |
| cortex-m55 | transpose_conv_float_stride1_f16 | 0.702 | 0.001 | 0.001 |
| cortex-m55 | depthwise_conv_float_single_input_channel_out8_f16 | 0.623 | 0.001 | 0.001 |
| cortex-m55 | convolve_float_default_f16 | 0.620 | 0.001 | 0.001 |
| cortex-m55 | convolve_float_wrapper_f16 | 0.620 | 0.001 | 0.001 |
| cortex-m55 | batch_matmul_float_vector8_f16 | 0.605 | 0.001 | 0.001 |
| cortex-m55 | convolve_float_1xn_direct_k4_valid_f16 | 0.600 | 0.001 | 0.001 |
| cortex-m55 | convolve_float_1xn_generic_k4_valid_f16 | 0.600 | 0.001 | 0.001 |
| cortex-m55 | nn_activation_float_tanh_lutband_f32 | 0.599 | 2e-05 | 0.0 |
| cortex-m55 | nn_activation_float_tanh_lutband_neg_f32 | 0.599 | 2e-05 | 0.0 |
| cortex-m4 | nn_activation_float_tanh_lutband_f32 | 0.599 | 2e-05 | 0.0 |
| cortex-m4 | nn_activation_float_tanh_lutband_neg_f32 | 0.599 | 2e-05 | 0.0 |
| cortex-m55 | depthwise_conv_float_default_f16 | 0.592 | 0.001 | 0.001 |
| cortex-m55 | convolve_float_1d_k3_packed_no_bias_f16 | 0.590 | 0.001 | 0.001 |
| cortex-m55 | convolve_float_1d_k5_packed_f16 | 0.586 | 0.001 | 0.001 |
| cortex-m55 | convolve_float_1x1_same_f16 | 0.583 | 0.001 | 0.001 |
| cortex-m55 | convolve_float_1x1_valid_f16 | 0.583 | 0.001 | 0.001 |
| cortex-m55 | depthwise_conv_float_depth_multiplier2_f16 | 0.579 | 0.001 | 0.001 |
| cortex-m55 | convolve_float_patch_gemm_f16 | 0.578 | 0.002 | 0.002 |
| cortex-m55 | convolve_float_dilated_valid_f16 | 0.563 | 0.001 | 0.001 |
| cortex-m55 | depthwise_conv_float_fast_nt_t_f16 | 0.556 | 0.001 | 0.001 |
| cortex-m55 | batch_matmul_float_tail9_f16 | 0.554 | 0.001 | 0.001 |
| cortex-m55 | gru_unidirectional_float_pre_reset_multi_batch_f16 | 0.538 | 0.01 | 0.01 |
| cortex-m55 | convolve_float_1xn_direct_k3_valid_oc4_f16 | 0.520 | 0.001 | 0.001 |
| cortex-m55 | convolve_float_1xn_direct_k3_valid_tail_f16 | 0.518 | 0.001 | 0.001 |

## Skipped (no float comparison target)

- gru_unidirectional_error_missing_temp1_prereset_f16
- gru_unidirectional_error_missing_temp1_prereset_f32
- gru_unidirectional_error_negative_hidden_size_f16
- gru_unidirectional_error_negative_hidden_size_f32
- gru_unidirectional_error_negative_input_size_f16
- gru_unidirectional_error_negative_input_size_f32
- gru_unidirectional_error_negative_time_steps_f16
- gru_unidirectional_error_negative_time_steps_f32
- gru_unidirectional_error_null_input_f16
- gru_unidirectional_error_null_input_f32
- gru_unidirectional_error_null_output_f16
- gru_unidirectional_error_null_output_f32
- gru_unidirectional_error_null_params_f16
- gru_unidirectional_error_null_params_f32
- gru_unidirectional_error_stateful_batch_gt1_f16
- gru_unidirectional_error_stateful_batch_gt1_f32
- gru_unidirectional_error_zero_batch_size_f16
- gru_unidirectional_error_zero_batch_size_f32
- quantize_float_f32_s16_vec33
- quantize_float_f32_s8_vec17

## Conclusion

No case anywhere in the 387-case float suite exceeds its configured
tolerance budget (max observed fraction is 0.976, still under 1.0). The
tightest headroom is concentrated in FP16 convolution-family kernels
(DepthwiseConv, Convolve, TransposeConv) at the default `1e-3/1e-3`
FP16 tolerance, and in the FP32 `tanh_lutband` cases at the tight
`2e-5/0` explicit override — both are close to their budget but pass
cleanly with real margin at the FVP measurement precision available
here. Raw per-case data (all 579 rows) is in
`issue-53-float-headroom-raw.csv` alongside this report.

This closes item 1 of issue #53's re-scope. Item 2 (mutation-testing
the previously-untested op families) and item 3/4 (tanh_f16 blind spot)
remain open follow-up work.
