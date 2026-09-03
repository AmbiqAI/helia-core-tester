# Issue #53, item 2: mutation-test the previously-untested op families

## Methodology

For each of the operator families the original audit never mutated
(DepthwiseConv was already covered in a prior pass), a single targeted bug
was injected into the FP16 CMSIS-NN kernel, the family's float suite was
run via `helia_core_tester full --op <Family> --cpu cortex-m55 --suite
float --float-precision both` on the `ghcr.io/ambiqai/ns-cmsis-nn-ci:latest`
FVP image, and the mutation was reverted immediately after observing the
result. Baseline (unmutated) runs for all 9 families were confirmed passing
first.

## Results

| Family | Kernel mutated | Mutation | Baseline | Mutated |
|---|---|---|---|---|
| TransposeConv | `arm_transpose_conv_f16.c` | Disabled bias add (`if (bias_data)` -> `if (0 && bias_data)`) | 387 pass | **4 failed** |
| AvgPool | `arm_avg_pool_f16.c` | Halved the MVE reciprocal-count scale factor | 387 pass | **3 failed** |
| MaxPool | `arm_max_pool_f16.c` | Replaced MVE `vmaxnmq` reduction with `vaddq` | 387 pass | **2 failed** |
| Transpose | `arm_transpose_f16.c` | Scalar fallback reads `in_row[(c+1) % cols]` instead of `in_row[c]` | 387 pass | **3 failed** |
| Minimum | `arm_minmax_common_f16.c` (shared) | Flipped scalar-path comparison (`>=` -> `<=`) | 387 pass | **3 failed** |
| Maximum | `arm_minmax_common_f16.c` (shared) | Same shared mutation as Minimum | 387 pass | **3 failed** |
| Pad | `arm_pad_f16.c` | Halved the leading batch-pad `memset` extent | 387 pass | **2 failed** |
| Reshape | `arm_reshape_f16.c` | `arm_memcpy_f16` copies one element short | 387 pass | **3 failed** |
| SVDF | `arm_svdf_f16.c` | Scaled the state-update accumulator by 0.5 | 387 pass | **3 failed** |

Every mutation was caught by at least one existing float test case at
default tolerances — no family needed a tightened tolerance or a new case
to gain discriminating power. All kernel changes were reverted immediately
after each run; no source changes are retained from this pass.

## Conclusion

This closes item 2 of issue #53's re-scope: DepthwiseConv (done in a prior
pass), TransposeConv, pooling (Avg+Max), Transpose, Min/Max, Pad, Reshape,
and SVDF all now have confirmed real discriminating power at their current
float default tolerances.
