#ifndef HCT_BENCHMARK_SERVER_BLOB_TYPES_H
#define HCT_BENCHMARK_SERVER_BLOB_TYPES_H

/* Wire codes for CASE_META blob roles and dtypes. Kept free of CMSIS-NN so the
 * generated catalog (per-kernel blob contracts) can use them. */

#define HCT_BLOB_ROLE_UNKNOWN 0u
#define HCT_BLOB_ROLE_INPUT_0 1u
#define HCT_BLOB_ROLE_WEIGHTS 2u
#define HCT_BLOB_ROLE_BIAS 3u
#define HCT_BLOB_ROLE_MULTIPLIER 4u
#define HCT_BLOB_ROLE_SHIFT 5u
#define HCT_BLOB_ROLE_INPUT_1 6u
#define HCT_BLOB_ROLE_INPUT_2 7u
#define HCT_BLOB_ROLE_META_0 8u
#define HCT_BLOB_ROLE_COUNT 9u

#define HCT_DTYPE_UNKNOWN 0u
#define HCT_DTYPE_S8 1u
#define HCT_DTYPE_S32 2u
#define HCT_DTYPE_S16 3u
#define HCT_DTYPE_S64 4u
#define HCT_DTYPE_BOOL 5u
#define HCT_DTYPE_F32 6u
#define HCT_DTYPE_F16 7u
/* S4 weight blobs are packed 2 nibbles/byte on the wire (same convention the host
 * uses -- see generated_test_bridge.py's expected_weight_bytes) and are never used
 * as an activation dtype, so they never flow through hct_dtype_size_bytes()/
 * hct_blob_element_count() -- weights blob sizing always comes from the wire's own
 * byte_length/alignment fields (see allocate_blob()), not a per-dtype element size. */
#define HCT_DTYPE_S4 8u

#endif
