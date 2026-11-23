/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Record mode (HCT_RECORD): the harness runs on the host against a library's
 * scalar build and, instead of comparing outputs with its golden, writes every
 * validated output and returned status to a capture file named by the
 * HCT_RECORD_PATH environment variable. helia_core_tester/record/capture.py
 * reads it and is the format's specification:
 *
 *   header   "HCTR" u16 version u16 reserved
 *   record   u8 kind, u8 dtype, u16 elem_size, u16 name_len, name bytes, body
 *     tensor       u8 mode, u8 flags, u8 ndim, u8 reserved, u32 dims[ndim],
 *                  i64 tolerance, f64 atol, f64 rtol, u32 nbytes,
 *                  actual[nbytes], golden[nbytes], mask[count] when flags & 1
 *     status       i32 status, i32 expected
 *     unrecordable (no body): the golden named is compared in a way record mode
 *                  cannot capture
 *   end      u8 0xFF, u32 record_count, u32 crc32 of every byte before it
 *
 * All integers are little-endian. A run that dies before
 * hct_record_finish() leaves no end record, which the reader rejects.
 */
#ifndef HCT_RECORD_H
#define HCT_RECORD_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    HCT_RECORD_KIND_TENSOR = 1,
    HCT_RECORD_KIND_STATUS = 2,
    HCT_RECORD_KIND_UNRECORDABLE = 3,
    HCT_RECORD_KIND_END = 0xFF,
};

enum {
    HCT_RECORD_DTYPE_UNKNOWN = 0,
    HCT_RECORD_DTYPE_I8 = 1,
    HCT_RECORD_DTYPE_U8 = 2,
    HCT_RECORD_DTYPE_I16 = 3,
    HCT_RECORD_DTYPE_U16 = 4,
    HCT_RECORD_DTYPE_I32 = 5,
    HCT_RECORD_DTYPE_U32 = 6,
    HCT_RECORD_DTYPE_I64 = 7,
    HCT_RECORD_DTYPE_U64 = 8,
    HCT_RECORD_DTYPE_F16 = 9,
    HCT_RECORD_DTYPE_F32 = 10,
    HCT_RECORD_DTYPE_F64 = 11,
    HCT_RECORD_DTYPE_BOOL = 12,
};

enum {
    HCT_RECORD_MODE_EXACT = 1,
    HCT_RECORD_MODE_TOLERANT = 2,
    HCT_RECORD_MODE_FLOAT = 3,
    HCT_RECORD_MODE_BOOL = 4,
};

#define HCT_RECORD_FLAG_MASK 1u

void hct_record_tensor(const char *name,
                       int dtype,
                       size_t elem_size,
                       int mode,
                       long long count,
                       const void *actual,
                       const void *golden,
                       const uint8_t *mask,
                       long long tolerance,
                       double atol,
                       double rtol);
void hct_record_status(const char *label, int status, int expected);
void hct_record_unrecordable(const char *golden);
void hct_record_finish(void);

#ifdef __cplusplus
}
#endif

#define HCT_RECORD_SIGNED_DTYPE(size) \
    ((size) == 1 ? HCT_RECORD_DTYPE_I8 : (size) == 2 ? HCT_RECORD_DTYPE_I16 \
     : (size) == 4 ? HCT_RECORD_DTYPE_I32 : (size) == 8 ? HCT_RECORD_DTYPE_I64 : HCT_RECORD_DTYPE_UNKNOWN)
#define HCT_RECORD_UNSIGNED_DTYPE(size) \
    ((size) == 1 ? HCT_RECORD_DTYPE_U8 : (size) == 2 ? HCT_RECORD_DTYPE_U16 \
     : (size) == 4 ? HCT_RECORD_DTYPE_U32 : (size) == 8 ? HCT_RECORD_DTYPE_U64 : HCT_RECORD_DTYPE_UNKNOWN)

#define HCT_RECORD_DTYPE_OF(element) \
    _Generic((element), \
        HELIA_FLOAT16_GENERIC_ROWS(HCT_RECORD_DTYPE_F16) \
        float: HCT_RECORD_DTYPE_F32, \
        double: HCT_RECORD_DTYPE_F64, \
        _Bool: HCT_RECORD_DTYPE_BOOL, \
        char: HCT_RECORD_SIGNED_DTYPE(sizeof(char)), \
        signed char: HCT_RECORD_DTYPE_I8, \
        unsigned char: HCT_RECORD_DTYPE_U8, \
        short: HCT_RECORD_SIGNED_DTYPE(sizeof(short)), \
        unsigned short: HCT_RECORD_UNSIGNED_DTYPE(sizeof(short)), \
        int: HCT_RECORD_SIGNED_DTYPE(sizeof(int)), \
        unsigned int: HCT_RECORD_UNSIGNED_DTYPE(sizeof(int)), \
        long: HCT_RECORD_SIGNED_DTYPE(sizeof(long)), \
        unsigned long: HCT_RECORD_UNSIGNED_DTYPE(sizeof(long)), \
        long long: HCT_RECORD_SIGNED_DTYPE(sizeof(long long)), \
        unsigned long long: HCT_RECORD_UNSIGNED_DTYPE(sizeof(long long)), \
        default: HCT_RECORD_DTYPE_UNKNOWN)

/* The golden's C name is the record's name, so the reader can pair it with the header array. */
#define HCT_RECORD_VALIDATED(mode, actual, expected, mask, size, tolerance, atol, rtol) \
    hct_record_tensor(#expected, HCT_RECORD_DTYPE_OF((expected)[0]), sizeof((expected)[0]), (mode), \
                      (long long)(size), (actual), (expected), (const uint8_t *)(mask), (long long)(tolerance), \
                      (double)(atol), (double)(rtol))

#undef HELIA_VALIDATE_EXPECTED_STATUS
#define HELIA_VALIDATE_EXPECTED_STATUS(label, status, expected_status) \
    do { \
        int helia_status = (int)(status); \
        int helia_expected_status = (int)(expected_status); \
        hct_record_status((label), helia_status, helia_expected_status); \
        if (helia_status != helia_expected_status) { \
            if (helia_expected_status == (int)ARM_CMSIS_NN_SUCCESS) { \
                return helia_test_status_failure((label), helia_status); \
            } \
            return helia_test_expected_status_failure((label), helia_status, helia_expected_status); \
        } \
    } while (0)

#undef HELIA_VALIDATE_EXACT_INTS
#define HELIA_VALIDATE_EXACT_INTS(actual, expected, size, max_reports, failures) \
    do { \
        HELIA_ASSERT_INT_VALIDATOR_INPUT(actual); \
        HELIA_ASSERT_INT_VALIDATOR_INPUT(expected); \
        HCT_RECORD_VALIDATED(HCT_RECORD_MODE_EXACT, (actual), expected, NULL, (size), 0, 0.0, 0.0); \
        (void)(max_reports); \
        (void)(failures); \
    } while (0)

#undef HELIA_VALIDATE_TOLERANT_INTS
#define HELIA_VALIDATE_TOLERANT_INTS(actual, expected, size, tolerance, max_reports, failures) \
    do { \
        HELIA_ASSERT_INT_VALIDATOR_INPUT(actual); \
        HELIA_ASSERT_INT_VALIDATOR_INPUT(expected); \
        HCT_RECORD_VALIDATED(HCT_RECORD_MODE_TOLERANT, (actual), expected, NULL, (size), (tolerance), 0.0, 0.0); \
        (void)(max_reports); \
        (void)(failures); \
    } while (0)

#undef HELIA_VALIDATE_FLOATS_MASKED
#define HELIA_VALIDATE_FLOATS_MASKED(actual, expected, mask, n, atol, rtol, max_reports, failures) \
    do { \
        HCT_RECORD_VALIDATED(HCT_RECORD_MODE_FLOAT, (actual), expected, (mask), (n), 0, (atol), (rtol)); \
        (void)(max_reports); \
        (void)(failures); \
    } while (0)

#undef HELIA_VALIDATE_FLOATS
#define HELIA_VALIDATE_FLOATS(actual, expected, size, atol, rtol, max_reports, failures) \
    HELIA_VALIDATE_FLOATS_MASKED((actual), expected, NULL, (size), (atol), (rtol), (max_reports), (failures))

#undef HELIA_VALIDATE_BOOLEANS
#define HELIA_VALIDATE_BOOLEANS(actual, expected, size, max_reports, failures) \
    do { \
        HCT_RECORD_VALIDATED(HCT_RECORD_MODE_BOOL, (actual), expected, NULL, (size), 0, 0.0, 0.0); \
        (void)(max_reports); \
        (void)(failures); \
    } while (0)

/* Per-element bit checks carry no array to capture; the case stays on its existing golden. */
#undef HELIA_VALIDATE_FLOAT_BITS
#define HELIA_VALIDATE_FLOAT_BITS(actual, expected, positive_inf, max_ulp, index, max_reports, failures) \
    do { \
        (void)(actual); \
        (void)(positive_inf); \
        (void)(max_ulp); \
        (void)(index); \
        (void)(max_reports); \
        (void)(failures); \
        hct_record_unrecordable(#expected); \
    } while (0)

/* The forwarding macros expand their operands once more, so they must pass `expected` unparenthesised. */
#undef HELIA_VALIDATE_OUTPUTS_EXACT_INT
#define HELIA_VALIDATE_OUTPUTS_EXACT_INT(actual, expected, size, tolerance, atol, rtol, max_reports, failures) \
    HELIA_VALIDATE_EXACT_INTS((actual), expected, (size), (max_reports), (failures))
#undef HELIA_VALIDATE_OUTPUTS_TOLERANT_INT
#define HELIA_VALIDATE_OUTPUTS_TOLERANT_INT(actual, expected, size, tolerance, atol, rtol, max_reports, failures) \
    HELIA_VALIDATE_TOLERANT_INTS((actual), expected, (size), (tolerance), (max_reports), (failures))
#undef HELIA_VALIDATE_OUTPUTS_FLOAT
#define HELIA_VALIDATE_OUTPUTS_FLOAT(actual, expected, size, tolerance, atol, rtol, max_reports, failures) \
    HELIA_VALIDATE_FLOATS((actual), expected, (size), (atol), (rtol), (max_reports), (failures))
#undef HELIA_VALIDATE_OUTPUTS_BOOL
#define HELIA_VALIDATE_OUTPUTS_BOOL(actual, expected, size, tolerance, atol, rtol, max_reports, failures) \
    HELIA_VALIDATE_BOOLEANS((actual), expected, (size), (max_reports), (failures))
#undef HELIA_VALIDATE_OUTPUTS
#define HELIA_VALIDATE_OUTPUTS(mode, actual, expected, size, tolerance, atol, rtol, max_reports, failures) \
    HELIA_VALIDATE_OUTPUTS_##mode((actual), expected, (size), (tolerance), (atol), (rtol), (max_reports), (failures))

#endif /* HCT_RECORD_H */
