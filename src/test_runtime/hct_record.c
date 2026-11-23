/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Capture writer for record mode (see hct_record.h). Host only: it needs a
 * file system. Any write error ends the process with HCT_RECORD_EXIT so a
 * partial capture is never mistaken for a run.
 */
#include "hct_record.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HCT_RECORD_EXIT 3
#define HCT_RECORD_VERSION 1u
#define HCT_RECORD_MAX_NAME 65535u

static FILE *hct_record_file;
static uint32_t hct_record_crc = 0xFFFFFFFFu;
static uint32_t hct_record_count;
static int hct_record_finished;
static const char *hct_record_last_unrecordable;

static void hct_record_fail(const char *what)
{
    fprintf(stderr, "HCT_RECORD_ERROR: %s\n", what);
    fflush(stderr);
    exit(HCT_RECORD_EXIT);
}

static void hct_record_crc_update(const uint8_t *bytes, size_t len)
{
    uint32_t crc = hct_record_crc;
    for (size_t i = 0; i < len; ++i) {
        crc ^= bytes[i];
        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
        }
    }
    hct_record_crc = crc;
}

static void hct_record_write(const void *data, size_t len)
{
    if (len == 0) {
        return;
    }
    if (fwrite(data, 1, len, hct_record_file) != len) {
        hct_record_fail("short write to the capture file");
    }
    hct_record_crc_update((const uint8_t *)data, len);
}

static void hct_record_u8(uint32_t value)
{
    uint8_t byte = (uint8_t)value;
    hct_record_write(&byte, 1);
}

static void hct_record_u16(uint32_t value)
{
    uint8_t bytes[2] = {(uint8_t)value, (uint8_t)(value >> 8)};
    hct_record_write(bytes, sizeof(bytes));
}

static void hct_record_u32(uint32_t value)
{
    uint8_t bytes[4] = {(uint8_t)value, (uint8_t)(value >> 8), (uint8_t)(value >> 16), (uint8_t)(value >> 24)};
    hct_record_write(bytes, sizeof(bytes));
}

static void hct_record_u64(uint64_t value)
{
    hct_record_u32((uint32_t)value);
    hct_record_u32((uint32_t)(value >> 32));
}

static void hct_record_f64(double value)
{
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    hct_record_u64(bits);
}

static void hct_record_open(void)
{
    if (hct_record_file != NULL) {
        return;
    }
    if (hct_record_finished) {
        hct_record_fail("record after the capture was finished");
    }
    const char *path = getenv("HCT_RECORD_PATH");
    if (path == NULL || path[0] == '\0') {
        hct_record_fail("HCT_RECORD_PATH is not set");
    }
    hct_record_file = fopen(path, "wb");
    if (hct_record_file == NULL) {
        hct_record_fail("cannot open HCT_RECORD_PATH for writing");
    }
    hct_record_write("HCTR", 4);
    hct_record_u16(HCT_RECORD_VERSION);
    hct_record_u16(0);
}

static void hct_record_head(int kind, int dtype, size_t elem_size, const char *name)
{
    if (name == NULL) {
        hct_record_fail("record without a name");
    }
    size_t name_len = strlen(name);
    if (name_len == 0 || name_len > HCT_RECORD_MAX_NAME) {
        hct_record_fail("record name is empty or too long");
    }
    if (elem_size > 0xFFFFu) {
        hct_record_fail("element size does not fit the record");
    }
    hct_record_open();
    hct_record_u8((uint32_t)kind);
    hct_record_u8((uint32_t)dtype);
    hct_record_u16((uint32_t)elem_size);
    hct_record_u16((uint32_t)name_len);
    hct_record_write(name, name_len);
    ++hct_record_count;
}

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
                       double rtol)
{
    if (count < 0) {
        hct_record_fail("negative element count");
    }
    if (elem_size == 0 || (unsigned long long)count > 0xFFFFFFFFull / elem_size) {
        hct_record_fail("tensor does not fit the record");
    }
    if (count > 0 && (actual == NULL || golden == NULL)) {
        hct_record_fail("tensor record without data");
    }
    uint32_t nbytes = (uint32_t)((unsigned long long)count * elem_size);
    hct_record_head(HCT_RECORD_KIND_TENSOR, dtype, elem_size, name);
    hct_record_u8((uint32_t)mode);
    hct_record_u8(mask != NULL ? HCT_RECORD_FLAG_MASK : 0u);
    hct_record_u8(1);
    hct_record_u8(0);
    hct_record_u32((uint32_t)count);
    hct_record_u64((uint64_t)tolerance);
    hct_record_f64(atol);
    hct_record_f64(rtol);
    hct_record_u32(nbytes);
    hct_record_write(actual, nbytes);
    hct_record_write(golden, nbytes);
    if (mask != NULL) {
        hct_record_write(mask, (size_t)count);
    }
}

void hct_record_status(const char *label, int status, int expected)
{
    hct_record_head(HCT_RECORD_KIND_STATUS, HCT_RECORD_DTYPE_I32, 4, label);
    hct_record_u32((uint32_t)status);
    hct_record_u32((uint32_t)expected);
}

void hct_record_unrecordable(const char *golden)
{
    /* A per-element validator names the same call site once per element. */
    if (golden == hct_record_last_unrecordable) {
        return;
    }
    hct_record_last_unrecordable = golden;
    hct_record_head(HCT_RECORD_KIND_UNRECORDABLE, HCT_RECORD_DTYPE_UNKNOWN, 0, golden);
}

void hct_record_finish(void)
{
    if (hct_record_finished) {
        return;
    }
    hct_record_open();
    hct_record_finished = 1;
    uint32_t records = hct_record_count;
    hct_record_u8(HCT_RECORD_KIND_END);
    hct_record_u32(records);
    uint32_t crc = hct_record_crc ^ 0xFFFFFFFFu;
    uint8_t bytes[4] = {(uint8_t)crc, (uint8_t)(crc >> 8), (uint8_t)(crc >> 16), (uint8_t)(crc >> 24)};
    if (fwrite(bytes, 1, sizeof(bytes), hct_record_file) != sizeof(bytes) || fclose(hct_record_file) != 0) {
        hct_record_fail("cannot finish the capture file");
    }
    hct_record_file = NULL;
}
