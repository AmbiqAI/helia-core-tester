#ifndef HCT_BENCHMARK_SERVER_MRAM_H
#define HCT_BENCHMARK_SERVER_MRAM_H

/* Image, pool, gap, store at MRAM top. */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HCT_MRAM_POOL_BYTES (512u * 1024u)
/* D-cache line; blobs start on one. */
#define HCT_MRAM_LINE_BYTES 32u

/* Store bytes; 0 builds it out. */
#ifndef HCT_BLOB_STORE_BYTES
#define HCT_BLOB_STORE_BYTES 0u
#endif

/* Pool start, or 0 when it overlaps. */
uintptr_t hct_mram_pool_base(void);

/* Host-sent identity of one blob. */
typedef struct
{
    uint32_t length;
    uint32_t crc32;
    /* First 8 bytes of SHA-256. */
    uint64_t digest;
} hct_store_key_t;

/* Stored bytes for key, or NULL. */
const uint8_t *hct_store_find(const hct_store_key_t *key);

/* Add data under key; wipe when full. */
void hct_store_save(const hct_store_key_t *key, const uint8_t *data);

#ifdef __cplusplus
}
#endif

#endif
