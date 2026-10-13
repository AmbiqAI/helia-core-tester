#include "benchmark_server_mram.h"

#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "hctp_protocol.h"

#if defined(HCT_PLACEMENT_MRAM) || HCT_BLOB_STORE_BYTES > 0

#ifdef HELIA_HARDWARE_BUILD
#include "am_mcu_apollo.h"
#else
/* Host harness: stubbed MRAM HAL. */
#include "hct_mram_stub.h"
#endif

/* NSX SBL script MCU_MRAM ends. */
#if defined(HCT_MRAM_END)
/* Host harness supplies its own. */
#elif defined(AM_PART_APOLLO330P)
#define HCT_MRAM_END 0x00600000u
#else
#define HCT_MRAM_END 0x00800000u
#endif

extern uint32_t _init_data, _sdata, _edata, _init_data_sram, _ssdata, _sedata;

/* First 64 KiB boundary past the image. */
static uintptr_t pool_start(void)
{
    const uintptr_t data_end = (uintptr_t)&_init_data + ((uintptr_t)&_edata - (uintptr_t)&_sdata);
    const uintptr_t sram_end = (uintptr_t)&_init_data_sram + ((uintptr_t)&_sedata - (uintptr_t)&_ssdata);
    const uintptr_t image_end = data_end > sram_end ? data_end : sram_end;
    return (image_end + 0xFFFFu) & ~(uintptr_t)0xFFFFu;
}

uintptr_t hct_mram_pool_base(void)
{
    const uintptr_t base = pool_start();
    return (base + HCT_MRAM_POOL_BYTES <= HCT_MRAM_END) ? base : 0u;
}
#else
uintptr_t hct_mram_pool_base(void)
{
    return 0u;
}
#endif

#if HCT_BLOB_STORE_BYTES > 0

/* Head, then header-plus-data entries. */
#define HCT_STORE_MAGIC 0x31534248u
#define HCT_STORE_ENTRY_MAGIC 0x45534248u
#define HCT_STORE_VERSION 1u

typedef struct
{
    uint32_t magic;
    uint32_t version;
    uint32_t base;
    uint32_t bytes;
    uint32_t generation;
    uint32_t reserved[2];
    /* CRC32 of the fields above. */
    uint32_t check;
} hct_store_head_t;

typedef struct
{
    uint32_t magic;
    uint32_t generation;
    uint32_t length;
    uint32_t crc32;
    uint32_t digest_lo;
    uint32_t digest_hi;
    uint32_t reserved;
    /* CRC32 of the fields above. */
    uint32_t check;
} hct_store_entry_t;

_Static_assert(sizeof(hct_store_head_t) == HCT_MRAM_LINE_BYTES, "head fills one cache line");
_Static_assert(sizeof(hct_store_entry_t) == HCT_MRAM_LINE_BYTES, "entry header fills one cache line");
_Static_assert(HCT_BLOB_STORE_BYTES % HCT_MRAM_LINE_BYTES == 0u, "store size is whole lines");
/* Head, entry header, one data line. */
_Static_assert(HCT_BLOB_STORE_BYTES >= 3u * HCT_MRAM_LINE_BYTES, "store holds one entry");

static uintptr_t store_base(void)
{
    return (uintptr_t)HCT_MRAM_END - HCT_BLOB_STORE_BYTES;
}

/* Store clears the image and pool. */
static bool store_fits(void)
{
    return pool_start() + HCT_MRAM_POOL_BYTES <= store_base();
}

static uint32_t record_check(const void *record)
{
    return hctp_crc32((const uint8_t *)record, HCT_MRAM_LINE_BYTES - sizeof(uint32_t));
}

static bool head_valid(const hct_store_head_t *head)
{
    return head->magic == HCT_STORE_MAGIC && head->version == HCT_STORE_VERSION &&
           head->base == (uint32_t)store_base() && head->bytes == HCT_BLOB_STORE_BYTES &&
           head->check == record_check(head);
}

static uint32_t padded(uint32_t length)
{
    return (length + HCT_MRAM_LINE_BYTES - 1u) & ~(HCT_MRAM_LINE_BYTES - 1u);
}

/* Program, then read back. */
static bool store_program(uintptr_t address, const uint8_t *data, uint32_t length)
{
    static uint32_t stage[128];
    uint32_t done = 0u;
    while (done < length)
    {
        const uint32_t span = (length - done) > sizeof(stage) ? (uint32_t)sizeof(stage) : (length - done);
        const uint32_t rows = (span + 15u) & ~15u;
        memset(stage, 0xFF, rows);
        memcpy(stage, data + done, span);
        if (am_hal_mram_main_program(AM_HAL_MRAM_PROGRAM_KEY, stage, (uint32_t *)(address + done), rows / 4u) != 0)
        {
            return false;
        }
        done += span;
    }
    SCB_InvalidateDCache_by_Addr((volatile void *)address, (int32_t)padded(length));
    return memcmp((const void *)address, data, length) == 0;
}

static bool head_write(uint32_t generation)
{
    hct_store_head_t head;
    memset(&head, 0, sizeof(head));
    head.magic = HCT_STORE_MAGIC;
    head.version = HCT_STORE_VERSION;
    head.base = (uint32_t)store_base();
    head.bytes = HCT_BLOB_STORE_BYTES;
    head.generation = generation;
    head.check = record_check(&head);
    return store_program(store_base(), (const uint8_t *)&head, sizeof(head));
}

/* New generation; first slot must not match. */
static uint32_t next_generation(const hct_store_head_t *head)
{
    const hct_store_entry_t *first = (const hct_store_entry_t *)(store_base() + HCT_MRAM_LINE_BYTES);
    const uint32_t generation = head->generation + 1u;
    return generation == first->generation ? generation + 1u : generation;
}

static bool entry_matches(const hct_store_entry_t *entry, const hct_store_key_t *key)
{
    return entry->length == key->length && entry->crc32 == key->crc32 &&
           entry->digest_lo == (uint32_t)key->digest && entry->digest_hi == (uint32_t)(key->digest >> 32);
}

/* Find a sound copy; return free slot. */
static uintptr_t store_scan(uint32_t generation, const hct_store_key_t *key, const hct_store_entry_t **hit)
{
    const uintptr_t end = store_base() + HCT_BLOB_STORE_BYTES;
    uintptr_t at = store_base() + HCT_MRAM_LINE_BYTES;
    *hit = NULL;
    while (at + HCT_MRAM_LINE_BYTES <= end)
    {
        const hct_store_entry_t *entry = (const hct_store_entry_t *)at;
        if (entry->magic != HCT_STORE_ENTRY_MAGIC || entry->generation != generation ||
            entry->check != record_check(entry) || padded(entry->length) > end - at - HCT_MRAM_LINE_BYTES)
        {
            break;
        }
        /* Skip copies that fail their CRC. */
        if (entry_matches(entry, key) && hctp_crc32((const uint8_t *)(entry + 1), entry->length) == entry->crc32)
        {
            *hit = entry;
            break;
        }
        at += HCT_MRAM_LINE_BYTES + padded(entry->length);
    }
    return at;
}

const uint8_t *hct_store_find(const hct_store_key_t *key)
{
    const hct_store_head_t *head = (const hct_store_head_t *)store_base();
    const hct_store_entry_t *hit;
    if (key->length == 0u || !store_fits() || !head_valid(head))
    {
        return NULL;
    }
    (void)store_scan(head->generation, key, &hit);
    return hit != NULL ? (const uint8_t *)(hit + 1) : NULL;
}

void hct_store_save(const hct_store_key_t *key, const uint8_t *data)
{
    const hct_store_head_t *head = (const hct_store_head_t *)store_base();
    const uintptr_t end = store_base() + HCT_BLOB_STORE_BYTES;
    const hct_store_entry_t *hit;
    hct_store_entry_t entry;
    uintptr_t slot;
    if (key->length == 0u || padded(key->length) > HCT_BLOB_STORE_BYTES - 2u * HCT_MRAM_LINE_BYTES || !store_fits())
    {
        return;
    }
    /* Stale or foreign head: start over. */
    if (!head_valid(head) && !head_write(next_generation(head)))
    {
        return;
    }
    slot = store_scan(head->generation, key, &hit);
    if (hit != NULL)
    {
        return;
    }
    /* Full: a new generation wipes it. */
    if (end - slot < HCT_MRAM_LINE_BYTES || padded(key->length) > end - slot - HCT_MRAM_LINE_BYTES)
    {
        if (!head_write(next_generation(head)))
        {
            return;
        }
        slot = store_base() + HCT_MRAM_LINE_BYTES;
    }
    if (!store_program(slot + HCT_MRAM_LINE_BYTES, data, key->length))
    {
        return;
    }
    memset(&entry, 0, sizeof(entry));
    entry.magic = HCT_STORE_ENTRY_MAGIC;
    entry.generation = head->generation;
    entry.length = key->length;
    entry.crc32 = key->crc32;
    entry.digest_lo = (uint32_t)key->digest;
    entry.digest_hi = (uint32_t)(key->digest >> 32);
    entry.check = record_check(&entry);
    (void)store_program(slot, (const uint8_t *)&entry, sizeof(entry));
}

#else

const uint8_t *hct_store_find(const hct_store_key_t *key)
{
    (void)key;
    return NULL;
}

void hct_store_save(const hct_store_key_t *key, const uint8_t *data)
{
    (void)key;
    (void)data;
}

#endif
