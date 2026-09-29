#include "benchmark_server_transport.h"

#include <stddef.h>
#include <stdint.h>

#include "am_mcu_apollo.h"
#include "SEGGER_RTT.h"

/* SEGGER_RTT.c owns the channel 0 buffers. */
enum
{
    HCT_RTT_UP_CHANNEL = 0,
    HCT_RTT_DOWN_CHANNEL = 0
};

static void hct_rtt_cache_clean(void)
{
#if defined(NSX_SOC_CORE_M55)
    SCB_CleanDCache();
#endif
}

static void hct_rtt_cache_invalidate(void)
{
#if defined(NSX_SOC_CORE_M55)
    SCB_InvalidateDCache();
#endif
}

static int32_t hct_rtt_init(void)
{
    /* Up blocks when full; down skips. */
    int up_result = SEGGER_RTT_SetFlagsUpBuffer(
        HCT_RTT_UP_CHANNEL, SEGGER_RTT_MODE_BLOCK_IF_FIFO_FULL);
    int down_result = SEGGER_RTT_SetFlagsDownBuffer(
        HCT_RTT_DOWN_CHANNEL, SEGGER_RTT_MODE_NO_BLOCK_SKIP);
    hct_rtt_cache_clean();
    return (up_result < 0 || down_result < 0) ? -1 : 0;
}

static size_t hct_rtt_write(const uint8_t *payload, size_t length)
{
    const unsigned written = SEGGER_RTT_Write(HCT_RTT_UP_CHANNEL, payload, (unsigned)length);
    hct_rtt_cache_clean();
    return (size_t)written;
}

static size_t hct_rtt_read(uint8_t *payload, size_t capacity)
{
    hct_rtt_cache_invalidate();
    return (size_t)SEGGER_RTT_Read(HCT_RTT_DOWN_CHANNEL, payload, (unsigned)capacity);
}

static const hct_transport_vtable_t g_hct_transport_rtt = {
    .init = hct_rtt_init,
    .write = hct_rtt_write,
    .read = hct_rtt_read,
};

const hct_transport_vtable_t *hct_transport_rtt(void)
{
    return &g_hct_transport_rtt;
}
