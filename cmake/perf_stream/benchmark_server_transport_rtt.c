#include "benchmark_server_transport.h"

#include <stddef.h>
#include <stdint.h>

#include "am_mcu_apollo.h"
#include "SEGGER_RTT.h"

enum
{
    HCT_RTT_UP_CHANNEL = 0,
    HCT_RTT_DOWN_CHANNEL = 0,
    /* Generous enough to hold a full burst of SAMPLE_RESULT/OUTPUT_CHUNK frames
     * queued by a single handler call (see hct_flush_outbound() in
     * benchmark_server_main.c) so BLOCK_IF_FIFO_FULL below rarely has to wait. */
    HCT_RTT_UP_BUFFER_BYTES = 8192,
    HCT_RTT_DOWN_BUFFER_BYTES = 512
};

__attribute__((aligned(16))) static uint8_t g_hct_rtt_up_buffer[HCT_RTT_UP_BUFFER_BYTES];
__attribute__((aligned(16))) static uint8_t g_hct_rtt_down_buffer[HCT_RTT_DOWN_BUFFER_BYTES];

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
    /* The up channel carries the reliable HCTP frame stream, so it must never
     * silently drop bytes when the host's RTT polling briefly falls behind a
     * burst of queued frames (SAMPLE_RESULT/OUTPUT_CHUNK bursts can be dozens
     * of frames back-to-back -- see hct_flush_outbound()). NO_BLOCK_SKIP used
     * to drop the tail of such bursts once the small ring buffer filled,
     * silently corrupting/truncating the frame stream and desyncing the
     * host's sequence-number tracking. BLOCK_IF_FIFO_FULL makes the firmware
     * wait for the host to drain space instead, which is safe here because
     * the host continuously polls for reads while a session is active. */
    int up_result = SEGGER_RTT_ConfigUpBuffer(
        HCT_RTT_UP_CHANNEL,
        "HCTP_UP",
        g_hct_rtt_up_buffer,
        HCT_RTT_UP_BUFFER_BYTES,
        SEGGER_RTT_MODE_BLOCK_IF_FIFO_FULL);
    int down_result = SEGGER_RTT_ConfigDownBuffer(
        HCT_RTT_DOWN_CHANNEL,
        "HCTP_DOWN",
        g_hct_rtt_down_buffer,
        HCT_RTT_DOWN_BUFFER_BYTES,
        SEGGER_RTT_MODE_NO_BLOCK_SKIP);
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
