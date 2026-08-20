#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "arm_nnfunctions.h"
#include "benchmark_server_adapter.h"
#include "benchmark_server_catalog.h"
#include "benchmark_server_messages.h"
#include "benchmark_server_session.h"
#include "benchmark_server_transport.h"
#include "nsx_system.h"

typedef struct
{
    const char *name;
    const void *address;
} hct_symbol_ref_t;

__attribute__((aligned(16))) static uint8_t g_hct_runtime_arena[HCT_SERVER_MAX_ARENA_BYTES];

static const hct_symbol_ref_t g_hct_symbol_refs[] = {
#include "kernel_symbol_refs.inc"
};

volatile uintptr_t g_hct_symbol_registry_anchor;
volatile size_t g_hct_symbol_registry_count;
volatile uint32_t g_hct_server_boot_status;
volatile uint32_t g_hct_last_hello_status;
volatile uint32_t g_hct_last_catalog_status;
volatile uint32_t g_hct_last_transport_init_status;
volatile uint32_t g_hct_last_transport_write_hello;
volatile uint32_t g_hct_last_transport_write_catalog;
volatile uint32_t g_hct_last_abs_dispatch_status;
volatile uint32_t g_hct_last_conv_dispatch_status;
volatile uint32_t g_hct_catalog_entry_count;
volatile uint32_t g_hct_last_transport_read_bytes;
volatile uint32_t g_hct_last_session_status;

static hct_server_session_t g_hct_session;
__attribute__((aligned(16))) static uint8_t g_hct_rx_buffer[2048u];
static size_t g_hct_rx_length = 0u;

static void hct_flush_outbound(hct_server_session_t *session, const hct_transport_vtable_t *transport)
{
    uint8_t frame[1024u];
    size_t frame_length = 0u;

    do
    {
        frame_length = hct_server_session_take_next_frame(session, frame, sizeof(frame));
        if (frame_length > 0u)
        {
            (void)transport->write(frame, frame_length);
        }
    } while (frame_length > 0u);
}

static void hct_poll_session(hct_server_session_t *session, const hct_transport_vtable_t *transport)
{
    hctp_frame_header_t header;
    /* The largest payload a complete frame could carry and still fit in our fixed-size
     * receive buffer (F005). Decoding against this bound -- rather than the protocol's
     * generic HCTP_DEFAULT_MAX_PAYLOAD (64 KiB) -- means we reject any header claiming a
     * payload too large for g_hct_rx_buffer up front, instead of accepting it and later
     * calling transport->read() with zero remaining capacity while we wait forever for
     * bytes that can never arrive. */
    const uint32_t max_payload_for_buffer = (uint32_t)(sizeof(g_hct_rx_buffer) - HCTP_HEADER_SIZE);

    /* Never call the transport with zero capacity (F005): if the buffer is already full
     * of an in-flight frame we can't yet consume, skip the read this poll instead of
     * passing a zero-length span to transport->read(). */
    if (g_hct_rx_length < sizeof(g_hct_rx_buffer))
    {
        g_hct_last_transport_read_bytes = (uint32_t)transport->read(g_hct_rx_buffer + g_hct_rx_length, sizeof(g_hct_rx_buffer) - g_hct_rx_length);
        g_hct_rx_length += g_hct_last_transport_read_bytes;
    }
    else
    {
        g_hct_last_transport_read_bytes = 0u;
    }

    while (g_hct_rx_length >= HCTP_HEADER_SIZE)
    {
        size_t frame_length;
        const hctp_status_t header_status = hctp_decode_header(g_hct_rx_buffer, HCTP_HEADER_SIZE, max_payload_for_buffer, &header);
        if (header_status != HCTP_STATUS_OK)
        {
            /* Resynchronize cleanly (F005): a header that is malformed, or whose
             * declared payload cannot possibly fit in our receive buffer, can never be
             * completed by reading more bytes into the same buffer. Discard everything
             * we've buffered so far and let the next poll start resynchronizing from a
             * clean slate, rather than wedging with a frame we can never finish. */
            g_hct_last_session_status = (uint32_t)header_status;
            g_hct_rx_length = 0u;
            break;
        }
        frame_length = HCTP_HEADER_SIZE + (size_t)header.payload_length;
        if (frame_length > sizeof(g_hct_rx_buffer))
        {
            /* Same resync rationale as above: max_payload_for_buffer should already
             * prevent this, but guard defensively against any future change to that
             * bound so a too-large-for-the-buffer frame can never silently stall. */
            g_hct_last_session_status = (uint32_t)HCTP_STATUS_OVERSIZED_PAYLOAD;
            g_hct_rx_length = 0u;
            break;
        }
        if (g_hct_rx_length < frame_length)
        {
            break;
        }
        g_hct_last_session_status = (uint32_t)hct_server_session_accept_frame(session, g_hct_rx_buffer, frame_length);
        memmove(g_hct_rx_buffer, g_hct_rx_buffer + frame_length, g_hct_rx_length - frame_length);
        g_hct_rx_length -= frame_length;
        hct_flush_outbound(session, transport);
    }
}

static uintptr_t hct_anchor_all_symbols(void)
{
    uintptr_t checksum = 0u;
    size_t index;

    for (index = 0u; index < (sizeof(g_hct_symbol_refs) / sizeof(g_hct_symbol_refs[0])); ++index)
    {
        checksum ^= ((uintptr_t)g_hct_symbol_refs[index].address >> 2) + (uintptr_t)index;
    }

    g_hct_symbol_registry_count = sizeof(g_hct_symbol_refs) / sizeof(g_hct_symbol_refs[0]);
    g_hct_symbol_registry_anchor = checksum;
    return checksum;
}

int main(void)
{
    const nsx_system_config_t system_cfg = {
        .perf_mode = NSX_PERF_HIGH,
        .enable_cache = true,
        .enable_sram = false,
        .debug = {.transport = NSX_DEBUG_NONE},
        .skip_bsp_init = true,
        .spot_mgr_profile = false,
    };
    const hct_transport_vtable_t *transport = hct_transport_rtt();
    size_t count = 0u;

    g_hct_server_boot_status = nsx_system_init(&system_cfg);
    (void)hct_anchor_all_symbols();
    (void)hct_benchmark_server_catalog(&count);
    g_hct_catalog_entry_count = (uint32_t)count;

    g_hct_last_abs_dispatch_status = (uint32_t)hct_link_smoke_invoke_abs_s8();
    g_hct_last_conv_dispatch_status = (uint32_t)hct_link_smoke_invoke_convolve_s8();
    g_hct_last_transport_init_status = (uint32_t)transport->init();
    hct_server_session_init(&g_hct_session, 0xC0DE1234u, 256u, (uint32_t)sizeof(g_hct_runtime_arena));
    g_hct_last_hello_status = HCTP_STATUS_OK;
    g_hct_last_catalog_status = HCTP_STATUS_OK;
    hct_flush_outbound(&g_hct_session, transport);

    for (;;)
    {
        hct_poll_session(&g_hct_session, transport);
    }
}
