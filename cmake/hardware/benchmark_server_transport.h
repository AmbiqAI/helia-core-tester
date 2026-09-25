#ifndef HCT_BENCHMARK_SERVER_TRANSPORT_H
#define HCT_BENCHMARK_SERVER_TRANSPORT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    int32_t (*init)(void);
    size_t (*write)(const uint8_t *payload, size_t length);
    size_t (*read)(uint8_t *payload, size_t capacity);
} hct_transport_vtable_t;

const hct_transport_vtable_t *hct_transport_rtt(void);

#ifdef __cplusplus
}
#endif

#endif
