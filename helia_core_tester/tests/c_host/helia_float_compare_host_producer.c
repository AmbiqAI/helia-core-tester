#include "helia_float_compare_host_producer.h"

#include <stdint.h>
#include <string.h>

/*
 * Assembled from bit patterns behind a volatile read: a literal NAN/INFINITY
 * is a compile-time constant that -ffinite-math-only lets the optimizer reason
 * away before it is ever stored.
 */
static volatile uint32_t helia_f32_nan_bits = 0x7FC00000u;
static volatile uint32_t helia_f32_pos_inf_bits = 0x7F800000u;
static volatile uint32_t helia_f32_neg_inf_bits = 0xFF800000u;

static float helia_f32_from_bits(uint32_t bits)
{
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

float helia_host_produce_f32(char kind, float finite_value)
{
    switch (kind) {
        case 'n':
            return helia_f32_from_bits(helia_f32_nan_bits);
        case 'p':
            return helia_f32_from_bits(helia_f32_pos_inf_bits);
        case 'm':
            return helia_f32_from_bits(helia_f32_neg_inf_bits);
        default:
            return finite_value;
    }
}

#ifdef HELIA_HOST_HAVE_F16

static volatile uint16_t helia_f16_nan_bits = 0x7E00u;
static volatile uint16_t helia_f16_pos_inf_bits = 0x7C00u;
static volatile uint16_t helia_f16_neg_inf_bits = 0xFC00u;

static helia_host_f16 helia_f16_from_bits(uint16_t bits)
{
    helia_host_f16 value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

helia_host_f16 helia_host_produce_f16(char kind, float finite_value)
{
    switch (kind) {
        case 'n':
            return helia_f16_from_bits(helia_f16_nan_bits);
        case 'p':
            return helia_f16_from_bits(helia_f16_pos_inf_bits);
        case 'm':
            return helia_f16_from_bits(helia_f16_neg_inf_bits);
        default:
            return (helia_host_f16)finite_value;
    }
}

#endif
