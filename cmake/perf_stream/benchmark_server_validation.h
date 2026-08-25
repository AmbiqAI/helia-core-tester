#ifndef HCT_BENCHMARK_SERVER_VALIDATION_H
#define HCT_BENCHMARK_SERVER_VALIDATION_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

static inline bool hct_checked_aligned_range(uint32_t offset,
                                             uint32_t alignment,
                                             uint32_t length,
                                             uint32_t capacity,
                                             uint32_t *aligned_offset,
                                             uint32_t *end_offset)
{
    uint64_t aligned;
    uint64_t end;
    if (aligned_offset == NULL || end_offset == NULL || alignment == 0u ||
        (alignment & (alignment - 1u)) != 0u)
    {
        return false;
    }
    aligned = ((uint64_t)offset + alignment - 1u) & ~((uint64_t)alignment - 1u);
    end = aligned + length;
    if (aligned > UINT32_MAX || end > capacity)
    {
        return false;
    }
    *aligned_offset = (uint32_t)aligned;
    *end_offset = (uint32_t)end;
    return true;
}

static inline bool hct_checked_shape_bytes(const int32_t *shape,
                                           int32_t rank,
                                           uint32_t element_size,
                                           uint32_t capacity,
                                           uint32_t *output_bytes)
{
    uint64_t product = 1u;
    int32_t index;
    if (shape == NULL || output_bytes == NULL || rank <= 0 || element_size == 0u)
    {
        return false;
    }
    for (index = 0; index < rank; ++index)
    {
        if (shape[index] <= 0)
        {
            return false;
        }
        product *= (uint32_t)shape[index];
        if (product > UINT32_MAX)
        {
            return false;
        }
    }
    product *= element_size;
    if (product > capacity || product > UINT32_MAX)
    {
        return false;
    }
    *output_bytes = (uint32_t)product;
    return true;
}

#endif
