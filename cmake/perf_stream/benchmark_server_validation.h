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

/* Convenience wrapper for hct_checked_shape_bytes() over a single-dimension (rank-1)
 * operand, e.g. a bias vector -- avoids every call site building its own one-element
 * shape array. */
static inline bool hct_checked_count_bytes(int32_t count,
                                           uint32_t element_size,
                                           uint32_t capacity,
                                           uint32_t *output_bytes)
{
    const int32_t shape[1] = {count};
    return hct_checked_shape_bytes(shape, 1, element_size, capacity, output_bytes);
}

/* Two packed 4-bit (S4) elements per byte, rounded up -- for validating an S4-packed
 * weights blob's received byte_length against its declared element-count shape (see
 * hct_checked_shape_bytes() above, which this mirrors for the sub-byte packed case). */
static inline bool hct_checked_packed4_bytes(const int32_t *shape,
                                             int32_t rank,
                                             uint32_t capacity,
                                             uint32_t *output_bytes)
{
    uint64_t product = 1u;
    uint64_t byte_count;
    int32_t index;
    if (shape == NULL || output_bytes == NULL || rank <= 0)
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
    byte_count = (product + 1u) / 2u;
    if (byte_count > capacity || byte_count > UINT32_MAX)
    {
        return false;
    }
    *output_bytes = (uint32_t)byte_count;
    return true;
}

#endif
