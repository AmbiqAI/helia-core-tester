/*
 * Opaque operand source for the HELIA_VALIDATE_FLOATS host driver (issue #75).
 *
 * The operands are produced in their own translation unit and consumed in
 * another, with no LTO, so the driver's compiler cannot see where a NaN or an
 * infinity came from. A driver that builds its own non-finite values in the
 * same function as the comparison proves less than it looks: the classifier
 * could be reading a value the optimizer already knows the class of.
 */
#ifndef HELIA_FLOAT_COMPARE_HOST_PRODUCER_H
#define HELIA_FLOAT_COMPARE_HOST_PRODUCER_H

#if defined(__FLT16_MAX__)
#define HELIA_HOST_HAVE_F16 1
typedef _Float16 helia_host_f16;
#endif

/* kind: 'n' NaN, 'p' +Inf, 'm' -Inf, anything else the supplied finite value. */
float helia_host_produce_f32(char kind, float finite_value);

#ifdef HELIA_HOST_HAVE_F16
helia_host_f16 helia_host_produce_f16(char kind, float finite_value);
#endif

#endif /* HELIA_FLOAT_COMPARE_HOST_PRODUCER_H */
