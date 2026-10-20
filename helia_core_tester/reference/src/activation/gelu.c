/*
 * Exact GELU, 0.5 * x * erfc(-x / sqrt(2)) (not the tanh approximation), evaluated in binary64 at the
 * exactly widened input and rounded once to the output type. The evaluation order is the kernel's, so
 * non-finite lanes follow IEEE 754: +Inf -> +Inf, NaN -> NaN, -Inf -> NaN (erfc(+Inf) is 0).
 */
#include <math.h>

#include "hct_ref_internal.h"

static int32_t gelu_float(const HctNoParams *p, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                          int32_t num_outputs, int32_t dtype)
{
    if (p == NULL)
    {
        return HCT_E_NULL;
    }
    int64_t count = 0;
    HCT_TRY(hct_unary_setup(inputs, num_inputs, outputs, num_outputs, dtype, dtype, &count));
    for (int64_t i = 0; i < count; ++i)
    {
        const double x = (double)hct_load_f32(&inputs[0], i);
        const double y = 0.5 * x * erfc(-x / sqrt(2.0));
        if (dtype == HCT_FLOAT16)
        {
            ((uint16_t *)outputs[0].data)[i] = hct_f64_to_f16(y);
        }
        else
        {
            ((float *)outputs[0].data)[i] = (float)y;
        }
    }
    return HCT_OK;
}

int32_t hct_ref_gelu_f32(const HctNoParams *params, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                         int32_t num_outputs)
{
    return gelu_float(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT32);
}

int32_t hct_ref_gelu_f16(const HctNoParams *params, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                         int32_t num_outputs)
{
    return gelu_float(params, inputs, num_inputs, outputs, num_outputs, HCT_FLOAT16);
}
