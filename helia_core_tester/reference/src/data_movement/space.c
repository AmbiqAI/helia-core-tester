/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Reshape, BatchToSpaceND / SpaceToBatchND, DepthToSpace / SpaceToDepth and
 * ResizeNearestNeighbor from TFLite's reference_ops loops, for any element type; 4-D NHWC with
 * two spatial block dims. BatchToSpaceND reads input batch (spatial_offset * out_batch + b), the
 * block offset major; DepthToSpace is TF's DCR order.
 */
#include <math.h>

#include "dm_common.h"

static int32_t check_4d(const HctTensor *in, int64_t *n_in, HctTensor *out, int64_t *n_out)
{
    HCT_TRY(hct_check_any_pair(in, n_in, out, n_out));
    return (in->rank == 4 && out->rank == 4) ? HCT_OK : HCT_E_SHAPE;
}

static int64_t at4(const HctTensor *t, int64_t b, int64_t h, int64_t w, int64_t c)
{
    return ((b * t->dims[1] + h) * t->dims[2] + w) * t->dims[3] + c;
}

int32_t hct_ref_reshape(const HctNoParams *params, const HctTensor *inputs, int32_t num_inputs, HctTensor *outputs,
                        int32_t num_outputs)
{
    if (params == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 1, outputs, num_outputs, 1));
    int64_t n_in = 0;
    int64_t n_out = 0;
    HCT_TRY(hct_check_any_pair(&inputs[0], &n_in, &outputs[0], &n_out));
    if (n_in != n_out)
    {
        return HCT_E_SHAPE;
    }
    for (int64_t i = 0; i < n_in; ++i)
    {
        hct_copy_elem(&outputs[0], i, &inputs[0], i);
    }
    return HCT_OK;
}

int32_t hct_ref_batch_to_space_nd(const HctNoParams *params, const HctTensor *inputs, int32_t num_inputs,
                                  HctTensor *outputs, int32_t num_outputs)
{
    if (params == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 3, outputs, num_outputs, 1));
    const HctTensor *in = &inputs[0];
    HctTensor *out = &outputs[0];
    int64_t n_in = 0;
    int64_t n_out = 0;
    int64_t n_crops = 0;
    HCT_TRY(check_4d(in, &n_in, out, &n_out));
    const int32_t *block = NULL;
    HCT_TRY(hct_check_i32_vector(&inputs[1], 2, &block));
    HCT_TRY(hct_check_tensor(&inputs[2], HCT_INT32, &n_crops));
    if (inputs[2].rank != 2 || inputs[2].dims[0] != 2 || inputs[2].dims[1] != 2)
    {
        return HCT_E_SHAPE;
    }
    const int32_t *crops = (const int32_t *)inputs[2].data;
    const int32_t bh = block[0];
    const int32_t bw = block[1];
    if (bh < 1 || bw < 1 || crops[0] < 0 || crops[1] < 0 || crops[2] < 0 || crops[3] < 0 ||
        in->dims[0] % (bh * bw) != 0)
    {
        return HCT_E_PARAM;
    }
    const int64_t out_batch = in->dims[0] / (bh * bw);
    if (out->dims[0] != out_batch || out->dims[1] != (int64_t)in->dims[1] * bh - crops[0] - crops[1] ||
        out->dims[2] != (int64_t)in->dims[2] * bw - crops[2] - crops[3] || out->dims[3] != in->dims[3])
    {
        return HCT_E_SHAPE;
    }
    for (int64_t ib = 0; ib < in->dims[0]; ++ib)
    {
        const int64_t ob = ib % out_batch;
        const int64_t spatial = ib / out_batch;
        for (int64_t ih = 0; ih < in->dims[1]; ++ih)
        {
            const int64_t oh = ih * bh + spatial / bw - crops[0];
            if (oh < 0 || oh >= out->dims[1])
            {
                continue;
            }
            for (int64_t iw = 0; iw < in->dims[2]; ++iw)
            {
                const int64_t ow = iw * bw + spatial % bw - crops[2];
                if (ow < 0 || ow >= out->dims[2])
                {
                    continue;
                }
                for (int64_t c = 0; c < in->dims[3]; ++c)
                {
                    hct_copy_elem(out, at4(out, ob, oh, ow, c), in, at4(in, ib, ih, iw, c));
                }
            }
        }
    }
    return HCT_OK;
}

int32_t hct_ref_space_to_batch_nd(const HctNoParams *params, const HctTensor *inputs, int32_t num_inputs,
                                  HctTensor *outputs, int32_t num_outputs)
{
    if (params == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 4, outputs, num_outputs, 1));
    const HctTensor *in = &inputs[0];
    HctTensor *out = &outputs[0];
    int64_t n_in = 0;
    int64_t n_out = 0;
    int64_t n_pad = 0;
    HCT_TRY(check_4d(in, &n_in, out, &n_out));
    const int32_t *block = NULL;
    HCT_TRY(hct_check_i32_vector(&inputs[1], 2, &block));
    HCT_TRY(hct_check_tensor(&inputs[2], HCT_INT32, &n_pad));
    HCT_TRY(hct_check_scalar_value(&inputs[3], in->dtype));
    if (inputs[2].rank != 2 || inputs[2].dims[0] != 2 || inputs[2].dims[1] != 2)
    {
        return HCT_E_SHAPE;
    }
    const int32_t *pad = (const int32_t *)inputs[2].data;
    const int32_t bh = block[0];
    const int32_t bw = block[1];
    if (bh < 1 || bw < 1 || pad[0] < 0 || pad[1] < 0 || pad[2] < 0 || pad[3] < 0)
    {
        return HCT_E_PARAM;
    }
    const int64_t ph = (int64_t)in->dims[1] + pad[0] + pad[1];
    const int64_t pw = (int64_t)in->dims[2] + pad[2] + pad[3];
    if (ph % bh != 0 || pw % bw != 0)
    {
        return HCT_E_PARAM;
    }
    if (out->dims[0] != (int64_t)in->dims[0] * bh * bw || out->dims[1] != ph / bh || out->dims[2] != pw / bw ||
        out->dims[3] != in->dims[3])
    {
        return HCT_E_SHAPE;
    }
    for (int64_t ob = 0; ob < out->dims[0]; ++ob)
    {
        const int64_t ib = ob % in->dims[0];
        const int64_t shift_w = (ob / in->dims[0]) % bw;
        const int64_t shift_h = (ob / in->dims[0]) / bw;
        for (int64_t oh = 0; oh < out->dims[1]; ++oh)
        {
            for (int64_t ow = 0; ow < out->dims[2]; ++ow)
            {
                const int64_t h = oh * bh + shift_h - pad[0];
                const int64_t w = ow * bw + shift_w - pad[2];
                const int inside = h >= 0 && h < in->dims[1] && w >= 0 && w < in->dims[2];
                for (int64_t c = 0; c < out->dims[3]; ++c)
                {
                    if (inside)
                    {
                        hct_copy_elem(out, at4(out, ob, oh, ow, c), in, at4(in, ib, h, w, c));
                    }
                    else
                    {
                        hct_copy_elem(out, at4(out, ob, oh, ow, c), &inputs[3], 0);
                    }
                }
            }
        }
    }
    return HCT_OK;
}

int32_t hct_ref_depth_to_space(const HctBlockParams *params, const HctTensor *inputs, int32_t num_inputs,
                               HctTensor *outputs, int32_t num_outputs)
{
    if (params == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 1, outputs, num_outputs, 1));
    const HctTensor *in = &inputs[0];
    HctTensor *out = &outputs[0];
    int64_t n_in = 0;
    int64_t n_out = 0;
    HCT_TRY(check_4d(in, &n_in, out, &n_out));
    const int64_t bs = params->block_size;
    if (bs < 1 || in->dims[3] % (bs * bs) != 0)
    {
        return HCT_E_PARAM;
    }
    const int64_t od = in->dims[3] / (bs * bs);
    if (out->dims[0] != in->dims[0] || out->dims[1] != in->dims[1] * bs || out->dims[2] != in->dims[2] * bs ||
        out->dims[3] != od)
    {
        return HCT_E_SHAPE;
    }
    for (int64_t b = 0; b < out->dims[0]; ++b)
    {
        for (int64_t oh = 0; oh < out->dims[1]; ++oh)
        {
            for (int64_t ow = 0; ow < out->dims[2]; ++ow)
            {
                for (int64_t c = 0; c < od; ++c)
                {
                    const int64_t ic = ((oh % bs) * bs + ow % bs) * od + c;
                    hct_copy_elem(out, at4(out, b, oh, ow, c), in, at4(in, b, oh / bs, ow / bs, ic));
                }
            }
        }
    }
    return HCT_OK;
}

int32_t hct_ref_space_to_depth(const HctBlockParams *params, const HctTensor *inputs, int32_t num_inputs,
                               HctTensor *outputs, int32_t num_outputs)
{
    if (params == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 1, outputs, num_outputs, 1));
    const HctTensor *in = &inputs[0];
    HctTensor *out = &outputs[0];
    int64_t n_in = 0;
    int64_t n_out = 0;
    HCT_TRY(check_4d(in, &n_in, out, &n_out));
    const int64_t bs = params->block_size;
    if (bs < 1 || in->dims[1] % bs != 0 || in->dims[2] % bs != 0)
    {
        return HCT_E_PARAM;
    }
    const int64_t id = in->dims[3];
    if (out->dims[0] != in->dims[0] || out->dims[1] != in->dims[1] / bs || out->dims[2] != in->dims[2] / bs ||
        out->dims[3] != id * bs * bs)
    {
        return HCT_E_SHAPE;
    }
    for (int64_t b = 0; b < out->dims[0]; ++b)
    {
        for (int64_t oh = 0; oh < out->dims[1]; ++oh)
        {
            for (int64_t ow = 0; ow < out->dims[2]; ++ow)
            {
                for (int64_t c = 0; c < out->dims[3]; ++c)
                {
                    const int64_t offset = c / id;
                    hct_copy_elem(out, at4(out, b, oh, ow, c), in,
                                  at4(in, b, oh * bs + offset / bs, ow * bs + offset % bs, c % id));
                }
            }
        }
    }
    return HCT_OK;
}

/* TFLite's GetNearestNeighbor, in binary32 as TFLite computes it. */
static int64_t nearest(int64_t x, int64_t in_size, float scale, int32_t align_corners, int32_t half_pixel)
{
    const float offset = half_pixel ? 0.5f : 0.0f;
    const float v = ((float)x + offset) * scale;
    int64_t r = align_corners ? (int64_t)roundf(v) : (int64_t)floorf(v);
    r = r < in_size - 1 ? r : in_size - 1;
    return half_pixel && r < 0 ? 0 : r;
}

int32_t hct_ref_resize_nearest_neighbor(const HctResizeParams *params, const HctTensor *inputs, int32_t num_inputs,
                                        HctTensor *outputs, int32_t num_outputs)
{
    if (params == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 1, outputs, num_outputs, 1));
    const HctTensor *in = &inputs[0];
    HctTensor *out = &outputs[0];
    int64_t n_in = 0;
    int64_t n_out = 0;
    HCT_TRY(check_4d(in, &n_in, out, &n_out));
    const int32_t ac = params->align_corners;
    const int32_t hp = params->half_pixel_centers;
    /* TFLite rejects both set; each is a boolean. */
    if ((ac != 0 && ac != 1) || (hp != 0 && hp != 1) || (ac && hp))
    {
        return HCT_E_PARAM;
    }
    if (out->dims[0] != in->dims[0] || out->dims[3] != in->dims[3] || in->dims[1] < 1 || in->dims[2] < 1 ||
        out->dims[1] < 1 || out->dims[2] < 1)
    {
        return HCT_E_SHAPE;
    }
    const int64_t ih = in->dims[1];
    const int64_t iw = in->dims[2];
    const int64_t oh_n = out->dims[1];
    const int64_t ow_n = out->dims[2];
    const float sh = (ac && oh_n > 1) ? (float)(ih - 1) / (float)(oh_n - 1) : (float)ih / (float)oh_n;
    const float sw = (ac && ow_n > 1) ? (float)(iw - 1) / (float)(ow_n - 1) : (float)iw / (float)ow_n;
    for (int64_t b = 0; b < out->dims[0]; ++b)
    {
        for (int64_t oh = 0; oh < oh_n; ++oh)
        {
            const int64_t y = nearest(oh, ih, sh, ac, hp);
            for (int64_t ow = 0; ow < ow_n; ++ow)
            {
                const int64_t x = nearest(ow, iw, sw, ac, hp);
                for (int64_t c = 0; c < out->dims[3]; ++c)
                {
                    hct_copy_elem(out, at4(out, b, oh, ow, c), in, at4(in, b, y, x, c));
                }
            }
        }
    }
    return HCT_OK;
}

/* ------------------------------------------------------------- ResizeBilinear (int8) */

/* TFLite's ComputeInterpolationValuesInteger: Q10 source position and its clamped neighbours. */
static void bilinear_axis(int32_t value, int32_t scale_10, int32_t half_pixel, int32_t in_size, int32_t *scaled,
                          int32_t *lo, int32_t *hi)
{
    *scaled = half_pixel ? value * scale_10 + scale_10 / 2 - (1 << 9) : value * scale_10;
    const int32_t lower = *scaled / (1 << 10);
    *lo = lower > 0 ? lower : 0;
    const int32_t upper = (*scaled + (1 << 10) - 1) / (1 << 10);
    *hi = upper < in_size - 1 ? upper : in_size - 1;
}

static int32_t bilinear_scale_10(int64_t in_size, int64_t out_size, int32_t align_corners)
{
    if (align_corners && out_size > 1)
    {
        return (int32_t)(((1 << 10) * (in_size - 1) + (out_size - 1) / 2) / (out_size - 1));
    }
    return (int32_t)(((1 << 10) * in_size + out_size / 2) / out_size);
}

/* TFLite reference_ops::ResizeBilinearInteger<int8_t> (the int8 path of the builtin reference kernel):
 * Q10 weights, a Q20 sum and a rounding divide by 2^20 with the half away from zero. */
int32_t hct_ref_resize_bilinear_s8(const HctResizeParams *params, const HctTensor *inputs, int32_t num_inputs,
                                   HctTensor *outputs, int32_t num_outputs)
{
    if (params == NULL)
    {
        return HCT_E_NULL;
    }
    HCT_TRY(hct_check_io(inputs, num_inputs, 1, outputs, num_outputs, 1));
    const HctTensor *in = &inputs[0];
    HctTensor *out = &outputs[0];
    if (in->dtype != HCT_INT8 || out->dtype != HCT_INT8)
    {
        return HCT_E_DTYPE;
    }
    int64_t n_in = 0;
    int64_t n_out = 0;
    HCT_TRY(check_4d(in, &n_in, out, &n_out));
    const int32_t ac = params->align_corners;
    const int32_t hp = params->half_pixel_centers;
    if ((ac != 0 && ac != 1) || (hp != 0 && hp != 1) || (ac && hp))
    {
        return HCT_E_PARAM;
    }
    if (out->dims[0] != in->dims[0] || out->dims[3] != in->dims[3] || in->dims[1] < 1 || in->dims[2] < 1 ||
        out->dims[1] < 1 || out->dims[2] < 1)
    {
        return HCT_E_SHAPE;
    }
    /* The Q10 positions are int32 in TFLite: keep every product inside that range. */
    if (in->dims[1] > (1 << 20) || in->dims[2] > (1 << 20) || out->dims[1] > (1 << 20) || out->dims[2] > (1 << 20))
    {
        return HCT_E_SHAPE;
    }
    const int32_t ih = (int32_t)in->dims[1];
    const int32_t iw = (int32_t)in->dims[2];
    const int32_t oh_n = (int32_t)out->dims[1];
    const int32_t ow_n = (int32_t)out->dims[2];
    const int32_t sh = bilinear_scale_10(ih, oh_n, ac);
    const int32_t sw = bilinear_scale_10(iw, ow_n, ac);
    if ((int64_t)sh * oh_n > INT32_MAX / 2 || (int64_t)sw * ow_n > INT32_MAX / 2)
    {
        return HCT_E_SHAPE;
    }
    const int8_t *src = (const int8_t *)in->data;
    for (int64_t b = 0; b < out->dims[0]; ++b)
    {
        for (int32_t oy = 0; oy < oh_n; ++oy)
        {
            int32_t y = 0;
            int32_t y0 = 0;
            int32_t y1 = 0;
            bilinear_axis(oy, sh, hp, ih, &y, &y0, &y1);
            const int64_t fy = y - (1 << 10) * y0;
            for (int32_t ox = 0; ox < ow_n; ++ox)
            {
                int32_t x = 0;
                int32_t x0 = 0;
                int32_t x1 = 0;
                bilinear_axis(ox, sw, hp, iw, &x, &x0, &x1);
                const int64_t fx = x - (1 << 10) * x0;
                for (int64_t c = 0; c < out->dims[3]; ++c)
                {
                    const int64_t sum = (int64_t)src[at4(in, b, y0, x0, c)] * ((1 << 10) - fy) * ((1 << 10) - fx) +
                                        (int64_t)src[at4(in, b, y1, x0, c)] * fy * ((1 << 10) - fx) +
                                        (int64_t)src[at4(in, b, y0, x1, c)] * ((1 << 10) - fy) * fx +
                                        (int64_t)src[at4(in, b, y1, x1, c)] * fy * fx;
                    const int64_t round = sum > 0 ? (1 << 19) : -(1 << 19);
                    hct_store_i32(out, at4(out, b, oy, ox, c), (int32_t)((sum + round) / (1 << 20)));
                }
            }
        }
    }
    return HCT_OK;
}
