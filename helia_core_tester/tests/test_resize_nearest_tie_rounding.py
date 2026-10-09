"""ResizeNearestNeighbor: the C reference's source index matches the kernel's float32 rule, and
float descriptors are refused (#127)."""

import shutil
import subprocess
import textwrap

import numpy as np
import pytest

from helia_core_tester.generation.ops.ReshapeFunctions.resize_nearest_neighbor import (
    OpResizeNearestNeighbor,
)
from helia_core_tester.generation.reference.call import ReferenceCall


def nearest_indices(in_size: int, out_size: int, align_corners: bool, half_pixel_centers: bool) -> np.ndarray:
    """Source index of every output row, read off the reference resize of an index column."""
    column = np.arange(in_size, dtype=np.int16).reshape(1, in_size, 1, 1)
    out = ReferenceCall(
        "resize_nearest_neighbor",
        {"align_corners": int(align_corners), "half_pixel_centers": int(half_pixel_centers)},
        {"input": column}, {"output": (1, out_size, 1, 1)}).output()
    return out.reshape(out_size).astype(int)


def _index(out_idx, in_size, out_size, align_corners, half_pixel_centers):
    return int(nearest_indices(in_size, out_size, align_corners, half_pixel_centers)[out_idx])

# GetNearestNeighbor transcribed from ns-cmsis-nn Include/arm_nnsupportfunctions.h, with the scale
# and offset setup of arm_resize_nearest_neighbor_s8.c lifted into main(), so the oracle is the
# kernel's own float32 arithmetic rather than a second Python formulation of the rule.
_KERNEL_HELPER_C = textwrap.dedent(
    """
    #include <math.h>
    #include <stdbool.h>
    #include <stdint.h>
    #include <stdio.h>
    #include <stdlib.h>
    #define ARM_NN_MIN(a, b) ((a) < (b) ? (a) : (b))
    #define ARM_NN_MAX(a, b) ((a) > (b) ? (a) : (b))
    static inline int32_t GetNearestNeighbor(const int input_value, const int32_t input_size, const float scale,
                                             const float offset, const bool align_corners,
                                             const bool half_pixel_centers)
    {
        const float scaled = ((float)input_value + offset) * scale;
        int32_t output_value = align_corners ? (int32_t)roundf(scaled) : (int32_t)floorf(scaled);
        output_value = ARM_NN_MIN(output_value, input_size - 1);
        if (half_pixel_centers)
        {
            output_value = ARM_NN_MAX(0, output_value);
        }
        return output_value;
    }
    int main(int argc, char **argv)
    {
        const int maxn = atoi(argv[1]);
        for (int ac = 0; ac < 2; ac++)
            for (int hp = 0; hp < 2; hp++)
                for (int in = 1; in <= maxn; in++)
                    for (int out = 1; out <= maxn; out++)
                    {
                        const float scale = (ac && out > 1) ? (float)(in - 1) / (float)(out - 1) : (float)in / (float)out;
                        const float offset = hp ? 0.5f : 0.0f;
                        for (int i = 0; i < out; i++)
                            printf("%d %d %d %d %d %d\\n", ac, hp, in, out, i, GetNearestNeighbor(i, in, scale, offset, ac, hp));
                    }
        return 0;
    }
    """
)


# These stay as the portable fallback for machines with no C compiler, where the exhaustive oracle
# below is skipped. Every expected value here came from that oracle.
@pytest.mark.parametrize(
    ("align_corners", "half_pixel_centers", "in_size", "out_size", "out_idx", "kernel"),
    [
        # Exact ties on the align_corners path: roundf goes away from zero, np.round went to even.
        (True, False, 3, 5, 1, 1),
        (True, False, 7, 13, 5, 3),
        (True, False, 3, 5, 3, 2),
        # Exact rationals that are not float32-representable: the float32 product lands below
        # the tie and the kernel takes the lower pixel, where double arithmetic takes the upper.
        (True, False, 14, 23, 11, 6),
        (True, False, 2, 83, 41, 0),
        # Same effect on the floor path, where the product lands just below an integer.
        (False, False, 2, 82, 41, 0),
        (False, False, 14, 46, 23, 6),
        (False, False, 26, 22, 11, 12),
        # One output pixel with half-pixel centres samples the middle, not index 0.
        (False, True, 2, 1, 0, 1),
        (False, True, 5, 1, 0, 2),
        # align_corners with half_pixel_centers is rejected by TFLite and by the reference.
    ],
)
def test_reference_matches_kernel_verified_points(align_corners, half_pixel_centers, in_size, out_size, out_idx, kernel):
    assert _index(out_idx, in_size, out_size, align_corners, half_pixel_centers) == kernel


def test_double_precision_would_disagree_on_the_float32_points():
    # Why the reference is float32: as exact rationals these land on the tie and on the integer,
    # so double rounds up, while the float32 product falls just below both and the kernel takes
    # the lower pixel.
    assert 11 * (14 - 1) / (23 - 1) == 6.5
    assert _index(11, 14, 23, True, False) == 6
    assert 41 * 2 / 82 == 1.0
    assert _index(41, 2, 82, False, False) == 0


@pytest.mark.skipif(shutil.which("cc") is None and shutil.which("gcc") is None, reason="needs a host C compiler for the kernel oracle")
def test_reference_matches_compiled_kernel_helper_exhaustively(tmp_path):
    src = tmp_path / "helper.c"
    src.write_text(_KERNEL_HELPER_C)
    exe = tmp_path / "helper"
    cc = shutil.which("cc") or shutil.which("gcc")
    subprocess.run(
        [cc, "-std=c99", "-Wall", "-Wextra", "-O0", "-ffp-contract=off", "-fno-fast-math",
         str(src), "-o", str(exe), "-lm"],
        check=True,
    )
    table = subprocess.run([str(exe), "48"], check=True, capture_output=True, text=True).stdout
    kernel = {}
    for line in table.splitlines():
        ac, hp, in_size, out_size, out_idx, idx = (int(v) for v in line.split())
        if ac and hp:
            continue  # TFLite (and the reference) reject the combination
        kernel.setdefault((ac, hp, in_size, out_size), []).append(idx)
    assert kernel
    mismatches = [(key, want, list(nearest_indices(key[2], key[3], bool(key[0]), bool(key[1]))))
                  for key, want in kernel.items()]
    assert [m for m in mismatches if m[1] != m[2]] == []


def test_generate_c_files_rejects_float_loudly_until_the_float_kernels_are_wired(tmp_path):
    # ValueError, not NotImplementedError: helia_core_tester/generation/test_ops.py swallows
    # NotImplementedError from this call with an INFO line, so a float descriptor would vanish
    # from the suite instead of failing the generation run.
    for dtype in ("FP32", "FP16"):
        op = OpResizeNearestNeighbor({"name": "resize_probe", "operator": "ResizeNearestNeighbor",
                                      "activation_dtype": dtype, "input_shape": [1, 2, 2, 1], "size": [3, 3]})
        with pytest.raises(ValueError, match="supports S8 and S16"):
            op.generate_c_files(tmp_path)
        assert op.reference is None
