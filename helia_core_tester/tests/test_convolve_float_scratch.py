"""The static scratch of a float Convolve case holds what the patch-GEMM sizer asks for."""

from __future__ import annotations

import re
from pathlib import Path

from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.test_ops import generate_test

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_long_patch_over_a_small_input_gets_eight_patch_rows(tmp_path: Path) -> None:
    name = "convolve_float_direct_fold_c20_k5_oc5_f16"
    desc = next(d for d in load_all_descriptors(str(_PROJECT_ROOT / "assets" / "descriptors")) if d["name"] == name)
    generate_test(desc, str(tmp_path))
    case_dir = next(p.parent for p in tmp_path.rglob(f"{name}.tflite"))
    source = "".join(p.read_text() for p in case_dir.rglob("*.[ch]"))

    size = int(re.search(rf"#define {name.upper()}_BUFFER_SIZE_MAX (\d+)", source).group(1))
    # arm_convolve_f16_get_buffer_size: 8 tile rows x (5 x 5 x 20) patch x 2 bytes; the tensors total 6800.
    assert size >= 8 * 5 * 5 * 20 * 2
