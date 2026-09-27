"""s4 depthwise headers carry the model's kernel height and width.

s4 cases take their filter shape from the descriptor, [H, W, I, M]. A kernel height of 1
must not be mistaken for the leading 1 of the TFLite layout [1, H, W, C].
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from helia_core_tester.generation.io.descriptors import load_all_descriptors
from helia_core_tester.generation.test_ops import generate_test

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "name",
    [
        "depthwise_conv_generic_c3_pad_1x5_batch2_s4",
        "depthwise_conv_generic_c3_pad_dil_1x2_s4",
        "depthwise_conv_opt_s4",
    ],
)
def test_filter_dims_match_the_model(name: str, tmp_path: Path) -> None:
    desc = next(d for d in load_all_descriptors(str(_PROJECT_ROOT / "assets" / "descriptors")) if d["name"] == name)
    generate_test(desc, str(tmp_path))
    case_dir = next(p.parent for p in tmp_path.rglob(f"{name}.tflite"))
    header = "".join(p.read_text() for p in (case_dir / "includes").glob("*.h"))
    dims = re.search(r"_filter_dims\b[^{]*\{([^}]*)\}", header).group(1)
    emitted = {k: int(v) for k, v in re.findall(r"\.(\w)\s*=\s*(\d+)", dims)}

    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(case_dir / f"{name}.tflite"))
    kernel_h, kernel_w = desc["filter_shape"][:2]
    filters = [d for d in interpreter.get_tensor_details() if len(d["shape"]) == 4 and d["shape"][0] == 1 and d["index"] not in (
        interpreter.get_input_details()[0]["index"], interpreter.get_output_details()[0]["index"])]
    assert len(filters) == 1
    assert tuple(filters[0]["shape"][1:3]) == (kernel_h, kernel_w)

    assert (emitted["h"], emitted["w"]) == (kernel_h, kernel_w)
