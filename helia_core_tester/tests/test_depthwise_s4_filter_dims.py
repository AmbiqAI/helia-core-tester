"""s4 depthwise headers carry the descriptor's kernel height and width.

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
    case_dir = next(p.parent for p in tmp_path.rglob("descriptor.yaml") if p.parent.name == name)
    header = "".join(p.read_text() for p in (case_dir / "includes").glob("*.h"))
    dims = re.search(r"_filter_dims\b[^{]*\{([^}]*)\}", header).group(1)
    emitted = {k: int(v) for k, v in re.findall(r"\.(\w)\s*=\s*(\d+)", dims)}

    import json

    record = json.loads((case_dir / f"{name}.reference.json").read_text())
    kernel_h, kernel_w = desc["filter_shape"][:2]
    # The reference kernel ran on a 1HWC filter of the descriptor's kernel size.
    assert tuple(record["params"]["filter_shape"][1:3]) == (kernel_h, kernel_w)
    assert record["params"]["filter_shape"][0] == 1

    assert (emitted["h"], emitted["w"]) == (kernel_h, kernel_w)
