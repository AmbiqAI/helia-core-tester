from __future__ import annotations

from pathlib import Path

import jinja2


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _templates_root() -> Path:
    return _repo_root() / "assets" / "templates"


def _render(template_name: str, context: dict[str, object]) -> str:
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_templates_root())),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env.get_template(template_name).render(**context)


def test_all_c_templates_use_standalone_harness_contract() -> None:
    template_paths = sorted(_templates_root().glob("**/*.c.j2"))
    assert template_paths

    for path in template_paths:
        text = path.read_text()
        assert '#include "unity.h"' not in text, path
        assert "void setUp(void)" not in text, path
        assert "void tearDown(void)" not in text, path
        assert "void test_" not in text, path
        assert "TEST_ASSERT" not in text, path
        assert '{% include "common/standalone/runtime_common.j2" %}' in text, path
        assert '{% include "common/standalone/main.j2" %}' in text, path
        assert "int32_t {{ prefix }}_{{ name }}_test_case_run(void)" in text, path


def test_rendered_templates_keep_helper_and_main_shape() -> None:
    rendered = {
        "relu": _render(
            "relu/relu.c.j2",
            {
                "name": "relu_smoke",
                "prefix": "relu",
                "input_dtype": "int8_t",
                "output_dtype": "int8_t",
                "output_size": 4,
                "input_offset": 0,
                "output_offset": 0,
                "output_mult": 1,
                "output_shift": 0,
                "kernel_fn": "arm_relu_s8",
            },
        ),
        "comparison": _render(
            "comparison/comparison.c.j2",
            {
                "name": "comparison_smoke",
                "prefix": "comparison",
                "input_dtype": "int8_t",
                "output_size": 4,
                "kernel_fn": "arm_equal_s8",
                "input_1_offset": 0,
                "input_1_mult": 1,
                "input_1_shift": 0,
                "input_2_offset": 0,
                "input_2_mult": 1,
                "input_2_shift": 0,
                "left_shift": 0,
            },
        ),
        "reducemax": _render(
            "reducemax/reducemax.c.j2",
            {
                "name": "reducemax_smoke",
                "prefix": "reducemax",
                "input_dtype": "int8_t",
                "output_dtype": "int8_t",
                "kernel_fn": "arm_reduce_max_s8",
                "output_dims": {"n": 1, "h": 1, "w": 1, "c": 1},
            },
        ),
        "conv2d": _render(
            "conv2d/conv2d.c.j2",
            {
                "name": "conv_smoke",
                "prefix": "conv",
                "input_dtype": "int8_t",
                "output_dtype": "int8_t",
                "buffer_size_max": 128,
                "kernel_fn": "arm_convolve_wrapper_s8",
                "kernel_get_buffer_size_fn": "arm_convolve_wrapper_s8_get_buffer_size",
                "output_dims": {"n": 1, "h": 1, "w": 1, "c": 1},
                "filter_dims": {"n": 1},
                "has_biases": False,
            },
        ),
        "pack": _render(
            "pack/pack.c.j2",
            {
                "name": "pack_smoke",
                "prefix": "pack",
                "input_dtype": "int8_t",
                "output_dtype": "int8_t",
                "output_size": 4,
                "outer_size": 2,
                "inner_size": 1,
                "num_tensors": 2,
            },
        ),
    }

    for name, text in rendered.items():
        assert '#include "unity.h"' not in text, name
        assert "void setUp(void)" not in text, name
        assert "void tearDown(void)" not in text, name
        assert "void test_" not in text, name
        assert "helia_test_platform_init();" in text, name
        assert "helia_test_finish(failures);" in text, name
        assert "int main(void)" in text, name
        assert "_test_case_run(void)" in text, name


def test_top_level_cmake_no_longer_uses_unity() -> None:
    text = (_repo_root() / "CMakeLists.txt").read_text()
    assert "unity_fetch" not in text
    assert "ThrowTheSwitch/Unity" not in text
    assert "FetchContent_Declare(unity" not in text
    assert "target_link_libraries(${TGT_NAME} PRIVATE cmsis-nn retarget cmsis_startup)" in text
