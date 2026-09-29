"""contract/render.py: prototype-ordered calls and the compile-time parity assert."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import jinja2
import pytest

from helia_core_tester.contract.ir import CONTRACT_RELPATH, ContractSet, STATUS_ABSENT, load_contract_set
from helia_core_tester.contract.render import (
    ContractRenderError,
    adjusted_type,
    contract_globals,
    function_type,
    render_call,
    render_parity_assert,
)
from helia_core_tester.generation.ops._shared.base import template_environment
from helia_core_tester.generation.utils.temp_sizer_probe import resolve_cmsis_nn_root
from helia_core_tester.tests.test_contract_ir import FIXTURE, HEADERS

# Enough of the ns-cmsis-nn types for the fixture prototypes to compile on the host.
STUB_TYPES = """\
#include <stdint.h>
typedef int arm_cmsis_nn_status;
typedef struct { void *buf; int32_t size; } cmsis_nn_context;
typedef struct { int32_t n, h, w, c; } cmsis_nn_dims;
typedef int arm_nn_tensor_layout;
typedef short float16_t;
#define ARM_NN_ENABLE_F16 1
"""


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    root = tmp_path / "ns-cmsis-nn"
    for relative, text in HEADERS.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(text)
    (root / CONTRACT_RELPATH).parent.mkdir(parents=True)
    shutil.copy(FIXTURE, root / CONTRACT_RELPATH)
    return root


def test_render_call_orders_by_prototype(checkout: Path) -> None:
    decl = load_contract_set(checkout).require("arm_elementwise_add_s8")
    text = render_call(decl, {"block_size": "N", "output": "out", "input_2_vect": "b", "input_1_vect": "a"})
    assert text == (
        "arm_elementwise_add_s8(\n"
        "        a, /* input_1_vect */\n"
        "        b, /* input_2_vect */\n"
        "        out, /* output */\n"
        "        N /* block_size */\n"
        "    )"
    )
    assert render_call(load_contract_set(checkout).require("arm_nn_fx_helper"), {}) == "arm_nn_fx_helper()"


@pytest.mark.parametrize(
    "args, fragment",
    [
        ({"input_1_vect": "a", "input_2_vect": "b", "output": "out"}, "missing ['block_size']"),
        ({"input_1_vect": "a", "input_2_vect": "b", "output": "out", "block_size": "N", "n": "1"}, "unknown ['n']"),
        ({"input_1_vect": "a", "input_2_vect": "", "output": "out", "block_size": "N"}, "input_2_vect: argument expression is empty"),
    ],
)
def test_render_call_rejects_mismatched_arguments(checkout: Path, args, fragment: str) -> None:
    decl = load_contract_set(checkout).require("arm_elementwise_add_s8")
    with pytest.raises(ContractRenderError, match=re_escape(fragment)):
        render_call(decl, args)


def re_escape(text: str) -> str:
    import re
    return re.escape(text)


def test_function_type_adjusts_arrays_and_function_pointers(checkout: Path) -> None:
    contracts = load_contract_set(checkout)
    f16 = contracts.require("arm_fx_pool_f16")
    assert adjusted_type(f16.param("dims")) == "const int32_t *"
    assert adjusted_type(f16.param("hook")) == "void (*)(int32_t)"
    assert function_type(f16) == "arm_cmsis_nn_status (const int32_t *, arm_nn_tensor_layout, void (*)(int32_t), float16_t *)"
    assert function_type(contracts.require("arm_nn_fx_helper")) == "void (void)"
    assert render_parity_assert(contracts.require("arm_nn_fx_helper")).startswith(
        "_Static_assert(__builtin_types_compatible_p(__typeof__(arm_nn_fx_helper), void (void)),")


def _compile(cc: str, source: str, tmp_path: Path, *flags: str) -> subprocess.CompletedProcess:
    path = tmp_path / "probe.c"
    path.write_text(source)
    return subprocess.run([cc, *flags, "-std=c11", "-fsyntax-only", str(path)], capture_output=True, text=True)


@pytest.fixture
def host_cc() -> str:
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        if os.environ.get("CI"):
            pytest.fail("no host C compiler on a CI runner")
        pytest.skip("no host C compiler")
    return cc


def test_parity_assert_compiles_and_catches_drift(checkout: Path, tmp_path: Path, host_cc: str) -> None:
    contracts = load_contract_set(checkout)
    asserts = "\n".join(render_parity_assert(contracts.require(name)) for name in
                        ("arm_elementwise_add_s8", "arm_fx_pool_s8", "arm_fx_pool_f16", "arm_nn_fx_helper"))
    headers = "".join(HEADERS.values())
    good = _compile(host_cc, STUB_TYPES + headers + "\n" + asserts + "\n", tmp_path)
    assert good.returncode == 0, good.stderr
    # The header drifts (block_size becomes int16_t) while the export still says int32_t.
    drifted = headers.replace("int8_t *output, const int32_t block_size);", "int8_t *output, const int16_t block_size);")
    assert drifted != headers
    bad = _compile(host_cc, STUB_TYPES + drifted + "\n" + asserts + "\n", tmp_path)
    assert bad.returncode != 0
    assert "arm_elementwise_add_s8: prototype differs from the kernel contract export" in bad.stderr


def test_parity_assert_against_the_real_headers(tmp_path: Path, host_cc: str) -> None:
    root = resolve_cmsis_nn_root()
    contracts = load_contract_set(root)
    if not contracts.present:
        if os.environ.get("HELIA_CORE_TESTER_REQUIRE_CONTRACT"):
            pytest.fail(f"HELIA_CORE_TESTER_REQUIRE_CONTRACT is set but {root} has no kernel contract")
        pytest.skip("no ns-cmsis-nn checkout with a kernel contract")
    # Every exported function under the guards the harness builds with (F32 and F16 on).
    kernels = [decl for decl in contracts.functions.values()
               if set(decl.guards) <= {"ARM_NN_ENABLE_F32", "ARM_NN_ENABLE_F16"}]
    source = ('#include "arm_nnfunctions.h"\n#include "arm_nnsupportfunctions.h"\n'
              + "\n".join(render_parity_assert(decl) for decl in kernels) + "\n")
    flags = ["-DARM_NN_ENABLE_F32=1", "-DARM_NN_ENABLE_F16=1", f"-I{root / 'Include'}"]
    result = _compile(host_cc, source, tmp_path, *flags)
    assert result.returncode == 0, result.stderr[:4000]
    arm_gcc = shutil.which("arm-none-eabi-gcc")
    if arm_gcc is not None:
        # Hard float as the tester's CMake builds it; soft float has no float16_t and the
        # f16 headers refuse to compile, which would be a toolchain finding, not a parity one.
        result = _compile(arm_gcc, source, tmp_path, "-mcpu=cortex-m55", "-mthumb",
                          "-mfloat-abi=hard", "-mfpu=auto", *flags)
        assert result.returncode == 0, result.stderr[:4000]


def test_jinja_globals_render_and_refuse_an_absent_contract(checkout: Path) -> None:
    env = jinja2.Environment()
    env.globals.update(contract_globals(lambda: load_contract_set(checkout)))
    rendered = env.from_string(
        "{{ contract_call('arm_nn_fx_helper', {}) }};\n{{ contract_parity_assert('arm_nn_fx_helper') }}"
    ).render()
    assert rendered.startswith("arm_nn_fx_helper();\n_Static_assert(")
    with pytest.raises(ContractRenderError, match="unknown \\['x'\\]"):
        env.from_string("{{ contract_call('arm_nn_fx_helper', {'x': '1'}) }}").render()
    absent = jinja2.Environment()
    absent.globals.update(contract_globals(lambda: ContractSet(status=STATUS_ABSENT, root=checkout, path=None)))
    with pytest.raises(ContractRenderError, match="has no Tests/KernelContracts/kernel_contracts.json"):
        absent.from_string("{{ contract_call('arm_nn_fx_helper', {}) }}").render()


def test_template_environment_installs_the_globals_once(tmp_path: Path) -> None:
    (tmp_path / "t.j2").write_text("x")
    env = template_environment(str(tmp_path))
    assert env is template_environment(str(tmp_path))
    assert callable(env.globals["contract_call"]) and callable(env.globals["contract_parity_assert"])
