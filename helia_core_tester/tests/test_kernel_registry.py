"""assets/kernel_registry.yaml is the single source of the kernel ids the host sends and the
firmware dispatches on: the committed header block, the catalog and the registry must agree,
and scripts/generate_kernel_catalog.py must refuse a registry that contradicts itself or the
ns-cmsis-nn kernel contract export."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from helia_core_tester.contract.ir import CONTRACT_RELPATH
from helia_core_tester.hardware.kernel_registry import (
    AmbiguousKernelError,
    KernelRegistryError,
    UnknownKernelError,
    load_kernel_registry,
    lookup_kernel_id,
)
from helia_core_tester.tests.test_contract_ir import FIXTURE as CONTRACT_FIXTURE
from helia_core_tester.tests.test_contract_ir import HEADERS as CONTRACT_HEADERS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GENERATOR_SCRIPT = PROJECT_ROOT / "scripts" / "generate_kernel_catalog.py"
REGISTRY_PATH = PROJECT_ROOT / "assets" / "kernel_registry.yaml"
ADAPTERS_H_PATH = PROJECT_ROOT / "cmake" / "hardware" / "benchmark_server_adapters.h"
CATALOG_JSON_PATH = PROJECT_ROOT / "cmake" / "hardware" / "kernel_catalog.json"
DEFINE_RE = re.compile(r"^#define (HCT_KERNEL_ID_[A-Z0-9_]+) (\d+)u$", re.M)


def _load_generator():
    spec = importlib.util.spec_from_file_location("generate_kernel_catalog", GENERATOR_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = _load_generator()

# ids 6/7 must be Maximum/Minimum (the F008 invariant the generator asserts); the functions
# are the ones the shared contract fixture declares.
_ROWS = [
    (1, "HCT_KERNEL_ID_ADD_S8", "BasicMathFunctions", "Add", "S8", None, "arm_elementwise_add_s8"),
    (2, "HCT_KERNEL_ID_POOL_S8", "PoolingFunctions", "Pool", "S8", None, "arm_fx_pool_s8"),
    (3, "HCT_KERNEL_ID_POOL_F16", "PoolingFunctions", "Pool", "FP16", None, "arm_fx_pool_f16"),
    (4, "HCT_KERNEL_ID_POOL_S8_S4", "PoolingFunctions", "Pool", "S8", "S4", "arm_fx_pool_s8"),
    (5, "HCT_KERNEL_ID_HELPER", "NNSupportFunctions", "Helper", "S8", None, "arm_nn_fx_helper"),
    (6, "HCT_KERNEL_ID_MAXIMUM_S8", "BasicMathFunctions", "Maximum", "S8", None, "arm_elementwise_add_s8"),
    (7, "HCT_KERNEL_ID_MINIMUM_S8", "BasicMathFunctions", "Minimum", "S8", None, "arm_elementwise_add_s8"),
]

_HEADER_TEMPLATE = (
    "#ifndef FX_H\n#define FX_H\n#define HCT_BLOB_ROLE_INPUT_0 1u\n\n"
    f"{gen.KERNEL_ID_BLOCK_BEGIN}\n{gen.KERNEL_ID_BLOCK_END}\n\n"
    "static inline int fx(void) { return 0; }\n#endif\n"
)


def _registry_text(rows=_ROWS) -> str:
    lines = ["kernels:"]
    for kernel_id, c_define, family, operator, dtype, weight_dtype, function in rows:
        lines += [f"  - kernel_id: {kernel_id}", f"    c_define: {c_define}", f"    family: {family}",
                  f"    operator: {operator}", f"    dtype: {dtype}"]
        if weight_dtype is not None:
            lines.append(f"    weight_dtype: {weight_dtype}")
        lines.append(f"    cmsis_function: {function}")
    return "\n".join(lines) + "\n"


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "tester"
    (root / "assets").mkdir(parents=True)
    (root / "cmake" / "hardware").mkdir(parents=True)
    (root / "assets" / "kernel_registry.yaml").write_text(_registry_text())
    (root / gen.ADAPTERS_H_RELPATH).write_text(_HEADER_TEMPLATE)
    return root


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    root = tmp_path / "ns-cmsis-nn"
    for relative, text in CONTRACT_HEADERS.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(text)
    target = root / CONTRACT_RELPATH
    target.parent.mkdir(parents=True)
    shutil.copy(CONTRACT_FIXTURE, target)
    return root


def _set_registry(root: Path, text: str) -> None:
    (root / "assets" / "kernel_registry.yaml").write_text(text)


# --- the committed tree ---------------------------------------------------------------


def test_committed_header_defines_equal_the_registry() -> None:
    entries = load_kernel_registry(PROJECT_ROOT)
    defines = [(name, int(value)) for name, value in DEFINE_RE.findall(ADAPTERS_H_PATH.read_text(encoding="utf-8"))]
    assert defines == [(entry.c_define, entry.kernel_id) for entry in entries]
    assert len(entries) >= 173


def test_committed_catalog_json_equals_the_registry() -> None:
    entries = load_kernel_registry(PROJECT_ROOT)
    catalog = json.loads(CATALOG_JSON_PATH.read_text(encoding="utf-8"))
    assert [(row["kernel_id"], row["canonical_name"]) for row in catalog] == [
        (entry.kernel_id, entry.cmsis_function) for entry in entries
    ]


def test_generator_check_mode_passes_on_the_committed_tree() -> None:
    result = subprocess.run([sys.executable, str(GENERATOR_SCRIPT), "--check"], cwd=PROJECT_ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "kernel contract" in result.stdout


def test_committed_header_block_is_byte_stable() -> None:
    entries = load_kernel_registry(PROJECT_ROOT)
    header = ADAPTERS_H_PATH.read_text(encoding="utf-8")
    block = gen.render_kernel_id_block(entries)
    assert block == gen.render_kernel_id_block(entries)
    assert gen.splice_kernel_id_block(header, block) == header
    assert gen.splice_kernel_id_block(gen.splice_kernel_id_block(header, block), block) == header


def test_lookup_on_the_committed_registry() -> None:
    assert lookup_kernel_id(PROJECT_ROOT, family="BasicMathFunctions", operator="Abs", dtype="S8") == 1
    assert lookup_kernel_id(PROJECT_ROOT, family="ConvolutionFunctions", operator="Convolve", dtype="S8",
                            weight_dtype="S4") == 124
    with pytest.raises(UnknownKernelError):
        lookup_kernel_id(PROJECT_ROOT, family="BasicMathFunctions", operator="Abs", dtype="S4")


# --- the loader -----------------------------------------------------------------------


def test_loader_reads_every_column(project: Path) -> None:
    entries = load_kernel_registry(project)
    assert [entry.kernel_id for entry in entries] == [1, 2, 3, 4, 5, 6, 7]
    assert entries[3].weight_dtype == "S4" and entries[3].c_define == "HCT_KERNEL_ID_POOL_S8_S4"
    assert lookup_kernel_id(project, family="PoolingFunctions", operator="Pool", dtype="S8", weight_dtype="S4") == 4
    assert lookup_kernel_id(project, family="PoolingFunctions", operator="Pool", dtype="S8") == 2


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda t: t.replace("c_define: HCT_KERNEL_ID_POOL_F16", "c_define: HCT_KERNEL_ID_POOL_S8"),
         "c_define 'HCT_KERNEL_ID_POOL_S8' is used by kernel_id=2 and kernel_id=3"),
        (lambda t: t.replace("kernel_id: 3\n", "kernel_id: 2\n"), "kernel_id 2 is used by kernel_id=2 and kernel_id=2"),
        (lambda t: t.replace("c_define: HCT_KERNEL_ID_HELPER", "c_define: hct_kernel_id_helper"),
         "kernel_id=5 c_define 'hct_kernel_id_helper' must match"),
        (lambda t: t.replace("    c_define: HCT_KERNEL_ID_HELPER\n", ""), "kernel_id=5 c_define None must match"),
        (lambda t: t.replace("cmsis_function: arm_nn_fx_helper", "cmsis_function: fx_helper"),
         "kernel_id=5 cmsis_function 'fx_helper' must match"),
        (lambda t: t.replace("  - kernel_id: 5\n", "  - kernel_id: five\n"), "kernels[4] has no integer kernel_id"),
        (lambda t: t.replace("  - kernel_id: 5\n", "  - kernel_id: 0\n"), "kernel_id=0 must be positive"),
        (lambda t: t.replace("    dtype: S8\n    cmsis_function: arm_nn_fx_helper", "    cmsis_function: arm_nn_fx_helper"),
         "kernel_id=5 has no dtype"),
        (lambda t: "kernels: 3\n", "expected a mapping with a `kernels` list"),
        (lambda t: "kernels:\n  - 7\n", "kernels[0] is not a mapping"),
    ],
)
def test_loader_rejects_malformed_registries(project: Path, mutate, message: str) -> None:
    _set_registry(project, mutate(_registry_text()))
    with pytest.raises(KernelRegistryError, match=re.escape(message)):
        load_kernel_registry(project)


def test_loader_rejects_a_missing_registry(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_kernel_registry(tmp_path)


def test_lookup_reports_ambiguous_tuples(project: Path) -> None:
    rows = list(_ROWS) + [(8, "HCT_KERNEL_ID_ADD_S8_AGAIN", "BasicMathFunctions", "Add", "S8", None, "arm_elementwise_add_s8")]
    _set_registry(project, _registry_text(rows))
    with pytest.raises(AmbiguousKernelError):
        lookup_kernel_id(project, family="BasicMathFunctions", operator="Add", dtype="S8")


# --- the generator ----------------------------------------------------------------------


def test_generate_writes_the_block_and_check_then_passes(project: Path, capsys) -> None:
    assert gen.generate(check=True, project_root=project) == 1
    assert "kernel-id block is stale" in capsys.readouterr().err
    assert gen.generate(check=False, project_root=project) == 0
    header = (project / gen.ADAPTERS_H_RELPATH).read_text()
    assert header.startswith("#ifndef FX_H\n#define FX_H\n#define HCT_BLOB_ROLE_INPUT_0 1u\n\n")
    assert header.endswith("\n\nstatic inline int fx(void) { return 0; }\n#endif\n")
    assert DEFINE_RE.findall(header) == [(row[1], str(row[0])) for row in _ROWS]
    assert (project / gen.CATALOG_JSON_RELPATH).exists() and (project / gen.CATALOG_C_RELPATH).exists()
    assert gen.generate(check=True, project_root=project) == 0
    assert (project / gen.ADAPTERS_H_RELPATH).read_text() == header


def test_check_reports_each_stale_output(project: Path, capsys) -> None:
    assert gen.generate(check=False, project_root=project) == 0
    _set_registry(project, _registry_text().replace("HCT_KERNEL_ID_HELPER", "HCT_KERNEL_ID_HELPER_S8"))
    assert gen.generate(check=True, project_root=project) == 1
    err = capsys.readouterr().err
    assert "benchmark_server_adapters.h kernel-id block is stale" in err
    assert "kernel_catalog.json is stale" not in err
    _set_registry(project, _registry_text().replace("arm_nn_fx_helper", "arm_nn_fx_helper2"))
    assert gen.generate(check=True, project_root=project) == 1
    err = capsys.readouterr().err
    assert "kernel_catalog.json is stale" in err and "benchmark_server_catalog.c is stale" in err
    assert "kernel-id block is stale" not in err


def test_check_never_writes(project: Path) -> None:
    before = (project / gen.ADAPTERS_H_RELPATH).read_text()
    assert gen.generate(check=True, project_root=project) == 1
    assert (project / gen.ADAPTERS_H_RELPATH).read_text() == before
    assert not (project / gen.CATALOG_JSON_RELPATH).exists()


@pytest.mark.parametrize(
    ("header", "message"),
    [
        ("#define X 1\n", "must contain exactly one"),
        (_HEADER_TEMPLATE + gen.KERNEL_ID_BLOCK_END + "\n", "must contain exactly one"),
        (f"{gen.KERNEL_ID_BLOCK_END}\n{gen.KERNEL_ID_BLOCK_BEGIN}\n", "END marker precedes the BEGIN marker"),
    ],
)
def test_generate_refuses_a_header_without_a_single_marked_block(project: Path, header: str, message: str) -> None:
    (project / gen.ADAPTERS_H_RELPATH).write_text(header)
    with pytest.raises(ValueError, match=message):
        gen.generate(check=False, project_root=project)


def test_generate_refuses_a_missing_header(project: Path) -> None:
    (project / gen.ADAPTERS_H_RELPATH).unlink()
    with pytest.raises(ValueError, match="does not exist"):
        gen.generate(check=True, project_root=project)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda t: t.replace("operator: Maximum", "operator: Softmax"), "kernel_id=6 must be the Maximum kernel"),
        (lambda t: t.replace("operator: Minimum", "operator: Softmax"), "kernel_id=7 must be the Minimum kernel"),
        (lambda t: t.replace("    family: NNSupportFunctions\n", ""), "rows without a family: kernel_id=[5]"),
    ],
)
def test_generate_enforces_the_registry_invariants(project: Path, mutate, message: str) -> None:
    _set_registry(project, mutate(_registry_text()))
    with pytest.raises(ValueError, match=re.escape(message)):
        gen.generate(check=True, project_root=project)


def test_generate_validates_cmsis_functions_against_the_contract(project: Path, checkout: Path, capsys) -> None:
    assert gen.generate(check=False, project_root=project, cmsis_nn_root=checkout) == 0
    assert "all 7 cmsis_function names declared" in capsys.readouterr().out
    _set_registry(project, _registry_text().replace("cmsis_function: arm_nn_fx_helper", "cmsis_function: arm_nn_fx_gone"))
    with pytest.raises(ValueError, match=re.escape("kernel_id=5 arm_nn_fx_gone")) as info:
        gen.generate(check=True, project_root=project, cmsis_nn_root=checkout)
    assert str(checkout / CONTRACT_RELPATH) in str(info.value)


def test_generate_reports_an_absent_contract_and_carries_on(project: Path, tmp_path: Path, capsys) -> None:
    assert gen.generate(check=False, project_root=project, cmsis_nn_root=None) == 0
    assert "kernel contract absent (no ns-cmsis-nn checkout)" in capsys.readouterr().out
    bare = tmp_path / "old-checkout"
    (bare / "Include").mkdir(parents=True)
    assert gen.generate(check=True, project_root=project, cmsis_nn_root=bare) == 0
    assert f"kernel contract absent ({bare / CONTRACT_RELPATH})" in capsys.readouterr().out


def test_generate_fails_closed_on_a_corrupt_contract(project: Path, checkout: Path) -> None:
    (checkout / CONTRACT_RELPATH).write_text("{not json")
    result = subprocess.run(
        [sys.executable, str(GENERATOR_SCRIPT), "--check", "--project-root", str(project), "--cmsis-nn-root", str(checkout)],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "not valid JSON" in result.stderr


def test_script_exit_codes(project: Path, checkout: Path) -> None:
    def run(*extra: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(GENERATOR_SCRIPT), "--project-root", str(project), "--cmsis-nn-root", str(checkout), *extra],
            cwd=PROJECT_ROOT, capture_output=True, text=True,
        )

    assert run("--check").returncode == 1
    assert run().returncode == 0
    assert run("--check").returncode == 0
    _set_registry(project, _registry_text().replace("HCT_KERNEL_ID_POOL_F16", "HCT_KERNEL_ID_POOL_S8"))
    stale = run("--check")
    assert stale.returncode == 1 and "is used by kernel_id=2 and kernel_id=3" in stale.stderr
    assert not (project / gen.ADAPTERS_H_RELPATH).read_text().count("HCT_KERNEL_ID_POOL_S8 3u")
