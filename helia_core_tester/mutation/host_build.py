"""
Host build + execution of generated cases against a (possibly mutated)
ns-cmsis-nn tree.

This reuses the proven approach of the PR #88 review harness: the generated
test .c files and the real ns-cmsis-nn int kernels are compiled with the
host C compiler, with the Armv7E-M DSP intrinsics emulated by
host/dsp_shim.h (pre-included via -include so its macros pre-empt the
unavailable ACLE definitions). That makes a full mutation run take minutes
instead of the hours an FVP sweep would need. FVP-based scoring is an
explicit non-goal of this MVP (see issue #76).

Scope: the int suite, DSP ("Armv7E-M on host") build mode. Float kernels
(f16/f32) are excluded from the host library; f16 needs a host half-float
story and is deferred with the FVP leg.
"""

from __future__ import annotations

import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

_HOST_DIR = Path(__file__).resolve().parent / "host"
DSP_SHIM = _HOST_DIR / "dsp_shim.h"
HOST_FINISH = _HOST_DIR / "host_finish.c"

# Source subdirectories compiled into the host kernel library. Enough for the
# elementwise + convolution families scored by catalog v1; extend as the
# catalog grows.
KERNEL_SOURCE_DIRS = (
    "Source/BasicMathFunctions",
    "Source/ConvolutionFunctions",
    "Source/NNSupportFunctions",
    "Source/FullyConnectedFunctions",
    "Source/ActivationFunctions",
)

# Float kernels are not part of the host build (see module docstring).
_EXCLUDED_NAME_PARTS = ("f16", "fp16", "f32", "_flt")


class HostBuildError(RuntimeError):
    pass


@dataclass
class CaseResult:
    name: str
    family: str
    passed: bool
    detail: str = ""


def _run(cmd: Sequence[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def kernel_sources(tree_root: Path) -> List[Path]:
    files: List[Path] = []
    for rel in KERNEL_SOURCE_DIRS:
        directory = tree_root / rel
        if not directory.is_dir():
            raise HostBuildError(f"kernel source dir missing: {directory}")
        for f in sorted(directory.glob("*.c")):
            lowered = f.name.lower()
            if any(part in lowered for part in _EXCLUDED_NAME_PARTS):
                continue
            files.append(f)
    return files


def _base_cflags(tree_root: Path) -> List[str]:
    return [
        "-O1",
        "-g",
        "-fno-strict-aliasing",
        "-DARM_MATH_DSP",
        "-include",
        str(DSP_SHIM),
        "-I",
        str(tree_root / "Include"),
    ]


def build_kernel_lib(tree_root: Path, out_dir: Path, cc: str = "gcc", jobs: int = 8) -> Path:
    """Compile the int kernel sources into a static library. Returns its path."""
    obj_dir = out_dir / "obj"
    if obj_dir.exists():
        shutil.rmtree(obj_dir)
    obj_dir.mkdir(parents=True)
    cflags = _base_cflags(tree_root)

    def compile_one(src: Path) -> Optional[str]:
        obj = obj_dir / (src.stem + ".o")
        proc = _run([cc, "-c", *cflags, str(src), "-o", str(obj)])
        if proc.returncode != 0:
            return f"{src}:\n{proc.stderr}"
        return None

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        errors = [e for e in pool.map(compile_one, kernel_sources(tree_root)) if e]
    if errors:
        raise HostBuildError("kernel library compile failed:\n" + "\n".join(errors[:5]))

    lib = out_dir / "libnn_host.a"
    if lib.exists():
        lib.unlink()
    proc = _run(["ar", "rcs", str(lib), *[str(o) for o in sorted(obj_dir.glob("*.o"))]])
    if proc.returncode != 0:
        raise HostBuildError(f"ar failed: {proc.stderr}")
    return lib


def build_runtime_obj(tester_root: Path, out_dir: Path, cc: str = "gcc") -> Path:
    """Compile the shared test runtime once, with helia_test_finish renamed away
    so the exiting host implementation in host_finish.c takes its place."""
    src = tester_root / "src" / "test_runtime" / "helia_test_runtime.c"
    obj = out_dir / "helia_test_runtime_host.o"
    proc = _run(
        [
            cc,
            "-c",
            "-O1",
            "-Dhelia_test_finish=helia_test_finish_fvp_unused",
            "-I",
            str(tester_root / "src"),
            str(src),
            "-o",
            str(obj),
        ]
    )
    if proc.returncode != 0:
        raise HostBuildError(f"runtime compile failed:\n{proc.stderr}")
    return obj


def discover_cases(cases_roots: Iterable[Path]) -> List[Path]:
    """Find generated case directories (a dir containing exactly one top-level
    generated .c file plus an includes/ dir) under the given roots."""
    cases: List[Path] = []
    for root in cases_roots:
        root = Path(root)
        for c_file in sorted(root.rglob("*_*.c")):
            case_dir = c_file.parent
            if (case_dir / "includes").is_dir() and case_dir not in cases:
                cases.append(case_dir)
    return cases


def build_and_run_case(
    case_dir: Path,
    tree_root: Path,
    lib: Path,
    runtime_obj: Path,
    tester_root: Path,
    bin_dir: Path,
    cc: str = "gcc",
    timeout_s: int = 60,
) -> CaseResult:
    """Compile one generated case against the kernel library and execute it."""
    name = case_dir.name
    family = case_dir.parent.name
    sources = sorted(case_dir.glob("*.c"))
    if not sources:
        return CaseResult(name, family, False, "no case source found")
    binary = bin_dir / name
    cmd = [
        cc,
        *_base_cflags(tree_root),
        "-I",
        str(tester_root / "src"),
        "-I",
        str(case_dir),
        "-I",
        str(case_dir / "includes"),
        *[str(s) for s in sources],
        str(runtime_obj),
        str(HOST_FINISH),
        str(lib),
        "-lm",
        "-o",
        str(binary),
    ]
    proc = _run(cmd)
    if proc.returncode != 0:
        return CaseResult(name, family, False, f"compile failed:\n{proc.stderr[-2000:]}")
    try:
        run_proc = _run([str(binary)], timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return CaseResult(name, family, False, f"timeout after {timeout_s}s")
    if run_proc.returncode != 0:
        tail = (run_proc.stdout or "").strip().splitlines()
        return CaseResult(name, family, False, tail[-1] if tail else f"exit {run_proc.returncode}")
    return CaseResult(name, family, True)


def run_all_cases(
    case_dirs: Sequence[Path],
    tree_root: Path,
    lib: Path,
    runtime_obj: Path,
    tester_root: Path,
    bin_dir: Path,
    cc: str = "gcc",
    jobs: int = 8,
) -> List[CaseResult]:
    bin_dir.mkdir(parents=True, exist_ok=True)

    def one(case_dir: Path) -> CaseResult:
        return build_and_run_case(case_dir, tree_root, lib, runtime_obj, tester_root, bin_dir, cc=cc)

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(one, case_dirs))
