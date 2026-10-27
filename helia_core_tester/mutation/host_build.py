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

Scope: the int suite. Two build modes: "dsp" ("Armv7E-M on host", the
mutation default) and "m0" (no ARM_MATH_* define at all: ns-cmsis-nn's
pure-C cortex-m0 configuration, which needs no shim and is what the
pre-FVP host check runs). Float kernels (f16/f32) are excluded from the host
library; f16 needs a host half-float story and is deferred with the FVP leg.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from helia_core_tester.utils.host_compiler import find_host_cc

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
    # arm_requantize_s8_s8 lives here and is called by the issue #81
    # chunked-equivalence requantize cases.
    "Source/QuantizationFunctions",
    # The tester#72 guard mutants target the int pooling and SVDF kernels.
    "Source/PoolingFunctions",
    "Source/SVDFunctions",
)

BUILD_MODES = ("dsp", "m0")

# Float kernels are not part of the mutation host build (see module docstring).
_EXCLUDED_NAME_PARTS = ("f16", "fp16", "f32", "_flt")
# The host check keeps the f32 sources: int cases call f32<->int bridges such as
# arm_quantize_f32_s8, and plain-C f32 compiles everywhere. f16 stays out until
# the host _Float16 story is settled.
HOST_CHECK_EXCLUDED_NAME_PARTS = ("f16", "fp16")

# Host DSP kernels need the DSP size.
_MVE_SIZER = re.compile(r"\b(arm_\w+_get_buffer_size)_mve\b")


class HostBuildError(RuntimeError):
    pass


# Failure kinds for CaseResult. Only genuine behavioural divergence
# (KIND_CASE_FAIL) or a hang (KIND_TIMEOUT) may count as a mutant kill; a
# case binary that fails to compile or link under a mutant proves nothing
# about the case's discriminating power and must never be scored as a kill.
KIND_PASS = "pass"
KIND_CASE_FAIL = "case_fail"
KIND_TIMEOUT = "timeout"
KIND_COMPILE_FAILED = "compile_failed"
KIND_NO_SOURCE = "no_source"
KILL_KINDS = (KIND_CASE_FAIL, KIND_TIMEOUT)


@dataclass
class CaseResult:
    name: str
    family: str
    passed: bool
    detail: str = ""
    kind: str = KIND_CASE_FAIL

    @property
    def killed(self) -> bool:
        """True when this failure is evidence against the mutant."""
        return not self.passed and self.kind in KILL_KINDS


def _run(cmd: Sequence[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def all_int_source_dirs(tree_root: Path) -> List[str]:
    """Every Source/*Functions directory of the tree (the host check builds them all)."""
    source = tree_root / "Source"
    if not source.is_dir():
        raise HostBuildError(f"kernel source dir missing: {source}")
    return sorted(f"Source/{d.name}" for d in source.iterdir() if d.is_dir() and d.name.endswith("Functions"))


def kernel_sources(
    tree_root: Path,
    source_dirs: Optional[Sequence[str]] = None,
    excluded_name_parts: Sequence[str] = _EXCLUDED_NAME_PARTS,
) -> List[Path]:
    files: List[Path] = []
    for rel in KERNEL_SOURCE_DIRS if source_dirs is None else source_dirs:
        directory = tree_root / rel
        if not directory.is_dir():
            raise HostBuildError(f"kernel source dir missing: {directory}")
        for f in sorted(directory.glob("*.c")):
            lowered = f.name.lower()
            if any(part in lowered for part in excluded_name_parts):
                continue
            files.append(f)
    return files


def _resolve_cc(cc: Optional[str]) -> str:
    return cc if cc else find_host_cc()


def _base_cflags(tree_root: Path, mode: str = "dsp") -> List[str]:
    if mode not in BUILD_MODES:
        raise HostBuildError(f"unknown host build mode {mode!r} (expected one of {', '.join(BUILD_MODES)})")
    flags = ["-O1", "-g", "-fno-strict-aliasing"]
    if mode == "dsp":
        flags += ["-DARM_MATH_DSP", "-include", str(DSP_SHIM)]
    return flags + ["-I", str(tree_root / "Include")]


def host_sizer_defines(sources: Iterable[Path]) -> List[str]:
    """Map MVE sizers to the host build's."""
    names = {m for src in sources for m in _MVE_SIZER.findall(src.read_text())}
    # Plain sizers dispatch on ARM_MATH_* macros.
    return [f"-D{name}_mve={name}" for name in sorted(names)]


def build_kernel_lib(
    tree_root: Path,
    out_dir: Path,
    cc: Optional[str] = None,
    jobs: int = 8,
    mode: str = "dsp",
    source_dirs: Optional[Sequence[str]] = None,
    excluded_name_parts: Sequence[str] = _EXCLUDED_NAME_PARTS,
) -> Path:
    """Compile the int kernel sources into a static library. Returns its path."""
    cc = _resolve_cc(cc)
    obj_dir = out_dir / "obj"
    if obj_dir.exists():
        shutil.rmtree(obj_dir)
    obj_dir.mkdir(parents=True)
    cflags = _base_cflags(tree_root, mode)

    def compile_one(src: Path) -> Optional[str]:
        obj = obj_dir / (src.stem + ".o")
        proc = _run([cc, "-c", *cflags, str(src), "-o", str(obj)])
        if proc.returncode != 0:
            return f"{src}:\n{proc.stderr}"
        return None

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        errors = [e for e in pool.map(compile_one, kernel_sources(tree_root, source_dirs, excluded_name_parts)) if e]
    if errors:
        raise HostBuildError("kernel library compile failed:\n" + "\n".join(errors[:5]))

    lib = out_dir / "libnn_host.a"
    if lib.exists():
        lib.unlink()
    proc = _run(["ar", "rcs", str(lib), *[str(o) for o in sorted(obj_dir.glob("*.o"))]])
    if proc.returncode != 0:
        raise HostBuildError(f"ar failed: {proc.stderr}")
    return lib


def build_runtime_obj(tester_root: Path, tree_root: Path, out_dir: Path, cc: Optional[str] = None) -> Path:
    """Compile the shared test runtime once, with helia_test_finish renamed away
    so the exiting host implementation in host_finish.c takes its place."""
    cc = _resolve_cc(cc)
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
            # helia_test_runtime.h pulls in arm_nnfunctions.h (for
            # ARM_CMSIS_NN_SUCCESS), so it needs the kernel tree's Include dir too.
            "-I",
            str(tree_root / "Include"),
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
    cc: Optional[str] = None,
    timeout_s: int = 60,
    mode: str = "dsp",
) -> CaseResult:
    """Compile one generated case against the kernel library and execute it."""
    cc = _resolve_cc(cc)
    name = case_dir.name
    family = case_dir.parent.name
    sources = sorted(case_dir.glob("*.c"))
    if not sources:
        return CaseResult(name, family, False, "no case source found", kind=KIND_NO_SOURCE)
    binary = bin_dir / name
    cmd = [
        cc,
        *_base_cflags(tree_root, mode),
        *host_sizer_defines(sources),
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
        return CaseResult(name, family, False, f"compile failed:\n{proc.stderr[-2000:]}", kind=KIND_COMPILE_FAILED)
    try:
        run_proc = _run([str(binary)], timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return CaseResult(name, family, False, f"timeout after {timeout_s}s", kind=KIND_TIMEOUT)
    if run_proc.returncode != 0:
        tail = (run_proc.stdout or "").strip().splitlines()
        return CaseResult(name, family, False, tail[-1] if tail else f"exit {run_proc.returncode}", kind=KIND_CASE_FAIL)
    return CaseResult(name, family, True, kind=KIND_PASS)


def run_all_cases(
    case_dirs: Sequence[Path],
    tree_root: Path,
    lib: Path,
    runtime_obj: Path,
    tester_root: Path,
    bin_dir: Path,
    cc: Optional[str] = None,
    jobs: int = 8,
    mode: str = "dsp",
    timeout_s: int = 60,
) -> List[CaseResult]:
    cc = _resolve_cc(cc)
    bin_dir.mkdir(parents=True, exist_ok=True)

    def one(case_dir: Path) -> CaseResult:
        return build_and_run_case(
            case_dir, tree_root, lib, runtime_obj, tester_root, bin_dir, cc=cc, timeout_s=timeout_s, mode=mode
        )

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(one, case_dirs))
