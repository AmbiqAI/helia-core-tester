"""Pre-FVP host check: run every generated int case on the host against the
ns-cmsis-nn kernels before any FVP or board build.

ns-cmsis-nn's cortex-m0 configuration (no ARM_MATH_* define) is pure C, so the
generated harness compiles and runs natively against it, validating its golden
exactly as it does on the FVP. "m0" is the default; "dsp" additionally checks
the Armv7E-M routes through the mutation harness's dsp_shim.h.

The kernel library and the test runtime are cached under
artifacts/host_kernels/<key>/, keyed by the checkout's kernel sources, the build
mode and flags, and the compiler identity; built under a lock and published by
an atomic rename like the reference library.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from helia_core_tester.mutation import host_build
from helia_core_tester.utils.file_lock import exclusive_lock
from helia_core_tester.utils.host_compiler import compiler_identity, find_host_cc

HOST_KERNEL_MODES = host_build.BUILD_MODES
CACHE_ENV = "HCT_HOST_KERNELS_CACHE"
REPORT_SCHEMA = "helia-core-tester/host-check/1"
# Bump whenever what goes into a cached build changes without the key's other
# inputs changing (source selection, layout), so old builds are never reused.
_CACHE_SCHEMA = 2
_KERNEL_SUBTREES = ("Include", "Source")

# Kernel-relevant capabilities each host build provides. A case generated for a
# CPU with more (cortex-m55's mve) may encode choices the host build cannot honour.
HOST_MODE_CAPABILITIES = {"m0": frozenset(), "dsp": frozenset({"dsp"})}
_KERNEL_CAPABILITIES = frozenset({"dsp", "mve"})

STATUS_BLOCKING = "blocking"
STATUS_ADVISORY = "cpu_specific"
STATUS_NOT_APPLICABLE = "not_applicable"


class HostCheckError(RuntimeError):
    """The host check could not run (build failure, missing checkout, bad mode)."""


@dataclass(frozen=True)
class HostKernelLibrary:
    mode: str
    tree: Path
    library: Path
    runtime_obj: Path
    key: str
    compiler: str


@dataclass
class HostCheckReport:
    mode: str
    cases_roots: List[str]
    tree: str
    tree_identity: Dict[str, str]
    library_key: str
    compiler: str
    seed: Optional[int]
    total: int = 0
    passed: int = 0
    failures: List[Dict[str, str]] = field(default_factory=list)
    # Failures of cases whose harness the generator specialised for a capability
    # the host build lacks: reported, never blocking.
    advisory: List[Dict[str, str]] = field(default_factory=list)
    # Cases that require a capability the host build lacks: not run.
    not_applicable: List[Dict[str, str]] = field(default_factory=list)
    schema: str = REPORT_SCHEMA

    @property
    def ok(self) -> bool:
        return self.total > 0 and not self.failures

    def to_json(self) -> Dict[str, object]:
        return asdict(self) | {"ok": self.ok}


def tester_root() -> Path:
    return Path(__file__).resolve().parents[3]


def kernel_tree_identity(tree: Path) -> Dict[str, str]:
    """Identity of a checkout's kernel sources: its commit when Include/ and
    Source/ are clean, plus their content digest when they are not (or when the
    root is not itself a git top level)."""
    from helia_core_tester.generation.reuse import _git_output, _is_git_toplevel, _subtree_digest

    tree = Path(tree).resolve()
    if _is_git_toplevel(tree):
        head = (_git_output(tree, "rev-parse", "HEAD") or "").strip()
        status = _git_output(tree, "status", "--porcelain", "--", *_KERNEL_SUBTREES)
        if head and status is not None and not status.strip():
            return {"state": "git-clean", "commit": head}
        return {"state": "git-dirty", "commit": head, "content": _subtree_digest(tree, _KERNEL_SUBTREES)}
    return {"state": "content", "content": _subtree_digest(tree, _KERNEL_SUBTREES)}


def _runtime_inputs(root: Path) -> List[Path]:
    runtime_dir = root / "src" / "test_runtime"
    return sorted(p for p in runtime_dir.rglob("*") if p.is_file()) + [host_build.DSP_SHIM, host_build.HOST_FINISH]


def host_kernels_key(tree: Path, mode: str, compiler: str) -> str:
    root = tester_root()
    payload = {
        "schema": _CACHE_SCHEMA,
        "tree": kernel_tree_identity(tree),
        "mode": mode,
        "cflags": [f for f in host_build._base_cflags(tree, mode) if not f.startswith("/")],
        "excluded": list(host_build.HOST_CHECK_EXCLUDED_NAME_PARTS),
        "compiler": compiler_identity(compiler),
        "runtime": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in _runtime_inputs(root)},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def default_cache_root() -> Path:
    override = os.environ.get(CACHE_ENV, "").strip()
    if override:
        return Path(override)
    from helia_core_tester.core.discovery import find_repo_root

    return find_repo_root() / "artifacts" / "host_kernels"


def ensure_host_kernels(
    tree: Path,
    mode: str = "m0",
    cache_root: Optional[Path] = None,
    cc: Optional[str] = None,
    jobs: Optional[int] = None,
) -> HostKernelLibrary:
    """Build (or reuse) the host kernel library and test runtime for `tree`."""
    if mode not in HOST_KERNEL_MODES:
        raise HostCheckError(f"unknown host kernel mode {mode!r} (expected one of {', '.join(HOST_KERNEL_MODES)})")
    tree = Path(tree).resolve()
    if not (tree / "Include").is_dir() or not (tree / "Source").is_dir():
        raise HostCheckError(f"{tree} is not an ns-cmsis-nn checkout (needs Include/ and Source/)")
    cc = cc or find_host_cc()
    key = host_kernels_key(tree, mode, cc)
    root = Path(cache_root) if cache_root is not None else default_cache_root()
    final = root / key
    lib_name = f"libnn_{mode}.a"
    runtime_name = "helia_test_runtime_host.o"

    def result() -> HostKernelLibrary:
        return HostKernelLibrary(mode, tree, final / lib_name, final / runtime_name, key, cc)

    if (final / lib_name).is_file() and (final / runtime_name).is_file():
        return result()
    with exclusive_lock(root / f".{key}.lock"):
        if (final / lib_name).is_file() and (final / runtime_name).is_file():
            return result()
        root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=root))
        try:
            excluded = host_build.HOST_CHECK_EXCLUDED_NAME_PARTS
            source_dirs = host_build.all_int_source_dirs(tree)
            try:
                built = host_build.build_kernel_lib(
                    tree,
                    staging,
                    cc=cc,
                    jobs=jobs or os.cpu_count() or 4,
                    mode=mode,
                    source_dirs=source_dirs,
                    excluded_name_parts=excluded,
                )
                runtime = host_build.build_runtime_obj(tester_root(), tree, staging, cc=cc)
            except host_build.HostBuildError as exc:
                raise HostCheckError(f"host kernel build ({mode}) failed: {exc}") from exc
            built.rename(staging / lib_name)
            if runtime.name != runtime_name:
                runtime.rename(staging / runtime_name)
            shutil.rmtree(staging / "obj", ignore_errors=True)
            (staging / "flags.json").write_text(
                json.dumps(
                    {
                        "mode": mode,
                        "compiler": cc,
                        "compiler_identity": compiler_identity(cc),
                        "cflags": host_build._base_cflags(tree, mode),
                        "source_dirs": source_dirs,
                        "excluded_name_parts": list(excluded),
                        "tree": str(tree),
                        "tree_identity": kernel_tree_identity(tree),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            if final.exists():
                shutil.rmtree(final)
            os.replace(staging, final)
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
    return result()


def manifest_cases(cases_root: Path) -> Optional[tuple[List[tuple[Path, Optional[str]]], Optional[int]]]:
    """The runnable cases and run seed manifest.json records for a generated tree,
    or None when the tree has no manifest. These are exactly the cases the FVP
    build compiles (tests.cmake is written from the same entries)."""
    manifest_path = Path(cases_root) / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HostCheckError(f"unreadable manifest {manifest_path}: {exc}") from exc
    cases = []
    default_cpu = (manifest.get("filters") or {}).get("cpu")
    for entry in manifest.get("tests", []):
        if not entry.get("c_sources"):
            continue
        case_dir = Path(cases_root) / str(entry["relative_test_dir"])
        if not case_dir.is_dir():
            raise HostCheckError(f"manifest lists {case_dir}, which does not exist")
        cases.append((case_dir, entry.get("cpu") or default_cpu))
    seed = manifest.get("run_seed")
    return sorted(cases), (int(seed) if seed is not None else None)


def _cpu_from_path(root: Path) -> Optional[str]:
    """artifacts/generated_tests/<suite>/<cpu>: the tree's directory names its CPU."""
    from helia_core_tester.core.cpu_targets import get_cpu_profile

    try:
        get_cpu_profile(root.name)
    except (KeyError, ValueError):
        return None
    return root.name


def int_case_dirs(cases_roots: Iterable[Path]) -> tuple[List[tuple[Path, Optional[str]]], Optional[int]]:
    """(case dir, target cpu) to check under each root (from its manifest when
    present) and the run seed they were drawn from (None when unrecorded or mixed)."""
    cases: List[tuple[Path, Optional[str]]] = []
    seeds = set()
    for root in (Path(r) for r in cases_roots):
        if not root.is_dir():
            raise HostCheckError(f"generated cases root not found: {root}")
        listed = manifest_cases(root)
        if listed is None:
            cpu = _cpu_from_path(root)
            cases += [(case, cpu) for case in host_build.discover_cases([root])]
            seeds.add(None)
        else:
            cases += listed[0]
            seeds.add(listed[1])
    return cases, (seeds.pop() if len(seeds) == 1 else None)


@dataclass(frozen=True)
class CaseClass:
    status: str
    reason: str = ""


def classify_case(case_dir: Path, target_cpu: Optional[str], mode: str) -> CaseClass:
    """How a host run of this case under `mode` is judged.

    - not_applicable: the descriptor requires a capability the host build lacks;
      the harness would exercise a path the host kernels do not have.
    - cpu_specific: the generator specialised the harness for a capability of the
      target CPU that the host build lacks (an entry or fault reachability resolved
      for that CPU, or FullyConnected s8 folding the bias into MVE-only kernel
      sums, see OpFullyConnected._supports_weight_sum); its result is advisory.
    - blocking: everything else, including any case whose target CPU is unknown.
    """
    import yaml

    from helia_core_tester.core.cpu_targets import get_cpu_profile

    host_caps = HOST_MODE_CAPABILITIES[mode]
    descriptor_path = Path(case_dir) / "descriptor.yaml"
    desc: Dict[str, object] = {}
    if descriptor_path.is_file():
        try:
            desc = yaml.safe_load(descriptor_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise HostCheckError(f"unreadable case descriptor {descriptor_path}: {exc}") from exc
    required = {str(c).lower() for c in (desc.get("required_capabilities") or ())}
    missing = sorted((required & _KERNEL_CAPABILITIES) - host_caps)
    if missing:
        return CaseClass(STATUS_NOT_APPLICABLE, f"requires {', '.join(missing)}")
    if target_cpu is None:
        return CaseClass(STATUS_BLOCKING)
    try:
        target_caps = get_cpu_profile(target_cpu).capabilities & _KERNEL_CAPABILITIES
    except (KeyError, ValueError):
        return CaseClass(STATUS_BLOCKING)
    extra = sorted(target_caps - host_caps)
    if not extra:
        return CaseClass(STATUS_BLOCKING)
    why = f"generated for {target_cpu} ({', '.join(extra)})"
    if desc.get("entry"):
        return CaseClass(STATUS_ADVISORY, f"entry resolved {why}")
    if desc.get("fault"):
        return CaseClass(STATUS_ADVISORY, f"fault reachability decided {why}")
    if (
        str(desc.get("operator")) == "FullyConnected"
        and str(desc.get("activation_dtype", "S8")).upper() == "S8"
        and str(desc.get("weight_dtype", "S8")).upper() == "S8"
        and "mve" in extra
    ):
        return CaseClass(STATUS_ADVISORY, f"bias folded into MVE-only kernel sums, {why}")
    return CaseClass(STATUS_BLOCKING)


def headline(detail: str) -> str:
    """The most telling line of a failure detail: the first compiler/linker
    error or undefined symbol, else the last line (the harness's verdict)."""
    lines = [line.strip() for line in detail.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        lowered = line.lower()
        if lowered.startswith("undefined symbols") and index + 1 < len(lines):
            # The symbol is on the next line (ld64); GNU ld puts it inline.
            return f"{line} {lines[index + 1]}"
        if "error:" in lowered or "undefined" in lowered:
            return line
    return lines[-1] if lines else ""


def repro_hint(seed: Optional[int], name: str) -> str:
    return f"--seed {seed} --name {name}" if seed is not None else f"--name {name}"


def run_host_check(
    cases_roots: Sequence[Path],
    tree: Path,
    mode: str = "m0",
    seed: Optional[int] = None,
    jobs: Optional[int] = None,
    cc: Optional[str] = None,
    cache_root: Optional[Path] = None,
    work_dir: Optional[Path] = None,
    timeout_s: int = 60,
) -> HostCheckReport:
    """Compile and run every case under `cases_roots` against the host kernels.

    Every non-pass (case failure, timeout, compile failure, missing source) is a
    failure: a case the host cannot even build has not been checked.
    """
    listed, recorded_seed = int_case_dirs(cases_roots)
    seed = seed if seed is not None else recorded_seed
    classes = {case: classify_case(case, cpu, mode) for case, cpu in listed}
    cases = [case for case, cls in classes.items() if cls.status != STATUS_NOT_APPLICABLE]
    library = ensure_host_kernels(tree, mode, cache_root=cache_root, cc=cc, jobs=jobs)
    report = HostCheckReport(
        mode=mode,
        cases_roots=[str(Path(r)) for r in cases_roots],
        tree=str(library.tree),
        tree_identity=kernel_tree_identity(library.tree),
        library_key=library.key,
        compiler=f"{library.compiler} ({compiler_identity(library.compiler)})",
        seed=seed,
    )
    report.not_applicable = [
        {"family": case.parent.name, "name": case.name, "reason": cls.reason}
        for case, cls in sorted(classes.items())
        if cls.status == STATUS_NOT_APPLICABLE
    ]
    if not cases:
        return report
    own_work = work_dir is None
    bin_dir = Path(tempfile.mkdtemp(prefix="hct-host-check-")) if own_work else Path(work_dir) / f"bin-{mode}"
    try:
        results = host_build.run_all_cases(
            cases,
            library.tree,
            library.library,
            library.runtime_obj,
            tester_root(),
            bin_dir,
            cc=library.compiler,
            jobs=jobs or os.cpu_count() or 4,
            mode=mode,
            timeout_s=timeout_s,
        )
    finally:
        if own_work:
            shutil.rmtree(bin_dir, ignore_errors=True)
    report.total = len(results)
    report.passed = sum(1 for r in results if r.passed)
    by_dir = dict(zip(cases, results))
    for case_dir, result in sorted(by_dir.items(), key=lambda item: (item[1].family, item[1].name)):
        if result.passed:
            continue
        cls = classes[case_dir]
        record = {
            "family": result.family,
            "name": result.name,
            "kind": result.kind,
            "headline": headline(result.detail),
            "detail": result.detail[-2000:],
            "repro": repro_hint(seed, result.name),
        }
        if cls.status == STATUS_ADVISORY:
            report.advisory.append({**record, "reason": cls.reason})
        else:
            report.failures.append(record)
    return report


def write_report(report: HostCheckReport, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_json(), indent=2) + "\n", encoding="utf-8")
    return path
