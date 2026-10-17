"""Rerun candidate rules on what the preprocessor makes.

Line rules read source text, so they miss tokens the preprocessor
builds (`_Pra##gma`, `__attri##bute__`, nested macros, macros a
header defines and an unchanged unit uses). Every Source/ C and .S
unit, in the base and the candidate, goes through `gcc -E` for each
core in CONFIGS. Lines are credited to the kernel file they came from
(system headers drop out), and a rule whose count grows in a file is
a finding. Counts per file: swapping one existing hit for another
stays unseen. Each gcc -E run has a time and output cap; a unit
that hits either is a scan_error. The whole scan has a deadline and a
unit limit: past either, or at the first new failure, outstanding
runs are killed and one scan_error names the cause.

Clean unit scans are memoized by every input gcc -E reads: compiler
binaries and env, flags, flag files, the Source/ and Include/ listing,
and each file its line markers name, checked by content on reuse. A
cache dir also keeps base tree scans across runs.

With a build dir, `gcc -E` also runs with that build's own compile
args (board and SoC macros, include paths), and each kernel object
from its compile_commands.json must hold code in .text only, standard
sections only, and no SCS address (0xE0000000-0xE00FFFFF: DWT, PMU,
SysTick, SCB) in a literal pool, mov/movw/movt or data word. Addresses
built at run time stay out of reach. No loaded section may hold a
cps or an msr other than an APSR flag write at any halfword, and
none, decoded as Thumb whether marked code or data, may hold wfi/wfe.
Code sections are also decoded the way the core runs them: from the
start of each code span (mapping symbols and function symbols) with
exact Thumb sizes. No instruction may be cut by a label, data or the
section end, and no code span may fall into data or off the end.
Every symbol, relocation and direct branch into code must land on a
decoded instruction start; a branch or odd address into data is
refused. Direct branch targets are decoded from those starts, not from
a disassembler listing, so a branch written as data rows is followed
too; a branch leaving its section, or any branch future (bf, bfx,
bfl, bflx, bfcsel), is refused. In code, a data relocation must write
only bytes of one data span, and an instruction relocation must sit
on an instruction it is defined for (opcode and width); any other
relocation kind, or a relocation row the scan cannot read, is
refused. So bytes inside an instruction or in data are never entered
by name.

The link fills relocations in, so the linked image
(hardware/hct_benchmark_server.elf and its .map, GNU ld or lld) is
read too: each kernel code section the map places must equal the
object outside relocated bytes, and every instruction the link
changed gets the instruction and SCS checks again. A missing image or
map, a map in another format, or a map placing none of the kernel code
fails the scan. Jump-table entries, data reached by pointer, kernel
data sections in the image, branch targets the link resolves, and
addresses or code built at run time stay out of reach.
"""

from __future__ import annotations

import functools
import hashlib
import io
import json
import math
import mmap
import os
import re
import shlex
import selectors
import signal
import stat
import struct
import subprocess
import tarfile
import tempfile
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, ThreadPoolExecutor, wait
from itertools import islice
from pathlib import Path
from typing import Callable, Optional

from .toolchain import arm_tool, toolchain_spec

# Cores the boards use: MVE, DSP.
_COMMON = ("-mthumb", "-mfloat-abi=hard", "-Ofast", "-ffast-math", "-DARM_NN_ENABLE_F32=1")
CONFIGS = {
    "cortex-m55": ("-mcpu=cortex-m55", *_COMMON, "-DARM_NN_ENABLE_F16=1", "-DCMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY"),
    "cortex-m4": ("-mcpu=cortex-m4", "-mfpu=fpv4-sp-d16", *_COMMON),
}
# Caps per gcc -E run.
TIMEOUT_S = 30.0
OUTPUT_CAP = 64 << 20
# Output all workers may buffer.
SCAN_MEMORY_BUDGET = 256 << 20
# Whole-scan limits.
SCAN_DEADLINE_S = 300.0
MAX_UNITS = 2000
# Object size caps, checked before reading.
OBJECT_CAP = 16 << 20
OBJECTS_TOTAL_CAP = 128 << 20
# Dep variants kept per cache key.
CACHE_VARIANTS = 8
# Abort-file poll interval.
_POLL_S = 0.25
_MARKER = re.compile(r'^#\s*\d+\s+"([^"]*)".*$', re.MULTILINE)


def _units(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for ext in ("*.c", "*.S") for p in (root / "Source").rglob(ext))


class ScanStopped(Exception):
    """The scan hit a limit or failed."""

    def __init__(self, path: str, message: str) -> None:
        super().__init__(message)
        self.finding = {"rule": "scan_error", "path": path, "message": message}


def run_capped(cmd: list[str], cwd: Path, timeout: float, cap: int, abort: str = "") -> Optional[bytes]:
    """Stdout, or None on failure, cap or abort file."""
    proc = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, start_new_session=True)
    deadline = time.monotonic() + timeout
    chunks: list[bytes] = []
    size = 0
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(proc.stdout, selectors.EVENT_READ)
            while True:
                left = deadline - time.monotonic()
                if left <= 0 or (abort and os.path.exists(abort)):
                    return None
                if not selector.select(min(left, _POLL_S)):
                    continue
                chunk = os.read(proc.stdout.fileno(), 1 << 16)
                if not chunk:
                    break
                size += len(chunk)
                if size > cap:
                    return None
                chunks.append(chunk)
        if proc.wait(max(0.0, deadline - time.monotonic())) != 0:
            return None
    except subprocess.TimeoutExpired:
        return None
    finally:
        # Kill gcc and its cc1 child.
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
        proc.stdout.close()
    return b"".join(chunks)


def run_binutil(tool: str, args: list[str], deadline: float = math.inf, abort: str = "") -> str:
    """Bounded binutil stdout; raises when capped."""
    timeout = min(TIMEOUT_S, deadline - time.monotonic())
    if timeout <= 0:
        raise ValueError(f"{tool}: scan deadline passed")
    out = run_capped([arm_tool(tool), *args], Path.cwd(), timeout, OUTPUT_CAP, abort)
    if out is None:
        raise ValueError(f"{tool} failed or hit limits")
    return out.decode("utf-8", "replace")


def extract_tar(archive: bytes, dest: Path) -> None:
    """Extract regular files and dirs only.

    No extractall(filter=): Python 3.11.0-3.11.12 lack it.
    """
    root = dest.resolve()
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar.getmembers():
            target = (root / member.name).resolve()
            parts = Path(member.name).parts
            if Path(member.name).is_absolute() or ".." in parts or not (member.isfile() or member.isdir()):
                raise tarfile.TarError(f"unsafe tar member {member.name!r}")
            if target != root and root not in target.parents:
                raise tarfile.TarError(f"unsafe tar member {member.name!r}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                raise tarfile.TarError(f"unreadable tar member {member.name!r}")
            target.write_bytes(source.read())


def scan_workers() -> int:
    """Pool size that fits the budget."""
    return max(1, min(os.cpu_count() or 1, SCAN_MEMORY_BUDGET // OUTPUT_CAP))


def _preprocess(
    gcc: str, root: Path, unit: str, flags: tuple[str, ...], deadline: float, abort: str,
) -> Optional[tuple[dict[str, str], set[str]]]:
    """Kernel text by origin, plus every file read."""
    timeout = min(TIMEOUT_S, deadline - time.monotonic())
    if timeout <= 0:
        return None
    out = run_capped([gcc, "-E", *flags, "-IInclude", unit], root, timeout, OUTPUT_CAP, abort)
    if out is None:
        return None
    stdout = out.decode(errors="replace")
    by_file: dict[str, list[str]] = {}
    names: set[str] = set()
    markers = list(_MARKER.finditer(stdout))
    for marker, after in zip(markers, [*markers[1:], None]):
        name = marker.group(1)
        # -g adds a "cwd//" marker.
        if not name.startswith("<") and not name.endswith("/"):
            names.add(name)
        # System headers and builtins drop out.
        if name.startswith(("/", "<")):
            continue
        end = after.start() if after else len(stdout)
        by_file.setdefault(Path(name).as_posix(), []).append(stdout[marker.end():end])
    return {origin: "".join(chunks) for origin, chunks in by_file.items()}, names


def _unit_flags(config: tuple | dict, unit: str) -> tuple[str, ...]:
    return config.get(unit, config["*"]) if isinstance(config, dict) else config


# Worker-side counts, keyed by text digest.
_MEMO: dict[bytes, Counter] = {}


def _unit_counts(
    gcc: str, root: Path, unit: str, flags: tuple[str, ...], deadline: float, abort: str,
    rule_counts: Callable[[str], Counter],
) -> Optional[tuple[dict[str, Counter], Optional[dict[str, str]]]]:
    """Rule counts of one unit by origin, plus deps.

    Runs in the worker, so only counts reach the parent. Deps map
    each file gcc read to its digest; None means do not cache.
    """
    found = _preprocess(gcc, root, unit, flags, deadline, abort)
    if found is None:
        return None
    by_file, names = found
    counts = {}
    for origin, text in by_file.items():
        # Headers repeat across units.
        key = hashlib.blake2b(text.encode(errors="replace"), digest_size=16).digest()
        if key not in _MEMO:
            _MEMO[key] = rule_counts(text)
        counts[origin] = _MEMO[key]
    deps = {name: _file_digest(root, name, {}) for name in names}
    # Markers must name the unit itself.
    cacheable = unit in deps and all(deps.values())
    return counts, deps if cacheable else None


# --- scan cache -----------------------------------------------------------------------

# Clean unit scans, by input key.
_SCANS: dict[str, list[dict]] = {}
# Env vars gcc -E reads.
_GCC_ENV = ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "OBJC_INCLUDE_PATH", "GCC_EXEC_PREFIX",
            "COMPILER_PATH", "SOURCE_DATE_EPOCH", "DEPENDENCIES_OUTPUT", "SUNPRO_DEPENDENCIES")


def _file_digest(root: Path, name: str, seen: dict) -> Optional[str]:
    """sha256 of root/name, or None."""
    path = os.path.join(root, name)
    if path not in seen:
        seen[path] = None
        try:
            # Devices and FIFOs never block.
            fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        except OSError:
            return None
        # fdopen refuses directories.
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            return None
        with os.fdopen(fd, "rb") as handle:
            seen[path] = hashlib.file_digest(handle, "sha256").hexdigest()
    return seen[path]


@functools.lru_cache(maxsize=None)
def _compiler_id(gcc: str) -> str:
    """Driver, cc1 and gcc env."""
    real = Path(os.path.realpath(gcc))
    digest = hashlib.sha256(str(real).encode())
    # Standard layout: libexec/gcc/<target>/<version>/cc1.
    for path in [real, *sorted(real.parent.parent.glob("libexec/gcc/*/*/cc1"))]:
        digest.update(f"{path}\0{_file_digest(Path('/'), str(path), {})}\0".encode())
    for name in _GCC_ENV:
        digest.update(f"{name}={os.environ.get(name)}\0".encode())
    return digest.hexdigest()


@functools.lru_cache(maxsize=None)
def _rules_id(rule_counts: Callable[[str], Counter]) -> str:
    """Digest of the rule code."""
    import sys

    from . import c_lex

    files = {sys.modules[rule_counts.__module__].__file__, __file__, c_lex.__file__}
    digest = hashlib.sha256(f"{rule_counts.__module__}.{rule_counts.__qualname__}\0".encode())
    for name in sorted(files):
        digest.update(Path(name).read_bytes())
    return digest.hexdigest()


def _listing(root: Path) -> str:
    """Digest of every Source/ and Include/ path."""
    digest = hashlib.sha256()
    for top in ("Include", "Source"):
        for path in sorted((root / top).rglob("*")):
            rel = path.relative_to(root).as_posix()
            kind = "L" + os.readlink(path) if path.is_symlink() else "D" if path.is_dir() else "F"
            digest.update(f"{rel}\0{kind}\0".encode())
    return digest.hexdigest()


def _unit_key(base: str, root: Path, unit: str, flags: tuple[str, ...], seen: dict) -> str:
    """Key of one unit scan."""
    digest = hashlib.sha256(f"{base}\0{unit}\0{_file_digest(root, unit, seen)}\0".encode())
    for flag in flags:
        digest.update(f"{flag}\0".encode())
        # Flag files: -include, -imacros, @file.
        for name in (flag, flag[1:]):
            if name and os.path.isfile(os.path.join(root, name)):
                digest.update(f"{_file_digest(root, name, seen)}\0".encode())
    return digest.hexdigest()


def _cached(key: str, root: Path, cache: Optional[Path], seen: dict) -> Optional[dict[str, Counter]]:
    """Counts whose deps all match."""
    entries = _SCANS.get(key)
    if entries is None and cache is not None:
        try:
            entries = _SCANS[key] = json.loads((cache / f"{key}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            entries = None
    for entry in entries or ():
        if all(_file_digest(root, name, seen) == want for name, want in entry["deps"].items()):
            return {origin: Counter(found) for origin, found in entry["counts"].items()}
    return None


def _remember(key: str, deps: dict[str, str], counts: dict[str, Counter]) -> dict:
    entry = {"deps": deps, "counts": {origin: dict(found) for origin, found in counts.items()}}
    _SCANS.setdefault(key, []).append(entry)
    return entry


def _store(cache: Path, entries: dict[str, list[dict]]) -> None:
    """Add entries to disk; races keep one."""
    cache.mkdir(parents=True, exist_ok=True)
    for key, found in entries.items():
        path = cache / f"{key}.json"
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            old = []
        kept = [entry for entry in old if entry not in found] + found
        tmp = cache / f".{key}.{os.getpid()}"
        tmp.write_text(json.dumps(kept[-CACHE_VARIANTS:]), encoding="utf-8")
        tmp.replace(path)


def _tree_counts(
    gcc: str, root: Path, rule_counts: Callable[[str], Counter], configs: dict, deadline: float, abort: str,
    fatal: Callable[[str], bool] = lambda key: False, cache: Optional[Path] = None, store: bool = False,
) -> tuple[dict, set[str]]:
    """Max rule counts per origin, plus failed units.

    At most scan_workers() jobs are in flight. Raises TimeoutError past
    the deadline, ScanStopped on a fatal failure; either way after
    killing every outstanding run. Memoized units skip gcc; with
    store, clean results reach cache once the whole scan ends.
    """
    counts: dict[tuple[str, str], Counter] = {}
    failed: set[str] = set()
    base = f"{_compiler_id(gcc)}\0{_rules_id(rule_counts)}\0{_listing(root)}\0{TIMEOUT_S}\0{OUTPUT_CAP}"
    keys, todo, seen, new = {}, [], {}, {}
    for config in configs:
        for unit in _units(root):
            flags = _unit_flags(configs[config], unit)
            key = keys[(config, unit)] = _unit_key(f"{base}\0{config}", root, unit, flags, seen)
            hit = _cached(key, root, cache, seen)
            if hit is None:
                todo.append((config, unit))
                continue
            for origin, found_counts in hit.items():
                counts[(config, origin)] = counts.get((config, origin), Counter()) | found_counts
    jobs = iter(todo)
    window = scan_workers()
    pending: dict[Future, tuple[str, str]] = {}
    # Parsing gcc -E output is CPU bound.
    pool = ProcessPoolExecutor(max_workers=window)
    try:
        while True:
            for config, unit in islice(jobs, window - len(pending)):
                job = pool.submit(_unit_counts, gcc, root, unit, _unit_flags(configs[config], unit), deadline,
                                  abort, rule_counts)
                pending[job] = (config, unit)
            if not pending:
                break
            done, _ = wait(pending, timeout=max(0.0, deadline - time.monotonic()), return_when=FIRST_COMPLETED)
            if not done:
                raise TimeoutError
            for future in done:
                config, unit = pending.pop(future)
                found = future.result()
                if found is None:
                    if time.monotonic() >= deadline:
                        raise TimeoutError
                    failed.add(f"{config}:{unit}")
                    if fatal(f"{config}:{unit}"):
                        raise ScanStopped(unit, f"gcc -E failed for {config}:{unit}")
                    continue
                found, deps = found
                if deps is not None:
                    entry = _remember(keys[(config, unit)], deps, found)
                    new.setdefault(keys[(config, unit)], []).append(entry)
                for origin, found_counts in found.items():
                    key = (config, origin)
                    counts[key] = counts.get(key, Counter()) | found_counts
    except BaseException:
        # Running workers poll this file.
        Path(abort).touch()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    if store and cache is not None and new:
        _store(cache, new)
    return counts, failed


def preprocess_findings(
    tree: Path, archive: bytes, rule_counts: Callable[[str], Counter], configs: dict = CONFIGS,
    deadline_s: float = SCAN_DEADLINE_S, cache: Optional[Path] = None,
) -> list[dict]:
    """Rules that grow after gcc -E.

    archive: base Include/ and Source/ as a tar. cache: trusted dir
    for base scans; candidate scans stay in memory.
    """
    units = len(_units(tree))
    if units > MAX_UNITS:
        return [{"rule": "scan_error", "path": "", "message": f"too many units: {units} > {MAX_UNITS}"}]
    gcc = arm_tool("arm-none-eabi-gcc")
    deadline = time.monotonic() + deadline_s
    with tempfile.TemporaryDirectory() as tmp:
        base, abort = Path(tmp, "base"), str(Path(tmp, "abort"))
        try:
            base.mkdir()
            extract_tar(archive, base)
            before, base_failed = _tree_counts(gcc, base, rule_counts, configs, deadline, abort, cache=cache, store=True)
            # A unit the base built must build.
            after, _ = _tree_counts(gcc, tree, rule_counts, configs, deadline, abort,
                                    lambda key: key not in base_failed or key.startswith("build:"))
        except TimeoutError:
            return [{"rule": "scan_error", "path": "", "message": f"scan passed its {deadline_s:g} s deadline"}]
        except ScanStopped as stop:
            return [stop.finding]
        except (OSError, tarfile.TarError) as exc:
            return [{"rule": "scan_error", "path": "", "message": f"preprocess failed: {exc}"[:200]}]
    # A config that never runs scans nothing.
    findings = [{"rule": "scan_error", "path": "", "message": f"gcc -E failed for all of {config}"}
                for config in configs if _units(tree) and not any(key[0] == config for key in after)]
    seen: set[tuple[str, str]] = set()
    for (config, origin), found in sorted(after.items()):
        old = before.get((config, origin), Counter())
        for rule in sorted(rule for rule in found if found[rule] > old[rule] and (rule, origin) not in seen):
            seen.add((rule, origin))
            findings.append({"rule": rule, "path": origin, "text": f"after preprocessing for {config}"})
    return findings


SCS_LOW, SCS_HIGH = 0xE0000000, 0xE00FFFFF
# Sections the linker scripts map normally.
_SECTION_OK = re.compile(
    r"^(?:\.rela?(?=\.))?(?:(?:\.text|\.rodata|\.data|\.bss|\.debug_\w+)(?:\..*)?"
    r"|\.ARM\.attributes|\.ARM\.(?:exidx|extab)(?:\.text\..*)?|\.llvm_addrsig|\.comment|\.note\.GNU-stack"
    r"|\.group|\.(?:sym|str|shstr)tab|)$"
)
# Other names fail closed: no parser chasing.
# Little-endian words 0xE0000000-0xE00FFFFF.
_SCS_WORD = re.compile(rb"(?=[\x00-\xff]{2}[\x00-\x0f]\xe0)")
_PLAIN_NAME = re.compile(r"[A-Za-z0-9._$-]+")
# ELF flag bits, as readelf letters.
_FLAG_LETTERS = ((0x1, "W"), (0x2, "A"), (0x4, "X"), (0x10, "M"), (0x20, "S"), (0x40, "I"), (0x80, "L"),
                 (0x200, "G"), (0x400, "T"))
_PROGBITS, _NOBITS, _RELA, _REL = 1, 8, 4, 9
_ADDRESS_INSN = re.compile(r"\t(?:mov|movw|movt|ldr)\S*\t|\t\.word\t")
_IMMEDIATE = re.compile(r"#(-?(?:0x[0-9a-fA-F]+|\d+))|\.word\t(0x[0-9a-fA-F]+)")
# Sleep, as the linear decode sees it. Its 16-bit encodings also sit
# inside ordinary MVE loads (vldrw q5, [sp, #128] is ed9d bf20).
_SLEEP_INSN = re.compile(r"\twf[ie]\S*(?:\t|$)", re.IGNORECASE)
# Interrupt masks and msr, as raw encodings at every instruction start:
# cps (T1, T2) and msr (T1) to anything but APSR (SYSm 0), i.e. PRIMASK,
# BASEPRI, FAULTMASK, CONTROL, stack pointers and limits, _NS aliases.
_MASK_ENCODING = re.compile(rb"(?=[\x60-\x7f]\xb6|[\x80-\x9f\xa0-\xaf]\xf3[\x00-\xff][\x80-\x8f\xa0-\xaf])")
_DISASSEMBLY = re.compile(r"^Disassembly of section (\S+):$")
_INSN_LINE = re.compile(r"^\s*([0-9a-f]+):\t")
_RELOC_SECTION = re.compile(r"^Relocation section '.*' at offset")
_SYMBOL_ROW = re.compile(r"^\s*\d+:\s+([0-9a-f]+)\s+\d+\s+(\w+)\s+(\w+)\s+\w+\s+(\d+)\s+(\S+)")
# Mapping symbols mark code: $t, $a.
_CODE_SYMBOL = re.compile(r"^\$[ta](?:\.|$)")
_MAPPING_SYMBOL = re.compile(r"^\$([adt])(?:\.|$)")
# Symbol 0 (an absolute value) has no symbol columns.
_RELOC_ROW = re.compile(r"^([0-9a-f]{8})\s+[0-9a-f]{8}\s+(R_ARM_\w+)(?:\s+[0-9a-f]{8}\s+(\S+))?\s*$")
_ANY_RELOC_ROW = re.compile(r"^[0-9a-f]{8}\s")
_CALLS = frozenset(("R_ARM_THM_CALL", "R_ARM_THM_JUMP24", "R_ARM_THM_JUMP19", "R_ARM_CALL", "R_ARM_JUMP24"))
# Kernel tables stay below 1 MiB.
_ADDEND_MAX = 1 << 20
# Compile args that are not -E flags.
_DROP_ARGS = frozenset(("-c", "-MD", "-MMD", "-MP"))
_DROP_WITH_VALUE = frozenset(("-o", "-MF", "-MT", "-MQ"))


def _markers() -> tuple[str, str]:
    from .nsx_app import CMSIS_NN_MODULE, CMSIS_NN_PROJECT

    return f"/modules/{CMSIS_NN_MODULE}/", f"/modules/{CMSIS_NN_PROJECT}/"


def _kernel_units(build_dir: Path) -> list[tuple[str, Path, list[str]]]:
    """(source, object, args) per kernel unit."""
    entries = json.loads((build_dir / "compile_commands.json").read_text(encoding="utf-8"))
    markers = _markers()
    units = []
    for entry in entries:
        source = Path(entry["file"]).as_posix()
        marker = next((m for m in markers if m in source), None)
        rel = source.split(marker, 1)[1] if marker else ""
        if not rel.startswith("Source/"):
            continue
        args = entry.get("arguments") or shlex.split(entry["command"])
        output = entry.get("output") or args[args.index("-o") + 1]
        units.append((rel, Path(entry["directory"], output), args))
    return units


def _preprocess_args(args: list[str], source: str) -> tuple[str, ...]:
    """A unit's compile args as -E flags."""
    kept, skip, markers = [], False, _markers()
    for arg in args[1:]:
        if skip or arg in _DROP_ARGS or arg.endswith(source):
            skip = False
            continue
        skip = arg in _DROP_WITH_VALUE
        marker = next((m for m in markers if m in arg), None)
        # Kernel paths point at the scanned tree.
        if marker and arg.startswith("-I"):
            arg = "-I" + arg.split(marker, 1)[1]
        if not skip:
            kept.append(arg)
    return tuple(kept)


def _elf_headers(obj: Path) -> tuple[memoryview, list[tuple[int, ...]]]:
    """Mapped file and raw ELF32 section headers."""
    with open(obj, "rb") as handle:
        # The map outlives the handle.
        raw = memoryview(mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ))
    if raw[:4] != b"\x7fELF" or raw[4:6] != b"\x01\x01":
        raise ValueError(f"{obj.name}: not ELF32 little-endian")
    try:
        shoff, = struct.unpack_from("<I", raw, 0x20)
        entsize, count = struct.unpack_from("<HH", raw, 0x2E)
        headers = [struct.unpack_from("<10I", raw, shoff + i * entsize) for i in range(count or 1)]
        # Extended numbering keeps counts in section 0.
        count = count or headers[0][5]
        headers = [struct.unpack_from("<10I", raw, shoff + i * entsize) for i in range(count)]
    except (struct.error, IndexError) as exc:
        raise ValueError(f"{obj.name}: bad section table: {exc}") from exc
    return raw, headers


def _section_aligns(obj: Path) -> list[int]:
    """sh_addralign per section."""
    return [header[8] for header in _elf_headers(obj)[1]]


def elf_sections(obj: Path) -> list[tuple[str, int, str, memoryview, int]]:
    """(name, type, flags, data, info) per section.

    Read from the ELF32 little-endian section table, not readelf text,
    so no name spelling can hide a section. The file is mapped, not
    read: section data are views into it.
    """
    raw, headers = _elf_headers(obj)
    try:
        names_index, = struct.unpack_from("<H", raw, 0x32)
        names_index = headers[0][6] if names_index == 0xFFFF else names_index
        table = headers[names_index]
    except (struct.error, IndexError) as exc:
        raise ValueError(f"{obj.name}: bad section table: {exc}") from exc
    strings = bytes(raw[table[4]:table[4] + table[5]])
    sections = []
    for name_at, kind, flags, _, offset, size, _, info, _, _ in headers:
        if name_at >= len(strings) or (kind != _NOBITS and offset + size > len(raw)):
            raise ValueError(f"{obj.name}: section out of bounds")
        name = strings[name_at:strings.index(b"\0", name_at)].decode(errors="replace")
        data = raw[offset:offset + size] if kind != _NOBITS else memoryview(b"")
        sections.append((name, kind, "".join(letter for bit, letter in _FLAG_LETTERS if flags & bit), data, info))
    return sections


def _mask_name(first: int, second: Optional[int]) -> Optional[str]:
    """cps or non-APSR msr, by name and encoding, or None."""
    if first & 0xFFE0 == 0xB660:
        return f"{'cpsid' if first & 0x10 else 'cpsie'} {first:04x}"
    if second is None:
        return None
    if first & 0xFFE0 == 0xF380 and second & 0xFF:
        return f"msr {first:04x} {second:04x}"
    if first & 0xFFF0 == 0xF3A0 and second & 0x0600:
        return f"cps {first:04x} {second:04x}"
    return None


def _mask_hits(name: str, data: memoryview, align: int) -> list[str]:
    """cps or non-APSR msr encodings wherever an instruction could start.

    Even offsets only, unless the section may land at an odd address.
    """
    found = []
    for match in _MASK_ENCODING.finditer(data):
        at = match.start()
        if at & 1 and align > 1:
            continue
        first = _halfword(data, at)
        second = _halfword(data, at + 2) if first & 0xFFE0 != 0xB660 else None
        if text := _mask_name(first, second):
            found.append(f"{name}+{at:#x}: {text}")
    return found


def _sign(value: int, bits: int) -> int:
    return value - (1 << bits) if value >> (bits - 1) & 1 else value


def _imm25(first: int, second: int) -> int:
    """Offset of b.w (T4), bl or blx."""
    sign = first >> 10 & 1
    high = (1 - (second >> 13 & 1 ^ sign)) << 1 | (1 - (second >> 11 & 1 ^ sign))
    return _sign(sign << 24 | high << 22 | (first & 0x3FF) << 12 | (second & 0x7FF) << 1, 25)


def _imm21(first: int, second: int) -> int:
    """Offset of a conditional b.w (T3)."""
    jumps = (second >> 11 & 1) << 1 | second >> 13 & 1
    return _sign((first >> 10 & 1) << 20 | jumps << 18 | (first & 0x3F) << 12 | (second & 0x7FF) << 1, 21)


def _imm16(first: int, second: int) -> int:
    """imm4:i:imm3:imm8 of movw/movt."""
    return (first & 0xF) << 12 | (first >> 10 & 1) << 11 | (second >> 12 & 7) << 8 | second & 0xFF


def _thumb_addend(kind: str, data: memoryview, offset: int) -> Optional[int]:
    """REL addend of a Thumb branch or movw/movt, or None if unread."""
    if offset + 4 > len(data):
        return None
    first, second = struct.unpack_from("<HH", data, offset)
    if kind in ("R_ARM_THM_CALL", "R_ARM_THM_JUMP24"):
        return _imm25(first, second)
    if kind == "R_ARM_THM_JUMP19":
        return _imm21(first, second)
    return _sign(_imm16(first, second), 16)


_BRANCH_RELOCS = frozenset(("R_ARM_THM_CALL", "R_ARM_THM_JUMP24", "R_ARM_THM_JUMP19"))
_MOVE_RELOCS = frozenset(("R_ARM_THM_MOVW_ABS_NC", "R_ARM_THM_MOVT_ABS", "R_ARM_THM_MOVW_PREL_NC", "R_ARM_THM_MOVT_PREL"))
_WORD_RELOCS = frozenset(("R_ARM_ABS32", "R_ARM_REL32", "R_ARM_TARGET1", "R_ARM_TARGET2"))


def _reloc_addend(kind: str, data: memoryview, offset: int) -> Optional[int]:
    """Byte offset from the symbol that a relocation reaches, or None if unread."""
    if kind in _WORD_RELOCS:
        return int.from_bytes(data[offset:offset + 4], "little", signed=True)
    if kind == "R_ARM_PREL31":
        return _sign(int.from_bytes(data[offset:offset + 4], "little") & 0x7FFFFFFF, 31)
    if kind in _BRANCH_RELOCS or kind in _MOVE_RELOCS:
        addend = _thumb_addend(kind, data, offset)
        # The branch addend carries the -4 pipeline bias.
        return None if addend is None else addend + 4 * (kind in _BRANCH_RELOCS)
    return None


# Relocations that rewrite a Thumb instruction, by the bytes they rewrite.
_CODE_RELOCS = {
    **dict.fromkeys(("R_ARM_THM_JUMP11", "R_ARM_THM_JUMP8", "R_ARM_THM_JUMP6", "R_ARM_THM_PC8"), 2),
    **dict.fromkeys((*_BRANCH_RELOCS, *_MOVE_RELOCS, "R_ARM_THM_PC12"), 4),
}
# Relocations that write data, by the bytes they write. Any other kind in
# code is refused.
_DATA_RELOCS = {**dict.fromkeys((*_WORD_RELOCS, "R_ARM_PREL31"), 4), "R_ARM_ABS16": 2, "R_ARM_ABS8": 1}
# Relocations that set a branch target.
_BRANCHES = _CALLS | {"R_ARM_THM_JUMP11", "R_ARM_THM_JUMP8", "R_ARM_THM_JUMP6", "R_ARM_THM_BF16",
                      "R_ARM_THM_BF12", "R_ARM_THM_BF18"}


def _fits(kind: str, first: int, second: Optional[int]) -> bool:
    """The instruction is one the relocation is defined for.

    The linker writes only the immediate fields (and may turn bl into
    blx), so the result stays that instruction.
    """
    if second is None:
        return {
            "R_ARM_THM_JUMP11": first & 0xF800 == 0xE000,
            "R_ARM_THM_JUMP8": first & 0xF000 == 0xD000 and first >> 8 & 0xF < 0xE,
            "R_ARM_THM_JUMP6": first & 0xF500 == 0xB100,
            "R_ARM_THM_PC8": first & 0xF800 in (0x4800, 0xA000),
        }.get(kind, False)
    branch = first & 0xF800 == 0xF000 and second & 0x8000
    if kind == "R_ARM_THM_CALL":
        return bool(branch) and (second & 0xD000 == 0xD000 or second & 0xD001 == 0xC000)
    if kind == "R_ARM_THM_JUMP24":
        return bool(branch) and second & 0xD000 == 0x9000
    if kind == "R_ARM_THM_JUMP19":
        # cond 111x is the hint, msr and barrier space.
        return bool(branch) and second & 0xD000 == 0x8000 and first >> 6 & 0xF < 0xE
    if kind in ("R_ARM_THM_MOVW_ABS_NC", "R_ARM_THM_MOVW_PREL_NC"):
        return first & 0xFBF0 == 0xF240 and not second & 0x8000
    if kind in ("R_ARM_THM_MOVT_ABS", "R_ARM_THM_MOVT_PREL"):
        return first & 0xFBF0 == 0xF2C0 and not second & 0x8000
    if kind == "R_ARM_THM_PC12":
        # ldr, ldrb, ldrh, ldrsb, ldrsh (literal).
        return first & 0xFE1F == 0xF81F and first >> 5 & 3 != 3
    return False


def _reloc_problem(kind: str, layout: "Layout", data: memoryview, offset: int) -> Optional[str]:
    """Why a relocation in a code section is refused, or None.

    Data relocations must write only bytes of one data span; instruction
    relocations must sit on an instruction they are defined for.
    """
    starts, data_spans, _ = layout
    if kind in _DATA_RELOCS:
        end = offset + _DATA_RELOCS[kind]
        inside = end <= len(data) and any(low <= offset and end <= high for low, high in data_spans)
        return None if inside else "rewrites code it does not fit"
    width = _CODE_RELOCS.get(kind)
    if width is None:
        return "is not a supported code relocation"
    if starts.get(offset) != width:
        return "rewrites code it does not fit"
    second = _halfword(data, offset + 2) if width == 4 else None
    return None if _fits(kind, _halfword(data, offset), second) else "rewrites code it does not fit"


def _reloc_size(kind: str) -> int:
    """Bytes a relocation writes; 0 for R_ARM_NONE and unsupported kinds."""
    return _DATA_RELOCS.get(kind) or _CODE_RELOCS.get(kind) or 0


def _branch_target(at: int, first: int, second: Optional[int]) -> Optional[int]:
    """Where a direct Thumb branch at `at` goes, or None for any other instruction."""
    pc = at + 4
    if second is None:
        if first & 0xF800 == 0xE000:
            return pc + _sign((first & 0x7FF) << 1, 12)
        if first & 0xF000 == 0xD000 and first >> 8 & 0xF < 0xE:
            return pc + _sign((first & 0xFF) << 1, 9)
        if first & 0xF500 == 0xB100:
            return pc + ((first >> 9 & 1) << 6 | (first >> 3 & 0x1F) << 1)
        return None
    if first & 0xF800 != 0xF000 or not second & 0x8000:
        return None
    op = second & 0xD000
    if op in (0x9000, 0xD000):
        return pc + _imm25(first, second)
    if op == 0x8000 and first >> 6 & 0xF < 0xE:
        return pc + _imm21(first, second)
    if op == 0xC000 and not second & 1:
        # blx to Arm state.
        return (pc & ~3) + _imm25(first, second)
    if _loop_branch(first, second) and second >> 12 == 0xC:
        # wls/wlstp jump forward; le/letp (Rn 15) back.
        offset = (second >> 1 & 0x3FF) << 2 | (second >> 11 & 1) << 1
        return pc - offset if first & 0xF == 0xF else pc + offset
    return None


def _loop_branch(first: int, second: int) -> bool:
    """wls, wlstp, le, letp, dls, dlstp, lctp or vctp."""
    if first & 0xFF80 != 0xF000 or second & 0xD001 != 0xC001:
        return False
    return (first >> 4 & 7 <= 4 and (second >> 12 == 0xC or second == 0xE001)
            or first >> 4 & 7 <= 3 and second == 0xE801)


def _branch_future(first: int, second: Optional[int]) -> bool:
    """Any other encoding of the branch future and loop space.

    bf, bfx, bfl, bflx and bfcsel branch later, from another address.
    """
    return (second is not None and first & 0xF800 == 0xF000 and second & 0xD001 == 0xC001
            and not _loop_branch(first, second))


# Code section layout: instruction starts with their sizes, data spans, problems.
Layout = tuple[dict[int, int], tuple[tuple[int, int], ...], list[str]]


def _halfword(data: memoryview, at: int) -> int:
    return data[at] | data[at + 1] << 8


def _ends_flow(first: int, second: Optional[int]) -> bool:
    """An unconditional branch, return or trap: execution never reaches the next instruction.

    A missed spelling only costs a false finding.
    """
    if second is None:
        # b, bx, bxns, mov pc, add pc, pop {pc}, udf.
        return (first & 0xF800 == 0xE000 or first & 0xFF87 in (0x4700, 0x4704, 0x4687, 0x4487)
                or first & 0xFF00 in (0xBD00, 0xDE00))
    # b.w, ldm/pop.w with pc, ldr pc, tbb/tbh, udf.w.
    return (first & 0xF800 == 0xF000 and second & 0xD000 == 0x9000
            or first & 0xFFD0 in (0xE890, 0xE910) and bool(second & 0x8000)
            or first & 0xFF70 == 0xF850 and second >> 12 == 0xF
            or first & 0xFFF0 == 0xE8D0 and second & 0xFFE0 == 0xF000
            or first & 0xFFF0 == 0xF7F0 and second & 0xF000 == 0xA000)


def _sleep_name(first: int, second: Optional[int]) -> Optional[str]:
    """wfi/wfe in either encoding; should-be bits of the 32-bit hint are ignored."""
    if second is None:
        return {0xBF20: "wfe", 0xBF30: "wfi"}.get(first)
    if first & 0xFFF0 == 0xF3A0 and second & 0xD700 == 0x8000:
        return {2: "wfe", 3: "wfi"}.get(second & 0xFF)
    return None


def _thumb_layout(name: str, data: memoryview, marks: dict[int, str], labels: set[int]) -> Layout:
    """Instruction starts, data spans and problems of a code section.

    Mapping symbols split the section: $d starts data, $t or $a code, and
    bytes before the first one are code. Code is decoded the way the core
    runs it, from each code span's start with exact Thumb sizes, so a data
    row or a label never counts as an instruction start. A code span must
    not run into data or off the section end, and no instruction may be cut
    by a label, by data or by the section end.
    """
    size = len(data)
    points = sorted(marks.items())
    if not points or points[0][0]:
        points.insert(0, (0, "t"))
    spans: list[list] = []
    for number, (low, kind) in enumerate(points):
        high = points[number + 1][0] if number + 1 < len(points) else size
        code = kind != "d"
        if high <= low or low >= size:
            continue
        if spans and spans[-1][2] == code:
            spans[-1][1] = high
        else:
            spans.append([low, high, code])
    starts: dict[int, int] = {}
    problems = []
    for low, high, code in spans:
        if not code:
            continue
        at, conditional, ends = low, 0, False
        while at < high:
            width = 4 if at + 1 < size and data[at + 1] >> 3 >= 0x1D else 2
            if at + width > high:
                problems.append(f"instruction at {name}+{at:#x} is cut by a symbol or the section end")
                break
            starts[at] = width
            first = _halfword(data, at)
            second = _halfword(data, at + 2) if width == 4 else None
            # nop padding keeps the state of the instruction before it.
            if not (first == 0xBF00 or (first, second) == (0xF3AF, 0x8000)):
                ends = not conditional and _ends_flow(first, second)
            conditional = max(conditional - 1, 0)
            if second is None and first & 0xFF00 == 0xBF00 and first & 0xF:
                # it: 4 - trailing zeros of the mask instructions are conditional.
                conditional = 4 - ((first & -first & 0xF).bit_length() - 1)
            at += width
        else:
            if not ends:
                into = "the section end" if high == size else f"data at {name}+{high:#x}"
                problems.append(f"code at {name}+{low:#x} can run into {into}")
    for label in labels:
        for back in (1, 2, 3):
            if starts.get(label - back, 0) > back:
                problems.append(f"instruction at {name}+{label - back:#x} is cut by a symbol or the section end")
    data_spans = tuple((low, high) for low, high, code in spans if not code)
    return starts, data_spans, problems


def _landing(layout: Layout, size: int, at: int, branch: bool) -> Optional[str]:
    """Why code entered at `at` is unchecked, or None.

    Branches into data are refused; data may still point at data, but
    only at an even offset, which bx/blx cannot run.
    """
    starts, data_spans, _ = layout
    if at in starts or at & ~1 in starts or at == size:
        return None
    if any(low <= at < high for low, high in data_spans):
        return "in data" if branch or at & 1 else None
    return "inside an instruction"


# A global code symbol: section name, value, layout, section size.
Export = tuple[str, int, Layout, int]
# Every definition of a global code symbol, each with whether it is weak.
Exports = dict[str, list[tuple[Export, bool]]]
# A reference into another object: symbol, addend (None if unread), origin, whether it branches.
External = tuple[str, Optional[int], str, bool]
# A code section as the linked image must hold it: name, bytes, layout,
# marks, labels, relocations (offset, kind, whether the object scan accepted it).
LinkSection = tuple[str, memoryview, Layout, dict[int, str], set[int], tuple[tuple[int, str, bool], ...]]


def _object_hits(
    source: str, obj: Path, defined: frozenset[str], weak: frozenset[str], limits: tuple[float, str],
) -> tuple[list[dict], Exports, list[External], list[LinkSection]]:
    """Section, SCS and reference hits, global code symbols, references to other objects, code sections.

    `defined` and `weak` are the strong and weak global definitions of all kernel objects.
    """
    hits, sections, names = [], {}, {}
    # Locals resolve only inside this object.
    local = _defined_symbols([obj], kinds="bdrt", limits=limits)
    table = elf_sections(obj)
    for index, (name, kind, flags, _, _) in enumerate(table):
        names[str(index)] = name
        # Code may live in .text only.
        odd = index and not _PLAIN_NAME.fullmatch(name)
        if odd or not _SECTION_OK.match(name) or ("X" in flags and not name.startswith(".text")):
            hits.append({"rule": "object_section", "path": source, "text": f"{repr(name) if odd else name} {flags}"[:200]})
        if kind == _PROGBITS and "A" in flags:
            sections[index] = flags

    # Code outside .text, whatever its flags.
    homes, places = {}, {}
    marks: dict[int, dict[int, str]] = {}
    labels: dict[int, set[int]] = {}
    # Only global and weak definitions are visible to other objects.
    linked: dict[str, set[tuple[int, int, bool]]] = {}
    for row in run_binutil("arm-none-eabi-readelf", ["-sW", str(obj)], *limits).splitlines():
        match = _SYMBOL_ROW.match(row)
        if not match:
            continue
        value, kind, bind, index, symbol = match.groups()
        home = names.get(index, "")
        homes[symbol] = home
        if index.isdigit() and int(index) < len(table):
            at = int(value, 16)
            # Bit 0 of a Thumb function's value is the state tag, not an offset.
            if kind == "FUNC":
                at &= ~1
            places.setdefault(symbol, set()).add((int(index), at))
            labels.setdefault(int(index), set()).add(at)
            mapping = _MAPPING_SYMBOL.match(symbol)
            if mapping or kind == "FUNC":
                # A function runs whatever marks its bytes carry. Code wins
                # where marks clash: it is decoded and checked.
                seen = marks.setdefault(int(index), {})
                seen[at] = "d" if mapping and mapping.group(1) == "d" and seen.get(at, "d") == "d" else "t"
            if bind in ("GLOBAL", "WEAK"):
                linked.setdefault(symbol, set()).add((int(index), at, bind == "WEAK"))
        code = kind == "FUNC" or _CODE_SYMBOL.match(symbol)
        if code and not home.startswith(".text"):
            hits.append({"rule": "object_section", "path": source, "text": f"code {symbol} in {home}"})

    def hit(text: str) -> None:
        hits.append({"rule": "object_address", "path": source, "text": text[:200]})

    def instruction(text: str) -> None:
        hits.append({"rule": "object_instruction", "path": source, "text": text[:200]})

    # Instruction starts per code section, as the core decodes them.
    layouts = {index: _thumb_layout(entry[0], entry[3], marks.get(index, {}), labels.get(index, set()))
               for index, entry in enumerate(table) if "X" in entry[2]}
    for _, _, problems in layouts.values():
        for text in problems:
            instruction(text)
    for line in run_binutil("arm-none-eabi-objdump", ["-dz", "--no-show-raw-insn", str(obj)], *limits).splitlines():
        if not _ADDRESS_INSN.search(line):
            continue
        for match in _IMMEDIATE.finditer(line):
            value = int(match.group(1) or match.group(2), 0) & 0xFFFFFFFF
            # movt loads the top half.
            if SCS_LOW <= value <= SCS_HIGH or ("\tmovt" in line and SCS_LOW >> 16 <= value <= SCS_HIGH >> 16):
                hit(line.strip())

    def entry(origin: str, index: int, at: int, branch: bool) -> None:
        """Code entered at `at` must start a decoded instruction."""
        if index in layouts and (problem := _landing(layouts[index], len(table[index][3]), at, branch)):
            instruction(f"{origin} enters {table[index][0]}+{at:#x} {problem}")

    # -D decodes data too: bytes behind a $d mapping symbol, in .rodata or
    # .data, can still be branched to. Odd names are already findings.
    loaded = [table[index][0] for index in sections if _PLAIN_NAME.fullmatch(table[index][0])]
    slept: set[tuple[str, int]] = set()
    if loaded:
        selected = [arg for name in loaded for arg in ("-j", name)]
        heading_name = ""
        for line in run_binutil("arm-none-eabi-objdump", ["-D", "--no-show-raw-insn", *selected, str(obj)],
                                *limits).splitlines():
            if heading := _DISASSEMBLY.match(line):
                heading_name = heading.group(1)
            elif _SLEEP_INSN.search(line):
                instruction(line.strip())
                if row := _INSN_LINE.match(line):
                    slept.add((heading_name, int(row.group(1), 16)))
    # The same check on the code as the core decodes it.
    for index, (code_starts, _, _) in layouts.items():
        name, data = table[index][0], table[index][3]
        for at, width in sorted(code_starts.items()):
            second = _halfword(data, at + 2) if width == 4 else None
            if (mnemonic := _sleep_name(_halfword(data, at), second)) and (name, at) not in slept:
                instruction(f"{name}+{at:#x}: {mnemonic}")
    contents = {index: table[index][3] for index in sections}
    aligns = _section_aligns(obj)
    for index, data in contents.items():
        for text in _mask_hits(table[index][0], data, aligns[index]):
            instruction(text)
        # Any byte offset: loads may be unaligned.
        if "X" not in sections[index] and _SCS_WORD.search(data):
            hit(f"SCS address in {table[index][0]}")
    # readelf lists reloc sections in table order.
    targets = iter([info for _, kind, _, _, info in table if kind in (_REL, _RELA)])
    expected, seen_relocs = sum(kind in (_REL, _RELA) for _, kind, _, _, _ in table), 0
    target = None
    relocated: set[tuple[int, int]] = set()
    section_relocs: dict[int, list[tuple[int, str, bool]]] = {}
    externals: list[External] = []
    for line in run_binutil("arm-none-eabi-readelf", ["-rW", str(obj)], *limits).splitlines():
        if _RELOC_SECTION.match(line):
            seen_relocs += 1
            target = next(targets, None)
            continue
        row = _RELOC_ROW.match(line)
        if not row and _ANY_RELOC_ROW.match(line):
            raise ValueError(f"{obj.name}: unread relocation {line.strip()[:80]}")
        if not row or target not in contents:
            continue
        section = table[target][0]
        offset, kind, symbol = int(row.group(1), 16), row.group(2), row.group(3) or "*ABS*"
        # Only a branch relocation replaces the target the bytes encode.
        if kind in _BRANCHES:
            relocated.add((target, offset))
        data = contents[target]
        # Symbol plus addend can reach any address.
        if kind == "R_ARM_ABS32":
            addend = int.from_bytes(data[offset:offset + 4], "little", signed=True)
            if not 0 <= addend < _ADDEND_MAX:
                hit(f"{symbol}{addend:+#x} in {section}")
        origin = f"{kind} at {section}+{offset:#x}"
        # The linker rewrites relocated bytes, so in code only data and
        # instruction fields may be relocated.
        problem = None
        if target in layouts and kind != "R_ARM_NONE":
            if problem := _reloc_problem(kind, layouts[target], data, offset):
                instruction(f"{origin} {problem}")
            section_relocs.setdefault(target, []).append((offset, kind, not problem))
        addend = _reloc_addend(kind, data, offset)
        # A weak definition here may lose to another one at link time.
        own_weak = any(is_weak for _, _, is_weak in linked.get(symbol, ()))
        elsewhere = symbol not in places and symbol not in local
        if own_weak or (elsewhere and (symbol in defined or symbol in weak)):
            # Defined in another kernel object: checked once all are read.
            externals.append((symbol, addend, origin, kind in _BRANCHES))
        elif elsewhere and kind in _CALLS and (addend is None or addend & ~1):
            # Harness code is not scanned, so only its symbol entries are known.
            reach = "" if addend is None else f"{addend:+#x}"
            instruction(f"{origin} into {symbol}{reach} outside the kernels is unchecked")
        for index, value in () if own_weak else places.get(symbol, ()):
            if index not in layouts:
                continue
            if addend is None:
                instruction(f"{origin} into {table[index][0]} is unchecked")
            else:
                entry(origin, index, value + addend, kind in _BRANCHES)
        home = homes.get(symbol, "")
        if kind in _CALLS and home and not home.startswith(".text"):
            hits.append({"rule": "object_section", "path": source, "text": f"branch to {symbol} in {home}"})
        if kind not in _CALLS and not symbol.startswith(".") and symbol not in defined | local:
            hit(f"{kind} to {symbol} in {section}")
    if seen_relocs != expected:
        raise ValueError(f"{obj.name}: read {seen_relocs} of {expected} reloc sections")
    # Direct branches, decoded from the instruction starts the core runs.
    for index, (code_starts, _, _) in layouts.items():
        name, data = table[index][0], table[index][3]
        for at, width in sorted(code_starts.items()):
            first = _halfword(data, at)
            second = _halfword(data, at + 2) if width == 4 else None
            if _branch_future(first, second):
                instruction(f"branch future or loop at {name}+{at:#x} is unchecked")
            elif (index, at) not in relocated and (goal := _branch_target(at, first, second)) is not None:
                if 0 <= goal < len(data):
                    entry(f"branch at {name}+{at:#x}", index, goal, True)
                else:
                    # Whatever the link places there is unchecked.
                    instruction(f"branch at {name}+{at:#x} leaves the section")
    exports: Exports = {}
    for symbol, spots in linked.items():
        for index, value, is_weak in spots:
            if index in layouts:
                spot = (table[index][0], value, layouts[index], len(table[index][3]))
                exports.setdefault(symbol, []).append((spot, is_weak))
    code = [(table[index][0], table[index][3], layout, marks.get(index, {}), labels.get(index, set()),
             tuple(section_relocs.get(index, ()))) for index, layout in layouts.items()]
    return hits, exports, externals, code


def _external_hits(results: list[tuple[str, tuple]]) -> list[dict]:
    """References that enter another kernel object's code inside an instruction.

    A strong definition wins the link. With only weak ones, any of them
    may be linked or a harness definition may replace them all, so each
    weak body is checked and only the symbol's own entry is accepted.
    """
    exports: Exports = {}
    for _, (_, found, *_) in results:
        for symbol, spots in found.items():
            exports.setdefault(symbol, []).extend(spots)
    hits = []
    for source, (_, _, externals, *_) in results:
        for symbol, addend, origin, branch in externals:
            spots = exports.get(symbol)
            if not spots:
                continue
            strong = [spot for spot, is_weak in spots if not is_weak]
            texts = []
            for section, value, layout, size in strong or [spot for spot, _ in spots]:
                if addend is None:
                    texts.append(f"{origin} into {symbol} ({section}) is unchecked")
                elif problem := _landing(layout, size, value + addend, branch):
                    texts.append(f"{origin} enters {symbol}{addend:+#x} ({section}+{value + addend:#x}) {problem}")
            if not strong and (addend is None or addend & ~1):
                reach = "" if addend is None else f"{addend:+#x}"
                texts.append(f"{origin} into weak {symbol}{reach} is unchecked: another definition can replace it")
            hits += [{"rule": "object_instruction", "path": source, "text": text[:200]} for text in texts]
    return hits


# GNU ld: " .text.f  0xADDR  0xSIZE file", the name alone on its own line when long.
_GNU_INPUT = re.compile(r"^ (\S+)\s+0x([0-9a-fA-F]+)\s+0x([0-9a-fA-F]+)\s+(\S.*)$")
_GNU_NAME = re.compile(r"^ (\S+)$")
_GNU_WRAPPED = re.compile(r"^\s+0x([0-9a-fA-F]+)\s+0x([0-9a-fA-F]+)\s+(\S.*)$")
# lld: "VMA LMA Size Align", then the input as "file:(section)" indented by 8.
_LLD_HEADER = re.compile(r"^\s*VMA\s+LMA\s+Size\s+Align\s+Out\s+In\s+Symbol\s*$")
_LLD_INPUT = re.compile(r"^\s*([0-9a-fA-F]+)\s+[0-9a-fA-F]+\s+([0-9a-fA-F]+)\s+\d+ {9}(\S.*):\(([^()]*)\)$")
_ARCHIVE_MEMBER = re.compile(r"^.*\(([^()]+)\)$")


def _map_sections(text: str) -> list[tuple[str, str, int, int]]:
    """(object file name, section, address, size) of each input section a link map places.

    Reads GNU ld and lld maps; anything else is refused. An archive
    member is named by itself.
    """
    lines = text.splitlines()
    placed = []

    def add(file: str, section: str, address: str, size: str) -> None:
        member = _ARCHIVE_MEMBER.match(file.strip())
        placed.append((Path(member.group(1) if member else file.strip()).name, section, int(address, 16),
                       int(size, 16)))

    if lines and _LLD_HEADER.match(lines[0]):
        for line in lines[1:]:
            if match := _LLD_INPUT.match(line):
                add(match.group(3), match.group(4), match.group(1), match.group(2))
        return placed
    try:
        start = lines.index("Linker script and memory map")
    except ValueError:
        raise ValueError("link map is neither GNU ld nor lld") from None
    pending = None
    # Discarded input sections come before the memory map and are skipped.
    for line in lines[start + 1:]:
        if match := _GNU_INPUT.match(line):
            add(match.group(4), match.group(1), match.group(2), match.group(3))
        elif pending and (match := _GNU_WRAPPED.match(line)):
            add(match.group(3), pending, match.group(1), match.group(2))
        pending = match.group(1) if (match := _GNU_NAME.match(line)) else None
    return placed


def _image_bytes(raw: memoryview, headers: list[tuple[int, ...]], address: int, size: int) -> Optional[memoryview]:
    """Bytes the image loads at `address`, or None if no loaded section holds them all."""
    for _, kind, flags, base, offset, length, *_ in headers:
        if kind == _PROGBITS and flags & 0x2 and base <= address and address + size <= base + length:
            start = offset + address - base
            return raw[start:start + size] if start + size <= len(raw) else None
    return None


def _linked_problems(code: LinkSection, linked: memoryview, address: int) -> Optional[list[tuple[str, str]]]:
    """(rule, text) per linked instruction the link made forbidden, or None if the bytes are not this section's.

    Only relocated bytes may differ from the object. Each instruction
    with a changed byte gets the object checks again: no sleep, cps or
    msr, an instruction relocation still on its instruction, and no SCS
    address from a movt or a data word. Bytes a refused relocation
    writes are already findings and are not reported again.
    """
    name, data, layout, marks, labels, relocs = code
    if len(linked) != len(data):
        return None
    written = bytearray(len(data))
    # Refused writes win where they overlap; no ARM relocation writes more than 4 bytes.
    for offset, kind, ok in sorted(relocs, key=lambda reloc: not reloc[2]):
        end = min(offset + (_reloc_size(kind) if ok else 4), len(data))
        written[offset:end] = (b"\x01" if ok else b"\x02") * max(end - offset, 0)
    changed = {at for at, (old, new) in enumerate(zip(data, linked)) if old != new}
    if any(not written[at] for at in changed):
        return None
    changed = {at for at in changed if written[at] == 1}
    if not changed:
        return []
    found = []
    if 2 not in written and _thumb_layout(name, linked, marks, labels)[0] != layout[0]:
        found.append(("object_instruction", f"linked {name} at {address:#x} decodes differently from the object"))
    kinds = {offset: kind for offset, kind, ok in relocs if ok}
    for at, width in sorted(layout[0].items()):
        if not changed & set(range(at, at + width)) or 2 in written[at:at + width]:
            continue
        first = _halfword(linked, at)
        second = _halfword(linked, at + 2) if width == 4 else None
        where = f"linked {name}+{at:#x} ({address + at:#x})"
        kind = kinds.get(at, "")
        nop = (first, second) == (0xF3AF, 0x8000) and kind in ("R_ARM_THM_CALL", "R_ARM_THM_JUMP24")
        if bad := _sleep_name(first, second) or _mask_name(first, second):
            found.append(("object_instruction", f"{where}: {bad}"))
        elif kind in _CODE_RELOCS and not nop and not _fits(kind, first, second):
            # A call to an undefined weak symbol becomes nop.w.
            found.append(("object_instruction", f"{where}: {kind} left no instruction it fits"))
        elif second is not None and first & 0xFBF0 == 0xF2C0 and SCS_LOW >> 16 <= _imm16(first, second) <= SCS_HIGH >> 16:
            found.append(("object_address", f"{where}: movt {_imm16(first, second):#x}"))
    for offset, kind, ok in relocs:
        word = linked[offset:offset + 4]
        if ok and _DATA_RELOCS.get(kind) == 4 and len(word) == 4 and changed & set(range(offset, offset + 4)):
            if SCS_LOW <= int.from_bytes(word, "little") <= SCS_HIGH:
                found.append(("object_address", f"linked {name}+{offset:#x} ({address + offset:#x}): SCS address"))
    return found


def _linked_hits(build_dir: Path, results: list[tuple[str, Path, tuple]]) -> list[dict]:
    """The object checks on kernel code as the linked image holds it.

    Relocations are filled in at link time, so every kernel code section
    the link map places is read back from the image. It must equal the
    object outside the relocated bytes, and each instruction the link
    changed is checked again. A missing or unreadable image or map, or
    kernel code with none of it in the map, fails the scan.
    """
    from .firmware_build import elf_path, map_path

    image, link_map = elf_path(build_dir), map_path(build_dir)
    for path in (image, link_map):
        if not path.is_file():
            raise ValueError(f"linked image missing: {path}")
    placed = _map_sections(link_map.read_text(encoding="utf-8", errors="replace"))
    raw, headers = _elf_headers(image)
    sections: dict[tuple[str, str], list[tuple[str, LinkSection]]] = {}
    for source, obj, (*_, code) in results:
        for section in code:
            if len(section[1]):
                sections.setdefault((obj.name, section[0]), []).append((source, section))
    hits, seen = [], 0
    for file, name, address, size in placed:
        candidates = sections.get((file, name))
        if not candidates or not size:
            continue
        seen += 1
        linked = _image_bytes(raw, headers, address, size)
        outcomes = [] if linked is None else [(source, _linked_problems(code, linked, address))
                                               for source, code in candidates]
        # Objects sharing a file name are told apart by their bytes.
        match = next(((source, found) for source, found in outcomes if found is not None), None)
        if match is None:
            text = f"linked {name} of {file} at {address:#x} is not the object's code"
            hits.append({"rule": "object_instruction", "path": candidates[0][0], "text": text[:200]})
            continue
        hits += [{"rule": rule, "path": match[0], "text": text[:200]} for rule, text in match[1]]
    if not seen and sections:
        raise ValueError(f"no kernel code in link map {link_map}")
    return hits


def _defined_symbols(
    objects: list[Path], kinds: str = "BDRT", limits: tuple[float, str] = (math.inf, ""),
) -> frozenset[str]:
    """Definitions whose nm type is in `kinds`: strong global by default.

    Weak and common can lose to harness, so they are asked for apart.
    """
    out = run_binutil("arm-none-eabi-nm", ["--defined-only", *map(str, objects)], *limits)
    return frozenset(parts[2] for parts in map(str.split, out.splitlines()) if len(parts) == 3 and parts[1] in kinds)


def _size_error(objects: list[tuple[str, Path]]) -> Optional[str]:
    """Why the objects are too big."""
    sizes = [(obj.stat().st_size, source) for source, obj in objects]
    big = [source for size, source in sizes if size > OBJECT_CAP]
    if big:
        return f"object over {OBJECT_CAP >> 20} MiB: {big[0]}"
    if sum(size for size, _ in sizes) > OBJECTS_TOTAL_CAP:
        return f"objects over {OBJECTS_TOTAL_CAP >> 20} MiB in total"
    return None


def object_findings(build_dir: Path, deadline_s: float = SCAN_DEADLINE_S) -> tuple[list[dict], dict, dict]:
    """Object hits, a summary, per-unit -E flags."""
    from .firmware_build import built_record, nsx_app_dir

    summary = {"build_dir": str(build_dir), "count": 0,
               "kernels_hash": built_record(nsx_app_dir(build_dir)).get("kernels") or None}

    def error(message: str) -> tuple[list[dict], dict, dict]:
        return [{"rule": "scan_error", "path": "", "message": message[:200]}], summary, {}

    deadline = time.monotonic() + deadline_s
    with tempfile.TemporaryDirectory() as tmp:
        limits = (deadline, str(Path(tmp, "abort")))
        pool = ThreadPoolExecutor(max_workers=scan_workers())
        try:
            units = _kernel_units(build_dir)
            missing = [source for source, obj, _ in units if not obj.is_file()]
            if not units or missing:
                return error(f"kernel object missing: {missing[0]}" if missing else "no kernel objects found")
            too_big = _size_error([(source, obj) for source, obj, _ in units])
            if too_big:
                return error(too_big)
            objects = [obj for _, obj, _ in units]
            defined = _defined_symbols(objects, limits=limits)
            weak = _defined_symbols(objects, kinds="VW", limits=limits)
            jobs = [pool.submit(_object_hits, source, obj, defined, weak, limits) for source, obj, _ in units]
            done, late = wait(jobs, timeout=max(0.0, deadline - time.monotonic()))
            if late:
                return error(f"object scan passed its {deadline_s:g} s deadline")
            results = [(source, job.result()) for (source, _, _), job in zip(units, jobs)]
            findings = [hit for _, (hits, *_) in results for hit in hits] + _external_hits(results)
            findings += _linked_hits(build_dir, [(source, obj, result) for (source, obj, _), (_, result)
                                                 in zip(units, results)])
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
            return error(f"object scan failed: {exc}")
        finally:
            # Kill running binutils, drop queued jobs.
            Path(limits[1]).touch()
            pool.shutdown(wait=True, cancel_futures=True)
    summary["count"] = len(units)
    # gcc -E cannot take clang flags.
    if Path(units[0][2][0]).stem == toolchain_spec("atfe").compiler:
        return findings, summary, {}
    flags = {source: _preprocess_args(args, source) for source, _, args in units}
    # New units get the first unit's flags.
    return findings, summary, {"*": flags[units[0][0]], **flags}
