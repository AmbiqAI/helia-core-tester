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
built at run time stay out of reach.
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

from .toolchain import arm_tool

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
        if not name.startswith("<"):
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
        with os.fdopen(fd, "rb") as handle:
            if stat.S_ISREG(os.fstat(fd).st_mode):
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
    r"|\.ARM\.(?:attributes|exidx|extab)|\.comment|\.note\.GNU-stack|\.group|\.(?:sym|str|shstr)tab|)$"
)
# Other names fail closed: no parser chasing.
# Little-endian words 0xE0000000-0xE00FFFFF.
_SCS_WORD = re.compile(rb"(?=[\x00-\xff]{2}[\x00-\x0f]\xe0)")
_PLAIN_NAME = re.compile(r"[A-Za-z0-9._$]+")
# ELF flag bits, as readelf letters.
_FLAG_LETTERS = ((0x1, "W"), (0x2, "A"), (0x4, "X"), (0x10, "M"), (0x20, "S"), (0x40, "I"), (0x80, "L"),
                 (0x200, "G"), (0x400, "T"))
_PROGBITS, _NOBITS, _RELA, _REL = 1, 8, 4, 9
_ADDRESS_INSN = re.compile(r"\t(?:mov|movw|movt|ldr)\S*\t|\t\.word\t")
_IMMEDIATE = re.compile(r"#(-?(?:0x[0-9a-fA-F]+|\d+))|\.word\t(0x[0-9a-fA-F]+)")
_RELOC_SECTION = re.compile(r"^Relocation section '.*' at offset")
_SYMBOL_ROW = re.compile(r"^\s*\d+:\s+[0-9a-f]+\s+\d+\s+(\w+)\s+\w+\s+\w+\s+(\d+)\s+(\S+)")
# Mapping symbols mark code: $t, $a.
_CODE_SYMBOL = re.compile(r"^\$[ta](?:\.|$)")
_RELOC_ROW = re.compile(r"^([0-9a-f]{8})\s+[0-9a-f]{8}\s+(R_ARM_\w+)\s+[0-9a-f]{8}\s+(\S+)")
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


def elf_sections(obj: Path) -> list[tuple[str, int, str, memoryview, int]]:
    """(name, type, flags, data, info) per section.

    Read from the ELF32 little-endian section table, not readelf text,
    so no name spelling can hide a section. The file is mapped, not
    read: section data are views into it.
    """
    with open(obj, "rb") as handle:
        # The map outlives the handle.
        raw = memoryview(mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ))
    if raw[:4] != b"\x7fELF" or raw[4:6] != b"\x01\x01":
        raise ValueError(f"{obj.name}: not ELF32 little-endian")
    try:
        shoff, = struct.unpack_from("<I", raw, 0x20)
        entsize, count, names_index = struct.unpack_from("<HHH", raw, 0x2E)
        headers = [struct.unpack_from("<10I", raw, shoff + i * entsize) for i in range(count or 1)]
        # Extended numbering keeps counts in section 0.
        count = count or headers[0][5]
        headers = [struct.unpack_from("<10I", raw, shoff + i * entsize) for i in range(count)]
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


def _object_hits(source: str, obj: Path, defined: frozenset[str], limits: tuple[float, str]) -> list[dict]:
    """Section, SCS and reference hits."""
    hits, sections, names = [], {}, {}
    # Locals resolve only inside this object.
    local = _defined_symbols([obj], local=True, limits=limits)
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
    homes = {}
    for row in run_binutil("arm-none-eabi-readelf", ["-sW", str(obj)], *limits).splitlines():
        match = _SYMBOL_ROW.match(row)
        if not match:
            continue
        kind, index, symbol = match.groups()
        home = names.get(index, "")
        homes[symbol] = home
        code = kind == "FUNC" or _CODE_SYMBOL.match(symbol)
        if code and not home.startswith(".text"):
            hits.append({"rule": "object_section", "path": source, "text": f"code {symbol} in {home}"})

    def hit(text: str) -> None:
        hits.append({"rule": "object_address", "path": source, "text": text[:200]})

    for line in run_binutil("arm-none-eabi-objdump", ["-d", "--no-show-raw-insn", str(obj)], *limits).splitlines():
        if not _ADDRESS_INSN.search(line):
            continue
        for match in _IMMEDIATE.finditer(line):
            value = int(match.group(1) or match.group(2), 0) & 0xFFFFFFFF
            # movt loads the top half.
            if SCS_LOW <= value <= SCS_HIGH or ("\tmovt" in line and SCS_LOW >> 16 <= value <= SCS_HIGH >> 16):
                hit(line.strip())
    contents = {index: table[index][3] for index in sections}
    for index, data in contents.items():
        # Any byte offset: loads may be unaligned.
        if "X" not in sections[index] and _SCS_WORD.search(data):
            hit(f"SCS address in {table[index][0]}")
    # readelf lists reloc sections in table order.
    targets = iter([info for _, kind, _, _, info in table if kind in (_REL, _RELA)])
    expected, seen_relocs = sum(kind in (_REL, _RELA) for _, kind, _, _, _ in table), 0
    target = None
    for line in run_binutil("arm-none-eabi-readelf", ["-rW", str(obj)], *limits).splitlines():
        if _RELOC_SECTION.match(line):
            seen_relocs += 1
            target = next(targets, None)
            continue
        row = _RELOC_ROW.match(line)
        if not row or target not in contents:
            continue
        section = table[target][0]
        offset, kind, symbol = int(row.group(1), 16), row.group(2), row.group(3)
        # Symbol plus addend can reach any address.
        if kind == "R_ARM_ABS32":
            addend = int.from_bytes(contents[target][offset:offset + 4], "little", signed=True)
            if not 0 <= addend < _ADDEND_MAX:
                hit(f"{symbol}{addend:+#x} in {section}")
        home = homes.get(symbol, "")
        if kind in _CALLS and home and not home.startswith(".text"):
            hits.append({"rule": "object_section", "path": source, "text": f"branch to {symbol} in {home}"})
        if kind not in _CALLS and not symbol.startswith(".") and symbol not in defined | local:
            hit(f"{kind} to {symbol} in {section}")
    if seen_relocs != expected:
        raise ValueError(f"{obj.name}: read {seen_relocs} of {expected} reloc sections")
    return hits


def _defined_symbols(
    objects: list[Path], local: bool = False, limits: tuple[float, str] = (math.inf, ""),
) -> frozenset[str]:
    """Strong global (or local) definitions."""
    out = run_binutil("arm-none-eabi-nm", ["--defined-only", *map(str, objects)], *limits)
    # Weak and common can lose to harness.
    kinds = "bdrt" if local else "BDRT"
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
            defined = _defined_symbols([obj for _, obj, _ in units], limits=limits)
            jobs = [pool.submit(_object_hits, source, obj, defined, limits) for source, obj, _ in units]
            done, late = wait(jobs, timeout=max(0.0, deadline - time.monotonic()))
            if late:
                return error(f"object scan passed its {deadline_s:g} s deadline")
            findings = [hit for job in jobs for hit in job.result()]
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
            return error(f"object scan failed: {exc}")
        finally:
            # Kill running binutils, drop queued jobs.
            Path(limits[1]).touch()
            pool.shutdown(wait=True, cancel_futures=True)
    summary["count"] = len(units)
    flags = {source: _preprocess_args(args, source) for source, _, args in units}
    # New units get the first unit's flags.
    return findings, summary, {"*": flags[units[0][0]], **flags}
