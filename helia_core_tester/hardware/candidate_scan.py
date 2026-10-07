"""Rerun candidate rules on what the preprocessor makes.

Line rules read source text, so they miss tokens the preprocessor
builds (`_Pra##gma`, `__attri##bute__`, nested macros, macros a
header defines and an unchanged unit uses). Every Source/ C and .S
unit, in the base and the candidate, goes through `gcc -E` for each
core in CONFIGS. Lines are credited to the kernel file they came from
(system headers drop out), and a rule whose count grows in a file is
a finding. Counts per file: swapping one existing hit for another
stays unseen. Each gcc -E run has a time and output cap; a unit
that hits either is a scan_error.

With a build dir, `gcc -E` also runs with that build's own compile
args (board and SoC macros, include paths), and each kernel object
from its compile_commands.json must hold code in .text only, standard
sections only, and no SCS address (0xE0000000-0xE00FFFFF: DWT, PMU,
SysTick, SCB) in a literal pool, mov/movw/movt or data word. Addresses
built at run time stay out of reach.
"""

from __future__ import annotations

import io
import json
import os
import re
import shlex
import selectors
import signal
import subprocess
import tarfile
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import repeat
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
_MARKER = re.compile(r'^#\s*\d+\s+"([^"]*)".*$', re.MULTILINE)


def _units(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for ext in ("*.c", "*.S") for p in (root / "Source").rglob(ext))


def run_capped(cmd: list[str], cwd: Path, timeout: float, cap: int) -> Optional[bytes]:
    """Stdout, or None on failure or cap."""
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
                if left <= 0 or not selector.select(left):
                    return None
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


def run_binutil(tool: str, args: list[str]) -> str:
    """Bounded binutil stdout; raises when capped."""
    out = run_capped([arm_tool(tool), *args], Path.cwd(), TIMEOUT_S, OUTPUT_CAP)
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


def _preprocess(gcc: str, root: Path, unit: str, flags: tuple[str, ...]) -> Optional[dict[str, str]]:
    """Kernel text of one unit, by origin."""
    out = run_capped([gcc, "-E", *flags, "-IInclude", unit], root, TIMEOUT_S, OUTPUT_CAP)
    if out is None:
        return None
    stdout = out.decode(errors="replace")
    by_file: dict[str, list[str]] = {}
    markers = list(_MARKER.finditer(stdout))
    for marker, after in zip(markers, [*markers[1:], None]):
        name = marker.group(1)
        # System headers and builtins drop out.
        if name.startswith(("/", "<")):
            continue
        end = after.start() if after else len(stdout)
        by_file.setdefault(Path(name).as_posix(), []).append(stdout[marker.end():end])
    return {origin: "".join(chunks) for origin, chunks in by_file.items()}


def _unit_flags(config: tuple | dict, unit: str) -> tuple[str, ...]:
    return config.get(unit, config["*"]) if isinstance(config, dict) else config


def _tree_counts(gcc: str, root: Path, rule_counts: Callable[[str], Counter], configs: dict) -> tuple[dict, set[str]]:
    """Max rule counts per origin, plus failed units."""
    counts: dict[tuple[str, str], Counter] = {}
    memo: dict[str, Counter] = {}
    failed: set[str] = set()
    jobs = [(config, unit) for config in configs for unit in _units(root)]
    # Parsing gcc -E output is CPU bound.
    with ProcessPoolExecutor() as pool:
        # A dict config holds per-unit flags.
        flags = [_unit_flags(configs[config], unit) for config, unit in jobs]
        results = pool.map(_preprocess, repeat(gcc), repeat(root), [unit for _, unit in jobs], flags, chunksize=8)
        for (config, unit), by_file in zip(jobs, results):
            if by_file is None:
                failed.add(f"{config}:{unit}")
                continue
            for origin, text in by_file.items():
                # Headers repeat across units.
                if text not in memo:
                    memo[text] = rule_counts(text)
                key = (config, origin)
                counts[key] = counts.get(key, Counter()) | memo[text]
    return counts, failed


def preprocess_findings(
    tree: Path, archive: bytes, rule_counts: Callable[[str], Counter], configs: dict = CONFIGS,
) -> list[dict]:
    """Rules that grow after gcc -E.

    archive: base Include/ and Source/ as a tar.
    """
    gcc = arm_tool("arm-none-eabi-gcc")
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        try:
            extract_tar(archive, base)
            before, base_failed = _tree_counts(gcc, base, rule_counts, configs)
            after, failed = _tree_counts(gcc, tree, rule_counts, configs)
        except (OSError, tarfile.TarError) as exc:
            return [{"rule": "scan_error", "path": "", "message": f"preprocess failed: {exc}"[:200]}]
    findings = [{"rule": "scan_error", "path": unit.split(":", 1)[1], "message": f"gcc -E failed for {unit}"}
                for unit in sorted(failed - base_failed | {u for u in failed if u.startswith("build:")})]
    # A config that never runs scans nothing.
    findings += [{"rule": "scan_error", "path": "", "message": f"gcc -E failed for all of {config}"}
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
_SECTION_ROW = re.compile(r"^\s*\[\s*(\d+)\]\s+(\S*)\s+([A-Z][A-Z0-9_]*)\s+[0-9a-f]{8}\s+\S+\s+\S+\s+\S+\s+([A-Z]*)")
_ADDRESS_INSN = re.compile(r"\t(?:mov|movw|movt|ldr)\S*\t|\t\.word\t")
_IMMEDIATE = re.compile(r"#(-?(?:0x[0-9a-fA-F]+|\d+))|\.word\t(0x[0-9a-fA-F]+)")
_CONTENTS = re.compile(r"^Contents of section (\S+):")
_DUMP_ROW = re.compile(r"^ [0-9a-f]+ ((?:[0-9a-f]{2,8} ){1,4})")
_RELOC_SECTION = re.compile(r"^Relocation section '\.rela?(\S+)'")
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


def _section_bytes(obj: Path, names: list[str]) -> dict[str, bytes]:
    """Contents of the named sections."""
    if not names:
        return {}
    found: dict[str, bytearray] = {}
    current = None
    args = [arg for name in names for arg in ("-j", name)]
    for line in run_binutil("arm-none-eabi-objdump", ["-s", *args, str(obj)]).splitlines():
        header = _CONTENTS.match(line)
        if header:
            current = found.setdefault(header.group(1), bytearray())
            continue
        row = _DUMP_ROW.match(line)
        if row and current is not None:
            current += bytes.fromhex("".join(row.group(1).split()))
    return {name: bytes(data) for name, data in found.items()}


def _object_hits(source: str, obj: Path, defined: frozenset[str]) -> list[dict]:
    """Section, SCS and reference hits."""
    hits, sections, names = [], {}, {}
    # Locals resolve only inside this object.
    local = _defined_symbols([obj], local=True)
    for row in run_binutil("arm-none-eabi-readelf", ["-SW", str(obj)]).splitlines():
        match = _SECTION_ROW.match(row)
        if not match:
            continue
        index, name, kind, flags = match.groups()
        names[index] = name
        # Code may live in .text only.
        if not _SECTION_OK.match(name) or ("X" in flags and not name.startswith(".text")):
            hits.append({"rule": "object_section", "path": source, "text": f"{name} {flags}"})
        if kind == "PROGBITS" and "A" in flags:
            sections[name] = flags

    # Code outside .text, whatever its flags.
    homes = {}
    for row in run_binutil("arm-none-eabi-readelf", ["-sW", str(obj)]).splitlines():
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

    for line in run_binutil("arm-none-eabi-objdump", ["-d", "--no-show-raw-insn", str(obj)]).splitlines():
        if not _ADDRESS_INSN.search(line):
            continue
        for match in _IMMEDIATE.finditer(line):
            value = int(match.group(1) or match.group(2), 0) & 0xFFFFFFFF
            # movt loads the top half.
            if SCS_LOW <= value <= SCS_HIGH or ("\tmovt" in line and SCS_LOW >> 16 <= value <= SCS_HIGH >> 16):
                hit(line.strip())
    contents = _section_bytes(obj, list(sections))
    for name, data in contents.items():
        # Any byte offset: loads may be unaligned.
        words = (int.from_bytes(data[i:i + 4], "little") for i in range(len(data) - 3))
        if "X" not in sections[name] and any(SCS_LOW <= word <= SCS_HIGH for word in words):
            hit(f"SCS address in {name}")
    section = None
    for line in run_binutil("arm-none-eabi-readelf", ["-rW", str(obj)]).splitlines():
        header = _RELOC_SECTION.match(line)
        if header:
            section = header.group(1)
            continue
        row = _RELOC_ROW.match(line)
        if not row or section not in contents:
            continue
        offset, kind, symbol = int(row.group(1), 16), row.group(2), row.group(3)
        # Symbol plus addend can reach any address.
        if kind == "R_ARM_ABS32":
            addend = int.from_bytes(contents[section][offset:offset + 4], "little", signed=True)
            if not 0 <= addend < _ADDEND_MAX:
                hit(f"{symbol}{addend:+#x} in {section}")
        home = homes.get(symbol, "")
        if kind in _CALLS and home and not home.startswith(".text"):
            hits.append({"rule": "object_section", "path": source, "text": f"branch to {symbol} in {home}"})
        if kind not in _CALLS and not symbol.startswith(".") and symbol not in defined | local:
            hit(f"{kind} to {symbol} in {section}")
    return hits


def _defined_symbols(objects: list[Path], local: bool = False) -> frozenset[str]:
    """Strong global (or local) definitions."""
    out = run_binutil("arm-none-eabi-nm", ["--defined-only", *map(str, objects)])
    # Weak and common can lose to harness.
    kinds = "bdrt" if local else "BDRT"
    return frozenset(parts[2] for parts in map(str.split, out.splitlines()) if len(parts) == 3 and parts[1] in kinds)


def object_findings(build_dir: Path) -> tuple[list[dict], dict, dict]:
    """Object hits, a summary, per-unit -E flags."""
    from .firmware_build import built_record, nsx_app_dir

    summary = {"build_dir": str(build_dir), "count": 0,
               "kernels_hash": built_record(nsx_app_dir(build_dir)).get("kernels") or None}
    try:
        units = _kernel_units(build_dir)
        missing = [source for source, obj, _ in units if not obj.is_file()]
        if not units or missing:
            message = f"kernel object missing: {missing[0]}" if missing else "no kernel objects found"
            return [{"rule": "scan_error", "path": "", "message": message}], summary, {}
        defined = _defined_symbols([obj for _, obj, _ in units])
        with ThreadPoolExecutor() as pool:
            found = pool.map(lambda unit: _object_hits(unit[0], unit[1], defined), units)
            findings = [hit for hits in found for hit in hits]
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        return [{"rule": "scan_error", "path": "", "message": f"object scan failed: {exc}"[:200]}], summary, {}
    summary["count"] = len(units)
    flags = {source: _preprocess_args(args, source) for source, _, args in units}
    # New units get the first unit's flags.
    return findings, summary, {"*": flags[units[0][0]], **flags}
