"""Rerun candidate rules on what the preprocessor makes.

Line rules read source text, so they miss tokens the preprocessor
builds (`_Pra##gma`, `__attri##bute__`, nested macros, macros a
header defines and an unchanged unit uses). Every Source/ C and .S
unit, in the base and the candidate, goes through `gcc -E` for each
core in CONFIGS. Lines are credited to the kernel file they came from
(system headers drop out), and a rule whose count grows in a file is
a finding. Counts per file: swapping one existing hit for another
stays unseen.

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
import re
import shlex
import subprocess
import tarfile
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Callable, Optional

from .toolchain import arm_tool, run_tool

# Cores the boards use: MVE, DSP.
_COMMON = ("-mthumb", "-mfloat-abi=hard", "-Ofast", "-ffast-math", "-DARM_NN_ENABLE_F32=1")
CONFIGS = {
    "cortex-m55": ("-mcpu=cortex-m55", *_COMMON, "-DARM_NN_ENABLE_F16=1", "-DCMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY"),
    "cortex-m4": ("-mcpu=cortex-m4", "-mfpu=fpv4-sp-d16", *_COMMON),
}
_MARKER = re.compile(r'^#\s*\d+\s+"([^"]*)".*$', re.MULTILINE)


def _units(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for ext in ("*.c", "*.S") for p in (root / "Source").rglob(ext))


def _preprocess(gcc: str, root: Path, unit: str, flags: tuple[str, ...]) -> Optional[dict[str, str]]:
    """Kernel text of one unit, by origin."""
    done = subprocess.run([gcc, "-E", *flags, "-IInclude", unit], cwd=root, capture_output=True, text=True,
                          errors="replace", check=False)
    if done.returncode != 0:
        return None
    by_file: dict[str, list[str]] = {}
    markers = list(_MARKER.finditer(done.stdout))
    for marker, after in zip(markers, [*markers[1:], None]):
        name = marker.group(1)
        # System headers and builtins drop out.
        if name.startswith(("/", "<")):
            continue
        end = after.start() if after else len(done.stdout)
        by_file.setdefault(Path(name).as_posix(), []).append(done.stdout[marker.end():end])
    return {origin: "".join(chunks) for origin, chunks in by_file.items()}


def _tree_counts(gcc: str, root: Path, rule_counts: Callable[[str], Counter], configs: dict) -> tuple[dict, set[str]]:
    """Max rule counts per origin, plus failed units."""
    counts: dict[tuple[str, str], Counter] = {}
    memo: dict[str, Counter] = {}
    failed: set[str] = set()
    jobs = [(config, unit) for config in configs for unit in _units(root)]
    # Parsing gcc -E output is CPU bound.
    with ProcessPoolExecutor() as pool:
        flags = [configs[config] for config, _ in jobs]
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
            tarfile.open(fileobj=io.BytesIO(archive)).extractall(base, filter="data")
            before, base_failed = _tree_counts(gcc, base, rule_counts, configs)
            after, failed = _tree_counts(gcc, tree, rule_counts, configs)
        except (OSError, tarfile.TarError) as exc:
            return [{"rule": "scan_error", "path": "", "message": f"preprocess failed: {exc}"[:200]}]
    findings = [{"rule": "scan_error", "path": unit.split(":", 1)[1], "message": f"gcc -E failed for {unit}"}
                for unit in sorted(failed - base_failed)]
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
    r"^(?:\.rela?(?=\.))?(?:(?:\.text|\.rodata|\.data|\.bss)(?:\..*)?$|\.ARM\.(?:attributes|exidx|extab)|\.debug)"
    r"|^\.comment$|^\.note\.GNU-stack$|^\.group$|^\.(?:sym|str|shstr)tab$|^$"
)
_SECTION_ROW = re.compile(r"^\s*\[\s*\d+\]\s+(\S*)\s+([A-Z][A-Z0-9_]*)\s+[0-9a-f]{8}\s+\S+\s+\S+\s+\S+\s+([A-Z]*)")
_ADDRESS_INSN = re.compile(r"\t(?:mov|movw|movt|ldr)\S*\t|\t\.word\t")
_IMMEDIATE = re.compile(r"#(-?(?:0x[0-9a-fA-F]+|\d+))|\.word\t(0x[0-9a-fA-F]+)")
_DUMP_ROW = re.compile(r"^ [0-9a-f]+ ((?:[0-9a-f]{2,8} ){1,4})")
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
    kept, skip = [], False
    for arg in args[1:]:
        if skip or arg in _DROP_ARGS or arg.endswith(source):
            skip = False
            continue
        skip = arg in _DROP_WITH_VALUE
        marker = next((m for m in _markers() if m in arg), None)
        # Kernel paths point at the scanned tree.
        if marker and arg.startswith("-I"):
            arg = "-I" + arg.split(marker, 1)[1]
        if not skip:
            kept.append(arg)
    return tuple(kept)


def _object_hits(source: str, obj: Path) -> list[dict]:
    """Section and SCS hits in one object."""
    hits, data = [], []
    for row in run_tool("arm-none-eabi-readelf", ["-SW", str(obj)]).splitlines():
        match = _SECTION_ROW.match(row)
        if not match:
            continue
        name, kind, flags = match.groups()
        # Code may live in .text only.
        if not _SECTION_OK.match(name) or ("X" in flags and not name.startswith(".text")):
            hits.append({"rule": "object_section", "path": source, "text": f"{name} {flags}"})
        if kind == "PROGBITS" and "A" in flags and "X" not in flags:
            data += ["-j", name]
    for line in run_tool("arm-none-eabi-objdump", ["-d", "--no-show-raw-insn", str(obj)]).splitlines():
        if not _ADDRESS_INSN.search(line):
            continue
        for match in _IMMEDIATE.finditer(line):
            value = int(match.group(1) or match.group(2), 0) & 0xFFFFFFFF
            # movt loads the top half.
            if SCS_LOW <= value <= SCS_HIGH or ("\tmovt" in line and SCS_LOW >> 16 <= value <= SCS_HIGH >> 16):
                hits.append({"rule": "object_address", "path": source, "text": line.strip()[:200]})
    for line in run_tool("arm-none-eabi-objdump", ["-s", *data, str(obj)]).splitlines() if data else ():
        row = _DUMP_ROW.match(line)
        words = [int.from_bytes(bytes.fromhex(w), "little") for w in (row.group(1).split() if row else ()) if len(w) == 8]
        if any(SCS_LOW <= word <= SCS_HIGH for word in words):
            hits.append({"rule": "object_address", "path": source, "text": line.strip()[:200]})
    return hits


def object_findings(build_dir: Path) -> tuple[list[dict], dict, tuple[str, ...]]:
    """Object hits, a summary, the build's -E flags."""
    from .firmware_build import built_record, nsx_app_dir

    summary = {"build_dir": str(build_dir), "objects": 0,
               "kernels_hash": built_record(nsx_app_dir(build_dir)).get("kernels") or None}
    try:
        units = _kernel_units(build_dir)
        missing = [source for source, obj, _ in units if not obj.is_file()]
        if not units or missing:
            message = f"kernel object missing: {missing[0]}" if missing else "no kernel objects found"
            return [{"rule": "scan_error", "path": "", "message": message}], summary, ()
        with ThreadPoolExecutor() as pool:
            findings = [hit for hits in pool.map(lambda unit: _object_hits(*unit[:2]), units) for hit in hits]
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        return [{"rule": "scan_error", "path": "", "message": f"object scan failed: {exc}"[:200]}], summary, ()
    summary["objects"] = len(units)
    source, _, args = units[0]
    return findings, summary, _preprocess_args(args, source)
