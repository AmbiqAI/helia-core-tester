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
"""

from __future__ import annotations

import io
import os
import re
import selectors
import signal
import subprocess
import tarfile
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
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
            extract_tar(archive, base)
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
