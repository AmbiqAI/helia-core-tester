#!/usr/bin/env python3
"""Vendor the TFLM reference-kernel closure the hct_ref shim compiles against.

Walks every quoted #include reachable from helia_core_tester/reference_kernels/shim/*.cc
through an upstream google/tflite-micro checkout and a gemmlowp checkout, copies
exactly that closure (plus the companion .cc of each kernels/internal header that has one) into
reference_kernels/third_party/, and writes manifest.json (per-file sha256, the
compiled sources, both commits) and VENDOR.md.

Fails, copying nothing, when an include resolves outside the allow-listed roots
(anything under tensorflow/lite/micro that the shim does not stub, ruy, flatbuffers,
schema, ...): the vendored tree must stay the reference kernels and their headers.

    python3 scripts/vendor_tflm_reference.py --tflm <tflite-micro checkout> \
        --gemmlowp <gemmlowp checkout> [--check]

--check compares the would-be tree with the vendored one and exits non-zero on drift.
Run from a clean checkout of each upstream at the commit to vendor; never from a
fork whose license differs from upstream's.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REF_ROOT = REPO_ROOT / "helia_core_tester" / "reference_kernels"
SHIM_DIR = REF_ROOT / "shim"
STUBS_DIR = SHIM_DIR / "stubs"
THIRD_PARTY = REF_ROOT / "third_party"

INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.MULTILINE)

# Roots an include may resolve into, per upstream.
TFLM_ALLOWED = (
    "tensorflow/lite/kernels/internal/",
    "tensorflow/lite/kernels/op_macros.h",
    "tensorflow/lite/core/macros.h",
    "tensorflow/lite/core/c/common.h",
    "tensorflow/lite/core/c/c_api_types.h",
    "tensorflow/lite/core/c/builtin_op_data.h",
    "tensorflow/compiler/mlir/lite/core/c/",
)
TFLM_DENIED = (
    # Optimized kernels pull in ruy/eigen; the reference build never needs them.
    "tensorflow/lite/kernels/internal/optimized/",
)
TFLM_DENIED_EXCEPTIONS = ("tensorflow/lite/kernels/internal/optimized/neon_check.h",)
GEMMLOWP_ALLOWED = ("fixedpoint/", "internal/detect_platform.h")

LICENSE_FILES = {"tflite_micro": ("LICENSE",), "gemmlowp": ("LICENSE", "AUTHORS", "CONTRIBUTORS")}


class VendorError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head(checkout: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise VendorError(f"{checkout} is not a git checkout: {exc}") from exc
    status = subprocess.run(
        ["git", "-C", str(checkout), "status", "--porcelain"], capture_output=True, text=True, check=True
    )
    if status.stdout.strip():
        raise VendorError(f"{checkout} has local changes; vendor from a clean upstream checkout")
    return out.stdout.strip()


def _allowed(rel: str, allowed: tuple[str, ...], denied: tuple[str, ...] = (), exceptions: tuple[str, ...] = ()) -> bool:
    if rel in exceptions:
        return True
    if any(rel.startswith(prefix) for prefix in denied):
        return False
    return any(rel == prefix or rel.startswith(prefix) for prefix in allowed)


def _resolve(include: str, including: Path, roots: dict[str, Path]) -> tuple[str, str, Path]:
    """Return (origin, relative path, absolute path) for one quoted include."""
    stub = STUBS_DIR / include
    if stub.is_file():
        return "stub", include, stub
    shim = SHIM_DIR / include
    if shim.is_file():
        return "shim", include, shim
    # Relative to the including file (gemmlowp uses "./x.h" and "../internal/x.h").
    for origin, root in roots.items():
        try:
            including.resolve().relative_to(root.resolve())
        except ValueError:
            continue
        candidate = (including.parent / include).resolve()
        if candidate.is_file():
            return origin, candidate.relative_to(root.resolve()).as_posix(), candidate
    for origin, root in roots.items():
        candidate = root / include
        if candidate.is_file():
            return origin, include, candidate
    raise VendorError(f"unresolvable include {include!r} (from {including})")


def collect_closure(tflm: Path, gemmlowp: Path) -> tuple[dict[str, dict[str, Path]], list[str]]:
    roots = {"tflite_micro": tflm, "gemmlowp": gemmlowp}
    files: dict[str, dict[str, Path]] = {"tflite_micro": {}, "gemmlowp": {}}
    sources: list[str] = []
    pending: list[Path] = sorted(SHIM_DIR.glob("*.cc"))
    seen: set[Path] = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        for include in INCLUDE_RE.findall(current.read_text(encoding="utf-8", errors="replace")):
            origin, rel, path = _resolve(include, current, roots)
            if origin in ("stub", "shim"):
                pending.append(path)
                continue
            if origin == "tflite_micro" and not _allowed(rel, TFLM_ALLOWED, TFLM_DENIED, TFLM_DENIED_EXCEPTIONS):
                raise VendorError(f"{current.name} reaches {rel}, outside the reference-kernel allow-list")
            if origin == "gemmlowp" and not _allowed(rel, GEMMLOWP_ALLOWED):
                raise VendorError(f"{current.name} reaches gemmlowp/{rel}, outside the allow-list")
            if rel not in files[origin]:
                files[origin][rel] = path
                pending.append(path)
                companion = path.with_suffix(".cc")
                # Only the kernel library's own .cc files are compiled; the C API
                # headers (core/c) are used header-only.
                if (
                    origin == "tflite_micro"
                    and rel.startswith("tensorflow/lite/kernels/internal/")
                    and path.suffix == ".h"
                    and companion.is_file()
                ):
                    crel = Path(rel).with_suffix(".cc").as_posix()
                    if crel not in files[origin]:
                        files[origin][crel] = companion
                        sources.append(f"third_party/tflite_micro/{crel}")
                        pending.append(companion)
    return files, sorted(sources)


def _write_tree(dest: Path, tflm: Path, gemmlowp: Path, commits: dict[str, str]) -> None:
    files, sources = collect_closure(tflm, gemmlowp)
    upstream_roots = {"tflite_micro": tflm, "gemmlowp": gemmlowp}
    manifest_files: dict[str, str] = {}
    for origin, entries in files.items():
        for rel, path in sorted(entries.items()):
            target = dest / origin / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            manifest_files[f"third_party/{origin}/{rel}"] = _sha256(target)
        for name in LICENSE_FILES[origin]:
            src = upstream_roots[origin] / name
            if src.is_file():
                shutil.copyfile(src, dest / origin / name)
                manifest_files[f"third_party/{origin}/{name}"] = _sha256(dest / origin / name)
    manifest = {
        "schema": 1,
        "upstream": {
            "tflite_micro": {"repo": "https://github.com/tensorflow/tflite-micro", "commit": commits["tflite_micro"]},
            "gemmlowp": {"repo": "https://github.com/google/gemmlowp", "commit": commits["gemmlowp"]},
        },
        "sources": sources,
        "files": dict(sorted(manifest_files.items())),
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    (dest / "VENDOR.md").write_text(_vendor_md(commits, len(manifest_files), sources), encoding="utf-8")


def _vendor_md(commits: dict[str, str], count: int, sources: list[str]) -> str:
    source_lines = "\n".join(f"- `{s}`" for s in sources)
    return f"""# Vendored reference kernels

Generated by `scripts/vendor_tflm_reference.py`; do not edit by hand.

| Upstream | Commit | License |
|---|---|---|
| [google/tflite-micro](https://github.com/tensorflow/tflite-micro) | `{commits['tflite_micro']}` | Apache-2.0 (`tflite_micro/LICENSE`) |
| [google/gemmlowp](https://github.com/google/gemmlowp) | `{commits['gemmlowp']}` | Apache-2.0 (`gemmlowp/LICENSE`) |

The tflite-micro commit is the upstream main snapshot (2026-05-06) that helia-rt's
replant (`690a2d72`) synced; its `tensorflow/lite/kernels/internal` tree,
`reference/` and `reference/integer_ops/` included, is blob-identical to that
replant. The gemmlowp commit is the one tflite-micro pins in
`tensorflow/lite/micro/tools/make/third_party_downloads.inc`. Code is taken from
upstream only: never from the helia-rt tree, whose root license restricts it to
Ambiq CPUs.

{count} files, each hashed in `manifest.json` (checked by
`helia_core_tester/tests/test_reference_vendor.py`). Only the include closure of
`shim/*.cc` is vendored, plus the companion `.cc` of each header that has one;
nothing from `tensorflow/lite/micro`, ruy, flatbuffers or the schema.

Stubs in `shim/stubs/` replace the two headers outside that closure:
`ruy/profiler/instrumentation.h` (an empty `ScopeLabel`) and
`tensorflow/lite/micro/micro_log.h` (the `TF_LITE_STRIP_ERROR_STRINGS` no-op forms).

Compiled sources (besides `shim/*.cc`):

{source_lines}

Refresh:

```bash
git -C <tflite-micro> checkout <commit> && git -C <gemmlowp> checkout <commit>
python3 scripts/vendor_tflm_reference.py --tflm <tflite-micro> --gemmlowp <gemmlowp>
```
"""


def _tree_digest(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): _sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tflm", type=Path, required=True, help="upstream tflite-micro checkout")
    parser.add_argument("--gemmlowp", type=Path, required=True, help="upstream gemmlowp checkout")
    parser.add_argument("--check", action="store_true", help="fail if the vendored tree differs")
    args = parser.parse_args(argv)
    try:
        commits = {"tflite_micro": _git_head(args.tflm), "gemmlowp": _git_head(args.gemmlowp)}
        with tempfile.TemporaryDirectory() as tmp:
            staged = Path(tmp) / "third_party"
            _write_tree(staged, args.tflm.resolve(), args.gemmlowp.resolve(), commits)
            if args.check:
                if not THIRD_PARTY.is_dir() or _tree_digest(staged) != _tree_digest(THIRD_PARTY):
                    print("vendored reference tree is out of date; rerun without --check", file=sys.stderr)
                    return 1
                print("vendored reference tree is up to date")
                return 0
            if THIRD_PARTY.exists():
                shutil.rmtree(THIRD_PARTY)
            shutil.copytree(staged, THIRD_PARTY)
    except VendorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    manifest = json.loads((THIRD_PARTY / "manifest.json").read_text(encoding="utf-8"))
    print(f"vendored {len(manifest['files'])} files into {THIRD_PARTY.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
