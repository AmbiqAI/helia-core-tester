"""Load a library's pack modules (profile ``pack.modules``) into a frozen registry."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from helia_core_tester.pack import PackError, Registry, api_version_problem


@dataclass(frozen=True)
class LoadedPack:
    """A profile's pack: the spec, the registry its modules filled, and the files that decide it."""

    library: str
    spec: Any
    registry: Registry
    files: Tuple[Path, ...]


def _package_name(library: str, root: Path) -> str:
    digest = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:12]
    return f"_hct_pack_{re.sub(r'[^A-Za-z0-9_]', '_', library)}_{digest}"


def _load_module(package: str, name: str, path: Path) -> types.ModuleType:
    qualified = f"{package}.{name}"
    spec = importlib.util.spec_from_file_location(qualified, path)
    if spec is None or spec.loader is None:
        raise PackError(f"{path}: cannot be loaded as a Python module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    try:
        # Compiled from the source each time: no bytecode is read from or written into the library tree.
        code = compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        exec(code, module.__dict__)  # noqa: S102
    except (Exception, SystemExit) as exc:
        sys.modules.pop(qualified, None)
        raise PackError(f"{path}: import failed: {type(exc).__name__}: {exc}") from exc
    return module


def load_pack(profile: Any) -> LoadedPack:
    """Import every module the profile's pack names and run its register(); fail closed on any doubt.

    The modules load as a package rooted at the pack directory, so they import each other relatively
    (``from . import rules``); the directory is not put on sys.path, so a pack file cannot shadow a
    module hct or the standard library imports."""
    spec = profile.pack
    if spec is None:
        raise PackError(f"{profile.describe()}: its profile names no pack")
    package = _package_name(profile.library, spec.root)
    for stale in [key for key in sys.modules if key == package or key.startswith(package + ".")]:
        del sys.modules[stale]
    parent = types.ModuleType(package)
    parent.__path__ = [str(spec.root)]  # type: ignore[attr-defined]
    sys.modules[package] = parent
    registry = Registry()
    files = [] if profile.profile_path is None else [Path(profile.profile_path)]
    files.extend(Path(p) for p in (profile.signatures, profile.implementations) if p is not None)
    writes_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        _register_modules(package, spec, registry, files)
    finally:
        sys.dont_write_bytecode = writes_bytecode
    registry._module = "<frozen>"
    registry._frozen = True
    # Helpers a module imports and data a generator reads decide its cases too.
    files.extend(_pack_files(spec.root, files))
    return LoadedPack(library=profile.library, spec=spec, registry=registry, files=tuple(files))


def _pack_files(root: Path, known: list) -> list:
    return sorted(p for p in root.rglob("*") if p.is_file() and p not in known and "__pycache__" not in p.parts)


def _register_modules(package: str, spec: Any, registry: Registry, files: list) -> None:
    for name in spec.modules:
        path = spec.module_path(name)
        module = _load_module(package, name, path)
        files.append(path)
        problem = api_version_problem(getattr(module, "HCT_PACK_API", None))
        if problem:
            raise PackError(f"{path}: HCT_PACK_API: {problem}")
        register = getattr(module, "register", None)
        if not callable(register):
            raise PackError(f"{path}: defines no register(pack)")
        registry._module = name
        try:
            register(registry)
        except PackError:
            raise
        except (Exception, SystemExit) as exc:
            raise PackError(f"{path}: register() failed: {type(exc).__name__}: {exc}") from exc


_CACHE: Dict[Tuple[str, Tuple[Tuple[str, int, int], ...]], LoadedPack] = {}


def _fingerprint(profile: Any) -> Tuple[Tuple[str, int, int], ...]:
    spec = profile.pack
    paths = [Path(profile.profile_path)] if profile.profile_path is not None else []
    paths.extend(Path(p) for p in (profile.signatures, profile.implementations) if p is not None)
    paths.extend(_pack_files(spec.root, paths))
    entries = []
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            continue
        entries.append((str(path), stat.st_size, stat.st_mtime_ns))
    return tuple(entries)


def pack_of(profile: Any) -> LoadedPack:
    """load_pack, cached for the process until a pack file or the profile changes on disk."""
    key = (str(profile.profile_path), _fingerprint(profile))
    cached = _CACHE.get(key)
    if cached is None:
        cached = load_pack(profile)
        _CACHE.clear()
        _CACHE[key] = cached
    return cached


def current_pack() -> Optional[LoadedPack]:
    """The pack of the library generation runs against, or None when it runs hct's built-in one."""
    from helia_core_tester.core.harness import current_library

    profile = current_library()
    if profile is None or profile.pack is None:
        return None
    return pack_of(profile)


def files_digest(pack: Optional[LoadedPack]) -> str:
    """A digest of the files that decide the pack's cases (empty for the built-in pack)."""
    if pack is None:
        return ""
    digest = hashlib.sha256()
    for path in pack.files:
        digest.update(os.path.relpath(path, pack.spec.root).encode("utf-8", "surrogateescape") + b"\0")
        try:
            digest.update(path.read_bytes())
        except OSError as exc:
            raise PackError(f"{path}: cannot read: {exc.strerror}") from None
        digest.update(b"\0")
    return digest.hexdigest()
