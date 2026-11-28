"""Load a library's pack modules (profile ``pack.modules``) into a frozen registry."""

from __future__ import annotations

import hashlib
import importlib
import importlib.abc
import importlib.machinery
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


class _SourceOnlyLoader(importlib.machinery.SourceFileLoader):
    """Compiles the module from its source each time: no bytecode is read from or written into the pack."""

    def get_code(self, fullname):
        path = self.get_filename(fullname)
        return compile(self.get_data(path), path, "exec", dont_inherit=True)


class _PackFinder(importlib.abc.MetaPathFinder):
    """Finds the modules of each loaded pack's package, the ones register() runs and every helper
    they import (at load or later, from a generator), so all of them load from source only."""

    def __init__(self) -> None:
        self.roots: Dict[str, Path] = {}

    def find_spec(self, fullname, path=None, target=None):
        package, _, rest = fullname.partition(".")
        root = self.roots.get(package)
        if root is None or not rest:
            return None
        base = root.joinpath(*rest.split("."))
        if (base / "__init__.py").is_file():
            init = str(base / "__init__.py")
            return importlib.util.spec_from_file_location(
                fullname, init, loader=_SourceOnlyLoader(fullname, init), submodule_search_locations=[str(base)])
        source = base.parent / f"{base.name}.py"
        if source.is_file():
            return importlib.util.spec_from_file_location(
                fullname, str(source), loader=_SourceOnlyLoader(fullname, str(source)))
        return None


_FINDER = _PackFinder()


def _install_finder(package: str, root: Path) -> None:
    if _FINDER not in sys.meta_path:
        sys.meta_path.insert(0, _FINDER)
    _FINDER.roots[package] = root


def _load_module(package: str, name: str, path: Path) -> types.ModuleType:
    """The pack module `name`; one an earlier module already imported in this load is reused, not run again."""
    qualified = f"{package}.{name}"
    if not path.is_file():
        raise PackError(f"{path}: cannot be loaded as a Python module")
    try:
        return importlib.import_module(qualified)
    except (Exception, SystemExit) as exc:
        sys.modules.pop(qualified, None)
        raise PackError(f"{path}: import failed: {type(exc).__name__}: {exc}") from exc


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
    _install_finder(package, spec.root)
    registry = Registry()
    files = [] if profile.profile_path is None else [Path(profile.profile_path)]
    files.extend(Path(p) for p in (profile.signatures, profile.implementations) if p is not None)
    _register_modules(package, spec, registry, files)
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
