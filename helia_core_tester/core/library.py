"""The library under test: its profile (hct-library.yaml) and where its sources are.

A library hands hct its tests through a profile, ``hct-library.yaml``, in the directory given with
``--library``. hct reads only the fields it uses; an unknown field, a duplicate key or a value of the
wrong type refuses the profile rather than being ignored. Fields arrive as the engine starts to read
them, so a profile written for a newer hct is refused by an older one.

Fields (schema ``hct/library-profile/1``):

- ``schema``: ``hct/library-profile/1``.
- ``library``: the library's id, lowercase (``ns-cmsis-nn``).
- ``source_root``: the library checkout, relative to the profile's directory (default ``.``). The
  profile's directory must lie inside it: the pack belongs to the library it tests.
- ``contracts``: ``signatures`` and, optionally, ``implementations``: the library's kernel contract
  documents, relative to the profile's directory and inside ``source_root``. Without it, an
  ns-cmsis-nn profile reads ``Tests/KernelContracts/`` of its checkout.
- ``harness``: what a generated harness needs from the library (``core/harness.py``). Without it, an
  ns-cmsis-nn profile takes ns-cmsis-nn's; any other library has none and cannot generate tests.
- ``host_reference``: the library's host reference build (``core/host_reference.py``): ``project``, the
  CMake project that builds it (default ``.``); ``option``, the cache option that turns it on;
  ``manifest``, the file name it writes in the build directory; ``build_inputs``, what decides the build.
  ``project`` and ``build_inputs`` are relative to ``source_root`` and must exist inside it. Without it,
  an ns-cmsis-nn profile takes ns-cmsis-nn's; any other library has none and cannot record goldens.
- ``pack``: the library's tests (``helia_core_tester.pack``): ``api``, the pack API version it is written
  for; ``path``, the pack root (default ``.``); ``descriptors``, the directory of committed descriptors,
  relative to the pack root; ``modules``, the Python modules under the pack root that generate them.
  Every path must exist inside ``source_root``. Without it, an ns-cmsis-nn profile runs hct's built-in
  ns-cmsis-nn descriptors; any other library has none and cannot generate tests.

An ns-cmsis-nn checkout without a profile still loads, as the built-in ``legacy`` ns-cmsis-nn profile,
until ns-cmsis-nn ships its own.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import jsonschema
import yaml

from helia_core_tester.core.errors import ConfigurationError
from helia_core_tester.core.harness import HarnessProfile, HarnessProfileError, harness_for, legacy_harness
from helia_core_tester.core.host_reference import HostReferenceError, HostReferenceRecipe, parse_recipe
from helia_core_tester.pack import api_version_problem

PROFILE_FILE = "hct-library.yaml"
SCHEMA_ID = "hct/library-profile/1"
SCHEMA_PATH = Path(__file__).with_name("schemas") / "library-profile-1.json"
# Libraries hct carries a built-in pack for; any other runs from the pack its profile names.
SUPPORTED_LIBRARIES = ("ns-cmsis-nn",)
LEGACY_LIBRARY = "ns-cmsis-nn"
LEGACY_SIGNATURES = Path("Tests") / "KernelContracts" / "kernel_contracts.json"
LEGACY_IMPLEMENTATIONS = Path("Tests") / "KernelContracts" / "kernel_implementations.json"
MAX_PROFILE_BYTES = 256 * 1024


class LibraryProfileError(ConfigurationError):
    """The library or its profile cannot be used."""


@dataclass(frozen=True)
class PackSpec:
    """The profile's ``pack`` block, resolved: the pack root, its descriptors and its modules."""

    api: str
    root: Path
    descriptors: Path
    modules: tuple[str, ...] = ()

    def module_path(self, module: str) -> Path:
        return self.root / f"{module}.py"


@dataclass(frozen=True)
class LibraryProfile:
    """A loaded profile: which library, where its sources are, and where the profile came from."""

    library: str
    source_root: Path
    # The hct-library.yaml read, or None for the built-in legacy ns-cmsis-nn profile.
    profile_path: Optional[Path]
    # The contract documents the profile names, or None when it names none.
    signatures: Optional[Path] = None
    implementations: Optional[Path] = None
    harness: Optional[HarnessProfile] = None
    host_reference: Optional[HostReferenceRecipe] = None
    pack: Optional[PackSpec] = None

    @property
    def names_contracts(self) -> bool:
        return self.signatures is not None

    @property
    def legacy(self) -> bool:
        return self.profile_path is None

    def describe(self) -> str:
        origin = "built-in legacy profile" if self.legacy else f"profile {self.profile_path}"
        return f"{self.library} at {self.source_root} ({origin})"


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that refuses a mapping with the same key twice."""


def _construct_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    mapping: dict = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        try:
            hash(key)
        except TypeError:
            raise yaml.constructor.ConstructorError(
                None, None, f"unhashable key {key!r}", key_node.start_mark) from None
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                None, None, f"duplicate key {key!r}", key_node.start_mark)
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


@lru_cache(maxsize=1)
def _schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def looks_like_ns_cmsis_nn(root: Path) -> bool:
    return (root / "CMakeLists.txt").is_file() and (root / "Include" / "arm_nnfunctions.h").is_file()


def _read_profile(path: Path) -> Any:
    try:
        size = path.stat().st_size
        if size > MAX_PROFILE_BYTES:
            raise LibraryProfileError(f"{path}: {size} bytes; a profile is at most {MAX_PROFILE_BYTES}")
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise LibraryProfileError(f"{path}: not UTF-8 text: {exc.reason}") from None
    except OSError as exc:
        raise LibraryProfileError(f"{path}: cannot read: {exc.strerror}") from None
    try:
        return yaml.load(text, Loader=_UniqueKeyLoader)  # noqa: S506 - SafeLoader subclass
    except yaml.YAMLError as exc:
        raise LibraryProfileError(f"{path}: not valid YAML: {exc}") from None


def _validate(path: Path, data: Any) -> None:
    validator = jsonschema.Draft202012Validator(_schema())
    errors = sorted(validator.iter_errors(data), key=lambda error: list(error.absolute_path))
    if errors:
        first = errors[0]
        where = "/".join(str(part) for part in first.absolute_path) or "(top level)"
        more = f" (and {len(errors) - 1} more)" if len(errors) > 1 else ""
        raise LibraryProfileError(f"{path}: {where}: {first.message}{more}")


def _source_root(path: Path, value: str) -> Path:
    if os.path.isabs(value) or "\\" in value:
        raise LibraryProfileError(f"{path}: source_root {value!r} must be a relative POSIX path")
    try:
        profile_dir = path.parent.resolve()
        root = (profile_dir / value).resolve()
    except (OSError, ValueError, RuntimeError) as exc:
        raise LibraryProfileError(f"{path}: source_root {value!r} cannot be resolved ({exc})") from None
    if not root.is_dir():
        raise LibraryProfileError(f"{path}: source_root {value!r} is not a directory ({root})")
    if profile_dir != root and root not in profile_dir.parents:
        raise LibraryProfileError(
            f"{path}: source_root {value!r} ({root}) does not contain the profile; "
            "the pack must live inside the library it tests")
    return root


def _contract_path(path: Path, root: Path, key: str, value: str) -> Path:
    if os.path.isabs(value) or "\\" in value:
        raise LibraryProfileError(f"{path}: contracts.{key} {value!r} must be a relative POSIX path")
    try:
        target = (path.parent.resolve() / value).resolve()
    except (OSError, ValueError, RuntimeError) as exc:
        raise LibraryProfileError(f"{path}: contracts.{key} {value!r} cannot be resolved ({exc})") from None
    if root not in target.parents:
        raise LibraryProfileError(f"{path}: contracts.{key} {value!r} ({target}) is outside source_root {root}")
    return target


def _inside(path: Path, where: str, root: Path, value: str) -> Path:
    if os.path.isabs(value) or "\\" in value:
        raise LibraryProfileError(f"{where} {value!r} must be a relative POSIX path")
    if ".." in Path(value).parts:
        raise LibraryProfileError(f"{where} {value!r} must not contain '..'")
    try:
        target = (path / value).resolve()
    except (OSError, ValueError, RuntimeError) as exc:
        raise LibraryProfileError(f"{where} {value!r} cannot be resolved ({exc})") from None
    if target != root and root not in target.parents:
        raise LibraryProfileError(f"{where} {value!r} ({target}) is outside source_root {root}")
    return target


def pack_for(path: Path, data: Any, root: Path) -> Optional[PackSpec]:
    """The profile's ``pack`` block, checked against `root`, or None when it has none."""
    if data is None:
        return None
    where = f"{path}: pack"
    problem = api_version_problem(data["api"])
    if problem:
        raise LibraryProfileError(f"{where}: {problem}")
    pack_root = _inside(path.parent.resolve(), f"{where}.path", root, data.get("path", "."))
    if not pack_root.is_dir():
        raise LibraryProfileError(f"{where}.path {data.get('path', '.')!r} is not a directory ({pack_root})")
    descriptors = _inside(pack_root, f"{where}.descriptors", root, data["descriptors"])
    if not descriptors.is_dir():
        raise LibraryProfileError(f"{where}.descriptors {data['descriptors']!r} is not a directory ({descriptors})")
    spec = PackSpec(api=data["api"], root=pack_root, descriptors=descriptors, modules=tuple(data.get("modules", ())))
    for module in spec.modules:
        target = _inside(pack_root, f"{where}.modules", root, f"{module}.py")
        if not target.is_file():
            raise LibraryProfileError(f"{where}.modules: {module!r} has no {target}")
    return spec


def legacy_host_reference() -> HostReferenceRecipe:
    from helia_core_tester.contract import legacy_nn

    return parse_recipe(legacy_nn.LIBRARY, legacy_nn.HOST_REFERENCE, "ns-cmsis-nn built-in host reference")


def host_reference_for(library: str, data: Any, where: str, root: Path) -> Optional[HostReferenceRecipe]:
    """The profile's block, checked against `root`; ns-cmsis-nn's when an ns-cmsis-nn profile has
    none; else None."""
    if data is not None:
        try:
            return parse_recipe(library, data, where, root=root)
        except HostReferenceError as exc:
            raise LibraryProfileError(str(exc)) from None
    return legacy_host_reference() if library == LEGACY_LIBRARY else None


def load_profile(path: Path) -> LibraryProfile:
    """Load and check one hct-library.yaml."""
    path = Path(path)
    data = _read_profile(path)
    _validate(path, data)
    root = _source_root(path, data.get("source_root", "."))
    contracts = data.get("contracts") or {}
    try:
        harness = harness_for(data["library"], data.get("harness"), f"{path}: harness")
    except HarnessProfileError as exc:
        raise LibraryProfileError(str(exc)) from None
    return LibraryProfile(
        library=data["library"],
        source_root=root,
        profile_path=path.resolve(),
        signatures=_contract_path(path, root, "signatures", contracts["signatures"]) if contracts else None,
        implementations=_contract_path(path, root, "implementations", contracts["implementations"])
        if "implementations" in contracts else None,
        harness=harness,
        host_reference=host_reference_for(
            data["library"], data.get("host_reference"), f"{path}: host_reference", root),
        pack=pack_for(path, data.get("pack"), root),
    )


def legacy_profile(root: Path) -> LibraryProfile:
    """The built-in ns-cmsis-nn profile of a checkout that ships none."""
    return LibraryProfile(library=LEGACY_LIBRARY, source_root=Path(root).resolve(), profile_path=None,
                          harness=legacy_harness(), host_reference=legacy_host_reference())


def resolve_library(directory: Path, *, allow_legacy: bool = True) -> LibraryProfile:
    """The profile of the library in `directory`.

    `directory` holds hct-library.yaml. Without one, an ns-cmsis-nn checkout loads as the built-in
    legacy profile when `allow_legacy`; anything else is refused.
    """
    directory = Path(directory).expanduser()
    if not directory.is_dir():
        raise LibraryProfileError(f"library directory not found: {directory}")
    profile = directory / PROFILE_FILE
    if profile.exists() or profile.is_symlink():
        if not profile.is_file():
            raise LibraryProfileError(f"{profile}: not a regular file")
        return load_profile(profile)
    resolved = directory.resolve()
    if allow_legacy and looks_like_ns_cmsis_nn(resolved):
        return legacy_profile(resolved)
    raise LibraryProfileError(
        f"{directory} has no {PROFILE_FILE}"
        + (" and is not an ns-cmsis-nn checkout" if allow_legacy else ""))


def require_supported(profile: LibraryProfile) -> LibraryProfile:
    """Refuse a library this hct's engine cannot run yet.

    ns-cmsis-nn runs from hct's built-in pack or its own. Any other library runs from its own pack, so
    its profile must name the pack, the harness and the contracts its cases are generated against."""
    if profile.library not in SUPPORTED_LIBRARIES:
        missing = [name for name, value in (("pack", profile.pack), ("harness", profile.harness),
                                            ("contracts", profile.signatures)) if value is None]
        if missing:
            raise LibraryProfileError(
                f"library {profile.library!r} ({profile.describe()}) runs from its own pack, so its profile "
                f"must name {', '.join(missing)}")
        return profile
    # Generation reads ns-cmsis-nn's contracts from their default places until the engine reads the
    # profile's; a profile that names others would be silently ignored, so it is refused.
    defaults = (profile.source_root / LEGACY_SIGNATURES, profile.source_root / LEGACY_IMPLEMENTATIONS)
    if profile.names_contracts and (profile.signatures, profile.implementations) != defaults:
        named = (f"{profile.signatures} and {profile.implementations}" if profile.implementations
                 else f"{profile.signatures} and no implementation manifest")
        raise LibraryProfileError(
            f"{profile.profile_path}: contracts name {named}; this hct runs {profile.library} from "
            f"{defaults[0]} and {defaults[1]} only (hct contract validate reads others)")
    return profile
