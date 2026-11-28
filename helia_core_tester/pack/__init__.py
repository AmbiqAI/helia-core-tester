"""The pack API: how a library hands hct the generators of its own tests (``hct.pack``).

A library's profile names its pack (``pack:`` in hct-library.yaml): a directory of committed
descriptors (``hct/descriptor/1``) and the Python modules that turn each descriptor into a case. A
module declares the API version it is written for and registers its generators:

    HCT_PACK_API = "1.0"

    def register(pack):
        pack.generator("basic_math.binary", build_binary, entries=["arm_add_*", "arm_sub_*"])

`build(case)` gets a ``Case`` (the descriptor's entry, its contract parameters, its ``params`` and a
seeded generator) and returns a ``CaseSpec``: an array for every input pointer, the output pointer and
its element count, a value for every scalar, and the golden, or None when the golden is recorded from
the library's host reference build (``generate --golden record``).

The version is MAJOR.MINOR. hct runs a pack whose major is its own and whose minor is not newer; a
breaking change bumps the major.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from helia_core_tester.core.errors import ConfigurationError

API_VERSION: Tuple[int, int] = (1, 0)
DESCRIPTOR_SCHEMA = "hct/descriptor/1"

_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
_ENTRY_PATTERN_RE = re.compile(r"^[A-Za-z_*][A-Za-z0-9_*]*$")
_VERSION_RE = re.compile(r"^([0-9]+)\.([0-9]+)$")


class PackError(ConfigurationError):
    """A pack, its modules or one of its cases cannot be used."""


def api_version_problem(value: Any) -> Optional[str]:
    """Why `value` (MAJOR.MINOR) is not an API version this hct runs, or None."""
    match = _VERSION_RE.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        return f"API version {value!r} is not MAJOR.MINOR"
    major, minor = int(match.group(1)), int(match.group(2))
    if major != API_VERSION[0] or minor > API_VERSION[1]:
        return (f"API version {value} is not supported: this hct runs {API_VERSION[0]}.0 to "
                f"{API_VERSION[0]}.{API_VERSION[1]}")
    return None


@dataclass(frozen=True)
class Param:
    """One parameter of the entry, as the library's contract declares it."""

    name: str
    c_type: str
    direction: str
    pointer: bool
    element: str


@dataclass(frozen=True)
class Case:
    """What a generator is given for one descriptor."""

    name: str
    entry: str
    params: Tuple[Param, ...]
    values: Mapping[str, Any]
    rng: np.random.Generator
    seed: int
    target_cpu: str

    def param(self, name: str) -> Param:
        for param in self.params:
            if param.name == name:
                return param
        raise PackError(f"{self.name}: {self.entry} has no parameter {name!r}")


@dataclass(frozen=True)
class CaseSpec:
    """One case: every input pointer's array, the output pointer and its element count, every scalar,
    and the golden (None: recorded from the host reference)."""

    arrays: Mapping[str, np.ndarray]
    output: str
    output_count: int
    scalars: Mapping[str, Any] = field(default_factory=dict)
    golden: Optional[np.ndarray] = None


Build = Callable[[Case], CaseSpec]
DescriptorSource = Callable[[Any], Mapping[str, Sequence[Mapping[str, Any]]]]


@dataclass(frozen=True)
class Generator:
    key: str
    build: Build
    entries: Tuple[str, ...]
    module: str

    def claims(self, entry: str) -> bool:
        return any(fnmatch.fnmatchcase(entry, pattern) for pattern in self.entries)


class Registry:
    """What a pack module's register() is given; frozen once every module has registered."""

    def __init__(self) -> None:
        self._generators: Dict[str, Generator] = {}
        self._sources: List[Tuple[str, DescriptorSource]] = []
        self._module = "<none>"
        self._frozen = False

    def _check_open(self, what: str) -> None:
        if self._frozen:
            raise PackError(f"{what}: the pack registry is frozen once its modules are loaded")

    def generator(self, key: str, build: Build, *, entries: Iterable[str]) -> None:
        self._check_open(f"generator {key!r}")
        if not isinstance(key, str) or not _KEY_RE.fullmatch(key):
            raise PackError(f"{self._module}: generator key {key!r} is not a dotted identifier")
        if not callable(build):
            raise PackError(f"{self._module}: generator {key!r}: build is not callable")
        if isinstance(entries, str):
            raise PackError(f"{self._module}: generator {key!r}: entries must be a list, not a string")
        patterns = tuple(entries)
        if not patterns:
            raise PackError(f"{self._module}: generator {key!r} claims no entries")
        for pattern in patterns:
            if not isinstance(pattern, str) or not _ENTRY_PATTERN_RE.fullmatch(pattern):
                raise PackError(f"{self._module}: generator {key!r}: entry pattern {pattern!r} is not a C "
                                "identifier with optional *")
        if key in self._generators:
            raise PackError(f"{self._module}: generator {key!r} is already registered by "
                            f"{self._generators[key].module}")
        self._generators[key] = Generator(key, build, patterns, self._module)

    def descriptor_source(self, source: DescriptorSource) -> None:
        self._check_open("descriptor_source")
        if not callable(source):
            raise PackError(f"{self._module}: descriptor_source is not callable")
        self._sources.append((self._module, source))

    @property
    def generators(self) -> Mapping[str, Generator]:
        return dict(self._generators)

    @property
    def sources(self) -> Tuple[Tuple[str, DescriptorSource], ...]:
        return tuple(self._sources)

    def generator_for(self, entry: str, key: Optional[str], where: str) -> Generator:
        if key is not None:
            generator = self._generators.get(key)
            if generator is None:
                raise PackError(f"{where}: no generator {key!r} (the pack has: "
                                f"{', '.join(sorted(self._generators)) or 'none'})")
            if not generator.claims(entry):
                raise PackError(f"{where}: generator {key!r} does not claim {entry}")
            return generator
        claiming = [g for g in self._generators.values() if g.claims(entry)]
        if not claiming:
            raise PackError(f"{where}: no generator claims {entry}")
        if len(claiming) > 1:
            raise PackError(f"{where}: {entry} is claimed by {', '.join(sorted(g.key for g in claiming))}; "
                            "name one in the descriptor's generator field")
        return claiming[0]


__all__ = [
    "API_VERSION", "DESCRIPTOR_SCHEMA", "Case", "CaseSpec", "Generator", "PackError", "Param", "Registry",
    "api_version_problem",
]
