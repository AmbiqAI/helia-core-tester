"""What a generated harness needs from the library it tests: the profile's ``harness`` block.

Fields (``hct/library-profile/1``, all names C identifiers):

- ``includes``: headers every generated test and the test runtime include, through ``hct_library.h``.
- ``header_includes``: headers each case's harness header includes.
- ``status``: the kernels' status ``type``; which of its ``codes`` (name: value) mean ``success`` and
  ``argument_error`` and, optionally, ``not_implemented`` (an entry that declines on the build).
- ``context``: the scratch context type, when the library's kernels take one.
- ``dims``: the tensor shape type, when the library has one.
- ``alias_groups``: group name: spellings of one parameter, so a harness value binds to any of them.

An ns-cmsis-nn profile without the block takes ns-cmsis-nn's (``contract/legacy_nn.py``); any other
library must state it. ``hct_library.h`` carries the includes and the two status macros the runtime
uses, and asserts every code against the library's headers, so a profile that drifts from them fails
to compile rather than misreading a status.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional

from helia_core_tester.core.errors import ConfigurationError

HEADER_NAME = "hct_library.h"
HEADER_ENV = "HCT_LIBRARY_HEADER_DIR"
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_INCLUDE_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_+.-]*(/[A-Za-z0-9_][A-Za-z0-9_+.-]*)*\.h")
INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1


class HarnessProfileError(ConfigurationError):
    """The profile's harness block cannot be used."""


@dataclass(frozen=True)
class StatusSet:
    type: str
    success: str
    argument_error: str
    not_implemented: Optional[str]
    codes: tuple[tuple[str, int], ...]

    def names(self) -> dict[int, str]:
        return {value: name for name, value in self.codes}

    def value(self, name: str) -> int:
        return dict(self.codes)[name]


@dataclass(frozen=True)
class HarnessProfile:
    library: str
    includes: tuple[str, ...]
    header_includes: tuple[str, ...]
    status: StatusSet
    context: Optional[str]
    dims: Optional[str]
    alias_groups: tuple[tuple[str, tuple[str, ...]], ...]

    def group_of(self) -> dict[str, tuple[str, ...]]:
        return {name: names for _, names in self.alias_groups for name in names}

    def group(self, key: str) -> tuple[str, ...]:
        return dict(self.alias_groups).get(key, ())

    def require(self, field: str) -> str:
        """`context`, `dims` or `not_implemented`, for a harness that uses it; refused when absent."""
        value = self.status.not_implemented if field == "not_implemented" else getattr(self, field)
        if value is None:
            raise HarnessProfileError(
                f"{self.library}: this harness needs the profile's harness "
                f"{'status.' if field == 'not_implemented' else ''}{field}, which it does not state")
        return value


def _ident(where: str, value: Any) -> str:
    if not isinstance(value, str) or not _IDENT_RE.fullmatch(value):
        raise HarnessProfileError(f"{where}: {value!r} is not a C identifier")
    return value


def _includes(where: str, value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise HarnessProfileError(f"{where}: needs a non-empty list of headers")
    for header in value:
        if not isinstance(header, str) or not _INCLUDE_RE.fullmatch(header) or ".." in header:
            raise HarnessProfileError(f"{where}: {header!r} is not a relative header name (name.h)")
    if len(set(value)) != len(value):
        raise HarnessProfileError(f"{where}: lists a header twice")
    return tuple(value)


def _status(where: str, data: Any) -> StatusSet:
    if not isinstance(data, Mapping):
        raise HarnessProfileError(f"{where}: needs type, success, argument_error and codes")
    codes = data.get("codes")
    if not isinstance(codes, Mapping) or not codes:
        raise HarnessProfileError(f"{where}.codes: needs at least one name: value")
    pairs = []
    for name, value in codes.items():
        _ident(f"{where}.codes", name)
        if isinstance(value, bool) or not isinstance(value, int) or not INT32_MIN <= value <= INT32_MAX:
            raise HarnessProfileError(f"{where}.codes.{name}: {value!r} is not a 32-bit integer")
        pairs.append((name, value))
    values = [value for _, value in pairs]
    if len(set(values)) != len(values):
        dup = sorted({value for value in values if values.count(value) > 1})
        raise HarnessProfileError(f"{where}.codes: value(s) {dup} name more than one code")
    roles = {}
    for role in ("success", "argument_error", "not_implemented"):
        if role not in data:
            if role == "not_implemented":
                roles[role] = None
                continue
            raise HarnessProfileError(f"{where}: missing {role}")
        name = _ident(f"{where}.{role}", data[role])
        if name not in codes:
            raise HarnessProfileError(f"{where}.{role}: {name} is not one of its codes")
        roles[role] = name
    named = [name for name in roles.values() if name is not None]
    if len(set(named)) != len(named):
        raise HarnessProfileError(f"{where}: success, argument_error and not_implemented name the same code")
    return StatusSet(
        type=_ident(f"{where}.type", data.get("type")),
        success=roles["success"],
        argument_error=roles["argument_error"],
        not_implemented=roles["not_implemented"],
        codes=tuple(pairs),
    )


def _alias_groups(where: str, data: Any) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if data is None:
        return ()
    if not isinstance(data, Mapping):
        raise HarnessProfileError(f"{where}: needs group: [spellings]")
    seen: dict[str, str] = {}
    groups = []
    for key, names in data.items():
        _ident(where, key)
        if not isinstance(names, list) or len(names) < 2:
            raise HarnessProfileError(f"{where}.{key}: needs at least two spellings")
        for name in names:
            _ident(f"{where}.{key}", name)
            if name in seen:
                raise HarnessProfileError(f"{where}: {name} is in both {seen[name]} and {key}")
            seen[name] = key
        groups.append((key, tuple(names)))
    return tuple(groups)


def parse_harness(library: str, data: Any, where: str = "harness") -> HarnessProfile:
    """A harness block, checked past what the schema can say (codes, roles, group overlaps)."""
    if not isinstance(data, Mapping):
        raise HarnessProfileError(f"{where}: needs includes, header_includes and status")
    unknown = sorted(set(data) - {"includes", "header_includes", "status", "context", "dims", "alias_groups"})
    if unknown:
        raise HarnessProfileError(f"{where}: unknown field(s) {unknown}")
    for key in ("includes", "header_includes", "status"):
        if key not in data:
            raise HarnessProfileError(f"{where}: missing {key}")
    return HarnessProfile(
        library=library,
        includes=_includes(f"{where}.includes", data["includes"]),
        header_includes=_includes(f"{where}.header_includes", data["header_includes"]),
        status=_status(f"{where}.status", data["status"]),
        context=_ident(f"{where}.context", data["context"]) if data.get("context") is not None else None,
        dims=_ident(f"{where}.dims", data["dims"]) if data.get("dims") is not None else None,
        alias_groups=_alias_groups(f"{where}.alias_groups", data.get("alias_groups")),
    )


def legacy_harness() -> HarnessProfile:
    from helia_core_tester.contract import legacy_nn

    return parse_harness(legacy_nn.LIBRARY, legacy_nn.HARNESS, "ns-cmsis-nn built-in harness")


def harness_for(library: str, data: Any, where: str) -> Optional[HarnessProfile]:
    """The profile's block; ns-cmsis-nn's when an ns-cmsis-nn profile has none; else None."""
    if data is not None:
        return parse_harness(library, data, where)
    from helia_core_tester.contract import legacy_nn

    return legacy_harness() if library == legacy_nn.LIBRARY else None


def harness_of_root(root: Optional[Path]) -> HarnessProfile:
    """The harness block of the library at `root` (its hct-library.yaml, else ns-cmsis-nn's)."""
    return _harness_of_root(str(Path(root).expanduser().resolve()) if root is not None else None)


@lru_cache(maxsize=8)
def _harness_of_root(root: Optional[str]) -> HarnessProfile:
    from helia_core_tester.core.library import PROFILE_FILE, resolve_library

    # A root without a profile is read as ns-cmsis-nn, as generation reads its contracts.
    if root is None or not (Path(root) / PROFILE_FILE).exists() and not (Path(root) / PROFILE_FILE).is_symlink():
        return legacy_harness()
    profile = resolve_library(Path(root))
    if profile.harness is None:
        raise HarnessProfileError(
            f"{profile.describe()}: its profile has no harness block; hct needs one to generate tests for "
            f"{profile.library}")
    return profile.harness


def current_harness() -> HarnessProfile:
    """The harness block of the library generation runs against: CMSIS_NN_ROOT as given (any layout),
    else the nested checkout generation resolves, else ns-cmsis-nn's. A step that knows its checkout
    passes it to harness_of_root instead."""
    env_root = os.environ.get("CMSIS_NN_ROOT", "").strip()
    if env_root:
        if not Path(env_root).is_dir():
            raise HarnessProfileError(f"CMSIS_NN_ROOT={env_root} is not a directory")
        return harness_of_root(Path(env_root))
    from helia_core_tester.generation.utils.temp_sizer_probe import resolve_cmsis_nn_root

    return harness_of_root(resolve_cmsis_nn_root())


def render_header(harness: HarnessProfile) -> str:
    status = harness.status
    lines = [
        f"/* Generated by helia-core-tester from the {harness.library} library profile; do not edit. */",
        "#ifndef HCT_LIBRARY_H",
        "#define HCT_LIBRARY_H",
        "",
        *[f'#include "{header}"' for header in harness.includes],
        "",
        f"#define HCT_STATUS_SUCCESS {status.success}",
        f"#define HCT_STATUS_ARGUMENT_ERROR {status.argument_error}",
        "",
        "#ifndef __cplusplus",
    ]
    for name, value in status.codes:
        lines.append(f'_Static_assert((int){name} == {value}, "{name} is not {value}, as the profile says");')
    lines += ["#endif", "", "#endif", ""]
    return "\n".join(lines)


def _write_if_changed(path: Path, text: str) -> None:
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_header(directory: Path, harness: HarnessProfile) -> Path:
    """Write `directory`/hct_library.h (unchanged when already current); returns its path."""
    path = Path(directory) / HEADER_NAME
    _write_if_changed(path, render_header(harness))
    return path


def header_digest(harness: HarnessProfile) -> str:
    return hashlib.sha256(render_header(harness).encode("utf-8")).hexdigest()


def header_dir(harness: HarnessProfile, cache_root: Optional[Path] = None) -> Path:
    """A directory holding this harness's hct_library.h, keyed by its content, for host compiles."""
    if cache_root is None:
        override = os.environ.get(HEADER_ENV, "").strip()
        if override:
            cache_root = Path(override)
        else:
            from helia_core_tester.core.discovery import find_repo_root

            cache_root = find_repo_root() / "artifacts" / "hct_library"
    directory = Path(cache_root) / header_digest(harness)[:16]
    write_header(directory, harness)
    return directory
