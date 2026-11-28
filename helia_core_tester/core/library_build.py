"""The profile's ``build`` block: how the FVP tests build a library other than ns-cmsis-nn.

hct adds the library's CMake project (``project``, relative to ``source_root``) to its own and links its
tests against ``target``. ``definitions`` are cache values for every CPU; ``cpu_definitions`` adds more
for one CPU. A value may name ``{cmsis_path}``, the CMSIS_5 checkout the build uses. A name hct sets
itself is refused, as is a name set both for every CPU and for one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from helia_core_tester.core.errors import ConfigurationError

PLACEHOLDERS = ("cmsis_path",)
RESERVED_PREFIXES = ("CMAKE_", "HCT_", "HELIA_", "ENABLE_COVERAGE", "ARM_NN_")
RESERVED_NAMES = frozenset({
    "CMSIS_PATH", "CMSIS_NN_ROOT", "TARGET_CPU", "GENERATED_TESTS_DIR", "FVP_CORSTONE_300_PATH",
    "CMSIS_OPTIMIZATION_LEVEL",
})
_PLACEHOLDER = re.compile(r"\{([^{}]*)\}")
_UNSAFE = re.compile(r'[;"\\\n\r\x00$]')

Definitions = tuple[tuple[str, str], ...]


class BuildSpecError(ConfigurationError):
    """The profile's build block cannot be used."""


@dataclass(frozen=True)
class BuildSpec:
    """The profile's ``build`` block, resolved."""

    project: Path
    target: str
    definitions: Definitions = ()
    cpu_definitions: tuple[tuple[str, Definitions], ...] = ()

    def definitions_for(self, cpu: str, cmsis_path: Path) -> list[str]:
        """``NAME=VALUE`` for every definition `cpu` takes, placeholders expanded."""
        from helia_core_tester.core.cpu_targets import normalize_cpu

        canon = normalize_cpu(cpu)
        per_cpu = dict(self.cpu_definitions).get(canon, ())
        values = {"cmsis_path": str(Path(cmsis_path))}
        if _UNSAFE.search(values["cmsis_path"]):
            raise BuildSpecError(f"CMSIS path {cmsis_path} holds a character a CMake definition cannot carry")
        return [f"{name}={_PLACEHOLDER.sub(lambda m: values[m.group(1)], value)}"
                for name, value in self.definitions + per_cpu]


def _definitions(data: Optional[Mapping[str, Any]], where: str) -> Definitions:
    out = []
    for name, value in (data or {}).items():
        if name in RESERVED_NAMES or name.startswith(RESERVED_PREFIXES):
            raise BuildSpecError(f"{where}: {name} is set by hct itself")
        if not isinstance(value, str):
            raise BuildSpecError(f"{where}: {name} = {value!r} must be a string")
        if _UNSAFE.search(value):
            raise BuildSpecError(f"{where}: {name} = {value!r} holds a character a CMake definition cannot carry")
        unknown = sorted({m.group(1) for m in _PLACEHOLDER.finditer(value)} - set(PLACEHOLDERS))
        if unknown:
            raise BuildSpecError(f"{where}: {name} names {', '.join('{' + u + '}' for u in unknown)}; "
                                 f"the only placeholder is {{{PLACEHOLDERS[0]}}}")
        if "{" in _PLACEHOLDER.sub("", value) or "}" in _PLACEHOLDER.sub("", value):
            raise BuildSpecError(f"{where}: {name} = {value!r} has an unmatched brace")
        out.append((name, value))
    return tuple(out)


def build_for(data: Optional[Mapping[str, Any]], where: str, project: Optional[Path]) -> Optional[BuildSpec]:
    """The checked block, given its project already resolved inside source_root; None without one."""
    if data is None:
        return None
    from helia_core_tester.core.cpu_targets import normalize_cpu

    if project is None or not (project / "CMakeLists.txt").is_file():
        raise BuildSpecError(f"{where}.project {data.get('project', '.')!r} has no CMakeLists.txt ({project})")
    base = _definitions(data.get("definitions"), f"{where}.definitions")
    base_names = {name for name, _ in base}
    per_cpu: dict[str, Definitions] = {}
    for cpu, values in (data.get("cpu_definitions") or {}).items():
        try:
            canon = normalize_cpu(cpu)
        except ValueError:
            raise BuildSpecError(f"{where}.cpu_definitions: {cpu!r} is not a CPU hct builds for") from None
        if canon in per_cpu:
            raise BuildSpecError(f"{where}.cpu_definitions: {cpu!r} names {canon} again")
        defs = _definitions(values, f"{where}.cpu_definitions.{cpu}")
        both = sorted(base_names & {name for name, _ in defs})
        if both:
            raise BuildSpecError(f"{where}.cpu_definitions.{cpu}: {', '.join(both)} is also in definitions; "
                                 "set it in one place")
        per_cpu[canon] = defs
    return BuildSpec(project=project, target=data["target"], definitions=base,
                     cpu_definitions=tuple(sorted(per_cpu.items())))
