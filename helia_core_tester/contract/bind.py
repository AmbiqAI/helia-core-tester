"""Bind a kernel call's arguments by parameter name from what a harness can supply.

`contract_call` needs the template to name exactly the prototype's parameters, which ties a
template to one prototype. `bind` instead takes a pool: every value the operator's harness can
supply, keyed by parameter name. It picks exactly the parameters the prototype lists, in
prototype order, so one pool serves every public entry of an operator. A library may spell a
concept two ways (ns-cmsis-nn's `input` / `input_data`, `kernel` / `filter_data`, ...); the
profile's harness `alias_groups` let a pool value answer to any spelling of its group. No
exported function takes two names from one group (test_alias_groups_are_unambiguous_in_the_real_tree),
so an alias never guesses.

`check_types` compares the element types of a kernel's tensor pointers with the descriptor's
dtypes, so an entry of the wrong precision fails at generation, naming the parameter, rather
than compiling into a call that reinterprets the data.
"""

from __future__ import annotations

import re
from typing import Mapping, Optional

from helia_core_tester.contract.ir import FunctionDecl
from helia_core_tester.contract.render import ContractRenderError
from helia_core_tester.core.harness import HarnessProfile, current_harness

# The tensor roles check_types compares, each named by the alias group of the same name.
TENSOR_ROLES = ("input", "output", "filter", "bias", "input_1", "input_2")


def role_groups(harness: HarnessProfile) -> dict[str, tuple[str, ...]]:
    """{role: spellings} for the tensor roles the harness's alias groups name."""
    return {role: harness.group(role) for role in TENSOR_ROLES if harness.group(role)}


DTYPE_C_TYPES: dict[str, frozenset[str]] = {
    "S4": frozenset({"int8_t"}),  # packed two per byte
    "S8": frozenset({"int8_t"}),
    "S16": frozenset({"int16_t"}),
    "S32": frozenset({"int32_t"}),
    "S64": frozenset({"int64_t"}),
    "U16": frozenset({"uint16_t"}),  # binary16 bit patterns
    "FP32": frozenset({"float32_t", "float"}),
    "FP16": frozenset({"float16_t"}),
    "BOOL": frozenset({"bool", "uint8_t"}),
}
_ELEMENT_TYPES = frozenset().union(*DTYPE_C_TYPES.values())
_POINTER_RE = re.compile(r"^(?:const\s+)?(?P<base>\w+)\s*\*\s*(?:const\s*)?$")


class ContractBindError(ContractRenderError):
    """A kernel's parameters cannot be bound from the values a harness supplies."""


def pointer_element(c_type: str) -> Optional[str]:
    """The element type of a single-level pointer type, else None."""
    match = _POINTER_RE.match(" ".join(c_type.split()))
    return match.group("base") if match else None


def bind(decl: FunctionDecl, pool: Mapping[str, str],
         harness: Optional[HarnessProfile] = None) -> dict[str, str]:
    """{parameter: expression} for every parameter of `decl`, in prototype order."""
    group_of = (harness or current_harness()).group_of()
    bound: dict[str, str] = {}
    missing: list[str] = []
    for param in decl.params:
        if param.name in pool:
            source = param.name
        else:
            offered = [name for name in group_of.get(param.name, ()) if name in pool]
            if len(offered) > 1:
                raise ContractBindError(
                    f"{decl.name}.{param.name}: the harness offers several spellings {offered}; "
                    "keep one per alias group")
            if not offered:
                missing.append(f"{param.name} ({param.c_type})")
                continue
            source = offered[0]
        expression = pool[source]
        if not isinstance(expression, str) or not expression.strip():
            raise ContractBindError(f"{decl.name}.{param.name}: the harness value {source!r} is empty")
        bound[param.name] = expression.strip()
    if missing:
        raise ContractBindError(
            f"{decl.name}: the harness cannot supply {missing}; it offers {sorted(pool)}")
    return bound


def takes(decl: FunctionDecl, name: str, harness: Optional[HarnessProfile] = None) -> bool:
    """Whether `decl` takes the parameter `name` under any spelling of its alias group."""
    names = {param.name for param in decl.params}
    return bool(names & set((harness or current_harness()).group_of().get(name, (name,))))


def param_type(decl: FunctionDecl, name: str, harness: Optional[HarnessProfile] = None) -> str:
    """The C type of `name` (any spelling of its alias group) in `decl`, or '' when absent."""
    group = set((harness or current_harness()).group_of().get(name, (name,)))
    return next((param.c_type for param in decl.params if param.name in group), "")


def check_types(decl: FunctionDecl, roles: Mapping[str, str], harness: Optional[HarnessProfile] = None) -> None:
    """Fail when a tensor pointer's element type does not match the descriptor dtype for its
    role (input, output, filter, bias). Struct-typed and untyped parameters are not compared."""
    groups = role_groups(harness or current_harness())
    problems: list[str] = []
    for role, dtype in roles.items():
        if role not in groups:
            raise ContractBindError(f"{decl.name}: unknown tensor role {role!r}; known {sorted(groups)}")
        if str(dtype).upper() not in DTYPE_C_TYPES:
            raise ContractBindError(f"{decl.name}: unknown dtype {dtype!r} for {role}; known {sorted(DTYPE_C_TYPES)}")
    for param in decl.params:
        role = next((r for r, group in groups.items() if param.name in group), None)
        if role is None or role not in roles:
            continue
        element = pointer_element(param.c_type)
        if element is None or element not in _ELEMENT_TYPES:
            continue
        allowed = DTYPE_C_TYPES[str(roles[role]).upper()]
        if element not in allowed:
            problems.append(f"{param.name} is {param.c_type!r} but the {role} dtype is {roles[role]}")
    if problems:
        raise ContractBindError(f"{decl.name}: " + "; ".join(problems))
