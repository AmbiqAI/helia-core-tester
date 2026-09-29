"""Bind a kernel call's arguments by parameter name from what a harness can supply.

`contract_call` needs the template to name exactly the prototype's parameters, which ties a
template to one prototype. `bind` instead takes a pool: every value the operator's harness can
supply, keyed by parameter name. It picks exactly the parameters the prototype lists, in
prototype order, so one pool serves every public entry of an operator. ns-cmsis-nn spells a
few concepts two ways (`input` / `input_data`, `kernel` / `filter_data`, ...); ALIAS_GROUPS
lets a pool value answer to either spelling. No exported function takes two names from one
group (test_alias_groups_are_unambiguous_in_the_real_tree), so an alias never guesses.

`check_types` compares the element types of a kernel's tensor pointers with the descriptor's
dtypes, so an entry of the wrong precision fails at generation, naming the parameter, rather
than compiling into a call that reinterprets the data.
"""

from __future__ import annotations

import re
from typing import Mapping, Optional

from helia_core_tester.contract.ir import FunctionDecl
from helia_core_tester.contract.render import ContractRenderError

ALIAS_GROUPS: tuple[tuple[str, ...], ...] = (
    ("input_data", "input"),
    ("output_data", "output"),
    ("filter_data", "kernel"),
    ("bias_data", "bias"),
    ("input_1_data", "input_1_vect", "input1_data"),
    ("input_2_data", "input_2_vect", "input2_data"),
    ("out_offset", "output_offset"),
)
_GROUP_OF = {name: group for group in ALIAS_GROUPS for name in group}

# Tensor roles whose element type check_types compares, by the alias group that names them.
ROLE_GROUPS: dict[str, tuple[str, ...]] = {
    "input": _GROUP_OF["input_data"],
    "output": _GROUP_OF["output_data"],
    "filter": _GROUP_OF["filter_data"],
    "bias": _GROUP_OF["bias_data"],
}

DTYPE_C_TYPES: dict[str, frozenset[str]] = {
    "S4": frozenset({"int8_t"}),  # packed two per byte
    "S8": frozenset({"int8_t"}),
    "S16": frozenset({"int16_t"}),
    "S32": frozenset({"int32_t"}),
    "S64": frozenset({"int64_t"}),
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


def bind(decl: FunctionDecl, pool: Mapping[str, str]) -> dict[str, str]:
    """{parameter: expression} for every parameter of `decl`, in prototype order."""
    bound: dict[str, str] = {}
    missing: list[str] = []
    for param in decl.params:
        if param.name in pool:
            source = param.name
        else:
            offered = [name for name in _GROUP_OF.get(param.name, ()) if name in pool]
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


def takes(decl: FunctionDecl, name: str) -> bool:
    """Whether `decl` takes the parameter `name` under any spelling of its alias group."""
    names = {param.name for param in decl.params}
    return bool(names & set(_GROUP_OF.get(name, (name,))))


def check_types(decl: FunctionDecl, roles: Mapping[str, str]) -> None:
    """Fail when a tensor pointer's element type does not match the descriptor dtype for its
    role (input, output, filter, bias). Struct-typed and untyped parameters are not compared."""
    problems: list[str] = []
    for role, dtype in roles.items():
        if role not in ROLE_GROUPS:
            raise ContractBindError(f"{decl.name}: unknown tensor role {role!r}; known {sorted(ROLE_GROUPS)}")
        if str(dtype).upper() not in DTYPE_C_TYPES:
            raise ContractBindError(f"{decl.name}: unknown dtype {dtype!r} for {role}; known {sorted(DTYPE_C_TYPES)}")
    for param in decl.params:
        role = next((r for r, group in ROLE_GROUPS.items() if param.name in group), None)
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
