"""Render a kernel call and its compile-time parity check from the contract export.

`render_call` writes the argument list in prototype order from a name -> C expression
mapping, so a template names what it passes and the contract decides the order; a
missing or extra name is an error before anything is rendered. `render_parity_assert`
emits a `_Static_assert(__builtin_types_compatible_p(__typeof__(f), <type>))` for the
same prototype, so a header that drifts from the export (or an export rendered from the
wrong checkout) is a compile error on the FVP and on hardware, not a silently wrong call.

Both are exposed to templates as the Jinja globals `contract_call(symbol, args)` and
`contract_parity_assert(symbol)` (see `contract_globals`). A checkout without the export
makes those globals raise: a template that has moved to the contract cannot fall back to
a hand-written call without reintroducing the drift the contract removes.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Mapping, Optional

from helia_core_tester.contract.ir import ContractError, ContractSet, FunctionDecl, ParamDecl, load_contract_set
from helia_core_tester.generation.utils.temp_sizer_probe import resolve_cmsis_nn_root

_FUNCTION_POINTER_NAME_RE = re.compile(r"\(\s*\*\s*(\w+)\s*\)")


class ContractRenderError(ContractError):
    """A template asked for a call the contract cannot render."""


def adjusted_type(param: ParamDecl) -> str:
    """The parameter type as it appears in the function's type: arrays decay to
    pointers and a function-pointer declarator loses its name."""
    if param.extent:
        return f"{param.c_type} *"
    if param.is_function_pointer:
        return _FUNCTION_POINTER_NAME_RE.sub("(*)", param.c_type, count=1)
    return param.c_type


def render_call(decl: FunctionDecl, args: Mapping[str, str], *, indent: str = "        ") -> str:
    """`symbol(\\n<indent>expr, // name ...)` with the arguments in prototype order."""
    expected = [param.name for param in decl.params]
    given = list(args)
    missing = [name for name in expected if name not in args]
    extra = [name for name in given if name not in expected]
    if missing or extra:
        detail = []
        if missing:
            detail.append(f"missing {missing}")
        if extra:
            detail.append(f"unknown {extra}")
        raise ContractRenderError(
            f"{decl.name}: call arguments do not match the contract ({'; '.join(detail)}); "
            f"the prototype takes {expected}"
        )
    for name, expression in args.items():
        if not isinstance(expression, str) or not expression.strip():
            raise ContractRenderError(f"{decl.name}.{name}: argument expression is empty")
    if not expected:
        return f"{decl.name}()"
    lines = []
    for index, param in enumerate(decl.params):
        separator = "," if index + 1 < len(expected) else ""
        lines.append(f"{indent}{args[param.name].strip()}{separator} /* {param.name} */")
    return f"{decl.name}(\n" + "\n".join(lines) + "\n" + indent[:-4] + ")"


def function_type(decl: FunctionDecl) -> str:
    params = ", ".join(adjusted_type(param) for param in decl.params) or "void"
    return f"{decl.returns} ({params})"


def render_parity_assert(decl: FunctionDecl) -> str:
    """A compile-time check that the kernel the harness links has this prototype."""
    return (
        f"_Static_assert(__builtin_types_compatible_p(__typeof__({decl.name}), {function_type(decl)}),\n"
        f"               \"{decl.name}: prototype differs from the kernel contract export; "
        "rerun `python3 scripts/check_kernel_contract.py export` in ns-cmsis-nn and regenerate\");"
    )


def contract_globals(loader: Optional[Callable[[], ContractSet]] = None) -> dict[str, Callable]:
    """Jinja globals bound to a lazily loaded contract set (default: the resolved
    ns-cmsis-nn checkout). Loading happens on first use so templates that never call
    them cost nothing and old checkouts keep rendering."""
    cache: dict[str, ContractSet] = {}

    def contracts() -> ContractSet:
        if "set" not in cache:
            cache["set"] = loader() if loader is not None else load_contract_set(resolve_cmsis_nn_root())
        return cache["set"]

    def require(symbol: str) -> FunctionDecl:
        current = contracts()
        if not current.present:
            raise ContractRenderError(
                f"{symbol}: this template renders its call from the kernel contract, but the "
                f"ns-cmsis-nn checkout ({current.root or 'unresolved'}) has no "
                "Tests/KernelContracts/kernel_contracts.json; it needs an ns-cmsis-nn that carries "
                "the export (AmbiqAI/ns-cmsis-nn#549 or later)"
            )
        return current.require(symbol)

    def contract_call(symbol: str, args: Mapping[str, str], indent: str = "        ") -> str:
        return render_call(require(symbol), args, indent=indent)

    def contract_parity_assert(symbol: str) -> str:
        return render_parity_assert(require(symbol))

    return {"contract_call": contract_call, "contract_parity_assert": contract_parity_assert}
