"""Where the ns-cmsis-nn checkout comes from.

The hardware firmware compiles ns-cmsis-nn from source (CMake's `CMSIS_NN_ROOT`)
and the generate step reads the same checkout (LSTM unit-test data, header probes,
the sigmoid table). Both must point at one checkout, resolved once, in this order:

1. `--cmsis-nn-root PATH` on the hardware command;
2. the `CMSIS_NN_ROOT` environment variable;
3. the nested layout, where this repo is the `Tests/helia-core-tester` submodule
   of an ns-cmsis-nn checkout (CMakeLists.txt's historical `../..` default).

Anything else is an error that names the flag and the variable. A resolved path
must look like an ns-cmsis-nn checkout (`Include/` and `Source/`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

ENV_VAR = "CMSIS_NN_ROOT"
CLI_FLAG = "--cmsis-nn-root"

SELECTOR_CLI_ROOT = f"cli.{CLI_FLAG}"
SELECTOR_ENV = f"env.{ENV_VAR}"
SELECTOR_NESTED = "layout.nested"


class CmsisNnSourceError(RuntimeError):
    """No usable ns-cmsis-nn checkout could be resolved."""


@dataclass(frozen=True)
class CmsisNnSelection:
    """What the user asked for. `ref` is reserved for a pinned-commit selection and
    is rejected until the dependency baseline lands."""

    root: Optional[Path] = None
    ref: Optional[str] = None


@dataclass(frozen=True)
class ResolvedCmsisNn:
    root: Path
    selector: str
    """One of SELECTOR_CLI_ROOT / SELECTOR_ENV / SELECTOR_NESTED."""
    requested: Optional[str]
    """The value as the user gave it (before expansion), or None for the nested layout."""


def nested_layout_root(repo_root: Path) -> Path:
    """The ns-cmsis-nn root when this repo sits at `<ns-cmsis-nn>/Tests/helia-core-tester`."""
    return repo_root.resolve().parent.parent


def looks_like_checkout(path: Path) -> bool:
    return (path / "Include").is_dir() and (path / "Source").is_dir()


def validate_checkout(path: Path, *, origin: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise CmsisNnSourceError(f"ns-cmsis-nn checkout from {origin} does not exist: {resolved}")
    for sub in ("Include", "Source"):
        if not (resolved / sub).is_dir():
            raise CmsisNnSourceError(
                f"ns-cmsis-nn checkout from {origin} is missing '{sub}/': {resolved} "
                f"-- expected an ns-cmsis-nn repository with Include/ and Source/."
            )
    return resolved


def resolve_cmsis_nn(
    repo_root: Path,
    selection: Optional[CmsisNnSelection] = None,
    env: Optional[Mapping[str, str]] = None,
) -> ResolvedCmsisNn:
    """Resolve the ns-cmsis-nn checkout: flag > env > nested layout, else raise."""
    selection = selection or CmsisNnSelection()
    env = os.environ if env is None else env
    if selection.ref is not None:
        raise CmsisNnSourceError("--cmsis-nn-ref is not supported yet; pass --cmsis-nn-root PATH or set CMSIS_NN_ROOT.")
    if selection.root is not None:
        root = validate_checkout(selection.root, origin=CLI_FLAG)
        return ResolvedCmsisNn(root, SELECTOR_CLI_ROOT, str(selection.root))
    env_value = env.get(ENV_VAR)
    if env_value:
        root = validate_checkout(Path(env_value), origin=f"${ENV_VAR}")
        return ResolvedCmsisNn(root, SELECTOR_ENV, env_value)
    nested = nested_layout_root(repo_root)
    if looks_like_checkout(nested):
        return ResolvedCmsisNn(nested, SELECTOR_NESTED, None)
    raise CmsisNnSourceError(
        f"No ns-cmsis-nn checkout found: pass {CLI_FLAG} PATH or set {ENV_VAR} to an ns-cmsis-nn "
        f"checkout (a directory with Include/ and Source/). The nested "
        f"<ns-cmsis-nn>/Tests/helia-core-tester layout was not detected at {nested}."
    )


def describe(resolved: ResolvedCmsisNn) -> str:
    return f"{resolved.root} ({resolved.selector})"
