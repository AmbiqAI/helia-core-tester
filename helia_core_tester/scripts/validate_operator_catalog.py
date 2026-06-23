"""Validate grouped operator catalog contracts."""

from __future__ import annotations

import sys
from pathlib import Path

from helia_core_tester.core.discovery import find_repo_root
from helia_core_tester.generation.ops.catalog import iter_operator_specs, validate_catalog_paths


def validate_operator_contracts(repo_root: Path) -> list[str]:
    """Validate grouped catalog contracts for modules, descriptors, and templates."""
    errors = list(validate_catalog_paths(Path(repo_root)))
    for spec in iter_operator_specs():
        for descriptor_relpath in spec.descriptor_relpaths:
            descriptor_rel = Path(descriptor_relpath)
            if not descriptor_rel.parts or descriptor_rel.parts[0] != spec.family:
                errors.append(
                    f"Descriptor path for {spec.operator} must live under {spec.family}: {descriptor_relpath}"
                )
        if spec.template_relpath is not None:
            template_rel = Path(spec.template_relpath)
            if not template_rel.parts or template_rel.parts[0] != spec.family:
                errors.append(
                    f"Template path for {spec.operator} must live under {spec.family}: {spec.template_relpath}"
                )
        if spec.parity_kind == "extension" and spec.family != "TesterExtensions":
            errors.append(
                f"Extension operator {spec.operator} must live under TesterExtensions, found {spec.family}"
            )
        if spec.parity_kind == "cmsis" and spec.family == "TesterExtensions":
            errors.append(
                f"CMSIS operator {spec.operator} must not live under TesterExtensions"
            )
        if spec.artifact_family_dir != spec.family:
            errors.append(
                f"Artifact family mismatch for {spec.operator}: {spec.artifact_family_dir} != {spec.family}"
            )
    return sorted(set(errors))


def main(argv: list[str] | None = None) -> int:
    del argv
    repo_root = find_repo_root()
    errors = validate_operator_contracts(repo_root)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("Operator catalog validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
