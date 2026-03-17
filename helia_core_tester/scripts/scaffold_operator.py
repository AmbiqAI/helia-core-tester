"""Create grouped operator skeleton files for helia-core-tester developers."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from helia_core_tester.core.discovery import find_repo_root


def operator_to_module_basename(operator: str) -> str:
    """Convert a PascalCase operator name to the canonical snake_case module basename."""
    normalized = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", operator)
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", normalized)
    return normalized.replace("-", "_").lower()


def operator_to_class_name(operator: str) -> str:
    return f"Op{operator}"


def create_operator_skeleton(
    repo_root: Path,
    *,
    operator: str,
    family: str,
    parity_kind: str,
    module_basename: str | None = None,
    descriptor_stem: str | None = None,
    class_name: str | None = None,
    create_descriptor: bool = True,
    create_templates: bool = True,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Create grouped skeleton files for a new operator."""
    repo_root = Path(repo_root)
    module_basename = module_basename or operator_to_module_basename(operator)
    descriptor_stem = descriptor_stem or module_basename
    class_name = class_name or operator_to_class_name(operator)

    ops_family_dir = repo_root / "helia_core_tester" / "generation" / "ops" / family
    descriptors_family_dir = repo_root / "assets" / "descriptors" / family
    templates_family_dir = repo_root / "assets" / "templates" / family / module_basename

    module_path = ops_family_dir / f"{module_basename}.py"
    descriptor_path = descriptors_family_dir / f"{descriptor_stem}.yaml"
    header_template_path = templates_family_dir / f"{module_basename}.h.j2"
    source_template_path = templates_family_dir / f"{module_basename}.c.j2"

    created: dict[str, Path] = {}

    for directory in (ops_family_dir, descriptors_family_dir, templates_family_dir):
        directory.mkdir(parents=True, exist_ok=True)
    package_init = ops_family_dir / "__init__.py"
    if not package_init.exists():
        package_init.write_text(f'"""{family} operator package."""\n')

    if not overwrite and module_path.exists():
        raise FileExistsError(f"Module already exists: {module_path}")

    module_path.write_text(
        f'"""{operator} operation implementation."""\n\n'
        "from helia_core_tester.generation.ops._shared.base import OperationBase\n\n\n"
        f"class {class_name}(OperationBase):\n"
        f'    """{operator} operation."""\n\n'
        "    def build_keras_model(self):\n"
        f'        raise NotImplementedError("{operator} build_keras_model is not implemented.")\n\n'
        "    def generate_c_files(self, output_dir) -> None:\n"
        f'        raise NotImplementedError("{operator} generate_c_files is not implemented.")\n'
    )
    created["module"] = module_path
    created["package_init"] = package_init

    if create_descriptor:
        if overwrite or not descriptor_path.exists():
            descriptor_path.write_text(
                f"name: {descriptor_stem}_case_01_s8\n"
                f"operator: {operator}\n"
                "activation_dtype: S8\n"
                "weight_dtype: S8\n"
                "input_shape: [1]\n"
            )
        created["descriptor"] = descriptor_path

    if create_templates:
        if overwrite or not header_template_path.exists():
            header_template_path.write_text(
                "#pragma once\n\n"
                "/* TODO: implement generated API for {{ name }}. */\n"
            )
        if overwrite or not source_template_path.exists():
            source_template_path.write_text(
                '#include "{{ name }}_'
                f'{module_basename}'
                '.h"\n\n'
                "/* TODO: implement generated source for {{ name }}. */\n"
            )
        created["template_dir"] = templates_family_dir
        created["header_template"] = header_template_path
        created["source_template"] = source_template_path

    return created


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create grouped skeleton files for a tester operator.")
    parser.add_argument("operator")
    parser.add_argument("family")
    parser.add_argument("parity_kind", choices=("cmsis", "extension"))
    parser.add_argument("--module-basename")
    parser.add_argument("--descriptor-stem")
    parser.add_argument("--class-name")
    parser.add_argument("--no-descriptor", action="store_true")
    parser.add_argument("--no-templates", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    created = create_operator_skeleton(
        find_repo_root(),
        operator=args.operator,
        family=args.family,
        parity_kind=args.parity_kind,
        module_basename=args.module_basename,
        descriptor_stem=args.descriptor_stem,
        class_name=args.class_name,
        create_descriptor=not args.no_descriptor,
        create_templates=not args.no_templates,
        overwrite=args.overwrite,
    )

    for label, path in created.items():
        print(f"{label}: {path}")
    print(
        "catalog_hint: "
        f'{args.operator}: _spec("{args.operator}", "{args.family}", '
        f'"{args.module_basename or operator_to_module_basename(args.operator)}", '
        f'"{args.class_name or operator_to_class_name(args.operator)}", '
        f'"{args.family}/{args.descriptor_stem or (args.module_basename or operator_to_module_basename(args.operator))}.yaml", '
        f'"{args.family}/{args.module_basename or operator_to_module_basename(args.operator)}", '
        f'parity_kind="{args.parity_kind}")'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
