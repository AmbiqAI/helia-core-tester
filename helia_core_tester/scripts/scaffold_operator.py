"""Create grouped operator skeleton files and register operator metadata."""

from __future__ import annotations

import argparse
import ast
import json
import pprint
import re
from pathlib import Path
from typing import Any

from helia_core_tester.core.discovery import find_repo_root


SUPPORTED_DESCRIPTOR_PROFILES = (
    "single_input_unary",
    "dual_input_elementwise",
    "arg_reduction",
    "pool",
    "shape_transform",
    "custom",
)

PROFILE_DEFAULT_DESCRIPTOR_FIELDS = {
    "arg_reduction": [("input_shape", "[1]")],
    "custom": [("input_shape", "[1]")],
    "dual_input_elementwise": [("input_1_shape", "[1]"), ("input_2_shape", "[1]")],
    "pool": [("input_shape", "[1]"), ("pool_size", "[1]")],
    "shape_transform": [("input_shape", "[1]")],
    "single_input_unary": [("input_shape", "[1]")],
}


def operator_to_module_basename(operator: str) -> str:
    """Convert a PascalCase operator name to the canonical snake_case module basename."""
    normalized = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", operator)
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", normalized)
    return normalized.replace("-", "_").lower()


def operator_to_class_name(operator: str) -> str:
    return f"Op{operator}"


def _operator_to_builtin_name(operator: str, module_basename: str) -> str:
    del operator
    return module_basename.upper()


def _wrapper_function_name(module_basename: str) -> str:
    return f"build_{module_basename}" if module_basename.endswith("_op") else f"build_{module_basename}_op"


def _format_python_assignment(name: str, value: Any) -> str:
    return f"{name} = {pprint.pformat(value, width=100, sort_dicts=True)}\n"


def _replace_python_assignment(text: str, name: str, value: Any) -> str:
    module = ast.parse(text)
    lines = text.splitlines()
    for node in module.body:
        if isinstance(node, ast.Assign):
            targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target.id]
        else:
            continue
        if name not in targets:
            continue
        replacement = _format_python_assignment(name, value).splitlines()
        lines[node.lineno - 1 : node.end_lineno] = replacement
        return "\n".join(lines) + "\n"
    raise ValueError(f"Could not find assignment for {name}")


def _load_python_assignment(text: str, name: str) -> Any:
    module = ast.parse(text)
    for node in module.body:
        value = None
        if isinstance(node, ast.Assign):
            targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if name in targets:
                value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            value = node.value
        if value is not None:
            return ast.literal_eval(value)
    raise ValueError(f"Could not load assignment for {name}")


def _extract_catalog_entries(assignment_text: str) -> tuple[str, dict[str, tuple[str, str]], list[str]]:
    header, _, remainder = assignment_text.partition("{\n")
    body, _, footer = remainder.rpartition("}")
    del footer

    entries_by_operator: dict[str, tuple[str, str]] = {}
    family_order: list[str] = []
    blocks = re.finditer(r'(?ms)^\s*"(?P<operator>[^"]+)":\s*_spec\((?P<body>.*?)^\s*\),\s*$', body)
    for match in blocks:
        operator = match.group("operator")
        block = f'    "{operator}": _spec({match.group("body")}    ),\n'
        family_match = re.search(r'_spec\(\s*"[^"]+",\s*"([^"]+)"', block, re.DOTALL)
        if family_match is None:
            raise ValueError(f"Could not parse family for catalog entry {operator}")
        family = family_match.group(1)
        entries_by_operator[operator] = (family, block)
        if family not in family_order:
            family_order.append(family)

    return f"{header}{{\n", entries_by_operator, family_order


def _render_catalog_entry(
    *,
    operator: str,
    family: str,
    parity_kind: str,
    module_basename: str,
    class_name: str,
    descriptor_relpath: str | None,
    template_relpath: str | None,
) -> str:
    descriptor_literal = "None" if descriptor_relpath is None else f'"{descriptor_relpath}"'
    template_literal = "None" if template_relpath is None else f'"{template_relpath}"'
    parity_arg = '' if parity_kind == "cmsis" else f', parity_kind="{parity_kind}"'
    return (
        f'    "{operator}": _spec("{operator}", "{family}", "{module_basename}", "{class_name}", '
        f'{descriptor_literal}, {template_literal}{parity_arg}),\n'
    )


def _update_catalog(
    catalog_path: Path,
    *,
    operator: str,
    family: str,
    parity_kind: str,
    module_basename: str,
    class_name: str,
    descriptor_relpath: str | None,
    template_relpath: str | None,
    overwrite: bool,
) -> None:
    text = catalog_path.read_text()
    module = ast.parse(text)
    lines = text.splitlines()
    assign_node = next(
        node
        for node in module.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "OPERATOR_SPECS"
    )

    assignment_text = "\n".join(lines[assign_node.lineno - 1 : assign_node.end_lineno]) + "\n"
    header, entries_by_operator, family_order = _extract_catalog_entries(assignment_text)

    if operator in entries_by_operator and not overwrite:
        raise FileExistsError(f"Operator already exists in catalog: {operator}")

    entries_by_operator[operator] = (
        family,
        _render_catalog_entry(
            operator=operator,
            family=family,
            parity_kind=parity_kind,
            module_basename=module_basename,
            class_name=class_name,
            descriptor_relpath=descriptor_relpath,
            template_relpath=template_relpath,
        ),
    )
    if family not in family_order:
        family_order.append(family)

    grouped: list[str] = []
    for family_name in family_order:
        family_entries = [
            (name, block)
            for name, (entry_family, block) in entries_by_operator.items()
            if entry_family == family_name
        ]
        for _, block in sorted(family_entries, key=lambda item: item[0]):
            grouped.append(block.rstrip("\n"))

    replacement = [f"{header.rstrip()}\n", *grouped, "}\n"]
    lines[assign_node.lineno - 1 : assign_node.end_lineno] = replacement
    catalog_path.write_text("\n".join(lines) + "\n")


def _update_descriptor_metadata(
    descriptors_path: Path,
    *,
    operator: str,
    descriptor_profile: str,
    require_fields: tuple[str, ...],
    activation_dtype_const: str | None,
) -> None:
    text = descriptors_path.read_text()
    profile_map = dict(_load_python_assignment(text, "OPERATOR_DESCRIPTOR_PROFILES"))
    extra_fields = dict(_load_python_assignment(text, "OPERATOR_EXTRA_REQUIRED_FIELDS"))
    field_constraints = dict(_load_python_assignment(text, "OPERATOR_FIELD_CONSTRAINTS"))

    profile_map[operator] = descriptor_profile
    if require_fields:
        extra_fields[operator] = tuple(require_fields)
    else:
        extra_fields.pop(operator, None)

    if activation_dtype_const is not None:
        field_constraints[operator] = {"activation_dtype": activation_dtype_const}
    else:
        field_constraints.pop(operator, None)

    text = _replace_python_assignment(text, "OPERATOR_DESCRIPTOR_PROFILES", profile_map)
    text = _replace_python_assignment(text, "OPERATOR_EXTRA_REQUIRED_FIELDS", extra_fields)
    text = _replace_python_assignment(text, "OPERATOR_FIELD_CONSTRAINTS", field_constraints)
    descriptors_path.write_text(text)


def _find_schema_rule_index(schema: dict[str, Any], comment: str) -> int | None:
    for idx, rule in enumerate(schema.get("allOf", [])):
        if rule.get("$comment") == comment:
            return idx
    return None


def _remove_operator_from_schema_rules(schema: dict[str, Any], operator: str) -> None:
    for rule in schema.get("allOf", []):
        op_props = rule.get("if", {}).get("properties", {}).get("operator", {})
        if "enum" in op_props:
            op_props["enum"] = [value for value in op_props["enum"] if value != operator]


def _base_profile_schema_rule(profile: str) -> dict[str, Any]:
    comments = {
        "arg_reduction": "descriptor_profile:arg_reduction",
        "conv": "descriptor_profile:conv",
        "custom": "descriptor_profile:custom",
        "dual_input_elementwise": "descriptor_profile:dual_input_elementwise",
        "fully_connected": "descriptor_profile:fully_connected",
        "pool": "descriptor_profile:pool",
        "shape_transform": "descriptor_profile:shape_transform",
        "single_input_unary": "descriptor_profile:single_input_unary",
    }
    if profile == "fully_connected":
        return {
            "$comment": comments[profile],
            "if": {"properties": {"operator": {"enum": []}}},
            "then": {"required": ["input_shape", "filter_shape"]},
        }
    if profile == "conv":
        return {
            "$comment": comments[profile],
            "if": {"properties": {"operator": {"enum": []}}},
            "then": {"required": ["input_shape", "filter_shape"]},
        }
    if profile == "dual_input_elementwise":
        return {
            "$comment": comments[profile],
            "if": {"properties": {"operator": {"enum": []}}},
            "then": {"required": ["input_1_shape", "input_2_shape"]},
        }
    if profile == "pool":
        return {
            "$comment": comments[profile],
            "if": {"properties": {"operator": {"enum": []}}},
            "then": {
                "required": ["input_shape"],
                "anyOf": [
                    {"required": ["pool_size"]},
                    {"required": ["filter_shape"]},
                ],
            },
        }
    return {
        "$comment": comments[profile],
        "if": {"properties": {"operator": {"enum": []}}},
        "then": {"required": ["input_shape"]},
    }


def _update_schema(
    schema_path: Path,
    *,
    operator: str,
    descriptor_profile: str,
    require_fields: tuple[str, ...],
    activation_dtype_const: str | None,
) -> None:
    schema = json.loads(schema_path.read_text())
    enum_values = sorted(set(schema["properties"]["operator"]["enum"]) | {operator})
    schema["properties"]["operator"]["enum"] = enum_values
    _remove_operator_from_schema_rules(schema, operator)

    if descriptor_profile != "custom":
        profile_rule = _base_profile_schema_rule(descriptor_profile)
        profile_idx = _find_schema_rule_index(schema, profile_rule["$comment"])
        if profile_idx is None:
            schema["allOf"].append(profile_rule)
            profile_idx = len(schema["allOf"]) - 1
        rule_enum = schema["allOf"][profile_idx]["if"]["properties"]["operator"].setdefault("enum", [])
        if operator not in rule_enum:
            rule_enum.append(operator)
            rule_enum.sort()

    fields_comment = f"operator_required_fields:{operator}"
    fields_idx = _find_schema_rule_index(schema, fields_comment)
    if require_fields:
        rule = {
            "$comment": fields_comment,
            "if": {"properties": {"operator": {"const": operator}}},
            "then": {"required": list(require_fields)},
        }
        if fields_idx is None:
            schema["allOf"].append(rule)
        else:
            schema["allOf"][fields_idx] = rule
    elif fields_idx is not None:
        del schema["allOf"][fields_idx]

    constraint_comment = f"operator_activation_dtype_const:{operator}"
    constraint_idx = _find_schema_rule_index(schema, constraint_comment)
    if activation_dtype_const is not None:
        rule = {
            "$comment": constraint_comment,
            "if": {"properties": {"operator": {"const": operator}}},
            "then": {"properties": {"activation_dtype": {"const": activation_dtype_const}}},
        }
        if constraint_idx is None:
            schema["allOf"].append(rule)
        else:
            schema["allOf"][constraint_idx] = rule
    elif constraint_idx is not None:
        del schema["allOf"][constraint_idx]

    schema_path.write_text(json.dumps(schema, indent=2) + "\n")


def _descriptor_text(
    *,
    descriptor_stem: str,
    operator: str,
    descriptor_profile: str,
) -> str:
    lines = [
        f"name: {descriptor_stem}_case_01_s8",
        f"operator: {operator}",
        "activation_dtype: S8",
        "weight_dtype: S8",
    ]
    for field, default_value in PROFILE_DEFAULT_DESCRIPTOR_FIELDS[descriptor_profile]:
        lines.append(f"{field}: {default_value}")
    return "\n".join(lines) + "\n"


def _module_template(
    *,
    operator: str,
    class_name: str,
    module_basename: str,
    descriptor_profile: str,
    builder_op_name: str,
) -> str:
    common_header = (
        f'"""{operator} operation implementation."""\n\n'
        "from pathlib import Path\n"
        "from helia_core_tester.generation.ops._shared.base import OperationBase\n"
    )
    op_fn_name = _wrapper_function_name(module_basename)

    if descriptor_profile == "single_input_unary":
        return (
            common_header
            + "from helia_core_tester.generation.utils.litert_builder import build_unary_same_shape_op\n\n\n"
            + f"def {op_fn_name}(*, input_shape, dtype: str = \"int8\", output_dtype: str | None = None) -> bytes:\n"
            + "    return build_unary_same_shape_op(\n"
            + f"        op_name=\"{builder_op_name}\",\n"
            + "        input_shape=input_shape,\n"
            + "        dtype=dtype,\n"
            + "        output_dtype=output_dtype,\n"
            + "    )\n\n\n"
            + f"class {class_name}(OperationBase):\n"
            + f"    \"\"\"{operator} operation.\"\"\"\n\n"
            + "    def needs_keras_model(self) -> bool:\n"
            + "        return False\n\n"
            + "    def build_keras_model(self):\n"
            + f"        raise NotImplementedError(\"{operator} uses LiteRT-only model generation.\")\n\n"
            + "    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:\n"
            + f"        model_bytes = {op_fn_name}(\n"
            + "            input_shape=tuple(self.desc[\"input_shape\"]),\n"
            + "            dtype=self.tensor_litert_dtype(\"input\"),\n"
            + "            output_dtype=self.tensor_litert_dtype(\"output\"),\n"
            + "        )\n"
            + "        self._write_tflite_bytes(out_path, model_bytes)\n\n"
            + "    def generate_c_files(self, output_dir: Path) -> None:\n"
            + f"        raise NotImplementedError(\"{operator} generate_c_files is not implemented.\")\n"
        )

    if descriptor_profile == "dual_input_elementwise":
        return (
            common_header
            + "from helia_core_tester.generation.utils.litert_builder import build_binary_broadcast_op\n\n\n"
            + f"def {op_fn_name}(*, input_1_shape, input_2_shape, dtype: str = \"int8\") -> bytes:\n"
            + "    return build_binary_broadcast_op(\n"
            + f"        op_name=\"{builder_op_name}\",\n"
            + "        input_1_shape=input_1_shape,\n"
            + "        input_2_shape=input_2_shape,\n"
            + "        dtype=dtype,\n"
            + "    )\n\n\n"
            + f"class {class_name}(OperationBase):\n"
            + f"    \"\"\"{operator} operation.\"\"\"\n\n"
            + "    def needs_keras_model(self) -> bool:\n"
            + "        return False\n\n"
            + "    def build_keras_model(self):\n"
            + f"        raise NotImplementedError(\"{operator} uses LiteRT-only model generation.\")\n\n"
            + "    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:\n"
            + f"        model_bytes = {op_fn_name}(\n"
            + "            input_1_shape=tuple(self.desc[\"input_1_shape\"]),\n"
            + "            input_2_shape=tuple(self.desc[\"input_2_shape\"]),\n"
            + "            dtype=self.tensor_litert_dtype(\"input\"),\n"
            + "        )\n"
            + "        self._write_tflite_bytes(out_path, model_bytes)\n\n"
            + "    def generate_c_files(self, output_dir: Path) -> None:\n"
            + f"        raise NotImplementedError(\"{operator} generate_c_files is not implemented.\")\n"
        )

    if descriptor_profile == "arg_reduction":
        return (
            common_header
            + "from helia_core_tester.generation.utils.litert_builder import build_arg_reduction_op\n\n\n"
            + f"def {op_fn_name}(*, input_shape, axis: int = -1, dtype: str = \"int8\") -> bytes:\n"
            + "    return build_arg_reduction_op(\n"
            + f"        op_name=\"{builder_op_name}\",\n"
            + "        input_shape=input_shape,\n"
            + "        axis=axis,\n"
            + "        dtype=dtype,\n"
            + "    )\n\n\n"
            + f"class {class_name}(OperationBase):\n"
            + f"    \"\"\"{operator} operation.\"\"\"\n\n"
            + "    def needs_keras_model(self) -> bool:\n"
            + "        return False\n\n"
            + "    def build_keras_model(self):\n"
            + f"        raise NotImplementedError(\"{operator} uses LiteRT-only model generation.\")\n\n"
            + "    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:\n"
            + f"        model_bytes = {op_fn_name}(\n"
            + "            input_shape=tuple(self.desc[\"input_shape\"]),\n"
            + "            axis=self.desc.get(\"axis\", -1),\n"
            + "            dtype=self.tensor_litert_dtype(\"input\"),\n"
            + "        )\n"
            + "        self._write_tflite_bytes(out_path, model_bytes)\n\n"
            + "    def generate_c_files(self, output_dir: Path) -> None:\n"
            + f"        raise NotImplementedError(\"{operator} generate_c_files is not implemented.\")\n"
        )

    return (
        common_header
        + "\n\n"
        + f"def {op_fn_name}(**kwargs) -> bytes:\n"
        + "    del kwargs\n"
        + f"    raise NotImplementedError(\"{operator} LiteRT wrapper is not implemented. Replace {op_fn_name}().\")\n\n\n"
        + f"class {class_name}(OperationBase):\n"
        + f"    \"\"\"{operator} operation.\"\"\"\n\n"
        + "    def needs_keras_model(self) -> bool:\n"
        + "        return False\n\n"
        + "    def build_keras_model(self):\n"
        + f"        raise NotImplementedError(\"{operator} uses LiteRT-only model generation.\")\n\n"
        + "    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:\n"
        + f"        raise NotImplementedError(\"{operator} convert_to_tflite() must be implemented with {op_fn_name}().\")\n\n"
        + "    def generate_c_files(self, output_dir: Path) -> None:\n"
        + f"        raise NotImplementedError(\"{operator} generate_c_files is not implemented.\")\n"
    )


def create_operator_skeleton(
    repo_root: Path,
    *,
    operator: str,
    family: str,
    parity_kind: str,
    descriptor_profile: str | None = None,
    module_basename: str | None = None,
    descriptor_stem: str | None = None,
    class_name: str | None = None,
    builder_op_name: str | None = None,
    require_fields: tuple[str, ...] = (),
    activation_dtype_const: str | None = None,
    create_descriptor: bool = True,
    create_templates: bool = True,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Create grouped skeleton files for a new operator and register metadata."""
    repo_root = Path(repo_root)
    module_basename = module_basename or operator_to_module_basename(operator)
    descriptor_stem = descriptor_stem or module_basename
    class_name = class_name or operator_to_class_name(operator)
    builder_op_name = builder_op_name or _operator_to_builtin_name(operator, module_basename)

    if create_descriptor:
        if descriptor_profile is None:
            raise ValueError("descriptor_profile is required when creating descriptors")
        if descriptor_profile not in SUPPORTED_DESCRIPTOR_PROFILES:
            raise ValueError(f"Unsupported descriptor_profile: {descriptor_profile}")

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
        _module_template(
            operator=operator,
            class_name=class_name,
            module_basename=module_basename,
            descriptor_profile=descriptor_profile or "custom",
            builder_op_name=builder_op_name,
        )
    )
    created["module"] = module_path
    created["package_init"] = package_init

    if create_descriptor:
        if overwrite or not descriptor_path.exists():
            descriptor_path.write_text(
                _descriptor_text(
                    descriptor_stem=descriptor_stem,
                    operator=operator,
                    descriptor_profile=descriptor_profile or "custom",
                )
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

    catalog_path = repo_root / "helia_core_tester" / "generation" / "ops" / "catalog.py"
    _update_catalog(
        catalog_path,
        operator=operator,
        family=family,
        parity_kind=parity_kind,
        module_basename=module_basename,
        class_name=class_name,
        descriptor_relpath=f"{family}/{descriptor_stem}.yaml" if create_descriptor else None,
        template_relpath=f"{family}/{module_basename}" if create_templates else None,
        overwrite=overwrite,
    )
    created["catalog"] = catalog_path

    if create_descriptor:
        descriptors_path = repo_root / "helia_core_tester" / "generation" / "io" / "descriptors.py"
        schema_path = repo_root / "helia_core_tester" / "generation" / "descriptors" / "schema.json"
        _update_descriptor_metadata(
            descriptors_path,
            operator=operator,
            descriptor_profile=descriptor_profile or "custom",
            require_fields=require_fields,
            activation_dtype_const=activation_dtype_const,
        )
        _update_schema(
            schema_path,
            operator=operator,
            descriptor_profile=descriptor_profile or "custom",
            require_fields=require_fields,
            activation_dtype_const=activation_dtype_const,
        )
        created["descriptor_metadata"] = descriptors_path
        created["schema"] = schema_path

    return created


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create grouped skeleton files for a tester operator.")
    parser.add_argument("operator")
    parser.add_argument("family")
    parser.add_argument("parity_kind", choices=("cmsis", "extension"))
    parser.add_argument("--descriptor-profile", choices=SUPPORTED_DESCRIPTOR_PROFILES)
    parser.add_argument("--builder-op-name")
    parser.add_argument("--require-field", action="append", default=[])
    parser.add_argument("--activation-dtype-const")
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
        descriptor_profile=args.descriptor_profile,
        module_basename=args.module_basename,
        descriptor_stem=args.descriptor_stem,
        class_name=args.class_name,
        builder_op_name=args.builder_op_name,
        require_fields=tuple(args.require_field),
        activation_dtype_const=args.activation_dtype_const,
        create_descriptor=not args.no_descriptor,
        create_templates=not args.no_templates,
        overwrite=args.overwrite,
    )

    for label, path in created.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
