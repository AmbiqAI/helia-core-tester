"""Lazy operation registry keyed by canonical operator names."""

from helia_core_tester.generation.ops.catalog import OPERATOR_SPECS


OP_CLASS_SPECS = {
    operator: (spec.module_path, spec.class_name)
    for operator, spec in OPERATOR_SPECS.items()
}
