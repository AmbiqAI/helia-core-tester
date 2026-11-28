"""Render a pack case: a pack generator's CaseSpec through hct's generic harness.

The generator returns arrays and scalars by contract parameter name; every one is checked against
the entry's contract (direction, pointer or value, element type, range) before anything is written,
so a pack cannot pass the kernel what its signature does not declare.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from helia_core_tester.generation.ops._shared.base import OperationBase
from helia_core_tester.pack import Case, CaseSpec, PackError, Param
from helia_core_tester.pack.descriptors import FLOAT_SCALARS, INTEGER_SCALARS, element_type, is_pointer, signature_problem

STEM = "pack"

ELEMENT_NUMPY: Dict[str, Any] = {
    "q7_t": np.int8, "int8_t": np.int8, "uint8_t": np.uint8,
    "q15_t": np.int16, "int16_t": np.int16, "uint16_t": np.uint16,
    "q31_t": np.int32, "int32_t": np.int32, "uint32_t": np.uint32,
    "float32_t": np.float32, "float": np.float32, "float16_t": np.float16,
}


def contract_params(function: Any) -> tuple[Param, ...]:
    return tuple(Param(p.name, p.c_type, p.direction, is_pointer(p.c_type), element_type(p.c_type))
                 for p in function.params)


def scalar_literal(param: Param, value: Any, where: str) -> str:
    """A C literal for a by-value parameter, checked against its type."""
    if param.element in INTEGER_SCALARS:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise PackError(f"{where}: {param.name} ({param.element}) takes an integer, got {value!r}")
        info = np.iinfo(ELEMENT_NUMPY[param.element])
        if not info.min <= int(value) <= info.max:
            raise PackError(f"{where}: {param.name} = {int(value)} is outside {param.element} "
                            f"[{info.min}, {info.max}]")
        return f"{int(value)}u" if param.element.startswith("uint") else str(int(value))
    if param.element in FLOAT_SCALARS:
        if isinstance(value, bool) or not isinstance(value, (int, float, np.floating, np.integer)):
            raise PackError(f"{where}: {param.name} ({param.element}) takes a number, got {value!r}")
        try:
            as_float = float(value)
        except OverflowError:
            raise PackError(f"{where}: {param.name} = {value!r} overflows {param.element}") from None
        if not math.isfinite(as_float):
            raise PackError(f"{where}: {param.name} = {value!r}; a non-finite scalar is not supported")
        dtype = ELEMENT_NUMPY[param.element]
        with np.errstate(over="ignore"):
            stored = dtype(value)
        if not math.isfinite(float(stored)):
            raise PackError(f"{where}: {param.name} = {value!r} overflows {param.element}")
        literal = f"{float(stored)!r}f"
        return f"(float16_t){literal}" if param.element == "float16_t" else literal
    raise PackError(f"{where}: {param.name} is {param.c_type}, which pack scalars do not support yet")


class PackCaseGenerator(OperationBase):
    """One hct/descriptor/1 case, built by the pack generator its descriptor resolves to."""

    pack_case = True

    def __init__(self, desc: Dict[str, Any], seed: int = 1, target_cpu: str = "cortex-m55"):
        super().__init__(desc, seed, target_cpu=target_cpu)
        self.golden = "hct_ref"
        self.golden_pending = False

    def render_template(self, template_path: str, context: Dict[str, Any], return_context: bool = False):
        """A pack case renders hct's common templates only, not an operator's from the catalog."""
        from helia_core_tester.core.discovery import find_tester_templates_dir
        from helia_core_tester.generation.ops._shared.base import template_environment

        if not template_path.startswith("common/") or template_path.endswith(".c.j2"):
            raise PackError(f"{self.desc['name']}: a pack case renders through the generic harness, not "
                            f"{template_path}")
        rendered = template_environment(str(find_tester_templates_dir())).get_template(template_path).render(**context)
        return (rendered, dict(context)) if return_context else rendered

    def _build(self, function: Any, params: tuple[Param, ...]) -> CaseSpec:
        from helia_core_tester.pack.loader import current_pack

        pack = current_pack()
        if pack is None:
            raise PackError(f"{self.desc['name']}: a pack case needs the library's pack, and none is loaded")
        generator = pack.registry.generator_for(self.desc["entry"], self.desc["operator"], self.desc["name"])
        case = Case(name=self.desc["name"], entry=function.name, params=params,
                    values=dict(self.desc.get("params") or {}), rng=self.rng, seed=self.seed,
                    target_cpu=self.target_cpu)
        try:
            spec = generator.build(case)
        except PackError:
            raise
        except (Exception, SystemExit) as exc:
            raise PackError(f"{self.desc['name']}: generator {generator.key} ({generator.module}) failed: "
                            f"{type(exc).__name__}: {exc}") from exc
        if not isinstance(spec, CaseSpec):
            raise PackError(f"{self.desc['name']}: generator {generator.key} returned {type(spec).__name__}, "
                            "not a CaseSpec")
        return spec

    def _checked_arrays(self, spec: CaseSpec, params: tuple[Param, ...]) -> Dict[str, np.ndarray]:
        where = self.desc["name"]
        inputs = {p.name: p for p in params if p.pointer and p.direction == "in"}
        unsupported = [p.name for p in params if p.pointer and p.direction not in ("in", "out")]
        if unsupported:
            raise PackError(f"{where}: {', '.join(unsupported)} are {self.desc['entry']} in/out pointers, which "
                            "pack cases do not support yet")
        if set(spec.arrays) != set(inputs):
            raise PackError(f"{where}: arrays {sorted(spec.arrays)} must be exactly the input pointers "
                            f"{sorted(inputs)}")
        arrays: Dict[str, np.ndarray] = {}
        for name, values in spec.arrays.items():
            param = inputs[name]
            dtype = ELEMENT_NUMPY.get(param.element)
            if dtype is None:
                raise PackError(f"{where}: {name} is {param.c_type}, which pack arrays do not support yet")
            array = np.asarray(values)
            if array.dtype != np.dtype(dtype):
                raise PackError(f"{where}: {name} is {param.element} ({np.dtype(dtype).name}), but the generator "
                                f"gave {array.dtype.name}")
            arrays[name] = np.ascontiguousarray(array.reshape(-1))
        return arrays

    def generate_c_files(self, output_dir: Path) -> None:
        from helia_core_tester.contract.render import load_current_contracts
        from helia_core_tester.generation.harness import ArgumentPool, ArrayLiteral, Declaration, HarnessInput
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        name, entry = self.desc["name"], self.desc["entry"]
        function = load_current_contracts().functions.get(entry)
        if function is None:
            raise PackError(f"{name}: entry {entry} is not in the library's contract")
        problem = signature_problem(function)
        if problem:
            raise PackError(f"{name}: {problem}")
        decided = [key for key in ("entry_sizer", "autovectorize_declines") if self.desc.get(key)]
        if decided:
            raise PackError(f"{name}: the implementation manifest gives {entry} {', '.join(decided)}, which pack "
                            "cases do not support yet")
        params = contract_params(function)
        spec = self._build(function, params)
        arrays = self._checked_arrays(spec, params)
        outputs = [p for p in params if p.pointer and p.direction == "out"]
        if len(outputs) != 1 or spec.output != outputs[0].name:
            raise PackError(f"{name}: output {spec.output!r} must be {entry}'s one output pointer "
                            f"({', '.join(p.name for p in outputs) or 'none'})")
        output = outputs[0]
        out_dtype = ELEMENT_NUMPY.get(output.element)
        if out_dtype is None:
            raise PackError(f"{name}: {output.name} is {output.c_type}, which pack outputs do not support yet")
        if isinstance(spec.output_count, bool) or not isinstance(spec.output_count, (int, np.integer)) \
                or int(spec.output_count) < 0:
            raise PackError(f"{name}: output_count {spec.output_count!r} is not a non-negative integer")
        count = int(spec.output_count)
        scalars = {p.name: p for p in params if not p.pointer}
        if set(spec.scalars) != set(scalars):
            raise PackError(f"{name}: scalars {sorted(spec.scalars)} must be exactly {entry}'s value parameters "
                            f"{sorted(scalars)}")
        values = {key: scalar_literal(scalars[key], value, name) for key, value in spec.scalars.items()}

        if spec.golden is None:
            if self.golden != "record":
                raise PackError(f"{name}: generator gives no golden, so the case's golden comes from the library's "
                                "host reference build: generate it with --golden record")
            golden = np.zeros(count, dtype=out_dtype)
            self.golden_pending = True
        else:
            golden = np.asarray(spec.golden)
            if golden.dtype != np.dtype(out_dtype):
                raise PackError(f"{name}: golden is {golden.dtype.name}, but {output.name} is {output.element}")
            golden = golden.reshape(-1)
            if golden.size != count:
                raise PackError(f"{name}: golden has {golden.size} elements, output_count is {count}")

        builder = TemplateContextBuilder()

        def literal(array: np.ndarray, dtype: Any) -> str:
            return builder.format_array_as_c_literal(array if array.size else np.zeros(1, dtype=dtype))

        header = [Declaration(f"{name}_{param}", inputs_param.element, ArrayLiteral(literal(arrays[param],
                              ELEMENT_NUMPY[inputs_param.element])), array=True)
                  for param, inputs_param in ((p.name, p) for p in params if p.name in arrays)]
        header.append(Declaration(f"{name}_expected_output", output.element, ArrayLiteral(literal(golden, out_dtype)),
                                  array=True, comment="Expected output (golden)"))
        pool = ArgumentPool(
            name=name, values=values, header=header, output_param=output.name, output_count=f"({count})",
            inputs=tuple(HarnessInput(p.name, f"arg_{p.name}", f"{name}_{p.name}", ctype=p.element)
                         for p in params if p.name in arrays),
            benchmark=False, scratch_buffer=False,
            output_capacity=None if count > 0 else "1", output_poison=count == 0, output_untouched=count == 0,
        )
        input_elements = [p.element for p in params if p.name in arrays]
        context: Dict[str, Any] = {
            "name": name,
            "kernel_fn": entry,
            "input_dtype": input_elements[0] if input_elements else output.element,
            "output_dtype": output.element,
            "entry_scratch_bytes": 0,
            "validation_label": entry,
        }
        if "expected_status" in self.desc:
            context["expected_status"] = self.desc["expected_status"]
        self.render_harness_case(
            output_dir, stem=STEM, context=context, pool=pool, validation_key=f"pack/{self.desc['operator']}",
            label=entry, operator=str(self.desc["operator"]), sidecar=False,
        )
