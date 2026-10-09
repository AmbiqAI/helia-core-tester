#!/usr/bin/env python3
"""Fuzz reference goldens against the ns-cmsis-nn kernels on the host.

Draws random descriptors per operator family, generates each through the
reference-kernel path, runs the generated harness on the host kernels with the
comparison forced to exact, and tabulates exactness per family and dtype. The
table is the evidence behind every tolerance in generation/io/dtypes.py.

    uv run python scripts/ref_fuzz.py --op Convolve,FullyConnected --draws 200 --seed 7
    uv run python scripts/ref_fuzz.py --op all --draws 50 --host-kernels m0,dsp --cpu cortex-m4

Exit status: 0 when every draw is bit-exact, 1 otherwise (each failure prints
the descriptor that reproduces it), 2 on a usage or build error.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from helia_core_tester.generation import random_shapes  # noqa: E402
from helia_core_tester.generation.reference import host_kernels as hk  # noqa: E402
from helia_core_tester.generation.utils.temp_sizer_probe import resolve_cmsis_nn_root  # noqa: E402

EXACT = {"tolerance": 0}


def _shape(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))


def _fc(rng: np.random.Generator, index: int, dtype: str) -> Dict[str, Any]:
    features = _shape(rng, 1, 96)
    desc: Dict[str, Any] = {
        "operator": "FullyConnected",
        "name": f"fz_fc_{dtype.lower()}_{index:04d}",
        "activation_dtype": "S8" if dtype == "S4" else dtype,
        "weight_dtype": "S4" if dtype == "S4" else "S8",
        "input_shape": [_shape(rng, 1, 4), features],
        "filter_shape": [_shape(rng, 1, 24), features],
        "use_bias": bool(rng.random() < 0.8),
        "activation": str(rng.choice(["NONE", "RELU"])) if dtype == "S8" else "NONE",
    }
    if dtype == "S8":
        if rng.random() < 0.3:
            desc["hint"] = {"force_per_tensor": True}
        elif rng.random() < 0.2:
            desc["hint"] = {"extras": {"force_filter_offset": int(rng.integers(-6, 7))}}
    return desc


def _tconv(rng: np.random.Generator, index: int, dtype: str) -> Dict[str, Any]:
    cin = _shape(rng, 1, 24)
    return {
        "operator": "TransposeConv",
        "name": f"fz_tc_{dtype.lower()}_{index:04d}",
        "activation_dtype": dtype,
        "weight_dtype": "S8",
        "input_shape": [1, _shape(rng, 1, 6), _shape(rng, 1, 6), cin],
        "filter_shape": [_shape(rng, 1, 4), _shape(rng, 1, 4), _shape(rng, 1, 12), cin],
        "strides": [_shape(rng, 1, 3), _shape(rng, 1, 3)],
        "padding": str(rng.choice(["same", "valid"])),
        "use_bias": bool(rng.random() < 0.8),
    }


def _random_shape_sampler(op: str) -> Callable[[np.random.Generator, int, str], Dict[str, Any]]:
    def sample(rng: np.random.Generator, index: int, dtype: str) -> Dict[str, Any]:
        desc = random_shapes.sample_op(op, 1, int(rng.integers(1, 2**31)), "cortex-m55", 1 << 22)[0]
        desc["name"] = f"fz_{op.lower()}_{dtype.lower()}_{index:04d}"
        if dtype == "S16":
            # s16 is symmetric and its kernels take no fused activation clamp knobs.
            desc["activation_dtype"] = "S16"
            for key in ("activation_min", "activation_max"):
                desc.pop(key, None)
        if dtype == "S4":
            desc["weight_dtype"] = "S4"
            for key in ("calibration_range", "input_range", "weight_gain", "activation_min", "activation_max"):
                desc.pop(key, None)
            desc["activation"] = "NONE"
        return desc

    return sample


SAMPLERS = {
    "Convolve": (_random_shape_sampler("Convolve"), ("S8", "S16", "S4")),
    "DepthwiseConv": (_random_shape_sampler("DepthwiseConv"), ("S8", "S16", "S4")),
    "FullyConnected": (_fc, ("S8", "S16", "S4")),
    "TransposeConv": (_tconv, ("S8",)),
}


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--op", default="all", help=f"comma-separated, or all ({', '.join(SAMPLERS)})")
    parser.add_argument("--draws", type=int, default=50, help="draws per op and dtype")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--vs", choices=("kernel",), default="kernel", help="what the golden is checked against")
    parser.add_argument("--cpu", default="cortex-m0", help="target the cases are generated for")
    parser.add_argument("--host-kernels", default="m0", help="host kernel builds, comma-separated (m0, dsp)")
    parser.add_argument("--cmsis-nn-root", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None, help="also write the table here")
    args = parser.parse_args(argv)

    ops = list(SAMPLERS) if args.op == "all" else [o.strip() for o in args.op.split(",") if o.strip()]
    unknown = [o for o in ops if o not in SAMPLERS]
    if unknown or args.draws < 1:
        parser.error(f"unknown op(s) {unknown}" if unknown else "--draws must be positive")
    tree = args.cmsis_nn_root or resolve_cmsis_nn_root()
    if tree is None:
        parser.error("no ns-cmsis-nn checkout: pass --cmsis-nn-root or set CMSIS_NN_ROOT")
    modes = [m.strip() for m in args.host_kernels.split(",") if m.strip()]

    from helia_core_tester.generation.test_ops import generate_test

    rng = np.random.default_rng(args.seed)
    table: Dict[str, Dict[str, Any]] = {}
    failed = False
    with tempfile.TemporaryDirectory(prefix="hct-ref-fuzz-") as tmp:
        for op in ops:
            sampler, dtypes = SAMPLERS[op]
            for dtype in dtypes:
                root = Path(tmp) / f"{op}_{dtype}" / args.cpu
                descs, gen_errors = {}, []
                for index in range(args.draws):
                    desc = {**sampler(rng, index, dtype), "comparison": dict(EXACT)}
                    descs[desc["name"]] = desc
                    try:
                        generate_test(desc, str(root), seed=index + 1, cpu=args.cpu)
                    except Exception as exc:  # recorded per draw, never fatal
                        gen_errors.append({"name": desc["name"], "error": repr(exc)})
                row: Dict[str, Any] = {"draws": args.draws, "generation_errors": gen_errors}
                for mode in modes:
                    report = hk.run_host_check([root], tree, mode=mode, seed=args.seed)
                    fails = report.failures + report.advisory
                    row[mode] = {"run": report.total, "exact": report.passed, "failures": [f["name"] for f in fails]}
                    for failure in fails:
                        print(f"FAIL {op} {dtype} [{mode}] {failure['name']}: {failure['headline']}")
                        print("     " + json.dumps(descs.get(failure["name"]), sort_keys=True))
                    failed |= bool(fails)
                failed |= bool(gen_errors)
                for err in gen_errors:
                    print(f"GENERATION {op} {dtype} {err['name']}: {err['error']}")
                table[f"{op}/{dtype}"] = row
    print(f"\n{'op/dtype':28s} " + " ".join(f"{m:>14s}" for m in modes))
    for key, row in table.items():
        cells = " ".join(f"{row[m]['exact']:>6d}/{row[m]['run']:<7d}" for m in modes)
        print(f"{key:28s} {cells}  gen_errors={len(row['generation_errors'])}")
    if args.json:
        args.json.write_text(json.dumps({"seed": args.seed, "cpu": args.cpu, "table": table}, indent=2) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
