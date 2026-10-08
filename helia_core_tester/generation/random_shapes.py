"""Seeded held-out s8 conv shapes."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from helia_core_tester.core.cpu_targets import get_cpu_profile, normalize_cpu
from helia_core_tester.core.path_layout import artifacts_root
from helia_core_tester.hardware.generated_test_bridge import (
    _align_up,
    _depthwise_s8_scratch_bytes,
    _with_weight_sums,
)
from helia_core_tester.hardware.wrapper_route import conv_route, dw_route
from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

# Per-case cap; keeps a 50-case stream short.
MAX_MACS = 1_500_000
MAX_TRIES = 500
PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61)
MVE_CONV_ROUTES = ("arm_convolve_1x1_out_s8", "arm_convolve_s8_small_cin", "arm_convolve_s8_3x3_c16_s1")
CONV_ROUTES = (
    "arm_convolve_1x1_s8_fast",
    "arm_convolve_1x1_s8",
    "arm_convolve_1_x_n_s8",
    *MVE_CONV_ROUTES,
    "arm_convolve_s8",
)
# One input channel runs as conv.
DW_AS_CONV = "as_conv"
SECRET_ENV = "HCT_HIDDEN_SEED"
# Short secrets fall to brute force.
MIN_SECRET = 16
HIDDEN_ID_HEX = 12


@dataclass(frozen=True)
class Generator:
    """Random-shape source for one op."""

    dtype: str
    tag: str
    descriptor_file: str
    # RNG stream id; never reuse one.
    stream: int
    routes: Callable[[bool], list[str]]
    draw: Callable


@dataclass
class Layer:
    """One sampled layer, NHWC."""

    h: int
    w: int
    cin: int
    cout: int
    kh: int
    kw: int
    sh: int = 1
    sw: int = 1
    dh: int = 1
    dw: int = 1
    padding: str = "VALID"
    mult: int = 1

    def out_hw(self) -> tuple[int, int]:
        """Output height and width."""
        ekh, ekw = (self.kh - 1) * self.dh + 1, (self.kw - 1) * self.dw + 1
        if self.padding == "SAME":
            return -(-self.h // self.sh), -(-self.w // self.sw)
        return -(-(self.h - ekh + 1) // self.sh), -(-(self.w - ekw + 1) // self.sw)

    def pads(self) -> tuple[int, int]:
        """TFLite pad_before, as CMSIS uses."""
        if self.padding != "SAME":
            return 0, 0
        oh, ow = self.out_hw()
        ekh, ekw = (self.kh - 1) * self.dh + 1, (self.kw - 1) * self.dw + 1
        return max((oh - 1) * self.sh + ekh - self.h, 0) // 2, max((ow - 1) * self.sw + ekw - self.w, 0) // 2


def _size(rng: np.random.Generator, lo: int, hi: int) -> int:
    """A size biased to primes and odd values."""
    pool = [p for p in PRIMES if lo <= p <= hi]
    if pool and rng.random() < 0.4:
        return int(rng.choice(pool))
    return int(rng.integers(lo, hi + 1))


def _channels(rng: np.random.Generator, lo: int, hi: int) -> int:
    """Channels; mostly not a multiple of 4."""
    roll = rng.random()
    if roll < 0.15 and hi >= 16:
        return int(16 * rng.integers(max(1, -(-lo // 16)), hi // 16 + 1))
    if roll < 0.3 and hi >= 4:
        return int(4 * rng.integers(max(1, -(-lo // 4)), hi // 4 + 1))
    return _size(rng, lo, hi)


def _kernel(rng: np.random.Generator) -> tuple[int, int]:
    shape = rng.choice(["1xn", "nx1", "3x3", "square", "any"])
    n = int(rng.integers(2, 8))
    return {
        "1xn": (1, n), "nx1": (n, 1), "3x3": (3, 3), "square": (n, n),
        "any": (int(rng.integers(1, 8)), int(rng.integers(1, 8))),
    }[str(shape)]


def _stride(rng: np.random.Generator) -> tuple[int, int]:
    return int(rng.integers(1, 4)), int(rng.integers(1, 4))


def _padding(rng: np.random.Generator) -> str:
    return "SAME" if rng.random() < 0.5 else "VALID"


def _maybe_dilate(rng: np.random.Generator, layer: Layer, p: float = 0.5) -> Layer:
    """Keras needs stride 1 to dilate."""
    if rng.random() < p:
        layer.sh = layer.sw = 1
        layer.dh = int(rng.integers(1, 4))
        layer.dw = int(rng.integers(2 if layer.dh == 1 else 1, 4))
    return layer


def _conv_layer(rng: np.random.Generator, route: str) -> Layer:
    """A conv layer aimed at one route."""
    pad = _padding(rng)
    if route == "arm_convolve_1x1_s8_fast":
        return Layer(_size(rng, 1, 24), _size(rng, 1, 24), _channels(rng, 1, 96), _channels(rng, 1, 64), 1, 1, padding=pad)
    if route == "arm_convolve_1x1_s8":
        sh, sw = _stride(rng)
        return Layer(_size(rng, 2, 24), _size(rng, 2, 24), _channels(rng, 1, 96), _channels(rng, 1, 64), 1, 1, sh, sw, padding=pad)
    if route == "arm_convolve_1_x_n_s8":
        kw = int(rng.integers(2, 10))
        return Layer(1, _size(rng, 8, 96), _channels(rng, 1, 48), _channels(rng, 1, 48), 1, kw, *_stride(rng), padding=pad)
    if route == "arm_convolve_1x1_out_s8":
        h, w = _size(rng, 1, 7), _size(rng, 1, 7)
        return Layer(h, w, _channels(rng, 4, 64), _channels(rng, 1, 48), h, w, *_stride(rng), padding="VALID")
    if route == "arm_convolve_s8_small_cin":
        cin = int(rng.integers(1, 4))
        kw = int(rng.integers(1, 16 // cin + 1))
        kh = int(rng.integers(1, max(1, 48 // (kw * cin)) + 1))
        return Layer(_size(rng, 3, 32), _size(rng, 3, 32), cin, 4 * int(rng.integers(1, 9)), kh, kw, *_stride(rng), padding=pad)
    if route == "arm_convolve_s8_3x3_c16_s1":
        return Layer(_size(rng, 3, 20), _size(rng, 3, 20), 16, _channels(rng, 1, 48), 3, 3, padding=pad)
    kh, kw = _kernel(rng)
    layer = Layer(_size(rng, 2, 24), _size(rng, 2, 24), _channels(rng, 1, 48), _channels(rng, 1, 48), kh, kw, *_stride(rng), padding=pad)
    return _maybe_dilate(rng, layer)


def _dw_layer(rng: np.random.Generator, route: str) -> Layer:
    """A depthwise layer aimed at one route."""
    pad = _padding(rng)
    kh, kw = _kernel(rng)
    if route == DW_AS_CONV:
        mult = int(rng.integers(2, 5))
        layer = Layer(_size(rng, 2, 24), _size(rng, 2, 24), 1, mult, kh, kw, *_stride(rng), padding=pad, mult=mult)
        return _maybe_dilate(rng, layer)
    if route in ("arm_depthwise_conv_s8_opt", "arm_depthwise_conv_3x3_s8"):
        if route == "arm_depthwise_conv_3x3_s8" or rng.random() < 0.3:
            # W*C > 1440 overruns 3x3 scratch (#375).
            cin = _channels(rng, 24, 64)
            w = int(rng.integers(1440 // cin + 1, 1440 // cin + 12))
            return Layer(_size(rng, 3, 6), w, cin, cin, 3, 3, *_stride(rng), padding=pad)
        cin = _channels(rng, 1, 96)
        if rng.random() < 0.2:
            # 1D dilation keeps the opt route.
            return Layer(1, _size(rng, 8, 64), cin, cin, 1, kw, dw=int(rng.integers(2, 4)), padding=pad)
        return Layer(_size(rng, 2, 24), _size(rng, 2, 24), cin, cin, kh, kw, *_stride(rng), padding=pad)
    mult = int(rng.integers(1, 5))
    cin = _channels(rng, 2, 48)
    layer = Layer(_size(rng, 2, 24), _size(rng, 2, 24), cin, cin * mult, kh, kw, *_stride(rng), padding=pad, mult=mult)
    if mult == 1:
        layer.sh = layer.sw = 1
        layer.dh, layer.dw = int(rng.integers(2, 4)), int(rng.integers(1, 4))
        return layer
    return _maybe_dilate(rng, layer)


def _conv_routes(mve: bool) -> list[str]:
    return [r for r in CONV_ROUTES if mve or r not in MVE_CONV_ROUTES]


def _dw_routes(mve: bool) -> list[str]:
    first = DW_AS_CONV if mve else "arm_depthwise_conv_3x3_s8"
    return [first, "arm_depthwise_conv_s8_opt", "arm_depthwise_conv_s8"]


# Register new random-shape ops here.
GENERATORS: dict[str, Generator] = {
    "Convolve": Generator("S8", "conv", "ConvolutionFunctions/convolve.yaml", 0, _conv_routes, _conv_layer),
    "DepthwiseConv": Generator("S8", "dw", "ConvolutionFunctions/depthwise_conv.yaml", 1, _dw_routes, _dw_layer),
}
OPS = tuple(GENERATORS)


def _parts(value: str | None) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _matches(op: str, gen: Generator, wanted: str) -> bool:
    """Match --op as loaded descriptors do."""
    from helia_core_tester.generation.io.descriptors import descriptor_matches_op

    path = Path(gen.descriptor_file)
    probe = {"name": "", "operator": op, "_source_stem": path.stem, "_source_relpath": str(path)}
    # Case-name prefixes, e.g. rs7_conv.
    return descriptor_matches_op(probe, wanted) or re.fullmatch(rf"rs\d+_{gen.tag}(_\d+)?", wanted) is not None


def select_ops(op_filter: str | None = None, dtype_filter: str | None = None) -> tuple[str, ...]:
    """Registered ops matching generate filters."""
    from helia_core_tester.generation.io.dtypes import normalize_dtype

    ops, dtypes = _parts(op_filter), [normalize_dtype(d) for d in _parts(dtype_filter)]
    picked = tuple(
        op for op, gen in GENERATORS.items()
        if (not ops or any(_matches(op, gen, wanted) for wanted in ops)) and (not dtypes or gen.dtype in dtypes)
    )
    if not picked:
        have = ", ".join(f"{op} {gen.dtype}" for op, gen in GENERATORS.items())
        raise ValueError(f"No random shapes for that op/dtype; have {have}")
    return picked


def layer_route(op: str, layer: Layer, mve: bool) -> str:
    """The wrapper's direct callee."""
    oh, ow = layer.out_hw()
    i, o = (1, layer.h, layer.w, layer.cin), (1, oh, ow, layer.cout)
    stride, dil, pad = (layer.sh, layer.sw), (layer.dh, layer.dw), layer.pads()
    if op == "Convolve":
        return conv_route(i, (layer.cout, layer.kh, layer.kw, layer.cin), o, stride, pad, dil, mve)
    return dw_route(i, (1, layer.kh, layer.kw, layer.cout), o, stride, pad, dil, layer.mult, mve)


def footprint(op: str, layer: Layer) -> int:
    """Firmware workspace bytes, TCM placement."""
    oh, ow = layer.out_hw()
    idims = {"n": 1, "h": layer.h, "w": layer.w, "c": layer.cin}
    odims = {"n": 1, "h": oh, "w": ow, "c": layer.cout}
    if op == "Convolve":
        fdims = {"h": layer.kh, "w": layer.kw, "c": layer.cin, "n": layer.cout}
        scratch = TemplateContextBuilder.calculate_buffer_size_max(idims, fdims, odims, output_dtype="S8")
        scratch = _with_weight_sums(scratch, layer.cout)
        weights = layer.kh * layer.kw * layer.cin * layer.cout
    else:
        fdims = {"n": 1, "h": layer.kh, "w": layer.kw, "c": layer.cout}
        scratch = _depthwise_s8_scratch_bytes(idims, fdims, odims)
        weights = layer.kh * layer.kw * layer.cout
    used = 0
    # input, weights, bias, multiplier, shift
    for size, align in ((layer.h * layer.w * layer.cin, 1), (weights, 1), *[(4 * layer.cout, 4)] * 3):
        used = _align_up(used, align) + size
    used = _align_up(used, 16) + scratch
    return _align_up(used, 16) + oh * ow * layer.cout


def layer_macs(op: str, layer: Layer) -> int:
    oh, ow = layer.out_hw()
    depth = layer.cin if op == "Convolve" else 1
    return oh * ow * layer.cout * layer.kh * layer.kw * depth


def _second_moment(data: tuple[float, float], calib: tuple[float, float]) -> float:
    """E[x^2] of data clipped to calibration."""
    lo, hi = min(calib[0], 0.0), max(calib[1], 0.0)
    x = np.clip(np.linspace(*data, 257), lo, hi)
    return float(np.mean(x * x))


def _relu6_gain(op: str, layer: Layer, data: tuple, calib: tuple) -> float:
    """Gain putting pre-activation RMS near 6."""
    taps = layer.kh * layer.kw
    depth, outs = (layer.cin, layer.cout) if op == "Convolve" else (1, layer.mult)
    fan_avg = taps * (layer.cin + outs) / 2
    return 36.0 * fan_avg / (taps * depth * max(_second_moment(data, calib), 1e-6))


def _quant(rng: np.random.Generator, op: str, layer: Layer) -> dict[str, Any]:
    """Activation, clamp and quant knobs."""
    act = str(rng.choice(["NONE", "RELU", "RELU6"]))
    oh, ow = layer.out_hw()
    # Few outputs flatten under clamps.
    roomy = oh * ow * layer.cout >= 64
    # Dilated graphs keep the activation outside.
    act = act if roomy and (layer.dh, layer.dw) == (1, 1) else "NONE"
    knobs: dict[str, Any] = {"activation": act, "use_bias": bool(rng.random() < 0.85)}
    calib = (-32.0, 32.0)
    if rng.random() < 0.5:
        # Off-centre ranges give non-zero offsets.
        calib = (-float(rng.integers(0, 64)), float(rng.integers(1, 64)))
    data = calib
    if rng.random() < 0.25:
        # Wider data than calibration saturates.
        spread = float(rng.uniform(1.2, 2.5))
        data = (calib[0] * spread - 1.0, calib[1] * spread)
    knobs["calibration_range"], knobs["input_range"] = list(calib), list(data)
    if act == "RELU6":
        knobs["weight_gain"] = _relu6_gain(op, layer, data, calib) * float(rng.uniform(0.7, 2.0))
    elif rng.random() < 0.4:
        knobs["weight_gain"] = float(10 ** rng.uniform(-2, 2))
    if rng.random() < 0.25 and roomy:
        low = int(rng.integers(-128, 0))
        knobs["activation_min"] = low
        knobs["activation_max"] = int(rng.integers(low + 64, 128))
    return knobs


def _descriptor(op: str, name: str, layer: Layer, knobs: dict, seed: int, route: str) -> dict[str, Any]:
    desc: dict[str, Any] = {
        "operator": op,
        "name": name,
        "activation_dtype": "S8",
        "weight_dtype": "S8",
        "input_shape": [1, layer.h, layer.w, layer.cin],
        "strides": [layer.sh, layer.sw],
        "padding": layer.padding,
        "dilation": [layer.dh, layer.dw],
        "shape_seed": seed,
        "expected_route": route,
        **knobs,
    }
    if op == "Convolve":
        desc["filter_shape"] = [layer.kh, layer.kw, layer.cin, layer.cout]
    else:
        desc["filter_shape"] = [layer.kh, layer.kw, layer.cin, layer.mult]
        desc["depth_multiplier"] = layer.mult
    return desc


def sample_op(op: str, n: int, seed: int, cpu: str, workspace: int) -> list[dict[str, Any]]:
    """N descriptors for one op, cycling routes."""
    mve = get_cpu_profile(cpu).has_mve
    gen = GENERATORS[op]
    targets, draw = gen.routes(mve), gen.draw
    # One stream per op keeps ops independent.
    rng = np.random.default_rng([seed, gen.stream])
    cases = []
    for index in range(n):
        target = targets[index % len(targets)]
        for _ in range(MAX_TRIES):
            layer = draw(rng, target)
            oh, ow = layer.out_hw()
            if oh < 1 or ow < 1:
                continue
            route = layer_route(op, layer, mve)
            hit = route == target or (target == DW_AS_CONV and route in CONV_ROUTES)
            if hit and layer_macs(op, layer) <= MAX_MACS and footprint(op, layer) <= workspace:
                break
        else:
            raise RuntimeError(f"No {op} shape for route {target}")
        name = f"rs{seed}_{gen.tag}_{index:03d}"
        cases.append(_descriptor(op, name, layer, _quant(rng, op, layer), seed, route))
    return cases


def min_workspace(cpu: str) -> int:
    """Smallest board workspace for cpu."""
    from helia_core_tester.hardware.boards import load_board_table

    boards = load_board_table()
    sizes = [b.workspace_bytes for b in boards if normalize_cpu(b.cpu) == normalize_cpu(cpu)]
    # FVP-only CPUs: smallest of all.
    return min(sizes or [b.workspace_bytes for b in boards])


def sample_cases(n: int, seed: int, cpu: str = "cortex-m55", ops: tuple[str, ...] = OPS) -> list[dict[str, Any]]:
    """N descriptors per op, deterministic in seed."""
    workspace = min_workspace(cpu)
    return [case for op in ops for case in sample_op(op, n, seed, cpu, workspace)]


def route_counts(cases: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Cases per route, per op."""
    counts: dict[str, Counter] = {}
    for case in cases:
        counts.setdefault(case["operator"], Counter())[case["expected_route"]] += 1
    return {op: dict(sorted(c.items())) for op, c in sorted(counts.items())}


def write_cases(root: Path, cases: list[dict[str, Any]], header: dict[str, Any], cpu: str) -> Path:
    """Write descriptors and summary; return descriptors dir."""
    shutil.rmtree(root, ignore_errors=True)
    descriptors = root / "descriptors"
    by_file: dict[str, list] = {}
    for case in cases:
        by_file.setdefault(GENERATORS[case["operator"]].descriptor_file, []).append(case)
    for relpath, docs in by_file.items():
        path = descriptors / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump_all(docs, sort_keys=False))
    ops = {case["operator"]: case["activation_dtype"] for case in cases}
    summary = {**header, "cpu": normalize_cpu(cpu), "cases": len(cases), "ops": dict(sorted(ops.items())),
               "routes": route_counts(cases)}
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Random shapes ({json.dumps(header)}): {json.dumps(summary['routes'])}")
    return descriptors


def prepare_shapes(repo_root: Path, n: int, seed: int, cpu: str, ops: tuple[str, ...] = OPS) -> Path:
    """Sample and write cases; return descriptors dir."""
    root = artifacts_root(repo_root) / "random_shapes" / f"s{seed}" / normalize_cpu(cpu)
    return write_cases(root, sample_cases(n, seed, cpu, ops), {"shape_seed": seed}, cpu)


def hidden_secret(raw: str | None = None) -> bytes:
    """Secret seed, by default from the environment."""
    secret = (os.environ.get(SECRET_ENV, "") if raw is None else raw).strip()
    if len(secret) < MIN_SECRET:
        raise ValueError(f"{SECRET_ENV} needs {MIN_SECRET}+ characters")
    return secret.encode()


def seed_commitment(secret: bytes) -> str:
    """Public hash of the secret."""
    return hashlib.sha256(b"hct-hidden-commit\0" + secret).hexdigest()


def hidden_cases(n: int, secret: bytes, cpu: str, ops: tuple[str, ...] = OPS) -> list[dict[str, Any]]:
    """Secret-seeded cases with opaque ids."""
    seed = int.from_bytes(hashlib.sha256(b"hct-hidden-seed\0" + secret).digest(), "big")
    cases = sample_cases(n, seed, cpu, ops)
    for case in cases:
        # Keyed hash hides seed and index.
        case["name"] = "h" + hmac.new(secret, case["name"].encode(), "sha256").hexdigest()[:HIDDEN_ID_HEX]
        del case["shape_seed"]
    return cases


def check_hidden_paths(hidden_dir: Path, repo_root: Path, cpu: str, suite: str = "int") -> None:
    """Refuse hidden writes that could touch the checkout."""
    from helia_core_tester.core.path_layout import generated_tests_dir, generation_report_dir

    root, repo = Path(hidden_dir).resolve(), Path(repo_root).resolve()
    # Deleting either would reach the other.
    if root.is_relative_to(repo) or repo.is_relative_to(root):
        raise ValueError(f"{root} must neither hold nor sit inside the checkout")
    for target in (
        generated_tests_dir(root, normalize_cpu(cpu), suite=suite),
        hidden_root(root, cpu),
        generation_report_dir(root, normalize_cpu(cpu), suite=suite),
    ):
        _refuse_links(root, target)


def _refuse_links(root: Path, target: Path) -> None:
    """Refuse any link from root through target."""
    path = root
    for part in target.relative_to(root).parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"{path} is a symlink; hidden trees allow none")
    if not target.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(target, followlinks=False):
        for name in (*dirnames, *filenames):
            entry = Path(dirpath) / name
            if entry.is_symlink():
                raise ValueError(f"{entry} is a symlink; hidden trees allow none")
            # In-place writes would reach the other name.
            if entry.is_file() and entry.lstat().st_nlink > 1:
                raise ValueError(f"{entry} is hard-linked; hidden trees allow none")


def hidden_root(hidden_dir: Path, cpu: str) -> Path:
    """Hidden descriptors and summary for cpu."""
    # The hidden dir mirrors a tester root.
    return artifacts_root(hidden_dir) / "random_shapes" / normalize_cpu(cpu)


def prepare_hidden(hidden_dir: Path, n: int, cpu: str, ops: tuple[str, ...] = OPS) -> Path:
    """Write hidden cases; return descriptors dir."""
    secret = hidden_secret()
    root = hidden_root(hidden_dir, cpu)
    return write_cases(root, hidden_cases(n, secret, cpu, ops), {"seed_commitment": seed_commitment(secret)}, cpu)
