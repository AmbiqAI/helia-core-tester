"""
Simplified base operation class for TFLite model generation.
All operations inherit from this and implement build_keras_model().
"""

import json
import numpy as np
from typing import Callable, Dict, Any, List, Optional, Sequence, Tuple, Iterator
from abc import ABC, abstractmethod
from pathlib import Path
import jinja2

from helia_core_tester.core.discovery import find_tester_templates_dir
from helia_core_tester.generation.io.dtypes import (
    descriptor_dtype_to_c_type,
    descriptor_dtype_to_litert_dtype,
    get_resolved_tensor_dtype,
    resolve_comparison,
)
from helia_core_tester.generation.ops.catalog import template_candidates
from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

_JINJA2_ENV_CACHE: Dict[str, jinja2.Environment] = {}

try:
    import tensorflow as tf
except Exception:
    tf = None

# Context keys whose values are bulk C array literals (large multi-line
# strings holding the full input/expected-output tensor data as C source
# text). These are already present verbatim in the generated .c file, so the
# sidecar deliberately excludes them by suffix convention to stay small and
# avoid duplicating tensor payload data across two artifacts.
_BULK_ARRAY_FIELD_SUFFIXES: Tuple[str, ...] = (
    "_data_array",
    "_array_str",
    "expected_output_array",
)


def _is_bulk_array_field(key: str, value: Any) -> bool:
    if any(key.endswith(suffix) for suffix in _BULK_ARRAY_FIELD_SUFFIXES):
        return True
    # Fallback heuristic: any very long string value is almost certainly a
    # C array literal blob rather than a meaningful scalar/name field.
    return isinstance(value, str) and len(value) > 500


def _is_json_serializable(value: Any) -> bool:
    # allow_nan=False rejects NaN/Infinity, which Python emits as bare JSON
    # literals that no strict parser accepts. A non-finite scalar reaching the
    # context (the +/-INFINITY "no clamp" activation bounds do) is dropped from
    # the sidecar rather than written as `Infinity`. Nothing is lost: the
    # rendered form the .c file uses travels alongside as a string, e.g.
    # out_activation_min is dropped while out_activation_min_literal keeps
    # "-INFINITY".
    try:
        json.dumps(value, allow_nan=False)
        return True
    except (TypeError, ValueError):
        return False


class OperationBase(ABC):
    """
    Base class for all CMSIS-NN operations.
    
    Each operation must implement:
    1. build_keras_model() - Construct the Keras model
    2. convert_to_tflite() - Convert model to TFLite with operation-specific quantization
    """
    
    def __init__(self, desc: Dict[str, Any], seed: int = 1, target_cpu: str = "cortex-m55"):
        """
        Initialize operation with descriptor and random seed.
        
        Args:
            desc: YAML descriptor dictionary
            seed: Random seed for reproducible generation
        """
        self.desc = desc
        self.seed = seed
        self.target_cpu = target_cpu
        self.rng = np.random.default_rng(seed)
        self._litert_interpreter = None
        self._tflite_path = None
        self._input_mode_consumed = False
        self._nonfinite_policy_applied = False

    @abstractmethod
    def build_keras_model(self):
        """
        Build the Keras model for this operation.
        
        Returns:
            Keras model ready for TFLite conversion
        """
        pass

    def needs_keras_model(self) -> bool:
        """Return True if build_keras_model should be called for conversion."""
        return True

    def allow_no_tflite(self) -> bool:
        """Return True if this op can generate C/H without a .tflite."""
        return False

    def activation_name(self) -> str:
        """Return the normalized descriptor activation name."""
        return str(self.desc.get("activation", "NONE")).upper()

    def resolved_tensor_dtypes(self) -> Dict[str, str]:
        """Return descriptor tensor dtypes after normalization."""
        return dict(self.desc.get("resolved_tensor_dtypes", {}))

    def tensor_dtype(self, role: str, default: Optional[str] = None) -> str:
        """Return the resolved descriptor dtype for a tensor role."""
        return get_resolved_tensor_dtype(self.desc, role, default=default)

    def tensor_c_type(self, role: str, default: Optional[str] = None) -> str:
        """Return the C type for a tensor role."""
        return descriptor_dtype_to_c_type(self.tensor_dtype(role, default=default))

    def tensor_litert_dtype(self, role: str, default: Optional[str] = None) -> str:
        """Return the LiteRT builder dtype name for a tensor role."""
        return descriptor_dtype_to_litert_dtype(self.tensor_dtype(role, default=default))

    def comparison_config(self) -> Dict[str, Any]:
        """Return the resolved comparison configuration for descriptor outputs."""
        return resolve_comparison(self.desc, self.resolved_tensor_dtypes())

    def primary_execution_dtype(self) -> str:
        """Return the primary execution dtype for unary ops and activation-like kernels."""
        resolved = self.resolved_tensor_dtypes()
        if "input" in resolved:
            return resolved["input"]
        return str(self.desc.get("activation_dtype", "S8")).upper()

    def _write_tflite_bytes(self, out_path: str | Path, model_bytes: bytes) -> None:
        """Write converted LiteRT bytes to disk."""
        Path(out_path).write_bytes(model_bytes)

    def _seeded_rng(self) -> np.random.Generator:
        """Return a temporary deterministic RNG seeded from the op seed."""
        return np.random.default_rng(self.seed)

    # Which tokens a sweep may request. A descriptor names a subset because the token
    # set is a per-kernel contract question: ns-cmsis-nn documents NaN behaviour for
    # some kernels, declares it unsupported for others (sigmoid), and destroys it by
    # design on others (MVE tanh), so a kernel that disclaims NaN must not be handed one.
    NONFINITE_TOKEN_VALUES: Dict[str, float] = {
        "nan": float("nan"),
        "inf": float("inf"),
        "-inf": float("-inf"),
    }
    # Ordered so index i of a swept tensor always holds the same token for a given token
    # set, which is what lets a failing element index name the token that broke without
    # re-reading the generated array.
    DEFAULT_NONFINITE_TOKENS: Tuple[str, ...] = ("nan", "inf", "-inf")

    def input_mode(self) -> str:
        """Return the descriptor input-generation mode."""
        mode = str(self.desc.get("input_mode", "uniform")).strip().lower()
        if mode not in ("", "uniform"):
            suite = str(
                self.desc.get("_descriptor_suite") or self.desc.get("suite") or ""
            ).strip().lower()
            if suite != "float":
                raise ValueError(
                    f"input_mode {mode!r} is float-suite only, but descriptor "
                    f"{self.desc.get('name')!r} declares suite {suite or 'default'!r}"
                )
        return mode

    def nonfinite_tokens(self) -> Tuple[str, ...]:
        """Return the ordered token names this descriptor's sweep writes."""
        requested = self.desc.get("nonfinite_tokens")
        if requested is None:
            return self.DEFAULT_NONFINITE_TOKENS
        tokens = tuple(str(token).strip().lower() for token in requested)
        if not tokens:
            raise ValueError("nonfinite_tokens must name at least one token")
        unknown = [token for token in tokens if token not in self.NONFINITE_TOKEN_VALUES]
        if unknown:
            raise ValueError(
                f"Unsupported nonfinite_tokens {unknown}; "
                f"known tokens are {sorted(self.NONFINITE_TOKEN_VALUES)}"
            )
        if len(set(tokens)) != len(tokens):
            raise ValueError(f"nonfinite_tokens must be unique, got {list(tokens)}")
        return tokens

    def nonfinite_policy(self) -> str:
        """Return how a non-finite sweep's golden is compared: 'strict' or 'mask'."""
        declared = self.desc.get("nonfinite_policy")
        if declared is None:
            # No default: whether an uncontracted non-finite output may be pinned as a
            # golden is a per-kernel contract question, and a silent 'strict' default
            # answers it by accident for every descriptor that forgets to.
            if self.input_mode() == "nonfinite_sweep":
                raise ValueError(
                    f"Descriptor {self.desc.get('name')!r} uses input_mode "
                    "'nonfinite_sweep' without nonfinite_policy; declare 'strict' "
                    "(the header contracts the value, or the op is pure data movement) "
                    "or 'mask' (it does not)"
                )
            return "strict"
        policy = str(declared).strip().lower()
        if policy not in ("strict", "mask"):
            raise ValueError(
                f"Unsupported nonfinite_policy {policy!r}; expected 'strict' or 'mask'"
            )
        if policy != "strict" and self.input_mode() != "nonfinite_sweep":
            raise ValueError(
                f"Descriptor {self.desc.get('name')!r} sets nonfinite_policy {policy!r} "
                "without input_mode 'nonfinite_sweep'; the policy only governs the "
                "comparison of a swept case"
            )
        return policy

    def apply_nonfinite_policy(
        self,
        output_data: np.ndarray,
        *,
        reference: Callable[[Sequence[np.ndarray]], np.ndarray],
        inputs: Sequence[np.ndarray],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Return the golden to emit and the template context the policy adds.

        Under 'mask' the don't-care set is every lane whose reference output is
        non-finite, plus every lane a swept token can reach. The two are not the
        same set: a max-pool window holding +Inf reduces to a finite value at
        every other lane of that window's row, and the kernel's own reduction
        order decides which of them the token influences, so pinning those lanes
        would encode an uncontracted value just as much as pinning the Inf lane.
        Reachability is measured, not declared: the reference is re-run with the
        token positions replaced by finite probes and any output lane that moves
        between probes is reachable. Three probes rather than two because two can
        coincide on a symmetric op (|+p| == |-p| for Abs), which would read as
        unreachable. This is why the caller passes a reference callable instead
        of a precomputed output -- it makes the window, row and receptive-field
        cases fall out of whatever reference path the op already uses.

        The emitted golden carries 0.0 in the masked lanes so the golden stays
        finite; the input arrays still carry the tokens. Under 'strict' the
        reference is emitted unchanged and nothing is added.
        """
        if self.nonfinite_policy() != "mask":
            return output_data, {}
        if not np.issubdtype(output_data.dtype, np.floating):
            raise ValueError(
                "nonfinite_policy 'mask' needs a float output to classify, got dtype "
                f"{output_data.dtype}"
            )
        from helia_core_tester.generation.utils.template_context import TemplateContextBuilder

        reachable = self._nonfinite_reachable_lanes(reference, inputs, output_data.shape)
        mask = ~np.isfinite(output_data) | reachable
        if mask.all():
            raise ValueError(
                f"Descriptor {self.desc.get('name')!r} masks all {mask.size} output "
                "lanes, so the case would assert nothing beyond SUCCESS; widen the "
                "tensor or move the tokens so finite lanes survive"
            )
        masked_golden = np.where(mask, output_data.dtype.type(0.0), output_data)
        self._nonfinite_policy_applied = True
        return masked_golden, {
            "nonfinite_mask_array_str": TemplateContextBuilder.format_array_as_c_literal(
                mask.astype(np.uint8)
            ),
            "nonfinite_masked_lanes": int(mask.sum()),
        }

    def _nonfinite_reachable_lanes(
        self,
        reference: Callable[[Sequence[np.ndarray]], np.ndarray],
        inputs: Sequence[np.ndarray],
        output_shape: Tuple[int, ...],
    ) -> np.ndarray:
        """Return the boolean lanes a swept token can influence."""
        if reference is None or inputs is None:
            raise ValueError(
                f"Descriptor {self.desc.get('name')!r} uses nonfinite_policy 'mask', "
                "which needs the op's reference callable and its input tensors to "
                "measure which output lanes the tokens reach"
            )
        arrays = [np.asarray(arr) for arr in inputs]
        token_sites = [
            (index, ~np.isfinite(arr))
            for index, arr in enumerate(arrays)
            if np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all()
        ]
        if not token_sites:
            raise ValueError(
                f"Descriptor {self.desc.get('name')!r} uses nonfinite_policy 'mask' "
                "but no input handed to apply_nonfinite_policy() carries a token; "
                "the inputs passed are not the ones the sweep wrote"
            )

        # Probes sit just outside the sampled range so they are the extreme of every
        # window or row they enter, which is what makes a reducing op's output move.
        magnitude = 1.0
        for arr in arrays:
            if not np.issubdtype(arr.dtype, np.floating):
                continue
            finite = arr[np.isfinite(arr)]
            if finite.size:
                magnitude = max(magnitude, float(np.max(np.abs(finite.astype(np.float64)))))
        probes = (magnitude + 1.0, -(magnitude + 1.0), magnitude + 2.0)

        probe_outputs = []
        for probe in probes:
            probed = [np.array(arr, copy=True) for arr in arrays]
            for index, sites in token_sites:
                probed[index][sites] = probed[index].dtype.type(probe)
            result = np.asarray(reference(probed))
            probe_outputs.append(result.astype(np.float64).reshape(output_shape))

        reachable = np.zeros(output_shape, dtype=bool)
        for i in range(len(probe_outputs)):
            for j in range(i + 1, len(probe_outputs)):
                left, right = probe_outputs[i], probe_outputs[j]
                # Two NaN lanes are the same verdict, not a difference; every other
                # inequality counts, including a lane that is NaN under one probe only.
                reachable |= (left != right) & ~(np.isnan(left) & np.isnan(right))
        if not reachable.any():
            # A token that moves no output lane means the probes were swallowed before
            # they reached the output -- a two-sided activation clamp saturates all
            # three to the same bound -- so the measurement is not evidence that the
            # token is confined, and the mask it produces would be built from the
            # reference lanes alone.
            raise ValueError(
                f"Descriptor {self.desc.get('name')!r} uses nonfinite_policy 'mask' but "
                "no output lane moved between the finite probes, so reachability could "
                "not be measured; place the tokens where the probes stay inside the "
                "activation clamp, widen that clamp, or use 'strict' if the kernel does "
                "contract the value there"
            )
        return reachable

    def nonfinite_sweep_positions(self) -> Tuple[int, ...]:
        """Return the flat indices the non-finite sweep overwrites.

        The default is the leading run 0..k-1. `nonfinite_positions` overrides it
        element for element with `nonfinite_tokens`, which is the only way to reach
        a placement the leading run cannot express: one token per reduction group
        rather than all of them in group 0, or a pooling window that the padding
        only partly covers.
        """
        tokens = self.nonfinite_tokens()
        requested = self.desc.get("nonfinite_positions")
        if requested is None:
            return tuple(range(len(tokens)))
        if self.input_mode() != "nonfinite_sweep":
            raise ValueError(
                f"Descriptor {self.desc.get('name')!r} sets nonfinite_positions without "
                "input_mode 'nonfinite_sweep'; the positions only place a swept token"
            )
        positions = tuple(int(position) for position in requested)
        if len(positions) != len(tokens):
            raise ValueError(
                f"Descriptor {self.desc.get('name')!r} lists {len(positions)} "
                f"nonfinite_positions for {len(tokens)} nonfinite_tokens; the two are "
                "paired element for element"
            )
        if len(set(positions)) != len(positions):
            raise ValueError(
                f"nonfinite_positions must be unique, got {list(positions)}"
            )
        if any(position < 0 for position in positions):
            raise ValueError(
                f"nonfinite_positions must be non-negative flat indices, got {list(positions)}"
            )
        return positions

    def _apply_nonfinite_sweep(self, arr: np.ndarray) -> np.ndarray:
        """Overwrite the leading elements of a float tensor with the non-finite tokens."""
        if not np.issubdtype(arr.dtype, np.floating):
            raise ValueError(
                f"input_mode 'nonfinite_sweep' requires a float tensor, got dtype {arr.dtype}"
            )
        tokens = self.nonfinite_tokens()
        positions = self.nonfinite_sweep_positions()
        if arr.size < max(positions) + 1:
            raise ValueError(
                f"input_mode 'nonfinite_sweep' needs at least {max(positions) + 1} elements, "
                f"got shape {tuple(arr.shape)}"
            )
        # Finite neighbours from the uniform draw are kept deliberately: a mixed tensor
        # distinguishes a kernel that corrupts the lanes around a NaN (or predicates a
        # whole vector off) from one that merely mishandles the token itself.
        swept = arr.copy()
        flat = swept.reshape(-1)
        flat[list(positions)] = np.asarray(
            [self.NONFINITE_TOKEN_VALUES[token] for token in tokens], dtype=arr.dtype
        )
        return swept

    def _maybe_apply_input_mode(self, arr: np.ndarray) -> np.ndarray:
        """Apply the descriptor input-generation mode to a freshly sampled tensor."""
        mode = self.input_mode()
        if mode in ("", "uniform"):
            return arr
        if mode == "nonfinite_sweep":
            swept = self._apply_nonfinite_sweep(arr)
            self._input_mode_consumed = True
            return swept
        raise ValueError(f"Unsupported input_mode: {mode!r}")

    def assert_input_mode_consumed(self) -> None:
        """Fail if a non-uniform input_mode was requested but no tensor was ever swept.

        Several ops sample outside the shared helpers, so a descriptor can carry
        input_mode and generate an ordinary finite case that looks green while testing
        nothing. Silence there is worse than a generation failure.
        """
        if self.input_mode() in ("", "uniform"):
            return
        if not self._input_mode_consumed:
            raise ValueError(
                f"Descriptor {self.desc.get('name')!r} requested input_mode "
                f"{self.input_mode()!r} but {type(self).__name__} never applied it; "
                "the op samples its input outside the shared helpers"
            )
        # Same failure shape one step later: a mask-policy case whose op never asked for
        # the mask would silently fall back to strict and pin an uncontracted value.
        if self.nonfinite_policy() == "mask" and not self._nonfinite_policy_applied:
            raise ValueError(
                f"Descriptor {self.desc.get('name')!r} requested nonfinite_policy 'mask' "
                f"but {type(self).__name__} never called apply_nonfinite_policy()"
            )

    def _sample_uniform(
        self,
        shape: Tuple[int, ...] | list[int],
        *,
        low: float = -1.0,
        high: float = 1.0,
        dtype=np.float32,
    ) -> np.ndarray:
        """Sample reproducible uniform input data without mutating self.rng state."""
        sampled = self._seeded_rng().uniform(low, high, size=tuple(shape)).astype(dtype)
        return self._maybe_apply_input_mode(sampled)

    def _sample_dual_uniform_inputs(
        self,
        shape_1: Tuple[int, ...] | list[int],
        shape_2: Tuple[int, ...] | list[int],
        *,
        low: float = -1.0,
        high: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample two reproducible uniform input tensors without mutating self.rng state."""
        rng = self._seeded_rng()
        input_1 = rng.uniform(low, high, size=tuple(shape_1)).astype(np.float32)
        input_2 = rng.uniform(low, high, size=tuple(shape_2)).astype(np.float32)
        # Only the left operand is swept so the right stays finite: that keeps each
        # element a single-token propagation check (NaN+x, Inf*x) rather than a mix whose
        # result would be ambiguous about which operand drove it.
        return self._maybe_apply_input_mode(input_1), input_2

    # "Near zero" is absolute, not a fraction of the operand's own range: the
    # sign boundary the packed/scalar paths and the branchy kernels (PReLU,
    # min/max) split on sits at a fixed post-offset value, so an operand that
    # only comes within some fraction of its own large maximum is still nowhere
    # near it.
    NEAR_ZERO_MAX_ABS = 1

    # An operand with fewer than this many elements cannot hold a negative, a
    # near-zero and a positive value at once. Broadcast scalars and one- or
    # two-element rows are therefore out of scope for the span rule; the
    # operand they broadcast against still has to satisfy it.
    SIGN_SPAN_MIN_ELEMENTS = 3

    # Descriptor key that waives the span rule for an operand whose one-signed
    # data is the point of the case (a PReLU alpha baked into the model as a
    # positive slope, a relu-shaped range whose zero point pins the whole
    # domain to one side). It maps operand label -> reason, and the reason is
    # required: see README "Operand sign span" and issue #81 property 2.
    SIGN_SPAN_EXEMPT_KEY = "operand_sign_span_exempt"

    # Operand labels this operator submits to the span rule. A waiver keyed on
    # anything else waives nothing and reads as covering a gap it never
    # touches, so the labels are declared per operator and both the waiver keys
    # and the call sites are checked against them.
    SIGN_SPAN_OPERANDS: Tuple[str, ...] = ()

    _SIGN_SPAN_DTYPE_BOUNDS = {"int8": (-128, 127), "int16": (-32768, 32767)}

    def _sign_span_exemptions(self) -> Dict[str, str]:
        """Validated operand -> reason map from the span opt-out key."""
        exempt = self.desc.get(self.SIGN_SPAN_EXEMPT_KEY)
        if exempt is None:
            return {}
        if not isinstance(exempt, dict) or not exempt:
            raise ValueError(
                f"'{self.desc.get('name')}': {self.SIGN_SPAN_EXEMPT_KEY} must map each "
                f"intentionally one-signed operand to the reason it stays that way, "
                f"e.g. {{alpha: the model constant is a positive PReLU slope}}"
            )
        known = ", ".join(self.SIGN_SPAN_OPERANDS) or "(none)"
        for operand, reason in exempt.items():
            if str(operand) not in self.SIGN_SPAN_OPERANDS:
                raise ValueError(
                    f"'{self.desc.get('name')}': {self.SIGN_SPAN_EXEMPT_KEY}[{operand}] names an "
                    f"operand {type(self).__name__} never submits to the span rule, so it waives "
                    f"nothing; the checked operands are {known}"
                )
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(
                    f"'{self.desc.get('name')}': {self.SIGN_SPAN_EXEMPT_KEY}[{operand}] must be "
                    f"a non-empty reason string explaining why the operand is one-signed"
                )
        return {str(k): str(v) for k, v in exempt.items()}

    def _enforce_int_operand_sign_span(
        self,
        operands: Sequence[Tuple[str, np.ndarray, Any]],
        steerable: Sequence[str] = (),
    ) -> List[np.ndarray]:
        """
        Make every int operand span negative, near-zero and positive AFTER its
        input offset is applied (value - zero_point), and fail generation when
        that is impossible without a declared reason.

        A one-signed operand cannot discriminate the sign-dependent kernel
        paths: the packed DSP loop of ns-cmsis-nn#343 dropped the sign of
        value + input_offset, and PReLU/min/max branch on it directly. Uniform
        [-1, 1] float data plus a TFLite zero point does not guarantee the
        span, so it is enforced rather than assumed (issue #81 property 2).

        ``steerable`` names the operands that are runtime inputs whose data the
        generator owns, i.e. the ones whose golden is recomputed from the array
        returned here. Two kinds of operand stay out of it and are check-only:
        one baked into the TFLite model (a PReLU alpha), because the reference
        interpreter would still use the model's copy; and one the descriptor
        pins explicitly (``hint.extras.input_values``), because the pinned
        values are the case. Either can only be waived with
        ``operand_sign_span_exempt``.

        Returns the operand arrays in input order, steered where needed.
        """
        undeclared = sorted({str(label) for label, _, _ in operands} - set(self.SIGN_SPAN_OPERANDS))
        if undeclared:
            raise ValueError(
                f"'{self.desc.get('name')}': {type(self).__name__}.SIGN_SPAN_OPERANDS does not "
                f"declare {', '.join(undeclared)}; waiver keys are validated against it, so an "
                f"undeclared operand could never be exempted"
            )
        exemptions = self._sign_span_exemptions()
        steerable_set = set(steerable)
        result: List[np.ndarray] = []
        for label, data, zero_point in operands:
            array = np.asarray(data)
            flat = array.reshape(-1)
            zp = int(zero_point)
            if label in exemptions or flat.size < self.SIGN_SPAN_MIN_ELEMENTS:
                result.append(array)
                continue
            missing = self._sign_span_gaps(flat, zp)
            if not missing:
                result.append(array)
                continue
            attempted = label in steerable_set
            if attempted:
                # Plant only the missing regions: overwriting all three throws
                # away the elements that already carried the ones present, and
                # on a short operand those are the case. A targeted plant can
                # still consume the sole carrier of a region it was relying on,
                # so the full triple stays as the fallback.
                steered = self._steer_int_operand_sign_span(array, zp, missing)
                if self._sign_span_gaps(steered.reshape(-1), zp):
                    steered = self._steer_int_operand_sign_span(array, zp)
                if not self._sign_span_gaps(steered.reshape(-1), zp):
                    result.append(steered)
                    continue
            if attempted:
                why = (
                    f"steering cannot reach them: clipped to the dtype range, no representable "
                    f"value lands on the missing side of zero_point {zp}"
                )
            else:
                why = (
                    "the operand is model-baked or descriptor-pinned, so the generator may not "
                    "steer it"
                )
            raise ValueError(
                f"'{self.desc.get('name')}': {label} post-offset values "
                f"(n {flat.size}, min {int(flat.min()) - zp}, max {int(flat.max()) - zp}, "
                f"zero_point {zp}) do not span {' and '.join(missing)}, and {why}; a "
                f"one-signed operand cannot discriminate the sign-dependent "
                f"kernel paths (ns-cmsis-nn#343). Set {self.SIGN_SPAN_EXEMPT_KEY}[{label}] to "
                f"the reason it must stay one-signed."
            )
        return result

    @classmethod
    def _sign_span_gaps(cls, flat: np.ndarray, zero_point: int) -> List[str]:
        """Which of negative / near-zero / positive the post-offset data lacks."""
        post = flat.astype(np.int64) - zero_point
        gaps = []
        if not (post < 0).any():
            gaps.append("negative")
        if not (np.abs(post) <= cls.NEAR_ZERO_MAX_ABS).any():
            gaps.append("near-zero")
        if not (post > 0).any():
            gaps.append("positive")
        return gaps

    @classmethod
    def _steer_int_operand_sign_span(
        cls,
        array: np.ndarray,
        zero_point: int,
        missing: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """
        Plant post-offset values for the regions in ``missing`` (default: all
        three), at half the operand's own magnitude so the steered values stay
        inside the range the case was written for. Deterministic and
        independent of the RNG stream, so a re-run reproduces it exactly.

        One element is replaced per planted region, so an operand that lacks
        only the near-zero boundary keeps every other element it had.
        """
        bounds = cls._SIGN_SPAN_DTYPE_BOUNDS.get(str(array.dtype))
        if bounds is None:
            return array
        qmin, qmax = bounds
        post = array.reshape(-1).astype(np.int64) - zero_point
        step = max(1, int(np.abs(post).max()) // 2)
        by_region = {
            "negative": zero_point - step,
            "near-zero": zero_point,
            "positive": zero_point + step,
        }
        wanted = list(missing) if missing else list(by_region)
        planted = np.clip([by_region[region] for region in wanted], qmin, qmax)
        # Overwrite the elements closest to the offset rather than the leading
        # ones: a full-scale element carries saturation coverage that a
        # mid-range element does not, and on a 3- or 4-element operand the head
        # is the whole case. Stable sort keeps the choice reproducible.
        targets = np.argsort(np.abs(post), kind="stable")[: planted.size]
        steered = array.copy().reshape(-1)
        steered[targets] = planted.astype(array.dtype)
        return steered.reshape(array.shape)

    @staticmethod
    def _quant_param_scalar(quant_params: Optional[Dict[str, Any]], key: str, default: float | int) -> float | int:
        """Extract a scalar quantization value from LiteRT quantization metadata."""
        if not quant_params:
            return default
        value = quant_params.get(key, default)
        if isinstance(value, (list, tuple, np.ndarray)):
            return default if len(value) == 0 else value[0]
        return value
    
    def _apply_activation_quantization(self, converter) -> None:
        """Set converter for activation-only quantization (S8 or S16) from descriptor."""
        if tf is None:
            raise ImportError("tensorflow is required for TFLite conversion")
        activation_dtype = self.primary_execution_dtype()
        if activation_dtype == "FP32":
            converter.optimizations = []
            return
        if activation_dtype == "FP16":
            converter.optimizations = []
            converter.target_spec.supported_types = [tf.float16]
            return
        if activation_dtype == "S8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        elif activation_dtype == "S16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
            ]
            converter.inference_input_type = tf.int16
            converter.inference_output_type = tf.int16
        else:
            raise NotImplementedError(f"Unsupported activation_dtype: {activation_dtype}")

    def _representative_dataset_gen(self) -> Iterator[list]:
        """Yield representative batches from descriptor (single- or dual-input)."""
        for _ in range(100):
            if "input_shape" in self.desc:
                inp = self.rng.uniform(
                    -1.0, 1.0, size=self.desc["input_shape"]
                ).astype(np.float32)
                yield [inp]
            elif "input_1_shape" in self.desc and "input_2_shape" in self.desc:
                inp1 = self.rng.uniform(
                    -1.0, 1.0, size=self.desc["input_1_shape"]
                ).astype(np.float32)
                inp2 = self.rng.uniform(
                    -1.0, 1.0, size=self.desc["input_2_shape"]
                ).astype(np.float32)
                yield [inp1, inp2]
            else:
                shape = self.desc.get("input_shape", [1, 1, 1, 1])
                yield [self.rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)]

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        """
        Convert Keras model to TFLite with activation quantization.
        Override in subclasses for weight/operation-specific quantization.
        """
        if tf is None:
            raise ImportError("tensorflow is required for TFLite conversion")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        self._apply_activation_quantization(converter)
        converter.representative_dataset = self._representative_dataset_gen
        tflite_model = converter.convert()
        self._write_tflite_bytes(out_path, tflite_model)

    def _convert_with_activation_quantization(
        self,
        model,
        out_path: str,
        *,
        input_type=None,
        output_type=None,
        rep_seed: int,
    ) -> None:
        """
        Convert Keras model to TFLite with activation quantization for S8/S16.
        Allows overriding inference input/output types (e.g., float input or bool output).
        """
        if tf is None:
            raise ImportError("tensorflow is required for TFLite conversion")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        activation_dtype = self.primary_execution_dtype()
        if activation_dtype == "FP32":
            converter.optimizations = []
            if input_type is not None:
                converter.inference_input_type = input_type
            if output_type is not None:
                converter.inference_output_type = output_type
        elif activation_dtype == "FP16":
            converter.optimizations = []
            converter.target_spec.supported_types = [tf.float16]
            if input_type is not None:
                converter.inference_input_type = input_type
            if output_type is not None:
                converter.inference_output_type = output_type
        elif activation_dtype == "S8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int8]
            converter.inference_input_type = input_type or tf.int8
            converter.inference_output_type = output_type or tf.int8
        elif activation_dtype == "S16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
            ]
            converter.inference_input_type = input_type or tf.int16
            converter.inference_output_type = output_type or tf.int16
        else:
            raise NotImplementedError(f"Unsupported activation dtype for conversion: {activation_dtype}")

        def representative_data_gen():
            if activation_dtype in {"FP32", "FP16"}:
                return
            for _ in range(100):
                if "input_shape" in self.desc:
                    inputs = self.rng.uniform(-1.0, 1.0, size=self.desc["input_shape"]).astype(np.float32)
                    yield [inputs]
                elif "input_1_shape" in self.desc and "input_2_shape" in self.desc:
                    inputs1 = self.rng.uniform(-1.0, 1.0, size=self.desc["input_1_shape"]).astype(np.float32)
                    inputs2 = self.rng.uniform(-1.0, 1.0, size=self.desc["input_2_shape"]).astype(np.float32)
                    yield [inputs1, inputs2]
                else:
                    shape = self.desc.get("input_shape", [1, 1, 1, 1])
                    yield [self.rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)]
        if activation_dtype not in {"FP32", "FP16"}:
            converter.representative_dataset = representative_data_gen

        tflite_model = converter.convert()
        self._write_tflite_bytes(out_path, tflite_model)
    
    def load_litert_interpreter(self, tflite_path: str):
        """
        Load LiteRT interpreter from .tflite file.
        
        Args:
            tflite_path: Path to .tflite file
            
        Returns:
            LiteRT interpreter instance
        """
        from helia_core_tester.generation.utils.litert_utils import load_litert_interpreter
        
        if self._tflite_path != tflite_path or self._litert_interpreter is None:
            interpreter = load_litert_interpreter(tflite_path)
            self._litert_interpreter = interpreter
            self._tflite_path = tflite_path
        
        return self._litert_interpreter
    
    def load_litert_model(self, tflite_path: str, subgraph_index: int = 0):
        """
        Load TFLite model using LiteRT schema.
        
        Args:
            tflite_path: Path to .tflite file
            subgraph_index: Index of subgraph to use (default: 0)
            
        Returns:
            Tuple of (model, subgraph) from LiteRT schema
        """
        from helia_core_tester.generation.utils.litert_utils import load_litert_model
        
        if self._tflite_path != tflite_path or not hasattr(self, '_litert_model'):
            model, subgraph = load_litert_model(tflite_path, subgraph_index)
            self._litert_model = model
            self._litert_subgraph = subgraph
            self._tflite_path = tflite_path
        
        return self._litert_model, self._litert_subgraph
    
    def extract_quantization_params(self, tflite_path: str) -> Dict[str, Any]:
        """
        Extract quantization parameters from TFLite model using LiteRT schema.
        
        Args:
            tflite_path: Path to .tflite file (required)
            
        Returns:
            Dictionary with quantization parameters for input, output, and weights
        """
        from helia_core_tester.generation.utils.litert_utils import (
            load_litert_model, get_operator_tensors_from_litert
        )
        
        model, subgraph = load_litert_model(tflite_path)
        
        # Get first operator's tensors
        if len(subgraph.operators) == 0:
            raise ValueError("No operators found in model")
        
        op_tensors = get_operator_tensors_from_litert(model, subgraph, 0)
        
        # Get input quantization (first input tensor)
        input_quant = None
        if op_tensors['inputs']:
            input_quant = op_tensors['inputs'][0]['quantization']
        
        # Get output quantization (first output tensor)
        output_quant = None
        if op_tensors['outputs']:
            output_quant = op_tensors['outputs'][0]['quantization']
        
        # Get weight quantization (from weights tensor)
        weight_quant = None
        if op_tensors['weights'] is not None:
            # Find the weight tensor in inputs
            for input_tensor_info in op_tensors['inputs']:
                if input_tensor_info['data'] is not None and len(input_tensor_info['shape']) > 1:
                    weight_quant = input_tensor_info['quantization']
                    break
        
        return {
            'input': input_quant or {'scale': 1.0, 'zero_point': 0, 'per_channel': False},
            'output': output_quant or {'scale': 1.0, 'zero_point': 0, 'per_channel': False},
            'weight': weight_quant or input_quant or {'scale': 1.0, 'zero_point': 0, 'per_channel': False}
        }
    
    def extract_weights_biases(self, tflite_path: str) -> Dict[str, Optional[np.ndarray]]:
        """
        Extract weights and biases from TFLite model using LiteRT schema.
        
        Args:
            tflite_path: Path to .tflite file (required)
            
        Returns:
            Dictionary with 'weights' and 'biases' keys
        """
        from helia_core_tester.generation.utils.litert_utils import (
            load_litert_model, extract_weights_biases_from_litert
        )
        
        model, subgraph = load_litert_model(tflite_path)
        return extract_weights_biases_from_litert(model, subgraph, 0)

    def load_primary_operator_tensors(self, tflite_path: str) -> Dict[str, Any]:
        """Load the first operator tensors from a LiteRT model."""
        from helia_core_tester.generation.utils.litert_utils import get_operator_tensors_from_litert

        model, subgraph = self.load_litert_model(tflite_path)
        if len(subgraph.operators) == 0:
            raise ValueError("No operators found in model")
        return get_operator_tensors_from_litert(model, subgraph, 0)
    
    def run_inference(self, tflite_path: str, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference on model using LiteRT interpreter.
        
        Args:
            tflite_path: Path to .tflite file
            input_data: Input data as numpy array
            
        Returns:
            Output data as numpy array
        """
        from helia_core_tester.generation.utils.litert_utils import run_inference_litert
        
        return run_inference_litert(tflite_path, input_data, subgraph_index=0)
    
    @staticmethod
    def _ensure_shape_tuple(shape: Any) -> Optional[Tuple[int, ...]]:
        """Normalize shape to tuple; return None if shape is None."""
        if shape is None:
            return None
        return tuple(shape)

    def _write_op_outputs(
        self,
        output_dir: Path,
        op_suffix: str,
        h_tpl: str,
        c_tpl: str,
        context: Dict[str, Any],
        cmake_context: Dict[str, Any],
    ) -> None:
        """Write .h, .c, CMakeLists.txt, and a structured JSON sidecar from templates.

        The sidecar (Phase 1 of the generation/bridge unification plan) is
        rendered from the *same* fully-resolved render context used to
        produce the .c file -- including the validation_*/comparison fields
        computed by TemplateContextBuilder.build_validation_context() -- so it
        is provably in sync with what actually got compiled/executed, instead
        of being re-derived independently (as the hardware bridge's 22
        hand-written regex extractors currently do). It is intended to
        eventually let the hardware perf-stream bridge look up kernel name,
        call args, and tensor roles/tolerance structurally rather than
        re-parsing generated C source.
        """
        includes_api_dir = output_dir / "includes"
        includes_api_dir.mkdir(parents=True, exist_ok=True)
        name = context["name"]
        h_content = self.render_template(h_tpl, context)
        (includes_api_dir / f"{name}_{op_suffix}.h").write_text(h_content)
        c_content, resolved_c_context = self.render_template(
            c_tpl, context, return_context=True
        )
        (output_dir / f"{name}_{op_suffix}.c").write_text(c_content)
        cmake_content = self.render_template("common/CMakeLists.txt.j2", cmake_context)
        (output_dir / "CMakeLists.txt").write_text(cmake_content)
        sidecar_content = self._build_generation_sidecar(op_suffix, resolved_c_context)
        (output_dir / f"{name}_{op_suffix}.sidecar.json").write_text(
            json.dumps(sidecar_content, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )

    def _build_generation_sidecar(self, op_suffix: str, resolved_context: Dict[str, Any]) -> Dict[str, Any]:
        """Build the structured sidecar payload for a generated test case from
        the fully-resolved context that actually rendered the .c file.

        Strips out bulk C array-literal fields (already present verbatim in
        the generated .c file, no need to duplicate/bloat the sidecar with
        them) and any non-JSON-serializable values. What remains is the
        single source of truth for this exact case's: kernel name, scalar
        call args, tensor dtypes/roles, and resolved comparison/tolerance
        policy -- the latter guaranteed identical to the hardware bridge
        manifest's since both now derive from dtypes.py's unified table (see
        the tolerance-policy-unification work in this same phase).
        """
        name = resolved_context.get("name", "")
        operator = str(self.desc.get("operator", ""))
        comparison = self.comparison_config()
        scalars: Dict[str, Any] = {}
        for key, value in resolved_context.items():
            if _is_bulk_array_field(key, value):
                continue
            if not _is_json_serializable(value):
                continue
            scalars[key] = value
        return {
            "name": name,
            "operator": operator,
            "op_suffix": op_suffix,
            "kernel_fn": resolved_context.get("kernel_fn"),
            "comparison": comparison,
            "resolved_tensor_dtypes": self.resolved_tensor_dtypes(),
            "scalars": scalars,
        }

    def generate_input_data(self) -> np.ndarray:
        """
        Generate test input data.
        
        Returns:
            Input data as numpy array
        """
        input_shape = self.desc.get('input_shape', [1, 1, 1, 1])
        return self._seeded_rng().integers(-32, 32, size=input_shape).astype(np.float32)
    
    def render_template(
        self, template_path: str, context: Dict[str, Any], return_context: bool = False
    ):
        """
        Render Jinja template with context. Environment is cached per template directory.

        If ``return_context`` is True, returns ``(rendered_text, resolved_context)``
        where ``resolved_context`` is the exact context dict used for rendering
        (post build_validation_context() resolution for .c templates) -- this is
        what the generation sidecar is built from, so the sidecar is guaranteed
        to reflect what was actually rendered rather than a re-derived copy.
        """
        template_dir = str(find_tester_templates_dir())
        if template_dir not in _JINJA2_ENV_CACHE:
            _JINJA2_ENV_CACHE[template_dir] = jinja2.Environment(
                loader=jinja2.FileSystemLoader(template_dir),
                trim_blocks=True,
                lstrip_blocks=True,
            )
        env = _JINJA2_ENV_CACHE[template_dir]
        operator = str(self.desc.get("operator", ""))
        render_context = dict(context)
        if template_path.endswith(".c.j2"):
            render_context = TemplateContextBuilder.build_validation_context(
                template_path,
                render_context,
                self.desc,
            )
        for candidate in template_candidates(operator, template_path):
            try:
                template = env.get_template(candidate)
                rendered = template.render(**render_context)
                return (rendered, render_context) if return_context else rendered
            except jinja2.TemplateNotFound:
                continue
        raise jinja2.TemplateNotFound(template_path)
    
    def get_tensor_shapes_from_litert(self, tflite_path: str) -> Dict[str, Any]:
        """
        Get input and output tensor shapes using LiteRT schema.
        
        Args:
            tflite_path: Path to .tflite file
            
        Returns:
            Dictionary with 'input_shape' and 'output_shape' keys
        """
        from helia_core_tester.generation.utils.litert_utils import (
            load_litert_model, get_operator_tensors_from_litert
        )
        
        model, subgraph = load_litert_model(tflite_path)
        
        if len(subgraph.operators) == 0:
            raise ValueError("No operators found in model")
        
        op_tensors = get_operator_tensors_from_litert(model, subgraph, 0)
        
        input_shape = op_tensors['inputs'][0]['shape'] if op_tensors['inputs'] else None
        output_shape = op_tensors['outputs'][0]['shape'] if op_tensors['outputs'] else None
        
        if input_shape is None or output_shape is None:
            raise ValueError("Missing shapes from LiteRT")
        
        return {
            'input_shape': input_shape,
            'output_shape': output_shape
        }
    
    def get_shapes_from_litert(self, tflite_path: str, operator_index: int = 0) -> Dict[str, Any]:
        """
        Get input and output shapes using LiteRT schema (convenience wrapper).
        
        Args:
            tflite_path: Path to .tflite file
            operator_index: Index of the operator (default: 0)
            
        Returns:
            Dictionary with 'input_shapes' (list) and 'output_shapes' (list) keys
        """
        from helia_core_tester.generation.utils.litert_utils import (
            load_litert_model, get_input_output_shapes_from_litert
        )
        
        model, subgraph = self.load_litert_model(tflite_path)
        return get_input_output_shapes_from_litert(model, subgraph, operator_index)
    
    def get_quantization_from_litert(self, tflite_path: str, operator_index: int = 0) -> Dict[str, Any]:
        """
        Get input and output quantization parameters using LiteRT schema (convenience wrapper).
        
        Args:
            tflite_path: Path to .tflite file
            operator_index: Index of the operator (default: 0)
            
        Returns:
            Dictionary with 'input_quantizations' (list) and 'output_quantizations' (list) keys
        """
        from helia_core_tester.generation.utils.litert_utils import (
            get_input_output_quantization_from_litert
        )
        
        model, subgraph = self.load_litert_model(tflite_path)
        return get_input_output_quantization_from_litert(model, subgraph, operator_index)
    
    def generate_c_files(self, output_dir: Path) -> None:
        """
        Generate C and H files from templates.
        
        This method should be overridden by subclasses to implement
        operator-specific generation logic.
        
        Args:
            output_dir: Output directory for generated files
        """
        raise NotImplementedError("Subclasses must implement generate_c_files()")
