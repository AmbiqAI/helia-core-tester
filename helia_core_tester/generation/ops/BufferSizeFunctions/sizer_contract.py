"""
Sizer-contract cases: the documented return contract of the scratch queries
(issue #69).

Every scratch-consuming harness calls a `*_get_buffer_size()` at runtime, but
until now the only thing asserted about the answer was that it did not exceed a
Python re-derivation of the same C formula. That bound is the right size for the
static allocation and the wrong thing to assert against: an under-reporting
sizer, a wrong-but-smaller one, and the documented -1 sentinel all satisfy it.

These cases assert the part of the answer that does *not* require re-deriving
the formula: the sentinel contract the public header states outright. Three
probe kinds, all quoted from the header of the sizer under test:

  negative_dim     a dimension the selected route reads is negative -> -1
  overflow         a shape whose byte count would not fit in an int32_t -> -1
  degenerate_zero  a route the header says needs no scratch, or a shape the
                   header says truncates to no units -> 0

Deliberately NOT asserted: the exact byte count for a valid shape. Producing an
expected value would mean transcribing the kernel arithmetic into Python a
second time, which is the defect issue #69 describes (a formula bug agrees with
itself in both languages). The sentinel half is oracle-free: the header states
the answer, not the implementation.

Every probe here must answer the same on a plain-C, DSP and MVE build, because
one descriptor generates for all three. Where the header makes the answer
build-dependent -- "which routes compute a byte count is build-dependent" for
the conv and depthwise wrappers, and the MVE/DSP split for the fully connected,
SVDF kernel-sum and avgpool queries -- the probe either picks a shape the header
covers unconditionally or the descriptor carries required_capabilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from helia_core_tester.generation.ops._shared.base import OperationBase


# A probe is a self-contained C block: `decls` are statements, `call` is the
# expression under test, `expect` is the value the quoted header sentence
# promises, and `doc` is that sentence. `doc` is emitted into the generated
# source as the comment justifying `expect`, so the assertion and its warrant
# never drift apart.
_SIZERS: Dict[str, Dict[str, Any]] = {
    "convolve_wrapper_s8": {
        "symbol": "arm_convolve_wrapper_s8_get_buffer_size",
        "header": "Include/arm_nnfunctions.h",
        "probes": {
            "negative_dim": {
                "subject": "negative-channel return",
                "expect": -1,
                "doc": [
                    "arm_convolve_wrapper_s8_get_buffer_size(), Include/arm_nnfunctions.h:",
                    '"returns required buffer size in bytes, or -1 if the shape is out of',
                    'range - a dimension the selected route reads is negative".',
                    "A 3x3 filter over a 4-row input selects neither the 1x1 nor the 1xN",
                    "route, so the general route -- the one that reads input_dims->c -- is",
                    "the selected one on every build target.",
                ],
                "decls": [
                    "cmsis_nn_conv_params conv_params = {0};",
                    "conv_params.stride.w = 1;",
                    "conv_params.stride.h = 1;",
                    "conv_params.dilation.w = 1;",
                    "conv_params.dilation.h = 1;",
                    "const cmsis_nn_dims input_dims = {1, 4, 4, -1};",
                    "const cmsis_nn_dims filter_dims = {8, 3, 3, -1};",
                    "const cmsis_nn_dims output_dims = {1, 2, 2, 8};",
                ],
                "call": (
                    "arm_convolve_wrapper_s8_get_buffer_size("
                    "&conv_params, &input_dims, &filter_dims, &output_dims)"
                ),
            },
            "overflow": {
                "subject": "int32-overflow return",
                "expect": -1,
                "doc": [
                    "Same sentence: -1 when \"the required size would not fit in an",
                    'int32_t". The general route folds filter_dims->w * filter_dims->h *',
                    "input_dims->c into the byte count, so 3 * 3 * 2^30 elements overflow",
                    "on every build target.",
                ],
                "decls": [
                    "cmsis_nn_conv_params conv_params = {0};",
                    "conv_params.stride.w = 1;",
                    "conv_params.stride.h = 1;",
                    "conv_params.dilation.w = 1;",
                    "conv_params.dilation.h = 1;",
                    "const cmsis_nn_dims input_dims = {1, 4, 4, 1073741824};",
                    "const cmsis_nn_dims filter_dims = {8, 3, 3, 1073741824};",
                    "const cmsis_nn_dims output_dims = {1, 2, 2, 8};",
                ],
                "call": (
                    "arm_convolve_wrapper_s8_get_buffer_size("
                    "&conv_params, &input_dims, &filter_dims, &output_dims)"
                ),
            },
            "degenerate_zero": {
                "subject": "no-scratch route return",
                "expect": 0,
                "doc": [
                    "Same header entry: \"A route that needs no scratch buffer returns 0",
                    'for an in-range shape", and the @details name the route this shape',
                    'selects: "the 1x1 route that is not the fast variant needs no buffer',
                    'on any build". Unit padding-free 1x1 filter with a stride of 2 is',
                    "1x1 but not the fast variant, so the 0 is build-independent.",
                ],
                "decls": [
                    "cmsis_nn_conv_params conv_params = {0};",
                    "conv_params.stride.w = 2;",
                    "conv_params.stride.h = 2;",
                    "conv_params.dilation.w = 1;",
                    "conv_params.dilation.h = 1;",
                    "const cmsis_nn_dims input_dims = {1, 4, 4, 8};",
                    "const cmsis_nn_dims filter_dims = {16, 1, 1, 8};",
                    "const cmsis_nn_dims output_dims = {1, 2, 2, 16};",
                ],
                "call": (
                    "arm_convolve_wrapper_s8_get_buffer_size("
                    "&conv_params, &input_dims, &filter_dims, &output_dims)"
                ),
            },
        },
    },
    "depthwise_conv_wrapper_s8": {
        "symbol": "arm_depthwise_conv_wrapper_s8_get_buffer_size",
        "header": "Include/arm_nnfunctions.h",
        "probes": {
            "negative_dim": {
                "subject": "negative-channel return",
                "expect": -1,
                "doc": [
                    "arm_depthwise_conv_wrapper_s8_get_buffer_size(),",
                    "Include/arm_nnfunctions.h: \"-1 if the shape is out of range - a",
                    'dimension the selected route reads is negative". The shape below',
                    "selects the optimized depthwise route (input and output channels",
                    "equal, batch 1, unit dilation) and is not the 3x3 filter that the",
                    "same entry says short-circuits to 0 without range-checking.",
                ],
                "decls": [
                    "cmsis_nn_dw_conv_params dw_conv_params = {0};",
                    "dw_conv_params.stride.w = 1;",
                    "dw_conv_params.stride.h = 1;",
                    "dw_conv_params.dilation.w = 1;",
                    "dw_conv_params.dilation.h = 1;",
                    "const cmsis_nn_dims input_dims = {1, 8, 8, -1};",
                    "const cmsis_nn_dims filter_dims = {1, 5, 5, -1};",
                    "const cmsis_nn_dims output_dims = {1, 4, 4, -1};",
                ],
                "call": (
                    "arm_depthwise_conv_wrapper_s8_get_buffer_size("
                    "&dw_conv_params, &input_dims, &filter_dims, &output_dims)"
                ),
            },
            "overflow": {
                "subject": "int32-overflow return",
                "expect": -1,
                "doc": [
                    "Same sentence: -1 when \"the required size would not fit in an",
                    'int32_t". Both optimized legs fold filter_dims->w * filter_dims->h',
                    "into the byte count, so a 65536x65536 filter overflows either way.",
                    "The descriptor requires dsp: the same entry says which routes",
                    '"compute a byte count is build-dependent", and on a plain-C build',
                    "the optimized route needs no buffer and answers 0 for any shape.",
                ],
                "decls": [
                    "cmsis_nn_dw_conv_params dw_conv_params = {0};",
                    "dw_conv_params.stride.w = 1;",
                    "dw_conv_params.stride.h = 1;",
                    "dw_conv_params.dilation.w = 1;",
                    "dw_conv_params.dilation.h = 1;",
                    "const cmsis_nn_dims input_dims = {1, 8, 8, 8};",
                    "const cmsis_nn_dims filter_dims = {1, 65536, 65536, 8};",
                    "const cmsis_nn_dims output_dims = {1, 4, 4, 8};",
                ],
                "call": (
                    "arm_depthwise_conv_wrapper_s8_get_buffer_size("
                    "&dw_conv_params, &input_dims, &filter_dims, &output_dims)"
                ),
            },
        },
    },
    "fully_connected_s8": {
        "symbol": "arm_fully_connected_s8_get_buffer_size",
        "header": "Include/arm_nnfunctions.h",
        "probes": {
            "negative_dim": {
                "subject": "negative-channel return",
                "expect": -1,
                "doc": [
                    "arm_fully_connected_s8_get_buffer_size(), Include/arm_nnfunctions.h:",
                    '"-1 if filter_dims->c is negative or the required size would not fit',
                    'in an int32_t ... For an invalid filter_dims->c, returns -1 on every',
                    'build target."',
                ],
                "decls": [
                    "const cmsis_nn_dims filter_dims = {1, 1, 1, -1};",
                ],
                "call": "arm_fully_connected_s8_get_buffer_size(&filter_dims)",
            },
            "overflow": {
                "subject": "int32-overflow return",
                "expect": -1,
                "doc": [
                    "Same sentence. The documented figure is filter_dims->c *",
                    "sizeof(int32_t), so 2^29 channels is exactly the first channel count",
                    "whose byte count does not fit in an int32_t.",
                ],
                "decls": [
                    "const cmsis_nn_dims filter_dims = {1, 1, 1, 536870912};",
                ],
                "call": "arm_fully_connected_s8_get_buffer_size(&filter_dims)",
            },
        },
    },
    "svdf_s8_input_ctx": {
        "symbol": "arm_svdf_s8_input_ctx_get_buffer_size",
        "header": "Include/arm_nnfunctions.h",
        "probes": {
            "negative_dim": {
                "subject": "negative-batch return",
                "expect": -1,
                "doc": [
                    "arm_svdf_s8_input_ctx_get_buffer_size(), Include/arm_nnfunctions.h:",
                    '"-1 if either pointer is NULL, if input_dims->n or',
                    "weights_feature_dims->n is negative, or if the required size would",
                    'not fit in an int32_t". The same entry states the figure "does not',
                    'vary by build target".',
                ],
                "decls": [
                    "const cmsis_nn_dims input_dims = {-1, 1, 1, 4};",
                    "const cmsis_nn_dims weights_feature_dims = {4, 1, 1, 4};",
                ],
                "call": (
                    "arm_svdf_s8_input_ctx_get_buffer_size("
                    "&input_dims, &weights_feature_dims)"
                ),
            },
            "overflow": {
                "subject": "int32-overflow return",
                "expect": -1,
                "doc": [
                    "Same sentence. The documented figure is input_dims->n *",
                    "weights_feature_dims->n * sizeof(int32_t); 2^20 * 2^20 * 4 needs 42",
                    "bits.",
                ],
                "decls": [
                    "const cmsis_nn_dims input_dims = {1048576, 1, 1, 4};",
                    "const cmsis_nn_dims weights_feature_dims = {1048576, 1, 1, 4};",
                ],
                "call": (
                    "arm_svdf_s8_input_ctx_get_buffer_size("
                    "&input_dims, &weights_feature_dims)"
                ),
            },
            "degenerate_zero": {
                "subject": "degenerate-shape return",
                "expect": 0,
                "doc": [
                    "Same entry: \"0 is a valid return for a degenerate shape",
                    '(input_dims->n == 0)". Note the entry also says a 0 here does not',
                    "license passing { NULL, 0 } to the kernel, which is why the SVDF",
                    "harness allocates at least one byte.",
                ],
                "decls": [
                    "const cmsis_nn_dims input_dims = {0, 1, 1, 4};",
                    "const cmsis_nn_dims weights_feature_dims = {4, 1, 1, 4};",
                ],
                "call": (
                    "arm_svdf_s8_input_ctx_get_buffer_size("
                    "&input_dims, &weights_feature_dims)"
                ),
            },
        },
    },
    "svdf_s8_output_ctx": {
        "symbol": "arm_svdf_s8_output_ctx_get_buffer_size",
        "header": "Include/arm_nnfunctions.h",
        "probes": {
            "negative_dim": {
                "subject": "negative-feature-batch return",
                "expect": -1,
                "doc": [
                    "arm_svdf_s8_output_ctx_get_buffer_size(), Include/arm_nnfunctions.h:",
                    '"-1 if any pointer is NULL, if svdf_params->rank is zero, negative or',
                    "outside int16_t range, if input_dims->n or weights_feature_dims->n is",
                    'negative, or if the required size would not fit in an int32_t".',
                ],
                "decls": [
                    "cmsis_nn_svdf_params svdf_params = {0};",
                    "svdf_params.rank = 2;",
                    "const cmsis_nn_dims input_dims = {1, 1, 1, 4};",
                    "const cmsis_nn_dims weights_feature_dims = {-1, 1, 1, 4};",
                ],
                "call": (
                    "arm_svdf_s8_output_ctx_get_buffer_size("
                    "&svdf_params, &input_dims, &weights_feature_dims)"
                ),
            },
            "overflow": {
                "subject": "int32-overflow return",
                "expect": -1,
                "doc": [
                    "Same sentence. The documented figure is input_dims->n *",
                    "(weights_feature_dims->n / svdf_params->rank) * sizeof(int32_t); at",
                    "rank 1 that is 2^20 * 2^20 * 4.",
                ],
                "decls": [
                    "cmsis_nn_svdf_params svdf_params = {0};",
                    "svdf_params.rank = 1;",
                    "const cmsis_nn_dims input_dims = {1048576, 1, 1, 4};",
                    "const cmsis_nn_dims weights_feature_dims = {1048576, 1, 1, 4};",
                ],
                "call": (
                    "arm_svdf_s8_output_ctx_get_buffer_size("
                    "&svdf_params, &input_dims, &weights_feature_dims)"
                ),
            },
            "degenerate_zero": {
                "subject": "truncated-unit-count return",
                "expect": 0,
                "doc": [
                    "Same entry: \"A rank greater than weights_feature_dims->n truncates",
                    'the unit count to 0 and so returns 0." Rank 8 over 4 feature batches',
                    "is that shape, and the entry says the figure is the same on every",
                    "build target.",
                ],
                "decls": [
                    "cmsis_nn_svdf_params svdf_params = {0};",
                    "svdf_params.rank = 8;",
                    "const cmsis_nn_dims input_dims = {2, 1, 1, 4};",
                    "const cmsis_nn_dims weights_feature_dims = {4, 1, 1, 4};",
                ],
                "call": (
                    "arm_svdf_s8_output_ctx_get_buffer_size("
                    "&svdf_params, &input_dims, &weights_feature_dims)"
                ),
            },
        },
    },
    "avgpool_s8": {
        "symbol": "arm_avgpool_s8_get_buffer_size",
        "header": "Include/arm_nnfunctions.h",
        "probes": {
            "negative_dim": {
                "subject": "negative-channel return",
                "expect": -1,
                "doc": [
                    "arm_avgpool_s8_get_buffer_size(), Include/arm_nnfunctions.h: \"-1 if",
                    "ch_src is negative or the required size would not fit in an int32_t",
                    '... For an invalid ch_src it returns -1 on every build target."',
                    "The same entry notes arm_avgpool_s8() depends on that sentinel being",
                    'non-zero, since it "reads a non-zero size as ctx->buf is required".',
                ],
                "decls": [],
                "call": "arm_avgpool_s8_get_buffer_size(4, -1)",
            },
            "overflow": {
                "subject": "int32-overflow return",
                "expect": -1,
                "doc": [
                    "Same sentence. The documented figure is ch_src * sizeof(int32_t), so",
                    "2^29 channels is the first count whose byte count does not fit.",
                ],
                "decls": [],
                "call": "arm_avgpool_s8_get_buffer_size(4, 536870912)",
            },
        },
    },
}


class OpSizerContract(OperationBase):
    """Sentinel-contract cases for one family's scratch query (issue #69)."""

    def needs_keras_model(self) -> bool:
        return False

    def allow_no_tflite(self) -> bool:
        return True

    def build_keras_model(self):
        raise NotImplementedError(
            "SizerContract asserts a header contract about a query function; it has no model."
        )

    def convert_to_tflite(self, model, out_path: str, rep_seed: int) -> None:
        raise NotImplementedError(
            "SizerContract runs no kernel, so there is nothing to convert."
        )

    def _sizer(self) -> Dict[str, Any]:
        key = str(self.desc.get("sizer", "")).strip()
        if key not in _SIZERS:
            raise ValueError(
                f"SizerContract descriptor '{self.desc.get('name')}' names unknown sizer "
                f"'{key}'; known sizers: {sorted(_SIZERS)}"
            )
        return _SIZERS[key]

    def _probes(self, sizer: Dict[str, Any]) -> List[Dict[str, Any]]:
        raw = self.desc.get("probes")
        if isinstance(raw, str):
            raw = [raw]
        requested = [str(kind).strip() for kind in (raw or []) if str(kind).strip()]
        if not requested:
            raise ValueError(
                f"SizerContract descriptor '{self.desc.get('name')}' lists no probes; a case "
                "that asserts nothing about the sizer is worse than no case"
            )
        probes: List[Dict[str, Any]] = []
        for kind in requested:
            if kind not in sizer["probes"]:
                raise ValueError(
                    f"SizerContract descriptor '{self.desc.get('name')}': sizer "
                    f"'{self.desc.get('sizer')}' has no '{kind}' probe; available: "
                    f"{sorted(sizer['probes'])}"
                )
            probe = dict(sizer["probes"][kind])
            probe["kind"] = kind
            probes.append(probe)
        return probes

    def generate_c_files(self, output_dir: Path) -> None:
        name = self.desc["name"]
        sizer = self._sizer()
        probes = self._probes(sizer)

        context: Dict[str, Any] = {
            "name": name,
            "sizer_symbol": sizer["symbol"],
            "sizer_header": sizer["header"],
            "probes": probes,
            "validation_mode": "exact_int",
            # Prefixed so a failing assertion is classified as a sizer-contract
            # violation by the reporting parser rather than as a generic scalar
            # mismatch; the prefix is the parser's hook (issue #69).
            "validation_label": f"sizer contract {sizer['symbol']}",
        }
        cmake_context = {
            "name": name,
            "operator": self.desc.get("operator", "SizerContract"),
            "operator_name": "sizer_contract",
        }
        self._write_op_outputs(
            output_dir,
            "sizer_contract",
            "BufferSizeFunctions/sizer_contract/sizer_contract.h.j2",
            "BufferSizeFunctions/sizer_contract/sizer_contract.c.j2",
            context,
            cmake_context,
        )
