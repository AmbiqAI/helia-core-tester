"""Generation-time feature detection for the ns-cmsis-nn temp-buffer sizers.

ns-cmsis-nn#377 / helia-core-tester#71: the LSTM/GRU templates historically
hard-coded their scratch ("temp") buffer sizes with LSTM_BUFFER_SIZE-style
macros -- formulas that were merely incidentally large enough and that
silently drift when the kernels change. ns-cmsis-nn#381 gives every LSTM/GRU
temp buffer an owning ``*_get_buffer_size`` query. The tester wants to call
those queries on-target, but it must keep generating against ns-cmsis-nn
checkouts that predate them (a hard circular dependency: this repo's CI pins
ns-cmsis-nn main, and ns-cmsis-nn#381 cannot merge with failing testers).

So the switch is a generation-time feature probe: grep the configured
ns-cmsis-nn checkout's public ``Include/`` headers for declarations of the
exact sizer symbols. All symbols present -> the templates emit the sizer-calling,
sizer-validating variant. Any symbol absent (or no checkout resolvable at
all) -> the templates emit byte-identical legacy output, so a generate
against ns-cmsis-nn main is unchanged down to the last byte.

The checkout is resolved exactly the way lstm_data.py resolves its UnitTest
data root: the ``CMSIS_NN_ROOT`` environment variable (set from
Config.cmsis_nn_root / --cmsis-nn-root by the generation step) when present,
else the historical nested layout (ns-cmsis-nn/Tests/helia-core-tester/...).
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

# Public headers that can declare the sizers. Probing only named top-level
# public headers keeps the scan cheap and pins the contract to the public API
# (a symbol only present in Source/ would not be callable from a test).
_PUBLIC_HEADER_NAMES = (
    "arm_nnfunctions.h",
    "arm_nnfunctions_flt.h",
)


def resolve_cmsis_nn_root() -> Optional[Path]:
    """Return the ns-cmsis-nn checkout root used for generation, or None.

    Mirrors lstm_data._unit_test_data_root(): CMSIS_NN_ROOT wins, else the
    nested ns-cmsis-nn/Tests/helia-core-tester layout. Returns None when the
    resolved directory does not look like an ns-cmsis-nn checkout (no
    Include/), which callers must treat as "sizers absent".
    """
    env_root = os.environ.get("CMSIS_NN_ROOT")
    if env_root:
        root = Path(env_root).resolve()
    else:
        # .../ns-cmsis-nn/Tests/helia-core-tester/helia_core_tester/generation/utils/temp_sizer_probe.py
        # parents[3] == helia-core-tester repo root, parents[5] == ns-cmsis-nn.
        root = Path(__file__).resolve().parents[5]
    if not (root / "Include").is_dir():
        return None
    return root


_COMMENT_RE = re.compile(r"/\*.*?\*/|//[^\n]*", re.DOTALL)


def _strip_comments(text: str) -> str:
    """Remove C block and line comments so a doxygen mention -- even in
    code-form, ``arm_..._get_buffer_size(...)`` -- can never look like a
    declaration. (String literals do not occur in these headers, so the
    naive regex is exact here.)"""
    return _COMMENT_RE.sub(" ", text)


def _symbol_declared(symbol: str, text: str) -> bool:
    """True iff ``symbol`` appears in declaration shape -- the symbol
    followed (modulo whitespace) by an opening parenthesis, at a word
    boundary -- in the comment-stripped header text. Comment stripping
    closes the code-form-mention false positive twice flagged in review;
    the remaining residual (a function-like ``#define symbol(x)``) would
    still fail loudly at link time.
    """
    return re.search(rf"\b{re.escape(symbol)}\s*\(", _strip_comments(text)) is not None


def _public_header_corpus(cmsis_nn_root: Optional[Path]) -> Optional[str]:
    """Comment-bearing concatenation of the checkout's public Include/ headers,
    or None when no header is readable (callers must treat None as "every
    symbol absent" -- skipping is always the safe answer)."""
    if cmsis_nn_root is None or not (cmsis_nn_root / "Include").is_dir():
        return None
    corpus = []
    for header_name in _PUBLIC_HEADER_NAMES:
        header = cmsis_nn_root / "Include" / header_name
        if not header.is_file():
            continue
        try:
            corpus.append(header.read_text(errors="replace"))
        except OSError:
            continue
    if not corpus:
        return None
    return "\n".join(corpus)


def probe_header_symbols(symbols: Iterable[str], cmsis_nn_root: Optional[Path] = None) -> bool:
    """True iff every symbol is declared in the checkout's public Include/
    headers (declaration shape: ``symbol(`` -- see _symbol_declared).

    Missing checkout, missing Include/, or unreadable headers all mean False
    -- the legacy templates are always a safe answer.
    """
    root = cmsis_nn_root if cmsis_nn_root is not None else resolve_cmsis_nn_root()
    text = _public_header_corpus(root)
    if text is None:
        return False
    return all(_symbol_declared(symbol, text) for symbol in symbols)


def missing_header_symbols(symbols: Iterable[str], cmsis_nn_root: Optional[Path] = None) -> list[str]:
    """Return the subset of ``symbols`` NOT declared in the checkout's public
    Include/ headers, preserving input order (deduplicated).

    Per-symbol sibling of probe_header_symbols() for descriptor-level gating
    (helia-core-tester float mean / hard_swish support): one tester pin is
    used across several in-flight ns-cmsis-nn kernel branches, each declaring
    only its own kernels, so generation must skip exactly the descriptors
    whose kernels the target checkout does not declare. Missing checkout,
    missing Include/, or unreadable headers mean every symbol is reported
    missing -- skipping is always the safe answer.
    """
    ordered: list[str] = []
    seen = set()
    for symbol in symbols:
        name = str(symbol).strip()
        if name and name not in seen:
            ordered.append(name)
            seen.add(name)
    if not ordered:
        return []
    root = cmsis_nn_root if cmsis_nn_root is not None else resolve_cmsis_nn_root()
    text = _public_header_corpus(root)
    if text is None:
        return list(ordered)
    return [symbol for symbol in ordered if not _symbol_declared(symbol, text)]


def lstm_int_temp_expected_bytes(*, time_major: bool, batch_size: int, hidden_size: int) -> int:
    """Expected arm_lstm_unidirectional_{s8,s16}_temp{1,2}_get_buffer_size().

    Both integer LSTM datatypes stage int16_t gate vectors, and only the
    time-major layer multiplies the batch in (the batch-major branch
    re-invokes the step kernel with batch_size == 1). Mirrors
    arm_lstm_temp_bytes() / arm_lstm_temp_bytes_s16() in ns-cmsis-nn#381.
    """
    gate_batch = batch_size if time_major else 1
    return gate_batch * hidden_size * 2  # sizeof(int16_t)


def gru_temp1_expected_bytes(*, reset_after: bool, hidden_size: int, elem_bytes: int) -> int:
    """Expected arm_gru_unidirectional_{f32,f16}_temp1_get_buffer_size().

    temp1 stages the reset-scaled hidden state, needed only by the pre-reset
    formulation; the reset-after formulation reports 0. Mirrors the
    ns-cmsis-nn#381 implementations.
    """
    if reset_after:
        return 0
    return hidden_size * elem_bytes


def lstm_float_temp_expected_bytes() -> int:
    """Expected arm_lstm_unidirectional_{f32,f16}_temp{1,2}_get_buffer_size().

    The float LSTM steps compute their gates in automatics and never touch
    temp1/temp2 on any build path, so the queries report 0 (and the buffers
    may be NULL).
    """
    return 0


def detect_temp_sizers(symbols: Iterable[str], op_label: str) -> bool:
    """Probe + one log line so generate logs record the decision per family."""
    symbols = tuple(symbols)
    root = resolve_cmsis_nn_root()
    available = probe_header_symbols(symbols, cmsis_nn_root=root)
    logger.info(
        "%s temp-buffer sizer probe (%s): %s -> %s",
        op_label,
        root if root is not None else "no ns-cmsis-nn checkout resolved",
        ", ".join(symbols),
        "detected (emitting sizer-validating variant)"
        if available
        else "absent (emitting legacy byte-identical output)",
    )
    return available
