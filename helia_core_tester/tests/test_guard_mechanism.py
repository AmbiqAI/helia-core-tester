"""Executed contract for the issue #68 buffer-overrun guards.

The guard is split between a Jinja macro (helia_guard_declare, which has to
emit a #define) and C macros/functions in the shipped runtime, so neither
half is testable by reading it. This renders the Jinja half exactly as the
generated tests do, compiles it with the runtime and a stub kernel using the
host compiler, and checks the GuardBreach lines and failure counts for a
clean write, an overrun, an underrun, both at once, and the scratch-slack
canary that sits at the kernel's *queried* size rather than the declared end.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import jinja2
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
C_HOST_DIR = Path(__file__).resolve().parent / "c_host"
TEMPLATES_ROOT = PROJECT_ROOT / "assets" / "templates"

DECLS_TEMPLATE = """\
{% from "common/standalone/runtime_common.j2" import helia_guard_declare %}
{{ helia_guard_declare("int32_t", "sanity_output", 8) }}
{{ helia_guard_declare("uint8_t", "sanity_scratch", 64) }}
"""

EXPECTED = {
    "output_clean": (0, []),
    "output_overrun": (1, ["GuardBreach[sanity output]: overrun detected (canary corrupted)"]),
    "output_underrun": (1, ["GuardBreach[sanity output]: underrun detected (canary corrupted)"]),
    "output_both": (1, ["GuardBreach[sanity output]: underrun and overrun detected (canary corrupted)"]),
    "scratch_clean": (0, []),
    # One byte past the queried size lands in the slack, not the tail guard.
    "scratch_slack_overrun": (1, ["GuardBreach[sanity scratch slack]: overrun detected (canary corrupted)"]),
    # Past the declared end: both the slack canary and the tail guard fire.
    "scratch_slack_deep_overrun": (
        2,
        [
            "GuardBreach[sanity scratch]: overrun detected (canary corrupted)",
            "GuardBreach[sanity scratch slack]: overrun detected (canary corrupted)",
        ],
    ),
    "slack_no_room": (0, []),
    "slack_short": (0, []),
    "slack_short_overrun": (1, ["GuardBreach[sanity scratch slack]: overrun detected (canary corrupted)"]),
}


@pytest.fixture(scope="module")
def host_sanity_output(tmp_path_factory: pytest.TempPathFactory) -> str:
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("host C compiler not available")

    workdir = tmp_path_factory.mktemp("guard_mechanism")
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(TEMPLATES_ROOT)),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    decls = env.from_string(DECLS_TEMPLATE).render()
    (workdir / "helia_guard_host_decls.h").write_text(decls)

    binary = workdir / "helia_guard_host_sanity"
    subprocess.run(
        [
            cc,
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-I",
            str(workdir),
            "-I",
            str(C_HOST_DIR),
            "-I",
            str(PROJECT_ROOT / "src"),
            str(PROJECT_ROOT / "src" / "test_runtime" / "helia_test_runtime.c"),
            str(C_HOST_DIR / "helia_guard_host_sanity.c"),
            "-o",
            str(binary),
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )

    completed = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    assert "HOST_SANITY_DONE" in completed.stdout
    assert "POISON_MISSING" not in completed.stdout
    return completed.stdout


def _case_block(output: str, case_id: str) -> str:
    start = output.index(f"CASE {case_id}\n")
    end = output.index(f"RESULT {case_id} failures=", start)
    return output[start:end]


def _failure_count(output: str, case_id: str) -> int:
    match = re.search(rf"^RESULT {re.escape(case_id)} failures=(\d+)$", output, re.MULTILINE)
    assert match is not None, case_id
    return int(match.group(1))


@pytest.mark.parametrize("case_id", sorted(EXPECTED))
def test_guard_reports_exactly_the_breached_boundaries(
    host_sanity_output: str, case_id: str
) -> None:
    expected_failures, expected_lines = EXPECTED[case_id]
    block = _case_block(host_sanity_output, case_id)
    breach_lines = [line for line in block.splitlines() if line.startswith("GuardBreach[")]
    assert breach_lines == expected_lines
    assert _failure_count(host_sanity_output, case_id) == expected_failures


def test_rendered_declaration_hides_the_guard_behind_the_buffer_name() -> None:
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(TEMPLATES_ROOT)),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    decls = env.from_string(DECLS_TEMPLATE).render()
    assert "uint8_t head[HELIA_GUARD_BYTES];" in decls
    assert "int32_t body[8];" in decls
    assert "uint8_t tail[HELIA_GUARD_BYTES];" in decls
    assert "} sanity_output_guard;" in decls
    assert "#define sanity_output (sanity_output_guard.body)" in decls
