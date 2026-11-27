"""Stand-ins for the library under test, for tests that compile the real runtime on the host.

The runtime includes hct_library.h, which asserts every status code of the profile against the
library's headers, so a stub header must define all of them: these are built from the profile itself.
"""

from __future__ import annotations

from typing import List

from helia_core_tester.core.harness import legacy_harness
from helia_core_tester.mutation.host_build import library_header_flags as _flags

_CODES = legacy_harness().status.codes
NN_STATUS_ENUM = (
    "typedef enum { " + ", ".join(f"{name} = {value}" for name, value in _CODES) + " } arm_cmsis_nn_status;\n"
)
NN_STATUS_DEFINES = "".join(f"#define {name} {value}\n" for name, value in _CODES)


def library_header_flags() -> List[str]:
    """`-I` for the hct_library.h of the library under test."""
    return _flags()
