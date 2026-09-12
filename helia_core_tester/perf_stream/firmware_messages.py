"""Decode firmware-emitted HCTP control payloads."""

from __future__ import annotations

from dataclasses import dataclass

from .hctp import ByteReader

# HELLO capability_flags bits -- must match the HCT_CAP_* enum in
# cmake/perf_stream/benchmark_server_catalog.h.
CAP_CASE_STREAMING = 1 << 0
CAP_CORRECTNESS = 1 << 1
CAP_PERFORMANCE = 1 << 2
CAP_RTT_TRANSPORT = 1 << 3
CAP_KERNEL_CATALOG = 1 << 4
CAP_ABS_S8 = 1 << 5
# Set only when the firmware was built for a core with the Armv8.1-M PMU
# (__PMU_PRESENT == 1); absent on DWT-only targets such as Cortex-M4.
CAP_PMU_ARMV8M = 1 << 6


@dataclass(frozen=True)
class HelloPayload:
    build_id: str
    catalog_hash: bytes
    max_frame_payload: int
    runtime_arena_capacity: int
    transfer_mode: int
    output_mode: int
    board_id: str
    target_cpu: str
    transport_kind: int
    capability_flags: int
    # v2: number of 16-bit PMU event-counter slots (8 on Cortex-M55, 0 without a PMU)
    # and the largest frame payload the target's receive buffer can hold.
    pmu_counter_slots: int
    max_rx_payload: int

    @property
    def has_pmu(self) -> bool:
        return bool(self.capability_flags & CAP_PMU_ARMV8M)


@dataclass(frozen=True)
class CatalogEntry:
    kernel_id: int
    canonical_name: str
    operator_family: str
    api_version: int
    supported_dtype: str
    adapter_schema_version: int
    stateless: bool
    repeated_invocation_safe: bool
    mutates_input: bool
    scratch_bytes: int



def decode_hello_payload(payload: bytes) -> HelloPayload:
    reader = ByteReader(payload)
    return HelloPayload(
        build_id=reader.text(),
        catalog_hash=reader.fixed(32),
        max_frame_payload=reader.u32(),
        runtime_arena_capacity=reader.u32(),
        transfer_mode=reader.u8(),
        output_mode=reader.u8(),
        board_id=reader.text(),
        target_cpu=reader.text(),
        transport_kind=reader.u8(),
        capability_flags=reader.u32(),
        pmu_counter_slots=reader.u8(),
        max_rx_payload=reader.u32(),
    )



def decode_catalog_payload(payload: bytes) -> tuple[CatalogEntry, ...]:
    reader = ByteReader(payload)
    count = reader.u16()
    entries = []
    for _ in range(count):
        entries.append(
            CatalogEntry(
                kernel_id=reader.u32(),
                canonical_name=reader.text(),
                operator_family=reader.text(),
                api_version=reader.u16(),
                supported_dtype=reader.text(),
                adapter_schema_version=reader.u16(),
                stateless=bool(reader.u8()),
                repeated_invocation_safe=bool(reader.u8()),
                mutates_input=bool(reader.u8()),
                scratch_bytes=reader.u32(),
            )
        )
    return tuple(entries)
