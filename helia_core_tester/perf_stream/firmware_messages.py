"""Decode firmware-emitted HCTP control payloads."""

from __future__ import annotations

from dataclasses import dataclass

from .hctp import ByteReader


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
