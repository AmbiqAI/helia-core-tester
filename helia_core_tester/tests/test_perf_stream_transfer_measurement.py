from __future__ import annotations

import zlib

import pytest

from helia_core_tester.perf_stream.measurement import (
    CounterDescriptor,
    RawCounterValue,
    RawSample,
    StatefulKernelRestrictionError,
    auto_calibrate_iterations,
    compute_sample_statistics,
    normalize_samples,
    plan_counter_passes,
    resolve_counter_selection,
)
from helia_core_tester.perf_stream.transfer import (
    AlignmentError,
    ArenaTracker,
    BlobAccumulator,
    BlobCrcMismatchError,
    BlobTransferSpec,
    CaseTooLargeError,
    DuplicateChunkError,
    MissingChunkError,
    OutOfRangeChunkError,
    OverlappingChunkError,
)



def test_chunked_blob_transfer_and_finish() -> None:
    payload = b"abcdefgh"
    accumulator = BlobAccumulator(BlobTransferSpec(blob_id=1, byte_length=len(payload), expected_crc32=zlib.crc32(payload) & 0xFFFFFFFF))
    accumulator.add_chunk(0, b"abc")
    accumulator.add_chunk(3, b"de")
    accumulator.add_chunk(5, b"fgh")
    assert accumulator.finish() == payload



def test_duplicate_chunk_rejected() -> None:
    payload = b"abcd"
    accumulator = BlobAccumulator(BlobTransferSpec(blob_id=1, byte_length=4, expected_crc32=zlib.crc32(payload) & 0xFFFFFFFF))
    accumulator.add_chunk(0, b"ab")
    with pytest.raises(DuplicateChunkError):
        accumulator.add_chunk(0, b"ab")



def test_missing_chunk_rejected() -> None:
    payload = b"abcd"
    accumulator = BlobAccumulator(BlobTransferSpec(blob_id=1, byte_length=4, expected_crc32=zlib.crc32(payload) & 0xFFFFFFFF))
    accumulator.add_chunk(0, b"ab")
    with pytest.raises(MissingChunkError):
        accumulator.finish()



def test_overlapping_chunk_rejected() -> None:
    payload = b"abcdef"
    accumulator = BlobAccumulator(BlobTransferSpec(blob_id=1, byte_length=6, expected_crc32=zlib.crc32(payload) & 0xFFFFFFFF))
    accumulator.add_chunk(0, b"abcd")
    with pytest.raises(OverlappingChunkError):
        accumulator.add_chunk(2, b"ef")



def test_out_of_range_chunk_rejected() -> None:
    accumulator = BlobAccumulator(BlobTransferSpec(blob_id=1, byte_length=4, expected_crc32=0))
    with pytest.raises(OutOfRangeChunkError):
        accumulator.add_chunk(3, b"zz")



def test_crc_failure_rejected() -> None:
    payload = b"abcdef"
    accumulator = BlobAccumulator(BlobTransferSpec(blob_id=1, byte_length=6, expected_crc32=0x12345678))
    accumulator.add_chunk(0, payload)
    with pytest.raises(BlobCrcMismatchError):
        accumulator.finish()



def test_alignment_rejected() -> None:
    payload = b"abcdefgh"
    accumulator = BlobAccumulator(BlobTransferSpec(blob_id=1, byte_length=8, expected_crc32=zlib.crc32(payload) & 0xFFFFFFFF, required_alignment=4))
    with pytest.raises(AlignmentError):
        accumulator.add_chunk(2, b"cd")



def test_arena_tracker_rewind_and_case_too_large() -> None:
    arena = ArenaTracker(16)
    arena.reserve(8)
    arena.reserve(8)
    with pytest.raises(CaseTooLargeError):
        arena.reserve(1)
    arena.rewind()
    assert arena.used_bytes == 0



def test_normalization_and_statistics() -> None:
    normalized = normalize_samples(
        [
            RawSample(0, 4, 400, (RawCounterValue("ARM_PMU_CPU_CYCLES", 0x11, 1600),), "cpu_0"),
            RawSample(1, 4, 440, (RawCounterValue("ARM_PMU_CPU_CYCLES", 0x11, 1760),), "cpu_0"),
            RawSample(2, 4, 520, (RawCounterValue("ARM_PMU_CPU_CYCLES", 0x11, 2080),), "cpu_0"),
        ]
    )
    stats = compute_sample_statistics(normalized)

    assert normalized[0].cycles_per_invocation == 100.0
    assert normalized[1].counters_per_invocation["ARM_PMU_CPU_CYCLES"] == 440.0
    assert stats.median_cycles == 440.0
    assert stats.mad_cycles == 40.0
    assert stats.p90_cycles == pytest.approx(504.0)
    assert stats.p99_cycles == pytest.approx(518.4)



def test_overflow_and_unsupported_counter_propagation() -> None:
    normalized = normalize_samples(
        [
            RawSample(
                0,
                2,
                200,
                (
                    RawCounterValue("ARM_PMU_CPU_CYCLES", 0x11, 400, overflow=True),
                    RawCounterValue("ARM_PMU_MVE_INST_RETIRED", 0x200, 0, supported=False),
                ),
                "mixed_0",
            )
        ]
    )
    stats = compute_sample_statistics(normalized)

    assert normalized[0].overflow is True
    assert normalized[0].unsupported_counters == ("ARM_PMU_MVE_INST_RETIRED",)
    assert stats.valid_for_regression is False
    assert stats.overflow_detected is True



def test_multiple_pmu_passes_are_planned() -> None:
    counters = resolve_counter_selection({"cpu": "default", "memory": "default", "mve": "default"})
    passes = plan_counter_passes(counters)

    assert [perf_pass.name for perf_pass in passes] == ["cpu_0", "memory_0", "mve_0"]



def test_auto_calibration_limits_and_stateful_restriction() -> None:
    calibrated = auto_calibrate_iterations(base_cycles=80, min_cycles=1000, max_iterations=8, stateful=False)
    assert calibrated.iterations == 8
    assert calibrated.capped is True

    with pytest.raises(StatefulKernelRestrictionError):
        auto_calibrate_iterations(base_cycles=80, min_cycles=1000, max_iterations=8, stateful=True)



def test_unsupported_counter_group_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported counter group"):
        resolve_counter_selection({"bogus": "default"})
