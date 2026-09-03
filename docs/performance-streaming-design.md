# Streaming hardware performance testing design

## Scope and repository ownership

This design follows the existing repository boundaries.

- **helia-core-tester** owns host orchestration, plan/build metadata, case generation, correctness comparison, reports, and the streaming protocol host implementation.
- **ns-cmsis-nn** continues to own public kernel APIs and kernel implementations. The streaming path calls those APIs through runtime adapters; descriptor-specific arrays do not move into firmware.
- **neuralspotx** remains the build/flash/transport integration layer for Apollo targets. The streaming firmware is an NSX-built application/profile, not a custom side channel that bypasses NSX.
- **helia-profiler** owns reusable PMU/DWT, RTT transport, overflow handling, result provenance, and memory/section reporting concepts that the streaming pipeline should reuse where practical.

## Current baseline

Today helia-core-tester generates standalone C tests per descriptor and builds/runs one ELF per case. That model is good for correctness but scales poorly for hardware performance work because each case carries tensors in flash and requires separate build/flash/run orchestration.

## Target architecture

The default architecture is **one universal benchmark-server firmware per target profile**:

- board
- CPU
- toolchain
- optimization/LTO
- integer/F16/F32 feature set
- cache/memory placement policy
- protocol version

That firmware contains:

- the selected ns-cmsis-nn library build
- a kernel adapter registry/catalog
- the HCTP streaming protocol server
- transport glue (RTT first)
- PMU/DWT measurement support
- a bounded runtime arena
- compact catalog metadata

It does **not** contain descriptor-specific tensors or expected outputs.

## Phase 0 sizing rule

Before adding sharding, build a real linked firmware for Apollo510/Cortex-M55 and measure:

1. integer only
2. integer + F32
3. integer + F16
4. integer + F16 + F32

The proof artifact is the final ELF plus map/bin/size/nm/objdump reports and a machine-readable `memory_report.json`.

## Host/target split

### Host responsibilities

helia-core-tester will:

- generate streamable case bundles directly from the same NumPy/TFLite data used for standalone tests
- build ordered execution plans
- validate firmware HELLO/catalog compatibility
- stream case metadata and blobs in bounded chunks
- reconstruct streamed outputs
- reuse existing comparison rules
- compute statistics from raw samples
- write result bundles and JUnit/CSV/JSON artifacts

### Target responsibilities

Firmware will:

- expose a versioned HELLO + capability block
- expose a versioned kernel catalog with stable numeric IDs
- request cases/blobs from the host (target-driven pull)
- bind streamed blobs into validated adapter metadata
- run one correctness invocation
- stream outputs back once per case
- run warmups + measured iterations only after host correctness ACK
- collect DWT cycles and PMU samples outside transfer/protocol work
- rewind the arena and request the next case without rebooting

## Wire protocol

The device wire format is **HCTP**, a binary little-endian framed protocol.

### Header goals

- fixed-size header
- explicit payload length
- header CRC and payload CRC
- session ID and sequence ID
- bounded payloads
- explicit state validation

### Directionality

The protocol is target-driven after plan load:

1. target sends `HELLO`
2. host sends `HELLO_ACK`
3. host sends `LOAD_PLAN`
4. target sends `REQUEST_CASE`
5. host sends `CASE_META`
6. target requests blobs chunk-by-chunk
7. host sends `BLOB_CHUNK`
8. target sends `CASE_READY`
9. host sends `RUN_CORRECTNESS`
10. target streams output + correctness result
11. host sends `CORRECTNESS_ACK`
12. host sends `RUN_PERFORMANCE` when allowed
13. target streams raw sample results
14. target sends `CASE_COMPLETE`
15. loop until `SESSION_COMPLETE`

## Kernel catalog

Each target build emits a catalog entry per supported runtime adapter with:

- stable kernel ID
- canonical kernel name
- operator family
- API version
- supported dtypes/capabilities
- adapter schema version
- stateless/repeated-invocation safety
- mutation/reset behavior
- scratch sizing behavior
- optional route-trace support

The target `HELLO` includes a hash of the full catalog. The host refuses plans that reference missing IDs.

## Adapter model

Adapters are reusable public-API/operator-family bindings, not one function per descriptor. Descriptor data stays on the host; firmware rebuilds only when:

- a new adapter is introduced
- a schema changes
- a kernel/catalog ID mapping changes

Adapters must declare mutation/reset/repeatability semantics so performance loops stay correct for stateful kernels.

## Memory model

The firmware uses bounded memory only:

- no unbounded dynamic allocation
- aligned static protocol buffers
- bounded runtime arena
- explicit max blob size/rank/output size
- per-case validation before copy/use
- arena rewind after each case

Runtime RAM is sized for the **largest active case**, not all descriptors.

## Transport

Default transport is bidirectional SEGGER RTT. In the current live Apollo510 implementation, HCTP uses RTT channel 0 in both directions and does not multiplex console logs onto that channel. Transport abstraction mirrors the fake/loopback split so hardware and simulated paths share the same host session logic.

### Current RTT implementation status

- **Live-real on Apollo510:** the benchmark-server target now boots on real Apollo510 hardware, initializes the real `SEGGER_RTT` target sources from `neuralspotx/examples/coremark/src/rtt/`, emits HELLO over RTT, accepts host frames, requests blobs, and streams correctness/performance results back to the host.
- **Host implementation:** the host now has a real J-Link RTT transport using `pylink-square`. It resolves `_SEGGER_RTT` from the linked ELF and starts RTT with an explicit control-block address.
- **Observed limitation:** SEGGER CLI auto-discovery (`JLinkRTTLogger`) did not find the control block on this board/firmware, so the working hardware path currently uses explicit RTT block-address startup rather than auto-discovery.

## PMU/DWT reuse

helia-profiler already provides useful patterns to reuse:

- counter registry/group planning
- RTT control-block discovery and direct read/write
- overflow-aware result models
- section-size probing and memory reporting concepts

The streaming implementation should align with those semantics instead of inventing incompatible PMU naming or overflow behavior.

## Timing boundaries

Performance measurements must exclude:

- host-side plan construction
- transport framing and CRC work
- target-side frame parsing and blob-copy protocol handling
- output streaming back to the host

Performance measurements may include:

- adapter argument validation that is intrinsic to a kernel invocation
- required scratch-buffer zeroing/prepare steps when they are part of the kernel contract
- repeated kernel invocations within one calibrated sample window

The host/fake-target path still simulates timing for hardware-independent tests. The live Apollo510 firmware path now performs real DWT cycle capture and real PMU event capture after correctness passes.

## Adding a new adapter

1. Pick a stable public CMSIS-NN API and assign a stable numeric kernel ID.
2. Add a catalog entry in the firmware-side catalog (`benchmark_server_catalog.*`).
3. Define a compact adapter metadata schema:
   - scalar fields
   - streamed blobs
   - scratch requirements
   - repeated-invocation/stateful constraints
4. Add host-side case-bundle generation in `helia_core_tester/perf_stream/case_bundle.py`.
5. Add host-side fake-target support for the same operator so protocol tests stay hardware-independent.
6. Add firmware-side dispatch code that calls the real CMSIS-NN API.
7. Prove retention in the linked firmware image with `arm-none-eabi-nm`.
8. Add:
   - framing/transfer tests if the blob mix is new
   - end-to-end fake-target session tests
   - firmware byte-compat or host-harness tests when practical

Current examples:

- `arm_abs_s8`: full host fake-target slice + real firmware C dispatch + host C session harness
- `arm_convolve_s8`: full host fake-target slice + real firmware C dispatch compiled/linked into the benchmark-server image

## Firmware sizing methodology

Two sizing checkpoints now exist:

1. **Phase 0 universal size probe**
   - goal: prove the whole retained ns-cmsis-nn library fits for a target profile
   - artifact: `artifacts/perf_stream/phase0/*/memory_report.json`
2. **Real benchmark-server firmware image**
   - goal: measure the actual streaming skeleton with protocol, RTT binding, catalog, session state, and adapters
   - artifact: `artifacts/perf_stream/benchmark_server/memory_report.json`

Both reports are generated from:

- the final linked ELF
- `arm-none-eabi-size`
- `arm-none-eabi-nm`
- `arm-none-eabi-objdump -h`
- the real Apollo510 linker script memory regions

Reported percentages are computed against:

- `MCU_MRAM` for flash image bytes
- `MCU_TCM` for static TCM usage before heap

## Result bundle

The streaming run writes a portable bundle under:

`artifacts/reports/performance_stream/<session_id>/`

Key files:

- `session_manifest.json`
- `session_summary.json`
- `memory_report.json`
- `kernel_catalog.json`
- `cases.json`
- `case_summary.csv`
- `raw_samples.csv`
- `protocol_trace.jsonl`
- `correctness/<case_id>.json`
- `outputs/<case_id>.bin`
- `logs/host.log`
- `logs/target.log`
- `junit.xml`

## Real vs simulated status by layer

- **Host HCTP framing/CRC/session validation:** real and unit-tested in Python.
- **Host case generation/comparison/statistics:** real and unit-tested in Python.
- **Loopback/fake-target transport and end-to-end sessions:** simulated, but executed for real in tests.
- **Firmware build/profile sizing:** real cross-compiled Cortex-M55 Apollo510 artifacts.
- **Firmware HELLO/catalog frame construction:** real C implementation, byte-for-byte decoded by the Python HCTP decoder on the host.
- **Firmware session state machine:** real C implementation for `HELLO_ACK -> CAPABILITIES -> LOAD_PLAN -> REQUEST_CASE -> CASE_META -> REQUEST_BLOB* -> CASE_READY -> RUN_CORRECTNESS -> CORRECTNESS_RESULT/OUTPUT_* -> RUN_PERFORMANCE -> SAMPLE_RESULT -> CASE_COMPLETE -> SESSION_COMPLETE`; executed both in a host-compiled C harness (`arm_abs_s8`) and on real Apollo510 hardware (`arm_abs_s8` + `arm_convolve_s8`).
- **Firmware RTT transport binding:** real compile-time integration against neuralspotx's SEGGER RTT target sources; exercised on real Apollo510 hardware.
- **Firmware kernel adapter dispatch:** real C adapters compiled and linked against the real CMSIS-NN APIs. `arm_abs_s8` and `arm_convolve_s8` are now session-executed on real Apollo510 hardware.
- **Real flash/run/RTT/PMU data capture:** verified on Apollo510 for the current two-operator vertical slice.

## Incremental implementation plan

1. write this design and size-probe tooling
2. build a universal linked firmware size prototype
3. add host HCTP framing + loopback/fake transports
4. add one simple stateless adapter vertical slice
5. add correctness streaming + output reconstruction
6. add DWT/PMU sample transport and reporting
7. add one complex adapter with scratch/multi-blob metadata
8. integrate NSX flash/RTT session handling
9. expand catalog/adapters/tests/docs

## Verified and unverified boundaries

Current remaining boundary:

- Corstone-300 FVP execution may still be blocked by missing Linux-only binaries
- RTT auto-discovery via SEGGER CLI tools remains unreliable on this board/firmware; explicit `_SEGGER_RTT` address startup is the working path

Loopback/fake-target validation remains the hardware-independent proof path; Apollo510 live RTT now covers the first real-hardware proof path.

## Hardware smoke-test commands

Cross-build the benchmark-server firmware skeleton for Apollo510/Cortex-M55:

```bash
cmake -S . -B build/perf_stream/benchmark_server_hw \
  -DCMAKE_TOOLCHAIN_FILE=cmake/nsx/toolchains/arm-none-eabi-gcc.cmake \
  -DCMAKE_OSX_ARCHITECTURES= \
  -DHELIA_HARDWARE_BUILD=ON \
  -DHELIA_BUILD_GENERATED_TESTS=OFF \
  -DHELIA_BUILD_PERF_STREAM_BENCHMARK_SERVER=ON \
  -DHELIA_HARDWARE_BOARD=apollo510_evb \
  -DTARGET_CPU=cortex-m55 \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build/perf_stream/benchmark_server_hw --target hct_benchmark_server -j
```

Flash once through the NSX-generated SEGGER target:

```bash
cmake --build build/perf_stream/benchmark_server_hw --target hct_benchmark_server_flash
```

Run the real host-target RTT session and write a result bundle:

```bash
uv run python - <<'PY'
from pathlib import Path
from helia_core_tester.perf_stream.hardware_run import run_apollo510_stream_session
run_apollo510_stream_session(Path.cwd(), serial_no=1160002276, session_id='apollo510-live-session')
PY
```
