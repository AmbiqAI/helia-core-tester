# Performance streaming implementation report

## Scope completed in this working tree

This snapshot now includes both the hardware-independent proof path and a live Apollo510 hardware run of the streaming benchmark-server vertical slice.

## Implemented components

### Host-side

- Python HCTP framing with 32-byte headers, little-endian encoding, payload/header CRC32, session/sequence validation.
- Loopback and fake-target transports for deterministic tests.
- Real J-Link RTT transport via `pylink-square`, using the firmware `_SEGGER_RTT` symbol address from the linked ELF.
- Streamable case-bundle generation (binary blobs + `case_manifest.json`) for:
  - `arm_abs_s8`
  - `arm_convolve_s8`
- Host correctness reuse through Helia Core Tester’s existing comparison rules.
- Host-side PMU/DWT sample normalization and statistics:
  - median
  - MAD
  - p90
  - p99
  - unsupported/overflow propagation
- Result-bundle writer under `artifacts/reports/performance_stream/<session_id>/`.
- Real Apollo510 hardware runner: `helia_core_tester/perf_stream/hardware_run.py`.

### Firmware-side

- Apollo510/Cortex-M55 benchmark-server target: `hct_benchmark_server`.
- Real C HCTP encoder/decoder.
- Real firmware HELLO + kernel-catalog frame emission.
- Real firmware SEGGER RTT transport binding using neuralspotx RTT target sources.
- Real firmware session loop for:
  - `HELLO_ACK`
  - `CAPABILITIES`
  - `LOAD_PLAN`
  - `REQUEST_CASE`
  - `CASE_META`
  - `REQUEST_BLOB`
  - blob assembly + CRC/alignment validation
  - `CASE_READY`
  - `RUN_CORRECTNESS`
  - `CORRECTNESS_RESULT`
  - `OUTPUT_BEGIN/OUTPUT_CHUNK/OUTPUT_END`
  - `RUN_PERFORMANCE`
  - `SAMPLE_RESULT`
  - `CASE_COMPLETE`
  - `SESSION_COMPLETE`
- Real target adapter dispatch compiled and executed on hardware for:
  - `arm_abs_s8`
  - `arm_convolve_s8`
- Real DWT cycle capture on hardware.
- Real PMU event capture on hardware for:
  - `ARM_PMU_INST_RETIRED`
  - `ARM_PMU_MEM_ACCESS`
  - `ARM_PMU_MVE_INST_RETIRED`

## Supported operators in the current snapshot

### Executed end-to-end on real Apollo510 hardware

- `arm_abs_s8`
- `arm_convolve_s8`

### Catalogued and retained in the firmware image

- `arm_abs_s8`
- `arm_convolve_s8`
- `arm_add_s8`
- `arm_sub_s8`
- `arm_mul_s8`
- `arm_minimum_s8`
- `arm_maximum_s8`

## Final firmware size and margins

Measured from `artifacts/perf_stream/benchmark_server/memory_report.json`:

- Flash image: **332,452 / 4,128,768 bytes** (**8.05%**)  
- TCM static before heap: **56,212 / 507,904 bytes** (**11.07%**)  
- Heap available: **451,692 bytes**
- Gate result: **flash pass**, **TCM pass**

Primary artifacts:

- `build/perf_stream/benchmark_server_gcc2/perf_stream/hct_benchmark_server.elf`
- `build/perf_stream/benchmark_server_gcc2/perf_stream/hct_benchmark_server.bin`
- `build/perf_stream/benchmark_server_gcc2/perf_stream/hct_benchmark_server.map`
- `artifacts/perf_stream/benchmark_server/memory_report.json`

## Commands run

### Targeted perf-stream tests

```bash
uv run pytest -q \
  helia_core_tester/tests/test_perf_stream_hctp.py \
  helia_core_tester/tests/test_perf_stream_vertical_slice.py \
  helia_core_tester/tests/test_perf_stream_transfer_measurement.py \
  helia_core_tester/tests/test_perf_stream_c_wire_compat.py \
  helia_core_tester/tests/test_perf_stream_firmware_messages.py \
  helia_core_tester/tests/test_perf_stream_firmware_session.py \
  helia_core_tester/tests/test_perf_stream_result_bundle.py
```

Observed result:

- `35 passed in 1.19s`

### Full helia-core-tester suite

```bash
uv run pytest -q
```

Observed result:

- `11 failed, 241 passed, 1 warning in 6.90s`

Those 11 failures remain the same pre-existing/unrelated failures previously baseline-compared from a clean worktree.

### Benchmark-server build + flash

```bash
cmake --build build/perf_stream/benchmark_server_gcc2 --target hct_benchmark_server_flash
```

Real transcript captured at:

- `artifacts/perf_stream/hardware_probe/hct_benchmark_server_flash.txt`

### Real Apollo510 streaming session

```bash
uv run python - <<'PY'
from pathlib import Path
from helia_core_tester.perf_stream.hardware_run import run_apollo510_stream_session
run_apollo510_stream_session(Path.cwd(), serial_no=1160002276, session_id='apollo510-live-session')
PY
```

Real transcript captured at:

- `artifacts/perf_stream/hardware_probe/apollo510_hardware_run.txt`

## Real hardware evidence collected

### J-Link flash

Real flash succeeded through the NSX-generated target with the connected on-board probe:

- device: `AP510NFA-CBR`
- probe serial: `1160002276`
- interface: `SWD`
- speed: `4000 kHz`
- flash result: `O.K.`
- programmed range: `335872 bytes`
- reported program speed: `210 KB/s`

### Live HELLO / RTT session

The benchmark server emitted a real HELLO frame over RTT and completed a real host-target session using the same HCTP framing bytes as the Python implementation.

Observed real protocol sequence for the live run included:

- `RX:HELLO`
- `TX:HELLO_ACK`
- `TX:LOAD_PLAN`
- `RX:CAPABILITIES`
- `RX:REQUEST_CASE`
- `TX:CASE_META`
- `RX:REQUEST_BLOB`
- `TX:BLOB_CHUNK`
- `RX:CASE_READY`
- `TX:RUN_CORRECTNESS`
- `RX:CORRECTNESS_RESULT`
- `RX:OUTPUT_*`
- `TX:RUN_PERFORMANCE`
- `RX:SAMPLE_RESULT`
- `RX:CASE_COMPLETE`
- `RX:SESSION_COMPLETE`

### Real correctness results

From `artifacts/perf_stream/hardware_probe/apollo510_hardware_run.txt` / `artifacts/reports/performance_stream/apollo510-live-session/`:

- `abs_hw_live` (`arm_abs_s8`): correctness **passed**
- `conv_hw_live` (`arm_convolve_s8`): correctness **passed**

### Real performance samples captured on Apollo510

Final captured run (`apollo510-live-session`):

- `abs_hw_live`
  - median cycles: **6104.0**
  - MAD: **4.0**
  - p90: **6113.8**
  - p99: **6134.68**
  - PMU counters captured: CPU cycles, instructions retired, memory accesses, MVE instructions retired
- `conv_hw_live`
  - median cycles: **32463.0**
  - MAD: **6.0**
  - p90: **32486.4**
  - p99: **32487.84**
  - PMU counters captured: CPU cycles, instructions retired, memory accesses, MVE instructions retired

Representative raw samples from the real run:

- `abs_hw_live cpu_0`: `6137`, `6105`, `6105` cycles
- `abs_hw_live memory_0`: `1361`, `1360`, `1361` memory-access counts
- `conv_hw_live cpu_0`: `32467`, `32486`, `32445` cycles
- `conv_hw_live mve_0`: `1844`, `1844`, `1844` MVE-inst-retired counts

### Real result bundle

Generated from the live Apollo510 run:

- `artifacts/reports/performance_stream/apollo510-live-session/`
  - `session_manifest.json`
  - `session_summary.json`
  - `memory_report.json`
  - `kernel_catalog.json`
  - `cases.json`
  - `case_summary.csv`
  - `raw_samples.csv`
  - `protocol_trace.jsonl`
  - `junit.xml`
  - `correctness/*.json`
  - `outputs/*.bin`
  - `logs/host.log`
  - `logs/target.log`

## Important hardware debugging notes

Two genuine hardware issues were hit and resolved during bring-up:

1. **NSX flash target path bug**  
   The generated SEGGER flash target initially tried to load:
   `build/perf_stream/benchmark_server_gcc2/hct_benchmark_server.bin`  
   while the actual `.bin` lived under `.../perf_stream/hct_benchmark_server.bin`.  
   Fix: copy the built `.bin`/`.elf` into the build root after link so the NSX-generated flash target can find them.

2. **RTT auto-discovery failure from SEGGER CLI tools**  
   `JLinkRTTLogger` reported:
   `Searching for RTT Control Block...RTT Control Block not found. Cannot get data.`  
   even though `_SEGGER_RTT` existed in SRAM and the firmware was running.  
   Workaround used successfully: resolve `_SEGGER_RTT` from the linked ELF and start RTT explicitly with `pylink` using `rtt_start(block_address=...)`.

Additionally, the first live Conv2D correctness attempt stalled because the MVE `arm_convolve_s8` path requires a populated `weight_sum_ctx`; the firmware adapter/session path was updated to compute real weight sums with `arm_convolve_weight_sum()` before dispatch.

## Hardware verified vs unverified

### Verified live on Apollo510

- real cross-built firmware flashes and boots
- real HELLO emission over RTT
- real host RTT attach through J-Link
- real HCTP HELLO/ACK/plan/case/blob/correctness/performance/session messaging
- real single-flash persistent session across multiple operators
- real one-case-at-a-time blob pull from target
- real correctness execution for `arm_abs_s8`
- real correctness execution for `arm_convolve_s8`
- real output reconstruction on host
- real DWT cycle measurements from hardware
- real PMU event measurements from hardware
- real result-bundle generation from hardware data

### Still not verified / still partial

- PMU overflow handling on real hardware was not stress-tested to overflow.
- Auto-calibration with `iterations_per_sample = 0` exists in firmware logic but was not exercised in the live Apollo510 run; the live run used fixed `iterations=4` from the case timing plan.
- The current live session uses one common timing plan for both cases because the host `LOAD_PLAN` format is session-scoped; per-case live timing plans are not implemented yet.
- JLinkRTTLogger/JLinkRTTClient auto-discovery was not made to work; the working live path uses `pylink` + explicit RTT control-block address.
- FVP execution remains unverified here.

## Exact hardware smoke-test steps

1. Build / flash:

```bash
cmake --build build/perf_stream/benchmark_server_gcc2 --target hct_benchmark_server_flash
```

2. Run one live Apollo510 session and write the real result bundle:

```bash
uv run python - <<'PY'
from pathlib import Path
from helia_core_tester.perf_stream.hardware_run import run_apollo510_stream_session
result, bundle = run_apollo510_stream_session(Path.cwd(), serial_no=1160002276, session_id='apollo510-live-session')
print(bundle)
for case in result.cases:
    print(case.case_bundle.case_id, case.statistics.median_cycles)
PY
```

3. Inspect artifacts:

```bash
ls artifacts/reports/performance_stream/apollo510-live-session
cat artifacts/reports/performance_stream/apollo510-live-session/case_summary.csv
cat artifacts/reports/performance_stream/apollo510-live-session/raw_samples.csv
```
