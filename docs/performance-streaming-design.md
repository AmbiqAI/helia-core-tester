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

## Universal size-probe sizing rule

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
- validate firmware TARGET_INFO/catalog compatibility
- stream case metadata and blobs in bounded chunks
- reconstruct streamed outputs
- reuse existing comparison rules
- compute statistics from raw samples
- write result bundles and JUnit/CSV/JSON artifacts

### Target responsibilities

Firmware will:

- expose a versioned TARGET_INFO block (capabilities, PMU width, session limits)
- expose a versioned kernel catalog with stable numeric IDs
- request cases/blobs from the host (target-driven pull)
- bind streamed blobs into validated adapter metadata
- run one correctness invocation
- stream outputs back once per case
- run warmups + measured iterations only after the host's CORRECTNESS_ACK
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

1. target sends `TARGET_INFO`
2. host sends `TARGET_INFO_ACK`
3. target sends `KERNEL_CATALOG` (one or more pages)
4. host sends `SESSION_PLAN`
5. target sends `REQUEST_CASE`
6. host sends `CASE_META`
7. target requests blobs chunk-by-chunk (`REQUEST_BLOB`)
8. host sends `BLOB_CHUNK`
9. target sends `CASE_READY`
10. host sends `RUN_CORRECTNESS`
11. target streams output + correctness result
12. host sends `CORRECTNESS_ACK`
13. host sends `RUN_PERFORMANCE` when allowed
14. target streams raw sample results
15. target sends `CASE_COMPLETE`
16. loop until `SESSION_COMPLETE`

### Messages (HCTP v3)

Protocol version 3 (`hctp.SUPPORTED_VERSION` / `HCTP_SUPPORTED_VERSION`); a peer on
another version is refused at the header. Message ids are compact and in protocol
order; every payload is encoded and decoded on the host by exactly one pair of
functions in `helia_core_tester/hardware/wire.py`, which the host session and the
fake target both use, and which the host-compiled C harnesses check against the
firmware byte for byte.

| id | message | direction | payload |
| --- | --- | --- | --- |
| 1 | `TARGET_INFO` | target -> host | build id, catalog hash, capabilities, PMU width, session limits (below) |
| 2 | `TARGET_INFO_ACK` | host -> target | empty |
| 3 | `KERNEL_CATALOG` | target -> host | one page of catalog entries; `HCTP_FLAG_MORE` on every non-final page |
| 4 | `SESSION_PLAN` | host -> target | timing plan, PMU passes, case list (below) |
| 5 | `REQUEST_CASE` | target -> host | `u16 case_index` |
| 6 | `CASE_META` | host -> target | case id, kernel id, comparison config, scalars, blob descriptors, scratch bytes |
| 7 | `REQUEST_BLOB` | target -> host | `u32 blob_id, u32 offset, u16 max_length` |
| 8 | `BLOB_CHUNK` | host -> target | `u32 blob_id, u32 offset, raw data` |
| 9 | `CASE_READY` | target -> host | `u32 blob_id, u32 bytes_received` |
| 10 | `RUN_CORRECTNESS` | host -> target | empty |
| 11 | `CORRECTNESS_RESULT` | target -> host | `i32 status` |
| 12 | `OUTPUT_BEGIN` | target -> host | `u32 offset (0), u32 length` |
| 13 | `OUTPUT_CHUNK` | target -> host | `u32 offset, u32 length, bytes` |
| 14 | `OUTPUT_END` | target -> host | `u32 length, u32 checksum` |
| 15 | `CORRECTNESS_ACK` | host -> target | `u8 passed` (informational) |
| 16 | `RUN_PERFORMANCE` | host -> target | empty |
| 17 | `SAMPLE_RESULT` | target -> host | one sample of one pass (below) |
| 18 | `CASE_COMPLETE` | target -> host | `text case_id, u8, u8, u32 workspace_used_bytes` |
| 19 | `SESSION_COMPLETE` | target -> host | `u16 case_count` |
| 20 | `ERROR` | target -> host | `text message` |

All integers are little-endian; `text` is `u16 length + UTF-8 bytes`; `raw` is
`u32 length + bytes`.

`TARGET_INFO` (target -> host): `text build_id`, 32-byte catalog SHA-256,
`u32 max_frame_payload`, `u32 runtime_arena_capacity`, `u8 transfer_mode`,
`u8 output_mode`, `text board_id`, `text target_cpu`, `u8 transport_kind`,
`u32 capability_flags`, `u8 pmu_counter_slots`, `u32 max_rx_payload`,
`u16 max_cases_per_session`, `u8 max_passes`.
`capability_flags` bit 6 is `HCT_CAP_PMU_ARMV8M`, set only when the firmware was
built for a core whose device header declares `__PMU_PRESENT == 1`;
`pmu_counter_slots` is `__PMU_NUM_EVENTCNT` (8 on Cortex-M55, 0 without a PMU).
`max_rx_payload` is the largest frame payload the target's fixed receive buffer can
hold (`HCT_SERVER_RX_BUFFER_BYTES - HCTP_HEADER_SIZE`, 2016 today);
`max_cases_per_session` and `max_passes` are the firmware's `HCT_SERVER_MAX_CASES`
(32) and `HCT_SERVER_MAX_PASSES` (16). After the handshake the advertised values are
authoritative: the host derives its batching (`session.TargetLimits`) from every
session's `TARGET_INFO`, cuts each batch so the plan stays within all three, checks
its chained-pair planning rule (four counters per pass) against `pmu_counter_slots / 2`,
and refuses any later outbound payload (`CASE_META` included) larger than
`max_rx_payload`, which the firmware applies to every frame it receives. The host also
mirrors the two firmware constants as `measurement.MAX_CASES_PER_PLAN` and
`MAX_PASSES_PER_PLAN` (lockstep-tested against the header), but only as early-rejection
bounds so `--pmu-counters` can fail at option parsing, before generate, build and flash.

`SESSION_PLAN` (host -> target): `u16 case_count`, `u8 transfer_mode`, `u16 warmups`,
`u16 samples`, `u32 iterations_per_sample`, `u32 min_cycles`, `u32 max_iterations`,
`u8 pass_count` and per pass `text pass_name`, `u8 chained`,
`u8 counter_count`, `u16 event_id[counter_count]`, then per case `text case_id`,
`u32 kernel_id`. The firmware rejects `pass_count > max_passes`, `counter_count > 4`,
`case_count > max_cases_per_session` and (when it has a PMU) a pass needing more
slots than it advertised (`chained ? 2 * counter_count : counter_count`); event ids
are not validated against a list -- whatever the host asks for is programmed and
reported back.

`SAMPLE_RESULT` (target -> host, one per sample per pass): `u16 sample_index`,
`u32 iterations`, `u64 cycles`, `text pass_name`, `u8 counter_count`, then per
counter `text name`, `u16 event_id`, `u64 value`, `u8 overflow`, `u8 supported`.
`cycles` is the DWT `CYCCNT` delta around the timed loop. The first counter entry
is always `ARM_PMU_CPU_CYCLES` (event `0x0011`) read from the PMU cycle counter
`CCNTR`, with `overflow` = bit 31 of the PMU overflow status register. On Armv8.1-M
`PMU_CCNTR` and `DWT_CYCCNT` may alias the same underlying counter (unverified here:
not checked against the Armv8.1-M Architecture Reference Manual or the Cortex-M55
TRM). On Apollo510 `CCNTR` reads 120-250 cycles above the DWT delta per sample
(about 30 cycles per invocation at four iterations), which is consistent with the
order the firmware starts and reads the two either way, so `cycles` is not claimed
as an independent measurement. The u64 `cycles` field is kept for DWT-only targets
(where it is the only cycle source) and as a read-order sanity check against
`CCNTR`. The remaining counter entries are the pass's event counters in plan order.
The firmware sends every `name` empty and the host resolves
names from `assets/pmu/armv8m_pmu_events.json` by event id (unknown ids become
`event_0x....`). On a DWT-only build the cycle entry is the DWT value and every
event counter comes back `supported = 0`.

### PMU passes, chained counters and overflow

The host plans counters into passes (`measurement.plan_counter_passes`): per counter
group, at most four event counters per pass, named `<group>_<n>`. Each pass reruns
the case's warmups and samples with its counters programmed, so a selection like
`mve:all` (34 events) costs nine passes. `ARM_PMU_CPU_CYCLES` never occupies an
event-counter slot -- it is reported from `CCNTR` in every pass -- so it is stripped
when planning and a cycles-only selection still yields one empty `cpu_0` pass.

A plan carries at most `HCT_SERVER_MAX_PASSES` (16) passes. The target advertises
the limit in `TARGET_INFO` (`max_passes`) and `HostSession` refuses a longer pass
list at the handshake, before `TARGET_INFO_ACK`; the host also mirrors the constant
as `measurement.MAX_PASSES_PER_PLAN` so the `--pmu-counters` parser and
`session_runner.run_case_bundles` can refuse the selection before generate/build/
flash and before the probe is opened, and the fake target's `SESSION_PLAN` admission
applies the same bound. Every error names the planned passes. Passes are never split
across sessions, so `cpu:all memory:all mve:all` (5 + 4 + 9 = 18 passes) is an
error; select fewer counters per run. An empty name list (`mve:,`) is rejected the
same way instead of degrading to a cycles-only pass.

Armv8.1-M event counters are 16 bits wide. Passes are chained by default: counter
`i` is programmed into slot `2i` and slot `2i+1` is programmed with `ARM_PMU_CHAIN`
(event `0x001E`), which increments on the even slot's overflow, so the pair reads as
`(high << 16) | low`, a 32-bit counter. Four chained counters use all eight slots.
Per sample the firmware disables the PMU, resets the event counters and `CCNTR`,
clears the overflow status (`ARM_PMU_Set_CNTR_OVS(0xFFFFFFFF)`), enables the pass's
slots plus `CCNTR`, runs the timed loop, disables the counters, reads the values and
`ARM_PMU_Get_CNTR_OVS()`, and clears the bits that were set. A chained counter's
overflow is the odd slot's bit; an unchained counter's is its own slot's bit. Any
overflow in any sample of a case sets `overflow_detected` and clears
`valid_for_regression` for that case in the result bundle; the DWT cycle statistics
are unaffected.

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

The target `TARGET_INFO` includes a hash of the full catalog. The host refuses plans that reference missing IDs.

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

- **Live-real on Apollo510:** the benchmark-server target now boots on real Apollo510 hardware, initializes the real `SEGGER_RTT` target sources from `neuralspotx/examples/coremark/src/rtt/`, emits TARGET_INFO over RTT, accepts host frames, requests blobs, and streams correctness/performance results back to the host.
- **Host implementation:** the host now has a real J-Link RTT transport using `pylink-square`. It resolves `_SEGGER_RTT` from the linked ELF and starts RTT with an explicit control-block address.
- **Observed limitation:** SEGGER CLI auto-discovery (`JLinkRTTLogger`) did not find the control block on this board/firmware, so the working hardware path uses explicit RTT block-address startup rather than auto-discovery, with a host-side scan for the `SEGGER RTT` magic as the fallback. See "Flash and run" for the full reset/attach sequence.

## PMU/DWT reuse

helia-profiler already provides useful patterns to reuse:

- counter registry/group planning
- RTT control-block discovery and direct read/write
- overflow-aware result models
- section-size probing and memory reporting concepts

The streaming implementation aligns with those semantics instead of inventing
incompatible PMU naming or overflow behavior: the event catalog
(`assets/pmu/armv8m_pmu_events.json`) is transcribed from heliaPROFILER, the
`--pmu-counters GROUP:SELECTION` syntax is hpx's, and the firmware's per-sample
reset/clear-OVS/read/clear-set-bits sequence mirrors the hpx PMU profiler. The
firmware uses raw CMSIS `pmu_armv8.h` rather than the NSX PMU module.

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

The host/fake-target path still simulates timing for hardware-independent tests (the fake target honours the 16/32-bit counter widths so overflow handling is testable). The live Apollo510 firmware path performs real DWT cycle capture and real PMU event capture (CCNTR plus up to four chained event counters per pass) after correctness passes.

## Adding a new adapter

1. Pick a stable public CMSIS-NN API and assign a stable numeric kernel ID.
2. Add a catalog entry in the firmware-side catalog (`benchmark_server_catalog.*`).
3. Define a compact adapter metadata schema:
   - scalar fields
   - streamed blobs
   - scratch requirements
   - repeated-invocation/stateful constraints
4. Add host-side case-bundle generation in `helia_core_tester/hardware/case_bundle.py`.
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

1. **Universal size probe** (`memory_report.build_size_probe`)
   - goal: prove the whole retained ns-cmsis-nn library fits for a target profile
   - artifact: `artifacts/hardware/size_probe/<board>/<variant>/memory_report.json`
2. **Real benchmark-server firmware image** (`hardware memory-report`, `memory_report.generate_memory_report`)
   - goal: measure the actual streaming skeleton with protocol, RTT binding, catalog, session state, and adapters
   - artifact: `artifacts/hardware/benchmark_server/memory_report.json`, copied into every result bundle

Both reports come from one analysis (`helia_core_tester/hardware/memory_report.py`) of:

- the final linked ELF
- `arm-none-eabi-size`
- `arm-none-eabi-nm`
- `arm-none-eabi-objdump -h`
- the board's NSX linker script memory regions -- read back out of the `-T` flag in
  the build tree's generated Ninja file, i.e. the script the linker actually used,
  with the SoC's default script in the synced SDK module as the fallback. The
  flash/RAM region names (`flash_region`, `ram_region`) come from the board's row in
  `assets/hardware_boards.yaml`

Reported percentages are computed against:

- `MCU_MRAM` for flash image bytes
- `MCU_TCM` for static TCM usage before heap

## Kernel build parity

The point of running these cases on real silicon is to measure the kernels as they
will actually ship, which means the firmware has to compile ns-cmsis-nn the way
every other consumer does. It did not: the firmware used to be built by pointing
CMake at this repo's root `CMakeLists.txt` with `HELIA_HARDWARE_BUILD=ON`, which
`add_subdirectory()`-ed the kernel repo under a flag set assembled here rather than
by the kernels' own NSX module.

Building the firmware as an NSX app fixes that by construction: `nsx-cmsis-nn` is
added as a module, so it brings its own `nsx/CMakeLists.txt`, its own
`NSX_CMSIS_NN_OPTIMIZATION`, and the board's `nsx::board_flags`. The effective
kernel compile line on `apollo510_evb`/arm-none-eabi-gcc is now:

```
-O3 -DNDEBUG -std=gnu11 -Ofast -mthumb -mcpu=cortex-m55 -mfloat-abi=hard
-fshort-enums -ffunction-sections -fdata-sections -fomit-frame-pointer
-fno-exceptions -MMD -MP -Wall -g -O3 -ffast-math
```

with `-DCMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY`, `-DARM_NN_ENABLE_F32=1`,
`-DARM_NN_ENABLE_F16=1` and the board/SoC define set (`ARMCM55`, `AM_PART_APOLLO510`,
`NSX_SOC_HAS_MVE=1`, ...). That flag list is **byte-identical to heliaPROFILER's**
for the same board and toolchain, so a kernel number from this tool and one from hpx
are measurements of the same binary shape.

Three differences from the old line are worth naming:

- **`-mfpu` is gone.** The old path applied a global
  `add_compile_options(-mfloat-abi=hard -mfpu=fpv5-sp-d16)` to everything, including
  the kernels. `-mfpu=fpv5-sp-d16` names a scalar single-precision FPU, which is not
  what a Cortex-M55 with MVE has; `-mcpu=cortex-m55` already selects the right
  FP/MVE feature set and adding `-mfpu` on top only narrows it. NSX's board flags
  set `-mcpu` and `-mfloat-abi` and stop there. This is a correctness-of-intent fix,
  not a performance one — measured, it moves kernel time by single-digit percent and
  in one case (`arm_abs_s8`, default rescale) the wrong way.
- **Requantize inline assembly is ON.** `NSX_CMSIS_NN_USE_REQUANTIZE_INLINE_ASM`
  defaults to `OFF` in the module, and the old path never set it either way. The app
  forces it `ON`, matching hpx. `--no-requantize-inline-asm` is the A/B control; it
  changes the rendered `CMakeLists.txt`, hence the render digest, so a build
  directory cannot silently carry the other setting. Measured on this case set its
  effect is within ±0.1 %, i.e. inside noise.
- **`-O3 … -Ofast … -O3 -ffast-math` all appear on the kernel line.** `-O3 -DNDEBUG`
  comes from `CMAKE_BUILD_TYPE=Release`, `-Ofast` from the module's own
  `NSX_CMSIS_NN_OPTIMIZATION`, and the trailing `-O3 -ffast-math` from the board
  flags target's interface options, which are emitted last and therefore win. It is
  redundant but it is exactly what heliaPROFILER compiles with, byte for byte, so it
  is recorded here rather than "fixed" — diverging from hpx to tidy a flag list would
  cost the parity this section exists to establish.

Both kernel switches are written into the app's `CMakeLists.txt` above
`nsx_bootstrap_app()` rather than passed as `-D` at configure time: an `option()`
default cannot be overridden once its module has been added.

### The harness was compiled at -O0 (and the timed window includes it)

The old hardware build gave the kernel archive `-Ofast` and gave **every other
target no optimization flag at all**. `hct_benchmark_server`, `helia_test_runtime`
and `retarget` compiled with
`-mcpu=cortex-m55 -mthumb -mfloat-abi=hard -mfpu=fpv5-sp-d16` and nothing else, i.e.
at GCC's `-O0` default. The benchmark server's per-case dispatch, adapter shims and
session code sit *inside* the timed window, so every hardware number this repo has
ever produced carried unoptimized tester code in its measurement.

As an NSX app the tester sources pick up the board flags target's `-O3 -ffast-math`
like everything else, so that overhead is gone. Measured on `apollo510_evb` by
building this branch twice — once as it ships, once with only the tester-side sources
forced back to `-O0` via `set_source_files_properties(... COMPILE_OPTIONS "-O0")`
(source-file options are emitted after the board flags target's interface options,
which is the only placement where `-O0` wins) — the split is:

| contribution | median over 16 cases |
|---|---:|
| kernel flags + board/SoC defines (old → new, both with `-O0` harness) | −6.8 % |
| harness `-O0` → `-O3` (same kernels) | **−45.3 %** |
| total | −51.2 % |

The harness share scales inversely with case size, as it must: it is −1.4 % on the
186 k-cycle grouped depthwise convolution and −60 % on a 1.5 k-cycle fully-connected
case. `ARM_PMU_MVE_INST_RETIRED` is unchanged across the whole comparison (e.g.
17 290 in both legs for the depthwise case), confirming the kernel code paths
themselves did not move.

**Consequence: hardware numbers from before this change are not comparable with
numbers after it.** The bundle baseline resets here. A regression comparison must
start from a post-change run.

### A/B session ids

A firmware change that moves the numbers is worth measuring against the mechanism it
replaces, on the same board with the same generated cases. The convention is one
session id per (leg, repetition, case group) -- `ab-<leg>-<rep>-<group>`, e.g.
`ab-A-1-basicmath` -- with the legs interleaved rather than run in blocks, so probe
or thermal drift shows up as A-vs-A disagreement instead of as a result. Reject the
comparison if the two A repetitions disagree by more than ~0.1 % on a case; read a
kernel-level regression as >1 % slower with `ARM_PMU_INST_RETIRED` up and
`ARM_PMU_MVE_INST_RETIRED` down, which is the signature of a lost vectorisation
rather than of noise.

When the change under test touches anything the timed window compiles — not just the
kernels — add a third leg that isolates it, as the harness `-O0` measurement above
does. A whole-firmware A/B tells you the number moved; it does not tell you which
half of the firmware moved it, and the answer is not always the half you changed on
purpose.

## Result bundle

The streaming run writes a portable bundle under:

`artifacts/reports/hardware/<session_id>/`

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
- **Firmware TARGET_INFO/catalog frame construction:** real C implementation, byte-for-byte decoded by the Python HCTP decoder on the host.
- **Firmware session state machine:** real C implementation for `TARGET_INFO_ACK -> KERNEL_CATALOG -> SESSION_PLAN -> REQUEST_CASE -> CASE_META -> REQUEST_BLOB* -> CASE_READY -> RUN_CORRECTNESS -> CORRECTNESS_RESULT/OUTPUT_* -> RUN_PERFORMANCE -> SAMPLE_RESULT -> CASE_COMPLETE -> SESSION_COMPLETE`; executed both in a host-compiled C harness (`arm_abs_s8`) and on real Apollo510 hardware (`arm_abs_s8` + `arm_convolve_s8`).
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

## Flash and run

Flashing and resetting are heliaPROFILER's, mechanism for mechanism; the one
deliberate divergence is the build-id flash skip, which hpx has no equivalent of
and which is kept because a repeat `hardware run` on an unchanged build is the
common case here and a flash costs ~6 s of MRAM programming.

### The flash runs NSX's own recipe

`nsx_finalize_app()` writes a ready-made commander script per target at
`<build>/jlink/<target>/flash_cmds.jlink`:

```
ExitOnError 1
Reset
LoadFile "<build>/hct_benchmark_server.bin", 0x00410000
Reset
Go
Exit
```

`hardware flash` runs *that file*, verbatim, through `JLinkExe` with the resolved
probe serial (`-SelectEmuBySN`). It does not hand-roll a `loadfile`: hpx tried that
against the extension-less ELF and it **silently programmed nothing** on Apollo510 —
the board kept running the previous firmware while the host reported a successful
flash. It also does not go through the `<target>_flash` ninja target NSX generates,
because reaching a ninja target means owning a configured build tree at flash time,
which is what made the previous revision re-render the app on every flash (see
"Flash never rebuilds" below). The recipe is the proven artifact; the target is just
one way to run it.

Running someone else's script verbatim means vetting it first, the same way NSX's
`validate_flash_recipe` and hpx's `target/probe/flash.py` do
(`helia_core_tester/hardware/flash_recipe.py`):

- **`ExitOnError 1` must be armed before the first `LoadFile`.** Without it JLinkExe
  can fail a command and still exit zero, so a failed flash looks like a success;
  arming it afterwards protects nothing, because the commander runs a script top to
  bottom. Presence alone is therefore not the check — position is.
- **Some `LoadFile` must name this build's `.bin`, with an explicit address.**
  Recipes bake absolute paths, so a recipe left behind by an earlier build resolves
  happily and flashes a stale image while the run is attributed to the new build id.
  An addressless `LoadFile` is refused: J-Link accepts it and it does program flash,
  taking the destination from the image format — a destination the host cannot check,
  which is the one thing this gate exists to refuse.
- Quoted and unquoted paths, a `, reset|noreset` tail and a trailing `//` comment are
  all accepted, because JLinkExe accepts them; the grammar is hpx's, which is wider
  than NSX's own regex for exactly this reason.

Every refusal says *"Nothing was programmed"*: "refused before JLinkExe ran" and
"failed halfway through programming" call for opposite next steps.

### Verification

Two gates after the flash:

1. **Exit status**, which is trustworthy precisely because `ExitOnError 1` was
   verified, plus a text tripwire — one of `Flash download: Total` or `Skipped.
   Contents already match` must appear. A bare connection `O.K.` is printed before
   any programming and does not count.
2. **The flash bank.** J-Link prints `Flash download: Bank 0 @ 0x00410000: …`, the
   base of the bank it programmed (its format string carries one address for N
   ranges, so it is not a per-range destination). The recipe's address must fall in
   a bank J-Link named. That catches a build dir configured for another part —
   every Ambiq part's app load address is its bank base — but *not* a wrong address
   inside a bank J-Link did program; J-Link reports nothing finer. When no bank line
   appears at all the flash is allowed to proceed with a loud `UNVERIFIED FLASH
   DESTINATION` warning: the bank line corroborates the exit-status gate, and turning
   a J-Link rewording into a hard stop would block correct flashes with no evidence
   of a wrong one.

### The skip rule (tester-only)

A flash is skipped only when **both** halves agree: the host-side stamp
`<build>/.flashed-<serial>.sha256` matches the ELF's sha256 (this build dir last
flashed this probe with this image), *and* the board answers a short RTT session
with this build dir's `hct_build_id.txt` in TARGET_INFO. The stamp alone cannot know
what another build dir, clone or lab runner did to the same probe since. A missing
stamp, a missing build id, a different id or no TARGET_INFO at all all mean flash.
`--force` / `--force-flash` skips the question entirely.

### Flash never rebuilds

`hardware flash` renders nothing, configures nothing and compiles nothing. It
computes the render its options *imply* in memory and compares the digest with the
`.hct-nsx-app.json` state file the last `hardware build` wrote; a mismatch — an
edited baseline, a different `--cmsis-nn-root`, a changed kernel switch — is an
error naming `hardware build`, not an implicit rebuild. This is not hypothetical
tidiness: the previous revision re-rendered from the working tree on every flash,
and during the A/B that rebuilt one leg from the other leg's sources. `hardware run`
therefore calls build and flash as two steps and owns the order.

The probe serial is no longer passed to the build. It used to be, so NSX would bake
it into the generated flash target — which forced a CMake reconfigure on every run
with `--serial-no`. Nothing reads it now that the recipe is executed directly.

### Reset and RTT attach

Each batch of cases is one RTT session, and each session starts like an hpx capture:

0. **(discovery path only) pre-clean.** Apollo5 retains SRAM across reset, so a
   control block from a previously flashed firmware can outlive the reset and race
   the live one. Attach, blank the `SEGGER RTT` magic of every structurally valid
   block, release the probe; the reset that follows lets the current firmware
   republish its own.
1. **Reset through `JLinkExe`** with the script `r` / `g` / `exit` — *not*
   `pylink.reset()`. The Apollo510 secure bootloader checks for an attached debugger
   on the boot that follows a reset and will not start the application while one is
   there; the commander's exit releases the probe, and pylink's reset does not. This
   is the single most load-bearing detail in the run path.
2. **Settle** 0.25 s: the SBL phase is unobservable, and a short floor costs less
   than a failed attach.
3. **Attach pylink, retrying** until the target answers or 30 s elapse — a connect
   refused 200 ms after a reset means "still booting", not "board is gone". If the
   attach finds the core halted (a bare `JLinkExe r` without `g`, a previous debug
   session), it is resumed: a halted core publishes no RTT bytes, and the session
   would otherwise fail as a protocol timeout rather than as the stopped target it is.
4. **Start RTT at the control-block address linked into the firmware**, read from the
   ELF's `_SEGGER_RTT` symbol. This is hpx's `known_block_address` path, and hpx
   skips both the pre-clean and the scan on it for the reason that applies here too:
   the firmware re-initialises that fixed address on every boot, so a stale block
   elsewhere can never be selected. The host waits (up to 5 s) for a valid block to
   appear there rather than reading once; `.bss` is legitimately still zero while the
   SBL runs.
5. **Fallback:** if that address never comes alive, sweep the board's SRAM window
   (`rtt_scan_ranges` in `assets/hardware_boards.yaml`; DTCM `0x20000000+0x80000` for
   apollo510_evb, the same window hpx uses) for the `SEGGER RTT` magic and score the
   candidates — up-channel 0 named `HCTP_UP` dominates, then recent write activity,
   then buffer size. Only if that finds nothing does J-Link's own auto-scan run. The
   scan never wipes: the phase-0 wipe is safe only because a reset follows it.

`HCT_RTT_DISCOVERY=scan|address|auto` forces a path, which is how the discovery and
pre-clean code is exercised on hardware without deleting the ELF that address comes
from.

**No heartbeat hang detection.** hpx declares a run hung when the gap between
firmware lines exceeds a heartbeat timeout; HCTP has no periodic target signal to
time against — the firmware speaks when a case, sample or blob request is ready, and
a long case is legitimately silent for its whole duration. The existing per-read
timeouts stay the only liveness bound, and adding a "heartbeat" would mean adding a
protocol message first.

### One J-Link install

Flash, reset, probe inspection and the RTT transport all resolve the SEGGER install
the same way: `$HPX_JLINK_DLL` (the library), then `$JLINK_PATH` (the `JLinkExe`
binary or its directory), then `JLinkExe` on `PATH`, then pylink's own search for the
library. `JLinkExe` is looked for beside the resolved library as well, so
`$HPX_JLINK_DLL` alone is enough on the lab runners, and the resolved path is
exported to NSX as `$JLINK_PATH` at configure time. `helia_core_tester doctor`
prints the library, the commander with its version banner, and whether each board's
build dir carries a flash recipe.

## Hardware commands

All hardware work goes through the board-keyed `hardware` CLI group (`--board`
selects a row of `assets/hardware_boards.yaml`, which supplies the CPU, NSX board
name, SEGGER device name, SWD speed and the `build/hardware/<board>` build dir;
`--serial-no` is optional and falls back to `$HPX_JLINK_SERIAL`, then to the single
connected J-Link probe enumerated through pylink).

Build the benchmark-server firmware for the board. The firmware is an NSX app
(rendered into `build/hardware/<board>/nsx_app/`, then locked/synced/configured/built
through `neuralspotx.api` -- see "Kernel build parity" below), so the first build
resolves and vendors the SDK, board and kernel modules into that app:

```bash
uv run helia_core_tester hardware build --board apollo510_evb -j
```

Flash through the NSX-generated J-Link recipe (see "Flash and run" above for the
recipe, its validation and the bank check) -- skipped automatically when the
ELF's sha256 matches the last flash to the same probe from this build dir *and*
the board confirms it is running this build (every build carries a content-hash
build id in `<build_dir>/hct_build_id.txt`, stamped into the linked image after
the link by `scripts/patch_build_id.py` -- a sha256 over the whole flash image,
so it covers every linked library and the linker layout, not only the server
objects -- and advertised in TARGET_INFO; the skip path opens one short RTT session to
read it). Another build dir flashing the same probe, or no TARGET_INFO at all, means a
reflash. `--force` (or `hardware run --force-flash`) overrides, and every stream
also fails at TARGET_INFO if the board's build id is not the build dir's. A build dir
without `hct_build_id.txt` (firmware built before stamping) is refused by
`hardware stream` / `hardware run --skip-flash` unless
`--allow-unverified-firmware` is given:

```bash
uv run helia_core_tester hardware flash --board apollo510_evb
```

Stream the generated suite to the flashed firmware and write a result bundle
(or run generate -> build -> flash -> stream in one go with `hardware run`):

```bash
uv run helia_core_tester hardware stream --board apollo510_evb --suite both
uv run helia_core_tester hardware run --board apollo510_evb --precision fp16 --json
```

The two-kernel synthetic demo session (`arm_abs_s8` + `arm_convolve_s8`) is still
available as library code, `session_runner.run_demo_session()`, and is covered by
the fake-target tests; it is no longer a CLI command.

## Host modules

- `hctp.py`: framing (header, CRCs, sequence/session validation), `ByteWriter`/`ByteReader`.
- `wire.py`: the payload codec -- one encode/decode pair per message, shared by the
  host session and the fake target.
- `session.py`: `HostSession` (handshake, plan, per-case streaming) and `TargetLimits`,
  the batching limits derived from `TARGET_INFO`.
- `session_runner.py`: one RTT session per batch on a `BoardSpec`, case discovery
  from the generated-test tree, result-bundle writing.
- `jlink_cli.py`: the only place this package shells out to `JLinkExe` (flash
  script, `r`/`g` reset, probe inspection, version banner).
- `flash_recipe.py`: locating, validating and running NSX's `flash_cmds.jlink`, and
  checking the flash bank J-Link reports afterwards.
- `rtt_control.py`: scanning, scoring and blanking `SEGGER RTT` control blocks over
  SWD -- the fallback for when the linked control-block address cannot be used.
- `hardware_pipeline.py`: generate -> build -> flash -> stream orchestration behind
  `hardware run` / `hardware stream`.
- `memory_report.py`: the flash/RAM report and the universal size probe.
- `fake_target.py`: the host-side target double the deterministic tests run against.
