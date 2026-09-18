# SEGGER RTT target sources (vendored)

`hct_benchmark_server` talks HCTP over SEGGER RTT, so the target-side RTT
implementation has to be compiled into the firmware. These three files are
SEGGER's, copied verbatim from the `neuralspotx` CoreMark example that the
hardware build used to `configure_file()` them out of.

They are vendored here rather than fetched because the hardware firmware is
now built as an NSX app against the `neuralspotx` **wheel**, which ships the
build glue and the module registry but not the repository's `examples/`.
heliaPROFILER bundles its own copy for the same reason.

Do not edit them: SEGGER asks that the RTT sources stay unmodified so the
control block stays compatible with J-Link. The compile-time up-buffer size is
set with `-DBUFFER_SIZE_UP=...` from the app CMakeLists instead.
