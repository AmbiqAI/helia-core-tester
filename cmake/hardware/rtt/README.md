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
control block stays compatible with J-Link.

## Which buffers HCTP actually uses

The protocol does **not** run on SEGGER's default terminal channel, and nothing
in the generated app defines `BUFFER_SIZE_UP`. `benchmark_server_transport_rtt.c`
owns its own statics and installs them on RTT buffer index 1 with
`SEGGER_RTT_ConfigUpBuffer()` / `SEGGER_RTT_ConfigDownBuffer()`:

| what | symbol | size | set by |
|---|---|---|---|
| HCTP target → host | `g_hct_rtt_up_buffer` | 8192 B | `HCT_RTT_UP_BUFFER_BYTES` |
| HCTP host → target | `g_hct_rtt_down_buffer` | 512 B | `HCT_RTT_DOWN_BUFFER_BYTES` |

Both constants live in `benchmark_server_transport_rtt.c`. That is the only
place to change the transport's buffer sizes; `BUFFER_SIZE_UP` has no effect on
it. (Buffer 0's pointer and size cannot be changed at run time by design — which
is why the transport takes buffer 1 rather than resizing SEGGER's.)

`SEGGER_RTT_Conf.h`'s defaults still cost the image something: channel 0's
`_acUpBuffer` (1024 B) and `_acDownBuffer` (16 B) are static, so they are linked
in and sit in `.bss` unused — confirmed in the shipped image:

```
$ arm-none-eabi-nm -S --size-sort hct_benchmark_server.elf
2002c440 00000400 b _acUpBuffer          <- BUFFER_SIZE_UP, unused by HCTP
2002c430 00000010 b _acDownBuffer        <- BUFFER_SIZE_DOWN, unused by HCTP
2002a430 00002000 b g_hct_rtt_up_buffer  <- the transport's own
2002a230 00000200 b g_hct_rtt_down_buffer
```

Shrinking channel 0 would reclaim ~1 KB of TCM, but it changes `.bss` and
therefore the linked image, so it is left alone here rather than folded into a
change whose hardware numbers have to stay comparable.
