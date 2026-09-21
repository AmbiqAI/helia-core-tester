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

## Which buffer HCTP actually uses

The transport runs on RTT **channel 0** -- `HCT_RTT_UP_CHANNEL` and
`HCT_RTT_DOWN_CHANNEL` in `benchmark_server_transport_rtt.c` are both `0`, and
the host reads and writes index 0 to match (`JLinkRttTransport`'s
`up_buffer_index` / `down_buffer_index`). Nothing in the generated app defines
`BUFFER_SIZE_UP`.

`hct_rtt_init()` calls `SEGGER_RTT_ConfigUpBuffer(0, "HCTP_UP",
g_hct_rtt_up_buffer, 8192, BLOCK_IF_FIFO_FULL)`. **Only the flags take effect.**
SEGGER guards the storage assignment on a non-zero index:

```c
/* SEGGER_RTT.c, SEGGER_RTT_ConfigUpBuffer() -- ConfigDownBuffer() is identical */
pUp = &pRTTCB->aUp[BufferIndex];
if (BufferIndex) {                 /* <- channel 0 skips all of this */
  pUp->sName        = sName;
  pUp->pBuffer      = (char*)pBuffer;
  pUp->SizeOfBuffer = BufferSize;
  ...
}
pUp->Flags          = Flags;       /* <- channel 0 gets only this */
```

So channel 0 keeps SEGGER's own storage, sized by the `SEGGER_RTT_Conf.h`
defaults, and the mode is the one the transport asked for:

| channel 0 | storage in use | size | set by |
|---|---|---|---|
| target → host | `_acUpBuffer` | 1024 B | `BUFFER_SIZE_UP` |
| host → target | `_acDownBuffer` | 16 B | `BUFFER_SIZE_DOWN` |

`g_hct_rtt_up_buffer` (8192 B) and `g_hct_rtt_down_buffer` (512 B) are still
`static`, so they are linked in and occupy TCM `.bss`, but the RTT control block
never points at them:

```
$ arm-none-eabi-nm -S --size-sort hct_benchmark_server.elf
2002c440 00000400 b _acUpBuffer            <- in use
2002c430 00000010 b _acDownBuffer          <- in use
2002a430 00002000 b g_hct_rtt_up_buffer    <- allocated, never installed
2002a230 00000200 b g_hct_rtt_down_buffer  <- allocated, never installed
```

That is a firmware bug, not a documentation one: the transport asks for an 8 KB
up buffer and gets 1 KB, and 8704 bytes of `.bss` are dead. It works because
`BLOCK_IF_FIFO_FULL` *is* applied, so the firmware waits for the host to drain
the small ring rather than dropping frames -- which is also why the smaller
buffer has not shown up as corruption. Fixing it means either moving the
transport to channel 1 (a host-side change too, since the host is pinned to
index 0) or raising `BUFFER_SIZE_UP`/`BUFFER_SIZE_DOWN` and dropping the unused
statics. Either changes `.bss` and therefore the linked image, so it is recorded
here rather than folded into a change whose hardware numbers have to stay
comparable with the A/B legs they were measured against.
