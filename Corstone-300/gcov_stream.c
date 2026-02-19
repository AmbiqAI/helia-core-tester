#include <gcov.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

extern unsigned char UartPutc(unsigned char ch);

extern const struct gcov_info *const __gcov_info_start[];
extern const struct gcov_info *const __gcov_info_end[];

typedef struct {
    unsigned int line_len;
} gcov_emit_ctx_t;

static void uart_puts(const char *s)
{
    while (*s != '\0') {
        UartPutc((unsigned char)*s++);
    }
}

static void emit_hex_byte(uint8_t byte, gcov_emit_ctx_t *ctx)
{
    static const char HEX[] = "0123456789ABCDEF";
    UartPutc((unsigned char)HEX[(byte >> 4) & 0x0FU]);
    UartPutc((unsigned char)HEX[byte & 0x0FU]);

    ctx->line_len += 2U;
    if (ctx->line_len >= 120U) {
        UartPutc('\n');
        ctx->line_len = 0U;
    }
}

static void gcov_emit_bytes(const void *data, unsigned int len, void *arg)
{
    const uint8_t *bytes = (const uint8_t *)data;
    gcov_emit_ctx_t *ctx = (gcov_emit_ctx_t *)arg;
    unsigned int i;
    for (i = 0U; i < len; ++i) {
        emit_hex_byte(bytes[i], ctx);
    }
}

static void gcov_emit_filename(const char *filename, void *arg)
{
    __gcov_filename_to_gcfn(filename, gcov_emit_bytes, arg);
}

static void *gcov_allocate(unsigned int length, void *arg)
{
    (void)arg;
    return malloc(length);
}

void gcov_stream_dump(void)
{
    const struct gcov_info *const *info = __gcov_info_start;
    gcov_emit_ctx_t ctx = {0U};

    uart_puts("@@GCOV_BEGIN@@\n");
    for (; info < __gcov_info_end; ++info) {
        if (*info == NULL) {
            continue;
        }
        __gcov_info_to_gcda(*info, gcov_emit_filename, gcov_emit_bytes, gcov_allocate, &ctx);
    }
    if (ctx.line_len != 0U) {
        UartPutc('\n');
    }
    uart_puts("@@GCOV_END@@\n");
}