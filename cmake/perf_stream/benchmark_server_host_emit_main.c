#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "benchmark_server_messages.h"

static int write_file(const char *path, const uint8_t *data, size_t length)
{
    FILE *file = fopen(path, "wb");
    if (file == NULL)
    {
        return 1;
    }
    if (fwrite(data, 1u, length, file) != length)
    {
        fclose(file);
        return 2;
    }
    fclose(file);
    return 0;
}

int main(int argc, char **argv)
{
    uint8_t hello[512];
    uint8_t catalog[1024];
    size_t hello_len = 0u;
    size_t catalog_len = 0u;

    if (argc != 3)
    {
        fprintf(stderr, "usage: %s <hello.bin> <catalog.bin>\n", argv[0]);
        return 64;
    }
    if (hct_build_hello_frame(0xC0DE1234u, 0u, 256u, 32768u, hello, sizeof(hello), &hello_len) != HCTP_STATUS_OK)
    {
        return 65;
    }
    if (hct_build_catalog_frame(0xC0DE1234u, 1u, catalog, sizeof(catalog), &catalog_len) != HCTP_STATUS_OK)
    {
        return 66;
    }
    if (write_file(argv[1], hello, hello_len) != 0)
    {
        return 67;
    }
    if (write_file(argv[2], catalog, catalog_len) != 0)
    {
        return 68;
    }
    printf("hello=%zu catalog=%zu\n", hello_len, catalog_len);
    return 0;
}
