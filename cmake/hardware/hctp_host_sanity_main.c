#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hctp_protocol.h"

static int read_file(const char *path, uint8_t **out_data, size_t *out_size)
{
    FILE *file = fopen(path, "rb");
    long length;
    uint8_t *buffer;

    if (file == NULL)
    {
        return 1;
    }
    if (fseek(file, 0L, SEEK_END) != 0)
    {
        fclose(file);
        return 2;
    }
    length = ftell(file);
    if (length < 0)
    {
        fclose(file);
        return 3;
    }
    if (fseek(file, 0L, SEEK_SET) != 0)
    {
        fclose(file);
        return 4;
    }

    buffer = (uint8_t *)malloc((size_t)length);
    if ((buffer == NULL) && (length != 0))
    {
        fclose(file);
        return 5;
    }
    if (fread(buffer, 1u, (size_t)length, file) != (size_t)length)
    {
        free(buffer);
        fclose(file);
        return 6;
    }
    fclose(file);

    *out_data = buffer;
    *out_size = (size_t)length;
    return 0;
}

int main(int argc, char **argv)
{
    uint8_t *frame_bytes = NULL;
    size_t frame_length = 0u;
    hctp_frame_view_t frame;
    hctp_status_t status;
    uint8_t reencoded_header[HCTP_HEADER_SIZE];

    if (argc != 2)
    {
        fprintf(stderr, "usage: %s <frame.bin>\n", argv[0]);
        return 64;
    }
    if (read_file(argv[1], &frame_bytes, &frame_length) != 0)
    {
        fprintf(stderr, "failed to read %s\n", argv[1]);
        return 65;
    }

    status = hctp_decode_frame(frame_bytes, frame_length, HCTP_DEFAULT_MAX_PAYLOAD, &frame);
    if (status != HCTP_STATUS_OK)
    {
        fprintf(stderr, "decode failed: %d\n", (int)status);
        free(frame_bytes);
        return 66;
    }

    hctp_encode_header(reencoded_header, &frame.header);
    if (memcmp(reencoded_header, frame_bytes, HCTP_HEADER_SIZE) != 0)
    {
        fprintf(stderr, "header re-encode mismatch\n");
        free(frame_bytes);
        return 67;
    }

    printf(
        "magic=0x%08x version=%u type=%u flags=%u session=0x%08x sequence=%u payload_length=%u payload_crc=0x%08x header_crc=0x%08x\n",
        frame.header.magic,
        frame.header.protocol_version,
        frame.header.message_type,
        frame.header.flags,
        frame.header.session_id,
        frame.header.sequence_id,
        frame.header.payload_length,
        frame.header.payload_crc32,
        frame.header.header_crc32);

    free(frame_bytes);
    return 0;
}
