/* SPDX-License-Identifier: Apache-2.0; fixed-record numerical capture, no acoustic labels. */
#include "r1d_stream.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>

typedef struct { uint64_t rows; } capture_status;

static int write_u64_le(uint64_t value) {
    unsigned char bytes[8];
    for (unsigned i = 0; i < 8; ++i) {
        bytes[i] = (unsigned char)(value >> (8u * i));
    }
    return fwrite(bytes, 1, sizeof(bytes), stdout) == sizeof(bytes) ? 0 : 1;
}

static int capture_row(void *user, uint64_t frame, uint64_t source_end,
                       uint64_t delivery_end, const double *features,
                       const double *logits) {
    capture_status *status = (capture_status *)user;
    if (frame != status->rows || write_u64_le(frame) ||
        write_u64_le(source_end) || write_u64_le(delivery_end)) {
        return 1;
    }
    if (fwrite(features, sizeof(double), 400, stdout) != 400 ||
        fwrite(logits, sizeof(double), 2601, stdout) != 2601) {
        return 1;
    }
    status->rows++;
    return 0;
}

static unsigned char *read_exact(const char *path, size_t expected_bytes) {
    FILE *stream = fopen(path, "rb");
    unsigned char *buffer;
    size_t received;
    int trailing, failed;
    if (stream == NULL) {
        return NULL;
    }
    buffer = (unsigned char *)malloc(expected_bytes);
    if (buffer == NULL) {
        fclose(stream);
        return NULL;
    }
    received = fread(buffer, 1, expected_bytes, stream);
    trailing = fgetc(stream);
    failed = ferror(stream);
    fclose(stream);
    if (received != expected_bytes || trailing != EOF || failed) {
        free(buffer);
        return NULL;
    }
    return buffer;
}

int main(int argc, char **argv) {
    unsigned char *weights;
    unsigned char *tables;
    r1d_stream *stream;
    capture_status status = {0};
    unsigned char input[640];
    size_t received, consumed;
    uint64_t pcm_bytes = 0;
    int result = 0;

    if (argc != 3) {
        fputs("Usage: capture model.f32le frontend.f32le < mono16k-s16le.pcm\n",
              stderr);
        return 2;
    }
    weights = read_exact(argv[1], 3028860u);
    tables = read_exact(argv[2], 83840u);
    if (weights == NULL || tables == NULL) {
        free(weights);
        free(tables);
        fputs("invalid model or frontend payload size\n", stderr);
        return 2;
    }
    stream = r1d_create(weights, 3028860u, tables, 83840u,
                        capture_row, &status);
    free(weights);
    free(tables);
    if (stream == NULL) {
        fputs("model/frontend identity or execution environment rejected\n",
              stderr);
        return 2;
    }
    if (fwrite("R1DREF1\0", 1, 8, stdout) != 8) {
        r1d_destroy(stream);
        return 2;
    }

    while ((received = fread(input, 1, sizeof(input), stdin)) != 0) {
        result = r1d_feed(stream, input, received, &consumed);
        pcm_bytes += consumed;
        if (result != 0 || consumed != received) {
            result = 1;
            break;
        }
    }
    if (ferror(stdin)) {
        result = 1;
    }
    if (result == 0) {
        result = r1d_finish(stream);
    }
    if (fflush(stdout) != 0) {
        result = 1;
    }
    fprintf(stderr,
            "{\"status\":\"%s\",\"frames\":%" PRIu64
            ",\"pcm_bytes\":%" PRIu64 ",\"state_bytes\":%zu,"
            "\"synthetic_eof\":false}\n",
            result == 0 ? "complete" : "failed", status.rows, pcm_bytes,
            r1d_state_bytes());
    r1d_destroy(stream);
    return result == 0 ? 0 : 2;
}
