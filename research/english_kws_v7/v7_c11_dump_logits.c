#include "v7_kws_stream.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    FILE *output;
    uint64_t frames;
    int failed;
} dump_context;

static int read_all(const char *path, uint8_t **bytes, size_t *length)
{
    FILE *file;
    long end;
    uint8_t *buffer;
    size_t got;

    file = fopen(path, "rb");
    if (file == NULL) return -1;
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return -1;
    }
    end = ftell(file);
    if (end < 0 || (unsigned long)end > (unsigned long)SIZE_MAX) {
        fclose(file);
        return -1;
    }
    if (fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return -1;
    }
    buffer = (uint8_t *)malloc((size_t)end == 0u ? 1u : (size_t)end);
    if (buffer == NULL) {
        fclose(file);
        return -1;
    }
    got = fread(buffer, 1u, (size_t)end, file);
    if (got != (size_t)end || ferror(file)) {
        free(buffer);
        fclose(file);
        return -1;
    }
    if (fclose(file) != 0) {
        free(buffer);
        return -1;
    }
    *bytes = buffer;
    *length = got;
    return 0;
}

static void on_frame(void *user, const float logits[4], uint64_t frame_index,
                     uint64_t available_after_sample, const en_v7_ctc_event *event)
{
    dump_context *context = (dump_context *)user;
    (void)frame_index;
    (void)available_after_sample;
    (void)event;
    if (context->failed) return;
    if (fwrite(logits, sizeof(float), 4u, context->output) != 4u) {
        context->failed = 1;
        return;
    }
    ++context->frames;
}

int main(int argc, char **argv)
{
    uint8_t *pack_bytes = NULL;
    uint8_t *pcm_bytes = NULL;
    int16_t *pcm = NULL;
    size_t pack_length = 0u, pcm_length = 0u, sample_count, offset;
    uint32_t candidate_id;
    en_v7_kws_model *model = NULL;
    en_v7_kws_stream *stream = NULL;
    FILE *output = NULL;
    dump_context context;
    int status = 1;

    if (argc != 5 || (strcmp(argv[3], "1") != 0 && strcmp(argv[3], "2") != 0)) {
        fprintf(stderr, "usage: %s PACK PCM16 CANDIDATE_ID LOGITS_F32\n", argv[0]);
        return 2;
    }
    candidate_id = (uint32_t)(argv[3][0] - '0');
    if (read_all(argv[1], &pack_bytes, &pack_length) != 0 ||
        pack_length != EN_V7_KWS_PACK_BYTES) {
        fprintf(stderr, "invalid model pack\n");
        goto done;
    }
    if (read_all(argv[2], &pcm_bytes, &pcm_length) != 0 ||
        pcm_length == 0u || (pcm_length & 1u) != 0u) {
        fprintf(stderr, "invalid PCM16 input\n");
        goto done;
    }
    sample_count = pcm_length / 2u;
    pcm = (int16_t *)malloc(sample_count * sizeof(int16_t));
    model = (en_v7_kws_model *)calloc(1u, en_v7_kws_model_bytes());
    stream = (en_v7_kws_stream *)calloc(1u, en_v7_kws_stream_bytes());
    if (pcm == NULL || model == NULL || stream == NULL) goto done;

    for (offset = 0u; offset < sample_count; ++offset) {
        uint32_t value = (uint32_t)pcm_bytes[2u * offset] |
                         ((uint32_t)pcm_bytes[2u * offset + 1u] << 8);
        int32_t signed_value = (value & 0x8000u) != 0u
            ? (int32_t)value - 65536
            : (int32_t)value;
        pcm[offset] = (int16_t)signed_value;
    }

    if (en_v7_kws_model_load(model, pack_bytes, pack_length, candidate_id) != EN_V7_KWS_OK ||
        model->candidate_id != candidate_id) {
        fprintf(stderr, "model load failed\n");
        goto done;
    }
    if (en_v7_kws_stream_init(stream, model, UINT64_MAX / 2u, 0u) != EN_V7_KWS_OK) {
        fprintf(stderr, "stream init failed\n");
        goto done;
    }
    output = fopen(argv[4], "wb");
    if (output == NULL) {
        fprintf(stderr, "cannot create logits output: %s\n", strerror(errno));
        goto done;
    }
    context.output = output;
    context.frames = 0u;
    context.failed = 0;
    for (offset = 0u; offset < sample_count; ) {
        size_t count = sample_count - offset;
        if (count > EN_V7_KWS_MAX_FEED) count = EN_V7_KWS_MAX_FEED;
        if (en_v7_kws_feed(stream, pcm + offset, count, (uint64_t)offset,
                           on_frame, &context) != EN_V7_KWS_OK || context.failed) {
            fprintf(stderr, "stream feed failed at sample %zu\n", offset);
            goto done;
        }
        offset += count;
    }
    if (context.frames == 0u || fflush(output) != 0) {
        fprintf(stderr, "no complete frames or output flush failed\n");
        goto done;
    }
    status = 0;

done:
    if (output != NULL && fclose(output) != 0) status = 1;
    free(stream);
    free(model);
    free(pcm);
    free(pcm_bytes);
    free(pack_bytes);
    return status;
}
