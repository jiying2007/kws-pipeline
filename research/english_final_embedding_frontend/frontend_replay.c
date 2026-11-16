/* Bounded verification/file-fed harness. Models are supplied by the pure
 * exporter; no model data or graph is bundled in these preparation sources.
 * Build with optional -DEF_WITH_HEAD and the unchanged saved-head C library.
 * Usage: frontend_replay INPUT.pcm TRACE.bin 1,1279,257,4096 [HEAD.bin]
 * Input: exact PCM16LE, 16 kHz mono, no resampling, startup or EOF padding.
 * Trace: EFFTR01\0, then repeated LE u32 stage,u32 count,u64 sample,float[count].
 * Stages 1/2/3 = raw mel3104 / normalized2432 / embedding96.
 * Optional stage4 = [ready,event,logits3,probabilities3], count8, each embedding.
 */
#include "fixed_frontend.h"
#include "fixed_models.h"
#ifdef EF_WITH_HEAD
#include "english_causal_head.h"
#endif
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    FILE *trace;
    uint64_t embeddings, ready, events;
    int use_head;
#ifdef EF_WITH_HEAD
    ehead_model *model;
    ehead_state head;
#endif
} replay_context;

static int write_u32(FILE *f, uint32_t value) {
    unsigned char b[4];
    unsigned i;
    for (i = 0; i < 4; ++i) b[i] = (unsigned char)(value >> (8 * i));
    return fwrite(b, 1, 4, f) == 4 ? 0 : -1;
}

static int write_u64(FILE *f, uint64_t value) {
    return write_u32(f, (uint32_t)value) || write_u32(f, (uint32_t)(value >> 32));
}

static int feature_trace(void *ptr, uint32_t stage, uint64_t available,
                         const float *values, size_t count) {
    replay_context *ctx = (replay_context *)ptr;
    size_t i;
    if (count > UINT32_MAX || write_u32(ctx->trace, stage) ||
        write_u32(ctx->trace, (uint32_t)count) || write_u64(ctx->trace, available)) return -1;
    for (i = 0; i < count; ++i) {
        uint32_t bits;
        memcpy(&bits, &values[i], sizeof(bits));
        if (write_u32(ctx->trace, bits)) return -1;
    }
    return 0;
}

static int embedding_callback(void *ptr, uint64_t available, const float *embedding) {
    replay_context *ctx = (replay_context *)ptr;
    ++ctx->embeddings;
#ifdef EF_WITH_HEAD
    if (ctx->use_head) {
        ehead_output out;
        float values[8];
        unsigned i;
        memset(&out, 0, sizeof(out));
        if (ehead_step(ctx->model, &ctx->head, embedding, available, &out)) return -1;
        values[0] = (float)out.ready; values[1] = (float)out.keyword;
        for (i = 0; i < 3; ++i) { values[2 + i] = out.logits[i]; values[5 + i] = out.probabilities[i]; }
        if (out.ready) ++ctx->ready;
        if (out.keyword) ++ctx->events;
        if (feature_trace(ctx, 4, available, values, 8)) return -1;
    }
#else
    (void)available; (void)embedding;
#endif
    return 0;
}

static int parse_chunks(const char *text, size_t chunks[16], size_t *length, size_t *maximum) {
    const char *p = text;
    *length = *maximum = 0;
    while (*p) {
        char *end;
        unsigned long value;
        if (*length == 16 || *p < '0' || *p > '9') return -1;
        errno = 0; value = strtoul(p, &end, 10);
        if (errno || end == p || !value || value > 65536 || (*end && *end != ',')) return -1;
        chunks[(*length)++] = (size_t)value;
        if (value > *maximum) *maximum = (size_t)value;
        if (!*end) break;
        p = end + 1;
        if (!*p) return -1;
    }
    return *length ? 0 : -1;
}

#ifdef EF_WITH_HEAD
static int load_head(const char *path, replay_context *ctx) {
    FILE *f = fopen(path, "rb");
    unsigned char *bytes;
    int result = -1;
    if (!f) return -1;
    bytes = (unsigned char *)malloc(EHEAD_BLOB_BYTES);
    ctx->model = (ehead_model *)malloc(sizeof(*ctx->model));
    if (bytes && ctx->model && fread(bytes, 1, EHEAD_BLOB_BYTES, f) == EHEAD_BLOB_BYTES &&
        fgetc(f) == EOF && !ferror(f) && !ehead_load(ctx->model, bytes, EHEAD_BLOB_BYTES)) {
        ehead_reset(&ctx->head); ctx->use_head = 1; result = 0;
    }
    free(bytes); fclose(f);
    return result;
}
#endif

int main(int argc, char **argv) {
    ef_frontend *state = NULL;
    replay_context ctx;
    FILE *input = NULL;
    unsigned char *raw = NULL;
    int16_t *pcm = NULL;
    float *workspace = NULL;
    size_t chunks[16], chunk_count, maximum, step = 0, workspace_bytes;
    int result = 2, status = EF_INVALID;
    memset(&ctx, 0, sizeof(ctx));
#ifdef EF_WITH_HEAD
    if (argc != 4 && argc != 5) {
#else
    if (argc != 4) {
#endif
        fprintf(stderr, "usage: frontend_replay INPUT.pcm TRACE.bin CHUNK_PATTERN [HEAD.bin]\n");
        return 2;
    }
    if (parse_chunks(argv[3], chunks, &chunk_count, &maximum)) {
        fprintf(stderr, "invalid chunk pattern\n"); return 2;
    }
    workspace_bytes = ef_frontend_workspace_bytes(&ef_fixed_mel, &ef_fixed_embedding);
    if (!workspace_bytes) { fprintf(stderr, "invalid exported graph pair\n"); return 2; }
    state = (ef_frontend *)malloc(sizeof(*state));
    workspace = (float *)malloc(workspace_bytes);
    raw = (unsigned char *)malloc(2 * maximum);
    pcm = (int16_t *)malloc(maximum * sizeof(*pcm));
    if (!state || !workspace || !raw || !pcm) goto cleanup;
    if (ef_frontend_init(state, &ef_fixed_mel, &ef_fixed_embedding, workspace,
                         workspace_bytes / sizeof(float))) goto cleanup;
#ifdef EF_WITH_HEAD
    if (argc == 5 && load_head(argv[4], &ctx)) { fprintf(stderr, "invalid head blob\n"); goto cleanup; }
#endif
    input = fopen(argv[1], "rb");
    if (!input) { perror("PCM input"); goto cleanup; }
    ctx.trace = fopen(argv[2], "wbx");
    if (!ctx.trace) { perror("fresh trace output"); goto cleanup; }
    if (fwrite("EFFTR01\0", 1, 8, ctx.trace) != 8) goto cleanup;
    state->trace = feature_trace; state->trace_context = &ctx;
    for (;;) {
        size_t wanted = chunks[step % chunk_count], n = fread(raw, 1, 2 * wanted, input), i, consumed;
        if (ferror(input) || n % 2) { fprintf(stderr, "PCM read/odd byte error\n"); goto cleanup; }
        if (!n) break;
        for (i = 0; i < n / 2; ++i) {
            uint32_t u = (uint32_t)raw[2 * i] | (uint32_t)raw[2 * i + 1] << 8;
            pcm[i] = (int16_t)(u < 32768 ? (int32_t)u : (int32_t)u - 65536);
        }
        status = ef_frontend_feed(state, pcm, n / 2, state->observed, embedding_callback, &ctx, &consumed);
        if (status != EF_OK || consumed != n / 2) {
            fprintf(stderr, "frontend failure status=%d sample=%" PRIu64 " consumed=%zu\n", status, state->observed, consumed);
            goto cleanup;
        }
        ++step;
    }
    if (fclose(ctx.trace)) { ctx.trace = NULL; goto cleanup; }
    ctx.trace = NULL;
    printf("{\"samples\":%" PRIu64 ",\"embeddings\":%" PRIu64
           ",\"head_ready\":%" PRIu64 ",\"events\":%" PRIu64
           ",\"workspace_bytes\":%zu,\"frontend_state_bytes\":%zu,\"head_enabled\":%s,"
           "\"ssc305_qualification\":false}\n",
           state->observed, ctx.embeddings, ctx.ready, ctx.events,
           workspace_bytes, sizeof(*state), ctx.use_head ? "true" : "false");
    result = 0;
cleanup:
    if (input) fclose(input);
    if (ctx.trace) fclose(ctx.trace);
#ifdef EF_WITH_HEAD
    free(ctx.model);
#endif
    free(state); free(workspace); free(raw); free(pcm);
    return result;
}
