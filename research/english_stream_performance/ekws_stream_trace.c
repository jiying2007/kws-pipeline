#define _POSIX_C_SOURCE 200809L
#include "ekws_stream_trace.h"
#include "ekws_research_stream.h"
#include "english_causal_head.h"
#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* GNU ld --wrap observes the installed archive's undefined ehead_step
 * reference. The original function is called exactly once per embedding.
 * This single-owner diagnostic intentionally performs synchronous output I/O.
 */
enum { CHUNK_SAMPLES = 320, MIN_SAMPLES = 35200,
       MAX_SAMPLES = 300 * 16000, MAX_SDK_ALLOCATION = 64 * 1024 * 1024 };
_Static_assert(CHAR_BIT == 8 && sizeof(float) == 4 && sizeof(uint32_t) == 4,
               "trace requires 8-bit bytes and 32-bit floats");
_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128,
               "trace requires binary32 floats");
_Static_assert(EKWS_CLASSES == EKWS_TRACE_CLASSES && EHEAD_CLASSES == EKWS_TRACE_CLASSES,
               "trace class count must match both interfaces");
_Static_assert(EHEAD_DIM == EKWS_TRACE_EMBED_DIM && EHEAD_HISTORY == 16,
               "trace expects the pinned 96-dimensional, 16-history head");
_Static_assert(EKWS_SAMPLE_RATE == 16000 && EHEAD_FIRST_SAMPLES == 16000 &&
               EHEAD_HOP_SAMPLES == 1280, "trace expects the pinned sample clock");

typedef struct {
    FILE *out;
    uint64_t input_samples, embeddings, real_head_calls, scores, ready, events;
    uint64_t last_embedding_sample, last_event_sample, calls, bytes, records;
    uint32_t pending_keyword, pending_confidence_bits;
    int failed;
} trace_state;
static trace_state *active_trace;

/* Resolved by --wrap=ehead_step to the original installed implementation. */
int __real_ehead_step(const ehead_model *model, ehead_state *state,
                     const float embedding[EHEAD_DIM], uint64_t sample,
                     ehead_output *output);

static uint32_t float_bits(const float *value) {
    uint32_t bits;
    memcpy(&bits, value, sizeof(bits));
    return bits;
}
static void put_u32(unsigned char *p, size_t *at, uint32_t value) {
    unsigned i;
    for (i = 0; i < 4; ++i) { p[(*at)++] = (unsigned char)(value & 255u); value >>= 8; }
}
static void put_u64(unsigned char *p, size_t *at, uint64_t value) {
    unsigned i;
    for (i = 0; i < 8; ++i) { p[(*at)++] = (unsigned char)(value & 255u); value >>= 8; }
}
static void put_float(unsigned char *p, size_t *at, const float *value) {
    put_u32(p, at, float_bits(value));
}
static int fail_trace(trace_state *t) {
    if (t) t->failed = 1;
    return -1;
}
static int write_bytes(trace_state *t, const unsigned char *data, size_t count) {
    size_t written;
    if (!t || t->failed || !t->out) return fail_trace(t);
    written = fwrite(data, 1, count, t->out);
    t->bytes += written;
    if (written != count || ferror(t->out)) return fail_trace(t);
    return 0;
}
static size_t begin_record(unsigned char *p, unsigned char type, uint32_t payload) {
    size_t at = 0;
    p[at++] = type; put_u32(p, &at, payload);
    return at;
}
static int write_record(trace_state *t, const unsigned char *data, size_t count,
                        size_t expected) {
    if (count != expected || write_bytes(t, data, count)) return fail_trace(t);
    ++t->records;
    return 0;
}
static int write_header(trace_state *t) {
    unsigned char data[EKWS_TRACE_HEADER_BYTES];
    size_t at = 8;
    memcpy(data, "EKWSTRC1", 8);
    put_u32(data, &at, EKWS_TRACE_FORMAT_VERSION);
    put_u32(data, &at, EKWS_RESEARCH_ABI);
    put_u32(data, &at, EKWS_SAMPLE_RATE);
    put_u32(data, &at, CHUNK_SAMPLES);
    put_u32(data, &at, EKWS_TRACE_EMBED_DIM);
    put_u32(data, &at, EKWS_TRACE_CLASSES);
    put_u64(data, &at, t->input_samples);
    if (at != sizeof(data)) return fail_trace(t);
    return write_bytes(t, data, sizeof(data));
}

int __wrap_ehead_step(const ehead_model *model, ehead_state *state,
                     const float embedding[EHEAD_DIM], uint64_t sample,
                     ehead_output *output) {
    trace_state *t = active_trace;
    unsigned char data[EKWS_TRACE_RECORD_PREFIX_BYTES + EKWS_TRACE_EMBED_PAYLOAD_BYTES];
    size_t at;
    unsigned i;
    int result;
    if (!t || t->failed || !model || !state || !embedding || !output ||
        t->embeddings != t->scores || t->pending_keyword ||
        sample != EHEAD_FIRST_SAMPLES + t->embeddings * EHEAD_HOP_SAMPLES ||
        sample > t->input_samples) return fail_trace(t);
    at = begin_record(data, EKWS_TRACE_EMBEDDING, EKWS_TRACE_EMBED_PAYLOAD_BYTES);
    put_u64(data, &at, t->embeddings); put_u64(data, &at, sample);
    for (i = 0; i < EHEAD_DIM; ++i) {
        if (!isfinite(embedding[i])) return fail_trace(t);
        put_float(data, &at, &embedding[i]);
    }
    if (write_record(t, data, at, sizeof(data))) return -1;
    ++t->embeddings; t->last_embedding_sample = sample;
    result = __real_ehead_step(model, state, embedding, sample, output);
    if (result) return fail_trace(t);
    ++t->real_head_calls;
    return 0;
}

static int observe_score(void *context, const ekws_score *score) {
    trace_state *t = (trace_state *)context;
    unsigned char data[EKWS_TRACE_RECORD_PREFIX_BYTES + EKWS_TRACE_SCORE_PAYLOAD_BYTES];
    size_t at;
    unsigned i;
    uint32_t expected_ready;
    if (!t || t != active_trace || t->failed || !score ||
        t->embeddings != t->scores + 1 || t->real_head_calls != t->embeddings ||
        t->pending_keyword || score->available_after_sample != t->last_embedding_sample ||
        score->ready > 1 || score->keyword_id >= EKWS_CLASSES) return fail_trace(t);
    expected_ready = t->scores >= EHEAD_HISTORY - 1 ? 1u : 0u;
    if (score->ready != expected_ready || (!score->ready && score->keyword_id)) return fail_trace(t);
    at = begin_record(data, EKWS_TRACE_SCORE, EKWS_TRACE_SCORE_PAYLOAD_BYTES);
    put_u64(data, &at, t->scores);
    put_u32(data, &at, score->ready); put_u32(data, &at, score->keyword_id);
    put_u64(data, &at, score->available_after_sample);
    for (i = 0; i < EKWS_CLASSES; ++i) {
        if (!isfinite(score->logits[i])) return fail_trace(t);
        put_float(data, &at, &score->logits[i]);
    }
    for (i = 0; i < EKWS_CLASSES; ++i) {
        if (!isfinite(score->probabilities[i])) return fail_trace(t);
        put_float(data, &at, &score->probabilities[i]);
    }
    if (write_record(t, data, at, sizeof(data))) return -1;
    ++t->scores; t->ready += score->ready;
    t->pending_keyword = score->keyword_id;
    if (score->keyword_id) t->pending_confidence_bits = float_bits(&score->probabilities[score->keyword_id]);
    return 0;
}

static int observe_event(void *context, const ekws_event *event) {
    trace_state *t = (trace_state *)context;
    unsigned char data[EKWS_TRACE_RECORD_PREFIX_BYTES + EKWS_TRACE_EVENT_PAYLOAD_BYTES];
    size_t at;
    if (!t || t != active_trace || t->failed || !event || !t->scores ||
        !t->pending_keyword || event->keyword_id != t->pending_keyword ||
        event->available_after_sample != t->last_embedding_sample ||
        event->available_after_sample <= t->last_event_sample ||
        !isfinite(event->confidence) || float_bits(&event->confidence) != t->pending_confidence_bits)
        return fail_trace(t);
    at = begin_record(data, EKWS_TRACE_EVENT, EKWS_TRACE_EVENT_PAYLOAD_BYTES);
    put_u64(data, &at, t->events); put_u64(data, &at, t->scores - 1);
    put_u32(data, &at, event->keyword_id); put_u64(data, &at, event->available_after_sample);
    put_float(data, &at, &event->confidence);
    if (write_record(t, data, at, sizeof(data))) return -1;
    ++t->events; t->last_event_sample = event->available_after_sample; t->pending_keyword = 0;
    return 0;
}

static int write_end(trace_state *t, uint64_t accepted_samples) {
    unsigned char data[EKWS_TRACE_RECORD_PREFIX_BYTES + EKWS_TRACE_END_PAYLOAD_BYTES];
    size_t at = begin_record(data, EKWS_TRACE_END, EKWS_TRACE_END_PAYLOAD_BYTES);
    put_u64(data, &at, accepted_samples); put_u64(data, &at, t->embeddings);
    put_u64(data, &at, t->scores); put_u64(data, &at, t->ready);
    put_u64(data, &at, t->events); put_u64(data, &at, t->calls);
    return write_record(t, data, at, sizeof(data));
}

static int load_head(const char *path, unsigned char *head) {
    FILE *in = fopen(path, "rb");
    struct stat st;
    int ok;
    if (!in) return -1;
    ok = fstat(fileno(in), &st) == 0 && S_ISREG(st.st_mode) &&
         st.st_size == EKWS_HEAD_BLOB_BYTES &&
         fread(head, 1, EKWS_HEAD_BLOB_BYTES, in) == EKWS_HEAD_BLOB_BYTES &&
         fgetc(in) == EOF && !ferror(in);
    if (fclose(in)) ok = 0;
    return ok ? 0 : -1;
}
static int allocate_sdk(void **out, size_t alignment, size_t bytes) {
    if (!alignment || (alignment & (alignment - 1)) || !bytes || bytes > MAX_SDK_ALLOCATION) return -1;
    if (alignment < sizeof(void *)) alignment = sizeof(void *);
    return posix_memalign(out, alignment, bytes);
}

int main(int argc, char **argv) {
    trace_state trace = {0};
    ekws_callbacks callbacks;
    ekws_stream *stream = NULL;
    ekws_stats stats;
    unsigned char raw[CHUNK_SAMPLES * 2], *head = NULL;
    int16_t pcm[CHUNK_SAMPLES];
    void *arena = NULL, *workspace = NULL;
    size_t arena_bytes, workspace_bytes;
    uint64_t offset = 0, expected_embeddings = 0, expected_ready = 0, expected_bytes = 0;
    FILE *input = NULL;
    struct stat input_stat, final_stat;
    float one = 1.0f, negative_zero = -0.0f;
    int result = 2, output_created = 0, removed = 0, fd = -1;
    const char *stage = "arguments";
    if (argc != 4) {
        fputs("usage: ekws_stream_trace HEAD.bin MONO16K_S16LE.pcm OUTPUT.trace\n", stderr);
        goto done;
    }
    stage = "float_representation";
    if (float_bits(&one) != UINT32_C(0x3f800000) || float_bits(&negative_zero) != UINT32_C(0x80000000)) goto done;
    stage = "input_size";
    input = fopen(argv[2], "rb");
    if (!input || fstat(fileno(input), &input_stat) || !S_ISREG(input_stat.st_mode) ||
        input_stat.st_size < (off_t)MIN_SAMPLES * 2 ||
        input_stat.st_size > (off_t)MAX_SAMPLES * 2 || input_stat.st_size % 2) goto done;
    trace.input_samples = (uint64_t)input_stat.st_size / 2;
    expected_embeddings = 1 + (trace.input_samples - EHEAD_FIRST_SAMPLES) / EHEAD_HOP_SAMPLES;
    expected_ready = expected_embeddings - (EHEAD_HISTORY - 1);
    stage = "head_read";
    head = (unsigned char *)malloc(EKWS_HEAD_BLOB_BYTES);
    if (!head || load_head(argv[1], head)) goto done;
    stage = "allocation";
    arena_bytes = ekws_stream_bytes(); workspace_bytes = ekws_workspace_bytes();
    if (allocate_sdk(&arena, ekws_stream_alignment(), arena_bytes) ||
        allocate_sdk(&workspace, ekws_workspace_alignment(), workspace_bytes)) goto done;
    callbacks.event = observe_event; callbacks.event_context = &trace;
    callbacks.score = observe_score; callbacks.score_context = &trace;
    active_trace = &trace;
    stage = "init";
    if (ekws_stream_init(arena, arena_bytes, (float *)workspace, workspace_bytes,
                         head, EKWS_HEAD_BLOB_BYTES, &callbacks, &stream) != EKWS_OK) goto done;
    free(head); head = NULL;
    stage = "output_create_exclusive";
    fd = open(argv[3], O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0) goto done;
    output_created = 1;
    trace.out = fdopen(fd, "wb");
    if (!trace.out) { (void)close(fd); fd = -1; goto done; }
    fd = -1;
    stage = "output_unbuffered";
    if (setvbuf(trace.out, NULL, _IONBF, 0)) goto done;
    stage = "trace_header";
    if (write_header(&trace)) goto done;
    stage = "feed";
    while (offset < trace.input_samples) {
        size_t i, consumed = 0, count = (size_t)(trace.input_samples - offset);
        int fed;
        if (count > CHUNK_SAMPLES) count = CHUNK_SAMPLES;
        if (fread(raw, 1, count * 2, input) != count * 2 || ferror(input)) goto done;
        for (i = 0; i < count; ++i) {
            uint32_t u = (uint32_t)raw[i * 2] | (uint32_t)raw[i * 2 + 1] << 8;
            pcm[i] = (int16_t)(u < 32768 ? (int32_t)u : (int32_t)u - 65536);
        }
        fed = ekws_stream_accept_pcm16(stream, pcm, count, offset, &consumed);
        if (consumed > count) goto done;
        offset += consumed;
        if (fed != EKWS_OK || consumed != count || trace.failed) goto done;
        ++trace.calls;
    }
    stage = "input_eof";
    if (fgetc(input) != EOF || ferror(input) || fstat(fileno(input), &final_stat) ||
        final_stat.st_size != input_stat.st_size ||
        final_stat.st_mtim.tv_sec != input_stat.st_mtim.tv_sec ||
        final_stat.st_mtim.tv_nsec != input_stat.st_mtim.tv_nsec ||
        final_stat.st_ctim.tv_sec != input_stat.st_ctim.tv_sec ||
        final_stat.st_ctim.tv_nsec != input_stat.st_ctim.tv_nsec) goto done;
    if (fclose(input)) { input = NULL; goto done; }
    input = NULL;
    stage = "stats";
    if (ekws_stream_get_stats(stream, &stats) != EKWS_OK || trace.failed || trace.pending_keyword ||
        stats.accepted_samples != trace.input_samples || stats.next_input_sample != trace.input_samples ||
        stats.embeddings != expected_embeddings || trace.embeddings != stats.embeddings ||
        trace.real_head_calls != stats.embeddings || trace.scores != stats.embeddings ||
        stats.ready_outputs != expected_ready || trace.ready != stats.ready_outputs ||
        stats.last_embedding_sample != trace.last_embedding_sample ||
        stats.detected_events != trace.events || stats.delivered_events != trace.events ||
        stats.last_event_sample != trace.last_event_sample ||
        stats.resets || stats.discontinuities || stats.callback_failures || stats.feed_failures ||
        stats.poisoned || stats.counters_saturated ||
        trace.calls != (trace.input_samples + CHUNK_SAMPLES - 1) / CHUNK_SAMPLES) goto done;
    stage = "trace_end";
    if (write_end(&trace, stats.accepted_samples)) goto done;
    expected_bytes = EKWS_TRACE_HEADER_BYTES +
        trace.embeddings * (EKWS_TRACE_RECORD_PREFIX_BYTES + EKWS_TRACE_EMBED_PAYLOAD_BYTES) +
        trace.scores * (EKWS_TRACE_RECORD_PREFIX_BYTES + EKWS_TRACE_SCORE_PAYLOAD_BYTES) +
        trace.events * (EKWS_TRACE_RECORD_PREFIX_BYTES + EKWS_TRACE_EVENT_PAYLOAD_BYTES) +
        EKWS_TRACE_RECORD_PREFIX_BYTES + EKWS_TRACE_END_PAYLOAD_BYTES;
    if (trace.bytes != expected_bytes || trace.records != trace.embeddings + trace.scores + trace.events + 1) goto done;
    stage = "output_flush";
    if (fflush(trace.out) || ferror(trace.out) || fsync(fileno(trace.out))) goto done;
    if (fclose(trace.out)) { trace.out = NULL; goto done; }
    trace.out = NULL;
    stage = "output_size";
    if (stat(argv[3], &final_stat) || !S_ISREG(final_stat.st_mode) ||
        final_stat.st_size < 0 || (uint64_t)final_stat.st_size != trace.bytes) goto done;
    stage = "stdout_report";
    if (printf("{\"schema\":\"english-stream-numeric-trace-v1\",\"trace_completed\":true,"
               "\"format_magic\":\"EKWSTRC1\",\"format_version\":%u,\"abi\":%u,"
               "\"sample_rate_hz\":%u,\"encoding\":\"s16le_mono\",\"chunk_samples\":%u,"
               "\"input_samples\":%" PRIu64 ",\"input_bytes\":%" PRIu64 ","
               "\"input_repetitions\":1,\"resets\":0,\"eof_padding_samples\":0,"
               "\"feed_calls\":%" PRIu64 ",\"embedding_records\":%" PRIu64 ","
               "\"real_head_calls\":%" PRIu64 ",\"score_records\":%" PRIu64 ","
               "\"ready_outputs\":%" PRIu64 ",\"event_records\":%" PRIu64 ","
               "\"trace_records\":%" PRIu64 ",\"trace_bytes\":%" PRIu64 ","
               "\"end_record_written\":true,\"score_observer_enabled\":true,"
               "\"original_head_link_wrap\":true,\"performance_measured\":false}\n",
               EKWS_TRACE_FORMAT_VERSION, EKWS_RESEARCH_ABI, EKWS_SAMPLE_RATE, CHUNK_SAMPLES,
               trace.input_samples, trace.input_samples * 2, trace.calls, trace.embeddings,
               trace.real_head_calls, trace.scores, trace.ready, trace.events, trace.records, trace.bytes) < 0 ||
        fflush(stdout) || ferror(stdout)) goto done;
    result = 0;
done:
    active_trace = NULL;
    if (input && fclose(input) && !result) result = 2;
    if (trace.out && fclose(trace.out) && !result) result = 2;
    if (fd >= 0) (void)close(fd);
    if (result && output_created) {
        if (unlink(argv[3]) == 0 || errno == ENOENT) removed = 1;
    }
    if (result) {
        fprintf(stderr, "trace failed at %s after %" PRIu64 " accepted samples; no retry; partial output removed=%s\n",
                stage, offset, removed ? "true" : "false");
        if (!ferror(stdout)) {
            fprintf(stdout, "{\"schema\":\"english-stream-numeric-trace-v1\",\"trace_completed\":false,"
                    "\"failed_stage\":\"%s\",\"accepted_samples\":%" PRIu64 ","
                    "\"partial_output_removed\":%s,\"retry_count\":0,\"performance_measured\":false}\n",
                    stage, offset, removed ? "true" : "false");
            (void)fflush(stdout);
        }
    }
    free(head); free(arena); free(workspace);
    return result;
}
