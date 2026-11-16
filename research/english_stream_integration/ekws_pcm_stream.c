#define _POSIX_C_SOURCE 200809L
#include "ekws_research_stream.h"
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Hosted I/O belongs here, never in the stream library. The installed runner
 * is also the single-process integration consumer for each backend. */
enum { VERIFY_SAMPLES = 48000, VERIFY_WINDOWS = 26, MAX_CHUNK = 4096 };
static const size_t chunks[] = {1, 1279, 257, 4096};
typedef struct {
    const char *recording;
    uint64_t origin;
    ekws_stream *stream;
    ekws_score expected[VERIFY_WINDOWS], observed[VERIFY_WINDOWS];
    ekws_event events[VERIFY_WINDOWS];
    size_t scores, compared_scores, event_count;
    int verify, reentrant_checked, passed;
    double accept_wall_seconds;
    uint64_t accepted_samples;
    ekws_stats final_stats;
} case_context;

static uint32_t little32(const unsigned char *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
static uint64_t little64(const unsigned char *p) { return little32(p) | (uint64_t)little32(p + 4) << 32; }
static uint32_t float_bits(float value) { uint32_t bits; memcpy(&bits, &value, 4); return bits; }
static int same_float(float a, float b) { return float_bits(a) == float_bits(b); }
static int recording_id(const char *text) {
    size_t i, n = strlen(text);
    if (!n || n > 128) return 0;
    for (i = 0; i < n; ++i) {
        unsigned char c = (unsigned char)text[i];
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.')) return 0;
    }
    return 1;
}

static int event_callback(void *context, const ekws_event *event) {
    case_context *c = (case_context *)context;
    if (!c->verify) {
        if (printf("{\"recording\":\"%s\",\"keyword_id\":%" PRIu32
                   ",\"time_s\":%.9f,\"confidence\":%.9g,\"available_after_sample\":%" PRIu64
                   ",\"input_origin_sample\":%" PRIu64 ",\"research_only\":true,\"pronunciation_verified\":false}\n", c->recording, event->keyword_id,
                   (double)(event->available_after_sample - c->origin) / EKWS_SAMPLE_RATE,
                   (double)event->confidence, event->available_after_sample, c->origin) < 0 || fflush(stdout)) return -1;
        return 0;
    }
    if (c->event_count >= VERIFY_WINDOWS || !c->scores) return -1;
    {
        const ekws_score *expected = &c->expected[c->scores - 1];
        if (event->keyword_id != expected->keyword_id ||
            event->available_after_sample != expected->available_after_sample + c->origin ||
            !same_float(event->confidence, expected->probabilities[event->keyword_id])) return -1;
    }
    c->events[c->event_count++] = *event;
    return 0;
}

static int score_observer(void *context, const ekws_score *score) {
    case_context *c = (case_context *)context;
    const ekws_score *expected;
    unsigned k;
    if (c->scores >= VERIFY_WINDOWS) return -1;
    c->observed[c->scores] = *score;
    expected = &c->expected[c->scores++];
    if (score->available_after_sample != expected->available_after_sample + c->origin ||
        score->ready != expected->ready || score->keyword_id != expected->keyword_id) return -1;
    for (k = 0; k < EKWS_CLASSES; ++k)
        if (!same_float(score->logits[k], expected->logits[k]) ||
            !same_float(score->probabilities[k], expected->probabilities[k])) return -1;
    if (!c->reentrant_checked) {
        size_t consumed = 99;
        ekws_stats stats;
        if (ekws_stream_reset(c->stream) != EKWS_REENTRANT ||
            ekws_stream_discontinuity(c->stream, score->available_after_sample, EKWS_XRUN) != EKWS_REENTRANT ||
            ekws_stream_accept_pcm16(c->stream, NULL, 0, score->available_after_sample, &consumed) != EKWS_REENTRANT ||
            consumed != 0 || ekws_stream_get_stats(c->stream, &stats) != EKWS_REENTRANT) return -1;
        c->reentrant_checked = 1;
    }
    ++c->compared_scores;
    return 0;
}

static int read_exact_file(const char *name, unsigned char *buffer, size_t bytes) {
    FILE *f = fopen(name, "rb");
    int valid;
    if (!f) return -1;
    valid = fread(buffer, 1, bytes, f) == bytes && fgetc(f) == EOF && !ferror(f);
    if (fclose(f)) valid = 0;
    return valid ? 0 : -1;
}
static void decode_pcm(const unsigned char *bytes, int16_t *pcm, size_t count) {
    size_t i;
    for (i = 0; i < count; ++i) {
        uint32_t u = (uint32_t)bytes[2 * i] | (uint32_t)bytes[2 * i + 1] << 8;
        pcm[i] = (int16_t)(u < 32768 ? (int32_t)u : (int32_t)u - 65536);
    }
}
static int load_reference(const char *name, ekws_score rows[VERIFY_WINDOWS]) {
    static const uint32_t counts[] = {3104, 2432, 96, 8};
    FILE *f = fopen(name, "rb");
    unsigned char header[16], values[32];
    size_t i, stage, k;
    int valid = 0;
    if (!f) return -1;
    if (fread(header, 1, 8, f) != 8 || memcmp(header, "EFFTR01\0", 8)) goto done;
    for (i = 0; i < VERIFY_WINDOWS; ++i) {
        for (stage = 0; stage < 4; ++stage) {
            if (fread(header, 1, 16, f) != 16 || little32(header) != stage + 1 ||
                little32(header + 4) != counts[stage] || little64(header + 8) != 16000 + 1280 * i) goto done;
            if (stage != 3) {
                if (fseek(f, (long)(4 * counts[stage]), SEEK_CUR)) goto done;
            } else {
                float fields[8];
                if (fread(values, 1, 32, f) != 32) goto done;
                for (k = 0; k < 8; ++k) {
                    uint32_t bits = little32(values + 4 * k);
                    memcpy(&fields[k], &bits, 4);
                    if (!isfinite(fields[k])) goto done;
                }
                if (fields[0] != (float)(i >= 15) ||
                    !(fields[1] == 0.0f || fields[1] == 1.0f || fields[1] == 2.0f) ||
                    (i < 15 && fields[1] != 0.0f)) goto done;
                rows[i].ready = (uint32_t)fields[0]; rows[i].keyword_id = (uint32_t)fields[1];
                rows[i].available_after_sample = 16000 + 1280 * i;
                for (k = 0; k < EKWS_CLASSES; ++k) {
                    rows[i].logits[k] = fields[k + 2]; rows[i].probabilities[k] = fields[k + 5];
                }
            }
        }
    }
    valid = fgetc(f) == EOF && !ferror(f);
done:
    if (fclose(f)) valid = 0;
    return valid ? 0 : -1;
}

static double seconds(const struct timespec *a, const struct timespec *b) {
    return (double)(b->tv_sec - a->tv_sec) + (double)(b->tv_nsec - a->tv_nsec) / 1e9;
}
static int verify_case(case_context *c, const char *pcm_path, const char *trace_path, unsigned index) {
    unsigned char *bytes = (unsigned char *)malloc(2 * VERIFY_SAMPLES);
    int16_t *pcm = (int16_t *)malloc(sizeof(int16_t) * VERIFY_SAMPLES);
    size_t offset = 0, step = 0, consumed = 0, expected_events = 0, i;
    ekws_stats before, boundary;
    int status = -1;
    if (!bytes || !pcm || read_exact_file(pcm_path, bytes, 2 * VERIFY_SAMPLES) ||
        load_reference(trace_path, c->expected)) goto done;
    decode_pcm(bytes, pcm, VERIFY_SAMPLES);
    if (index < 2) {
        if (ekws_stream_reset(c->stream) != EKWS_OK) goto done;
    } else if (ekws_stream_discontinuity(c->stream, 48000, EKWS_ROUTE_CHANGE) != EKWS_OK) goto done;
    if (ekws_stream_get_stats(c->stream, &before) != EKWS_OK || before.next_input_sample != c->origin ||
        before.frontend_buffered_samples || before.head_history_count || before.poisoned) goto done;
    if (ekws_stream_accept_pcm16(c->stream, NULL, 0, c->origin, &consumed) != EKWS_OK || consumed ||
        ekws_stream_accept_pcm16(c->stream, pcm, 1, c->origin + 1, &consumed) != EKWS_CLOCK || consumed)
        goto done;
    if (ekws_stream_get_stats(c->stream, &boundary) != EKWS_OK ||
        boundary.accepted_samples != before.accepted_samples || boundary.embeddings != before.embeddings ||
        boundary.next_input_sample != c->origin || boundary.poisoned) goto done;
    while (offset < VERIFY_SAMPLES) {
        size_t count = chunks[step++ % 4];
        struct timespec start, end;
        int fed;
        if (count > VERIFY_SAMPLES - offset) count = VERIFY_SAMPLES - offset;
        if (clock_gettime(CLOCK_MONOTONIC, &start)) goto done;
        fed = ekws_stream_accept_pcm16(c->stream, pcm + offset, count, c->origin + offset, &consumed);
        c->accepted_samples += consumed;
        if (clock_gettime(CLOCK_MONOTONIC, &end)) goto done;
        c->accept_wall_seconds += seconds(&start, &end);
        if (fed != EKWS_OK || consumed != count) goto done;
        offset += consumed;
    }
    if (ekws_stream_accept_pcm16(c->stream, NULL, 0, c->origin + VERIFY_SAMPLES, &consumed) != EKWS_OK || consumed)
        goto done;
    if (ekws_stream_get_stats(c->stream, &c->final_stats) != EKWS_OK) goto done;
    for (i = 0; i < VERIFY_WINDOWS; ++i) expected_events += c->expected[i].keyword_id != 0;
    if (c->scores != VERIFY_WINDOWS || c->compared_scores != VERIFY_WINDOWS || !c->reentrant_checked || c->event_count != expected_events ||
        c->final_stats.accepted_samples - before.accepted_samples != VERIFY_SAMPLES ||
        c->final_stats.embeddings - before.embeddings != VERIFY_WINDOWS ||
        c->final_stats.ready_outputs - before.ready_outputs != 11 ||
        c->final_stats.delivered_events - before.delivered_events != c->event_count ||
        c->final_stats.next_input_sample != c->origin + VERIFY_SAMPLES ||
        c->final_stats.poisoned || !isfinite(c->accept_wall_seconds) || c->accept_wall_seconds < 0) goto done;
    c->passed = 1; status = 0;
done:
    (void)ekws_stream_get_stats(c->stream, &c->final_stats);
    free(bytes); free(pcm);
    return status;
}

static void print_verification(case_context reports[3], size_t attempted, int passed) {
    size_t i, j, k;
    printf("{\"schema\":\"english-stream-integration-v1\",\"passed\":%s,\"cases\":[", passed ? "true" : "false");
    for (i = 0; i < attempted; ++i) {
        const case_context *c = &reports[i];
        if (i) putchar(',');
        printf("{\"recording\":\"%s\",\"passed\":%s,\"input_origin_sample\":%" PRIu64
               ",\"accepted_samples\":%" PRIu64 ",\"accept_wall_seconds\":%.9f,\"compared_scores\":%zu,\"scores\":[",
               c->recording, c->passed ? "true" : "false", c->origin, c->accepted_samples, c->accept_wall_seconds, c->compared_scores);
        for (j = 0; j < c->scores; ++j) {
            const ekws_score *s = &c->observed[j];
            if (j) putchar(',');
            printf("{\"available_after_sample\":%" PRIu64 ",\"ready\":%" PRIu32 ",\"keyword_id\":%" PRIu32
                   ",\"logit_bits\":[", s->available_after_sample, s->ready, s->keyword_id);
            for (k = 0; k < EKWS_CLASSES; ++k) printf("%s%" PRIu32, k ? "," : "", float_bits(s->logits[k]));
            fputs("],\"probability_bits\":[", stdout);
            for (k = 0; k < EKWS_CLASSES; ++k) printf("%s%" PRIu32, k ? "," : "", float_bits(s->probabilities[k]));
            fputs("]}", stdout);
        }
        fputs("],\"events\":[", stdout);
        for (j = 0; j < c->event_count; ++j) {
            const ekws_event *e = &c->events[j];
            printf("%s{\"recording\":\"%s\",\"keyword_id\":%" PRIu32 ",\"time_s\":%.9f,\"confidence\":%.9g,"
                   "\"available_after_sample\":%" PRIu64 ",\"input_origin_sample\":%" PRIu64 "}",
                   j ? "," : "", c->recording, e->keyword_id,
                   (double)(e->available_after_sample - c->origin) / EKWS_SAMPLE_RATE, (double)e->confidence,
                   e->available_after_sample, c->origin);
        }
        printf("],\"cumulative_embeddings\":%" PRIu64 ",\"cumulative_ready_outputs\":%" PRIu64
               ",\"next_input_sample\":%" PRIu64 ",\"poisoned\":%" PRIu32 "}",
               c->final_stats.embeddings, c->final_stats.ready_outputs, c->final_stats.next_input_sample, c->final_stats.poisoned);
    }
    printf("],\"stream_bytes\":%zu,\"workspace_bytes\":%zu,\"new_ort_inference_calls\":0,"
           "\"ssc305_qualification\":false,\"formal_training_admitted\":false,\"model_promotion_allowed\":false,"
           "\"training_admitted\":false,\"human_gold\":false,\"trusted_negative\":false,"
           "\"pronunciation_verified\":false,\"qualification_allowed\":false,"
           "\"timing_scope\":\"sum of monotonic accept calls including in-memory verification callbacks; excludes file I/O; host only\"}\n",
           ekws_stream_bytes(), ekws_workspace_bytes());
}

int main(int argc, char **argv) {
    unsigned char *head = NULL;
    void *arena = NULL;
    float *workspace = NULL;
    ekws_stream *stream = NULL;
    ekws_callbacks callbacks;
    case_context reports[3], live;
    int verify = argc == 9 && !strcmp(argv[1], "--verify-cached"), result = 2;
    const char *head_path;
    size_t i, attempted = 0;
    memset(reports, 0, sizeof(reports)); memset(&live, 0, sizeof(live));
    if (!verify && (argc != 3 || !recording_id(argv[2]))) {
        fputs("usage: ekws_pcm_stream HEAD.bin RECORDING_ID < mono16k.pcm\n"
              "   or: ekws_pcm_stream --verify-cached HEAD.bin ZERO.pcm ZERO.trace SINE.pcm SINE.trace LCG.pcm LCG.trace\n", stderr);
        return 2;
    }
    head_path = argv[verify ? 2 : 1];
    head = (unsigned char *)malloc(EKWS_HEAD_BLOB_BYTES);
    arena = malloc(ekws_stream_bytes()); workspace = (float *)malloc(ekws_workspace_bytes());
    if (!head || !arena || !workspace || read_exact_file(head_path, head, EKWS_HEAD_BLOB_BYTES)) goto done;
    callbacks.event = event_callback; callbacks.event_context = &live;
    callbacks.score = verify ? score_observer : NULL; callbacks.score_context = &live;
    if (ekws_stream_init(arena, ekws_stream_bytes(), workspace, ekws_workspace_bytes(),
                         head, EKWS_HEAD_BLOB_BYTES, &callbacks, &stream) != EKWS_OK) goto done;
    if (verify) {
        static const char *names[] = {"zero", "sine400", "lcg14"};
        result = 0;
        for (i = 0; i < 3; ++i) {
            memset(&live, 0, sizeof(live)); live.recording = names[i]; live.verify = 1; live.stream = stream;
            live.origin = i == 2 ? 48000 : 0; attempted = i + 1;
            if (verify_case(&live, argv[3 + 2 * i], argv[4 + 2 * i], (unsigned)i)) result = 2;
            reports[i] = live;
            if (result) break;
        }
        print_verification(reports, attempted, result == 0);
        if (fflush(stdout)) result = 2;
    } else {
        unsigned char raw[640]; int16_t pcm[320]; uint64_t first = 0;
        live.recording = argv[2]; live.stream = stream;
        for (;;) {
            size_t got = fread(raw, 1, sizeof(raw), stdin), consumed = 0;
            if (ferror(stdin) || got % 2) goto done;
            if (!got) { result = 0; break; }
            decode_pcm(raw, pcm, got / 2);
            if (ekws_stream_accept_pcm16(stream, pcm, got / 2, first, &consumed) != EKWS_OK || consumed != got / 2) goto done;
            first += consumed;
        }
    }
done:
    if (verify && !attempted) print_verification(reports, 0, 0);
    if (result) fputs("English research stream failed; consumed input or emitted events must not be retried.\n", stderr);
    free(head); free(arena); free(workspace);
    return result;
}
