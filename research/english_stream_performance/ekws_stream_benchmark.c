#define _POSIX_C_SOURCE 200809L
#include "ekws_research_stream.h"
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

/* Installed-ABI consumer only. No model/source change, observer, reset, padding,
 * audio loop, callback I/O, or timing claim about annotated wake-word latency. */
enum { CHUNK_SAMPLES = 320, WARMUP_SAMPLES = 35200,
       MIN_SAMPLES = WARMUP_SAMPLES + EKWS_SAMPLE_RATE,
       MAX_SAMPLES = 300 * EKWS_SAMPLE_RATE };
typedef struct { double wall, cpu; size_t samples; } timing;
typedef struct { long rss_kib, pss_kib, process_peak_rss_kib; } memory;
typedef struct { uint64_t events, last_sample, digest; } event_counter;

static double elapsed(const struct timespec *a, const struct timespec *b) {
    return (double)(b->tv_sec - a->tv_sec) + (double)(b->tv_nsec - a->tv_nsec) / 1e9;
}
static uint64_t hash_word(uint64_t h, uint64_t value, unsigned bytes) {
    unsigned i;
    for (i = 0; i < bytes; ++i) {
        h ^= value & 255u; h *= UINT64_C(1099511628211); value >>= 8;
    }
    return h;
}
static int count_event(void *context, const ekws_event *event) {
    event_counter *c = (event_counter *)context;
    uint32_t bits;
    if (event->keyword_id < 1 || event->keyword_id >= EKWS_CLASSES ||
        !isfinite(event->confidence) || event->available_after_sample < c->last_sample)
        return -1;
    memcpy(&bits, &event->confidence, sizeof(bits));
    c->digest = hash_word(c->digest, event->keyword_id, 4);
    c->digest = hash_word(c->digest, event->available_after_sample, 8);
    c->digest = hash_word(c->digest, bits, 4);
    c->last_sample = event->available_after_sample; ++c->events;
    return 0;
}
static memory read_memory(void) {
    memory m = {-1, -1, -1};
    struct rusage usage;
    char line[256];
    FILE *f = fopen("/proc/self/smaps_rollup", "r");
    if (f) {
        while (fgets(line, sizeof(line), f)) {
            long v;
            if (sscanf(line, "Rss: %ld kB", &v) == 1) m.rss_kib = v;
            if (sscanf(line, "Pss: %ld kB", &v) == 1) m.pss_kib = v;
        }
        fclose(f);
    }
    if (m.rss_kib < 0 && (f = fopen("/proc/self/status", "r")) != NULL) {
        while (fgets(line, sizeof(line), f)) {
            long v;
            if (sscanf(line, "VmRSS: %ld kB", &v) == 1) m.rss_kib = v;
        }
        fclose(f);
    }
#ifdef __linux__
    if (getrusage(RUSAGE_SELF, &usage) == 0) m.process_peak_rss_kib = usage.ru_maxrss;
#else
    (void)usage;
#endif
    return m;
}
static void print_optional(long n) {
    if (n < 0) fputs("null", stdout); else printf("%ld", n);
}
static void print_memory(const memory *m) {
    fputs("{\"rss_kib\":", stdout); print_optional(m->rss_kib);
    fputs(",\"pss_kib\":", stdout); print_optional(m->pss_kib);
    fputs(",\"process_peak_rss_kib\":", stdout); print_optional(m->process_peak_rss_kib);
    putchar('}');
}
static int read_exact(const char *path, unsigned char *buffer, size_t count) {
    FILE *f = fopen(path, "rb");
    int ok;
    if (!f) return -1;
    ok = fread(buffer, 1, count, f) == count && fgetc(f) == EOF && !ferror(f);
    if (fclose(f)) ok = 0;
    return ok ? 0 : -1;
}
static int double_cmp(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static double quantile(const double *sorted, size_t count, double p) {
    size_t rank = (size_t)ceil(p * (double)count);
    return sorted[(rank ? rank : 1) - 1];
}
static void print_segment(const timing *t, size_t count, double *scratch) {
    size_t i, deadline_misses = 0;
    uint64_t samples = 0;
    double wall = 0, cpu = 0, duration;
    for (i = 0; i < count; ++i) {
        wall += t[i].wall; cpu += t[i].cpu; samples += t[i].samples;
        scratch[i] = t[i].wall;
        if (t[i].wall > (double)t[i].samples / EKWS_SAMPLE_RATE) ++deadline_misses;
    }
    duration = (double)samples / EKWS_SAMPLE_RATE;
    qsort(scratch, count, sizeof(*scratch), double_cmp);
    printf("{\"samples\":%" PRIu64 ",\"audio_seconds\":%.9f,\"calls\":%zu,"
           "\"accept_wall_seconds\":%.9f,\"accept_process_cpu_seconds\":%.9f,"
           "\"wall_rtf\":%.9f,\"cpu_seconds_per_audio_second\":%.9f,"
           "\"call_wall_p50_ms\":%.9f,\"call_wall_p95_ms\":%.9f,"
           "\"call_wall_p99_ms\":%.9f,\"call_wall_max_ms\":%.9f,"
           "\"calls_over_input_duration\":%zu,\"calls_over_input_duration_fraction\":%.9f}",
           samples, duration, count, wall, cpu, wall / duration, cpu / duration,
           1000 * quantile(scratch, count, .50), 1000 * quantile(scratch, count, .95),
           1000 * quantile(scratch, count, .99), 1000 * scratch[count - 1],
           deadline_misses, (double)deadline_misses / count);
}

int main(int argc, char **argv) {
    unsigned char *raw = NULL, *head = NULL;
    int16_t *pcm = NULL;
    void *arena = NULL;
    float *workspace = NULL;
    timing *times = NULL;
    double *scratch = NULL;
    ekws_stream *stream = NULL;
    ekws_stats stats, warmup_stats = {0};
    ekws_callbacks callbacks;
    event_counter counter = {0, 0, UINT64_C(14695981039346656037)};
    memory input_resident, after_init, after_warmup = {-1, -1, -1}, after_stream;
    struct timespec iw0, iw1, ic0, ic1;
    size_t samples = 0, offset = 0, calls = 0, total_calls, warmup_calls = 0;
    size_t arena_bytes = ekws_stream_bytes(), workspace_bytes = ekws_workspace_bytes();
    double init_wall = 0, init_cpu = 0;
    FILE *f = NULL;
    long bytes;
    int result = 2;
    const char *stage = "arguments";
    if (argc != 3) {
        fputs("usage: ekws_stream_benchmark HEAD.bin MONO16K_S16LE.pcm\n", stderr);
        return 2;
    }
    stage = "input_size";
    f = fopen(argv[2], "rb");
    if (!f || fseek(f, 0, SEEK_END) || (bytes = ftell(f)) < 0) goto done;
    if (fclose(f)) { f = NULL; goto done; }
    f = NULL;
    if (bytes % 2 || bytes / 2 < MIN_SAMPLES || bytes / 2 > MAX_SAMPLES) goto done;
    samples = (size_t)bytes / 2;
    total_calls = (samples + CHUNK_SAMPLES - 1) / CHUNK_SAMPLES;
    stage = "preload";
    raw = (unsigned char *)malloc(samples * 2);
    pcm = (int16_t *)malloc(samples * sizeof(*pcm));
    head = (unsigned char *)malloc(EKWS_HEAD_BLOB_BYTES);
    times = (timing *)calloc(total_calls, sizeof(*times));
    scratch = (double *)calloc(total_calls, sizeof(*scratch));
    if (!raw || !pcm || !head || !times || !scratch ||
        read_exact(argv[2], raw, samples * 2) || read_exact(argv[1], head, EKWS_HEAD_BLOB_BYTES)) goto done;
    for (offset = 0; offset < samples; ++offset) {
        uint32_t u = (uint32_t)raw[2 * offset] | (uint32_t)raw[2 * offset + 1] << 8;
        pcm[offset] = (int16_t)(u < 32768 ? (int32_t)u : (int32_t)u - 65536);
    }
    free(raw); raw = NULL; offset = 0;
    input_resident = read_memory();
    stage = "allocate_stream";
    arena = malloc(arena_bytes); workspace = (float *)malloc(workspace_bytes);
    if (!arena || !workspace) goto done;
    callbacks.event = count_event; callbacks.event_context = &counter;
    callbacks.score = NULL; callbacks.score_context = NULL;
    stage = "init";
    if (clock_gettime(CLOCK_MONOTONIC, &iw0) || clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ic0)) goto done;
    if (ekws_stream_init(arena, arena_bytes, workspace, workspace_bytes,
                         head, EKWS_HEAD_BLOB_BYTES, &callbacks, &stream) != EKWS_OK) goto done;
    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ic1) || clock_gettime(CLOCK_MONOTONIC, &iw1)) goto done;
    init_wall = elapsed(&iw0, &iw1); init_cpu = elapsed(&ic0, &ic1);
    free(head); head = NULL;
    after_init = read_memory();
    stage = "accept";
    while (offset < samples) {
        struct timespec w0, w1, c0, c1;
        size_t count = samples - offset, consumed = 0;
        int fed;
        if (count > CHUNK_SAMPLES) count = CHUNK_SAMPLES;
        if (clock_gettime(CLOCK_MONOTONIC, &w0) || clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &c0)) goto done;
        fed = ekws_stream_accept_pcm16(stream, pcm + offset, count, offset, &consumed);
        offset += consumed;
        if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &c1) || clock_gettime(CLOCK_MONOTONIC, &w1)) goto done;
        if (fed != EKWS_OK || consumed != count) goto done;
        times[calls].wall = elapsed(&w0, &w1); times[calls].cpu = elapsed(&c0, &c1);
        times[calls].samples = consumed;
        if (!isfinite(times[calls].wall) || !isfinite(times[calls].cpu) ||
            times[calls].wall < 0 || times[calls].cpu < 0) goto done;
        ++calls;
        if (offset == WARMUP_SAMPLES) {
            warmup_calls = calls;
            if (ekws_stream_get_stats(stream, &warmup_stats) != EKWS_OK) goto done;
            after_warmup = read_memory();
        }
    }
    after_stream = read_memory();
    stage = "stats";
    if (ekws_stream_get_stats(stream, &stats) != EKWS_OK || !warmup_calls || calls <= warmup_calls ||
        stats.accepted_samples != samples || stats.next_input_sample != samples ||
        warmup_stats.accepted_samples != WARMUP_SAMPLES || warmup_stats.embeddings != 16 ||
        warmup_stats.ready_outputs != 1 ||
        stats.embeddings != 1 + (samples - EKWS_SAMPLE_RATE) / 1280 ||
        stats.ready_outputs != stats.embeddings - 15 ||
        stats.delivered_events != counter.events || stats.detected_events != counter.events ||
        stats.resets || stats.discontinuities || stats.callback_failures || stats.feed_failures ||
        stats.poisoned || stats.counters_saturated) goto done;
    printf("{\"schema\":\"english-stream-performance-v1\",\"passed\":true,"
           "\"measurement_completed\":true,\"performance_budget_evaluated\":false,"
           "\"abi\":%u,\"sample_rate_hz\":%u,\"score_observer_enabled\":false,"
           "\"chunk_samples\":%u,\"warmup_samples\":%u,\"input_samples\":%zu,"
           "\"input_repetitions\":1,\"resets\":0,\"eof_padding_samples\":0,"
           "\"stream_bytes\":%zu,\"workspace_bytes\":%zu,"
           "\"preloaded_pcm_bytes\":%zu,\"timing_array_bytes\":%zu,"
           "\"init_wall_seconds\":%.9f,\"init_process_cpu_seconds\":%.9f,\"warmup\":",
           EKWS_RESEARCH_ABI, EKWS_SAMPLE_RATE, CHUNK_SAMPLES, WARMUP_SAMPLES, samples,
           arena_bytes, workspace_bytes, samples * sizeof(*pcm), total_calls * (sizeof(*times) + sizeof(*scratch)),
           init_wall, init_cpu);
    print_segment(times, warmup_calls, scratch);
    fputs(",\"steady\":", stdout); print_segment(times + warmup_calls, calls - warmup_calls, scratch);
    fputs(",\"whole_stream\":", stdout); print_segment(times, calls, scratch);
    printf(",\"warmup_output_counts\":{\"embeddings\":%" PRIu64 ",\"ready_outputs\":%" PRIu64
           ",\"events\":%" PRIu64 "},\"steady_output_counts\":{\"embeddings\":%" PRIu64
           ",\"ready_outputs\":%" PRIu64 ",\"events\":%" PRIu64 "}",
           warmup_stats.embeddings, warmup_stats.ready_outputs, warmup_stats.delivered_events,
           stats.embeddings - warmup_stats.embeddings, stats.ready_outputs - warmup_stats.ready_outputs,
           stats.delivered_events - warmup_stats.delivered_events);
    printf(",\"embeddings\":%" PRIu64 ",\"ready_outputs\":%" PRIu64 ",\"events\":%" PRIu64
           ",\"event_digest_fnv1a64\":\"%016" PRIx64 "\",\"memory\":{\"input_resident_before_stream_allocation\":",
           stats.embeddings, stats.ready_outputs, counter.events, counter.digest);
    print_memory(&input_resident); fputs(",\"after_init_and_head_blob_release\":", stdout); print_memory(&after_init);
    fputs(",\"after_warmup\":", stdout); print_memory(&after_warmup);
    fputs(",\"after_stream\":", stdout); print_memory(&after_stream);
    fputs("},\"timing_scope\":\"accept calls including constant-time event count/digest and clock overhead; excludes input I/O, PCM decoding, memory sampling and report I/O\","
          "\"memory_scope\":\"process observations include runner, preloaded input, timing arrays and shared/static code; peak is process lifetime high-water, not isolated KWS peak\","
          "\"call_quantiles\":\"nearest-rank; last partial chunk included\","
          "\"realtime_pacing\":false,\"word_end_latency_measured\":false,\"far_frr_measured\":false,"
          "\"ssc305_qualification\":false,\"model_promotion_allowed\":false}\n", stdout);
    if (fflush(stdout)) goto done;
    result = 0;
done:
    if (result) {
        fprintf(stderr, "benchmark failed at %s after %zu accepted samples; no retry performed\n", stage, offset);
    }
    if (f) fclose(f);
    free(raw); free(pcm); free(head); free(arena); free(workspace); free(times); free(scratch);
    return result;
}
