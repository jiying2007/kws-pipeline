#define _POSIX_C_SOURCE 200809L
/* Research only: actual Sherpa C ABI memory/CPU profile in its own process.
 * The parent verifies all source/model/runtime digests and starts this
 * executable separately. No embedded Python, TTS, ASR or synthetic audio.
 * These x64 host measurements NEVER qualify SSC305 resources.
 */
#include "pcm_kws.h"
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

enum { KWS_FEED = 320, MAX_CALLS = 20000, MAX_WAV_FRAMES = 16000 * 120 };
static unsigned event_count = 0;
static unsigned eof_only_count = 0;
static unsigned bad_event_count = 0;

static void stop(const char *why) {
    fprintf(stderr, "EN_NATIVE_PROFILE_STOP: %s\n", why);
    exit(2);
}
static void event_cb(void *user, const PcmKwsEvent *event) {
    (void)user;
    if (!event || !event->keyword ||
        strcmp(event->keyword, "HEY_MERO") != 0 || event->keyword_id != 3)
        bad_event_count++;
    else {
        event_count++;
        if (event->eof_flush) eof_only_count++;
    }
}
static double elapsed_ms(const struct timespec *a, const struct timespec *b) {
    return (double)(b->tv_sec - a->tv_sec) * 1000.0 +
           (double)(b->tv_nsec - a->tv_nsec) / 1000000.0;
}
static void mark(clockid_t clock, struct timespec *result) {
    if (clock_gettime(clock, result) != 0) stop("clock_gettime failed");
}
static unsigned long read_proc_rss(const char *key) {
    FILE *f = fopen("/proc/self/status", "r");
    char line[256];
    unsigned long n = 0;
    if (!f) stop("Linux proc status missing");
    while (fgets(line, sizeof(line), f)) {
        size_t len = strlen(key);
        if (strncmp(line, key, len) == 0) {
            char *end;
            const char *value = line + len;
            errno = 0;
            n = strtoul(value, &end, 10);
            if (errno != 0 || end == value) n = 0;
            break;
        }
    }
    fclose(f);
    if (!n) stop("Linux RSS key missing");
    return n; /* KiB */
}
static unsigned read_u16(FILE *f) {
    unsigned char b[2];
    if (fread(b, 1, 2, f) != 2) stop("short RIFF uint16");
    return (unsigned)b[0] | ((unsigned)b[1] << 8);
}
static uint32_t read_u32(FILE *f) {
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) stop("short RIFF uint32");
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) |
           ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}
static uint32_t begin_wav(FILE *f) {
    char root[4], wave[4], id[4];
    int saw_fmt = 0;
    if (fread(root, 1, 4, f) != 4 || memcmp(root, "RIFF", 4))
        stop("not original RIFF WAV");
    (void)read_u32(f);
    if (fread(wave, 1, 4, f) != 4 || memcmp(wave, "WAVE", 4))
        stop("not original WAVE");
    for (unsigned part = 0; part < 128; part++) {
        uint32_t bytes;
        if (fread(id, 1, 4, f) != 4) stop("WAV chunk missing");
        bytes = read_u32(f);
        if (memcmp(id, "fmt ", 4) == 0) {
            unsigned format, channels, rate, alignment, bits;
            if (saw_fmt || bytes < 16 || bytes > 1024)
                stop("invalid WAV fmt size");
            format = read_u16(f);
            channels = read_u16(f);
            rate = read_u32(f);
            (void)read_u32(f);
            alignment = read_u16(f);
            bits = read_u16(f);
            if (format != 1 || channels != 1 || rate != 16000 ||
                alignment != 2 || bits != 16)
                stop("source must be PCM16 mono16k");
            if (fseek(f, (long)(bytes - 16 + (bytes & 1)), SEEK_CUR))
                stop("WAV fmt seek failed");
            saw_fmt = 1;
        } else if (memcmp(id, "data", 4) == 0) {
            if (!saw_fmt || bytes % 2 || bytes < 800 ||
                bytes / 2 > MAX_WAV_FRAMES)
                stop("source PCM geometry");
            return bytes / 2;
        } else {
            if (bytes > 40000000U ||
                fseek(f, (long)(bytes + (bytes & 1)), SEEK_CUR))
                stop("WAV unknown chunk invalid");
        }
    }
    stop("source WAV data not found");
    return 0;
}
static int double_cmp(const void *a, const void *b) {
    const double l = *(const double*)a, r = *(const double*)b;
    return (l > r) - (l < r);
}
static double pct(const double *samples, size_t n, double p) {
    size_t i;
    if (!n) stop("no actual feed calls");
    i = (size_t)ceil((double)n * p) - 1;
    if (i >= n) i = n - 1;
    return samples[i];
}
int main(int argc, char **argv) {
    /* argc=7: encoder decoder joiner tokens keywords original_63_wav_list */
    const PcmKwsFiles files = {
        argc > 1 ? argv[1] : NULL, argc > 2 ? argv[2] : NULL,
        argc > 3 ? argv[3] : NULL, argc > 4 ? argv[4] : NULL,
        argc > 5 ? argv[5] : NULL
    };
    FILE *paths = NULL;
    PcmKws *kws = NULL;
    char error_buf[512] = {0};
    char path[4096];
    int16_t frame[KWS_FEED];
    double *wall = NULL, *cpu = NULL, wall_total = 0, cpu_total = 0;
    unsigned clip_count = 0;
    uint64_t sample_count = 0;
    size_t feeds = 0;
    unsigned late = 0;
    unsigned long rss_before, rss_loaded, rss_after;
    struct rusage usage;
    struct timespec t1, t2, c1, c2;

#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__)
    if (__BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__)
        stop("big-endian host requires explicit PCM16 swap");
#endif
    if (argc != 7) stop("expected pinned model files and absolute WAV list");
    paths = fopen(argv[6], "r");
    if (!paths) stop("WAV list missing");
    wall = (double*)calloc(MAX_CALLS, sizeof(double));
    cpu = (double*)calloc(MAX_CALLS, sizeof(double));
    if (!wall || !cpu) stop("bounded latency array allocation");
    rss_before = read_proc_rss("VmRSS:");
    if (pcm_kws_create(&files, event_cb, NULL, &kws,
                       error_buf, sizeof(error_buf)) != PCM_KWS_OK || !kws)
        stop("actual C KWS model creation failed");
    rss_loaded = read_proc_rss("VmRSS:");
    while (fgets(path, sizeof(path), paths)) {
        FILE *wav;
        uint32_t remaining;
        size_t npath = strlen(path);
        if (npath < 3 || path[npath - 1] != '\n')
            stop("WAV list line malformed or overlong");
        path[npath - 1] = '\0';
        if (path[0] != '/' || strstr(path, ".."))
            stop("WAV paths must be canonical absolute");
        wav = fopen(path, "rb");
        if (!wav) stop("source WAV missing");
        remaining = begin_wav(wav);
        if (pcm_kws_reset(kws, error_buf, sizeof(error_buf)) != PCM_KWS_OK)
            stop("native reset failed");
        sample_count += remaining;
        while (remaining) {
            const size_t request = remaining > KWS_FEED ? KWS_FEED : remaining;
            size_t n = fread(frame, sizeof(int16_t), request, wav);
            if (n != request) stop("short original WAV data");
            mark(CLOCK_MONOTONIC, &t1);
            mark(CLOCK_PROCESS_CPUTIME_ID, &c1);
            if (pcm_kws_feed(kws, frame, n, error_buf,
                             sizeof(error_buf)) != PCM_KWS_OK)
                stop("actual native feed failed");
            mark(CLOCK_PROCESS_CPUTIME_ID, &c2);
            mark(CLOCK_MONOTONIC, &t2);
            if (feeds == MAX_CALLS) stop("bounded number of native PCM feeds");
            wall[feeds] = elapsed_ms(&t1, &t2);
            cpu[feeds] = elapsed_ms(&c1, &c2);
            wall_total += wall[feeds];
            cpu_total += cpu[feeds];
            if (wall[feeds] > (double)n * 1000.0 / 16000.0)
                late++;
            feeds++;
            remaining -= (uint32_t)n;
        }
        if (pcm_kws_finish(kws, error_buf, sizeof(error_buf)) != PCM_KWS_OK)
            stop("actual native finish failed");
        if (fgetc(wav) != EOF) stop("unexpected original WAV trailing payload");
        fclose(wav);
        clip_count++;
        if (clip_count > 63) stop("original source clip count exceeded");
    }
    fclose(paths);
    rss_after = read_proc_rss("VmRSS:");
    if (getrusage(RUSAGE_SELF, &usage)) stop("getrusage failed");
    pcm_kws_destroy(kws);
    if (clip_count != 63 || sample_count != 2101760 || event_count != 5 ||
        eof_only_count != 0 || bad_event_count != 0)
        stop("source-locked native model and five original events mismatch");
    qsort(wall, feeds, sizeof(double), double_cmp);
    qsort(cpu, feeds, sizeof(double), double_cmp);
    printf("{\"schema\":\"english63-real-native-c-only-resource-profile-v1\",");
    printf("\"status\":\"HOST_NATIVE_KWS_RESOURCE_MEASURED_NOT_SSC305\",");
    printf("\"source_wavs\":%u,\"source_samples\":%" PRIu64 ",",
           clip_count, sample_count);
    printf("\"actual_english_events\":%u,\"bad_events\":%u,\"eof_events\":%u,",
           event_count, bad_event_count, eof_only_count);
    printf("\"native_feed_calls\":%zu,\"deadline_misses_at_input_chunk_duration\":%u,",
           feeds, late);
    printf("\"rss_kib\":{\"before_model_create\":%lu,\"after_model_create\":%lu,"
           "\"after_pcm_replay\":%lu,\"process_peak\":%ld},",
           rss_before, rss_loaded, rss_after, usage.ru_maxrss);
    printf("\"host_native_wall_feed_ms\":{\"p50\":%.8f,\"p95\":%.8f,"
           "\"p99\":%.8f,\"max\":%.8f,\"sum\":%.8f},",
           pct(wall, feeds, .50), pct(wall, feeds, .95), pct(wall, feeds, .99),
           wall[feeds-1], wall_total);
    printf("\"host_native_cpu_feed_ms\":{\"p50\":%.8f,\"p95\":%.8f,"
           "\"p99\":%.8f,\"max\":%.8f,\"sum\":%.8f},",
           pct(cpu, feeds, .50), pct(cpu, feeds, .95), pct(cpu, feeds, .99),
           cpu[feeds-1], cpu_total);
    printf("\"original_audio_seconds\":131.36,"
           "\"native_feed_cpu_seconds_per_audio_second\":%.9f,",
           (cpu_total/1000.0)/131.36);
    printf("\"model_load_included_in_feed_timing\":false,"
           "\"python_audio_buffers_in_profiled_process\":false,"
           "\"ssc305_cpu_rss_qualified\":false,"
           "\"human_far_frr_qualified\":false,"
           "\"model_promotion_allowed\":false}\n");
    free(cpu);
    free(wall);
    return 0;
}
