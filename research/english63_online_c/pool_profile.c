#define _GNU_SOURCE
#include "en63_stream.h"
#include "en63_generated.h"
#include <errno.h>
#include <inttypes.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#ifndef EN63_POOL_TIME_MAJOR
#define EN63_POOL_TIME_MAJOR 0
#endif
#if EN63_POOL_TIME_MAJOR != 0 && EN63_POOL_TIME_MAJOR != 1
#error "EN63_POOL_TIME_MAJOR must be 0 or 1"
#endif

#define SAMPLES 2101760u
#define FRAMES 13134u
#define CALLS 13136u
#define FNV_OFFSET UINT64_C(14695981039346656037)
#define FNV_PRIME UINT64_C(1099511628211)

typedef struct {
    FILE *trace;
    uint64_t frames, events, signature;
    int error;
} collector;

static void put32(unsigned char *p, uint32_t value)
{
    unsigned i;
    for (i = 0; i < 4; ++i) p[i] = (unsigned char)(value >> (8*i));
}

static void put64(unsigned char *p, uint64_t value)
{
    put32(p, (uint32_t)value);
    put32(p+4, (uint32_t)(value >> 32));
}

static void putfloat(unsigned char *p, float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    put32(p, bits);
}

static void collect(void *user, uint64_t frame, uint64_t sample,
                    const float features[32], const float logits[3],
                    const float probabilities[3], uint32_t proposal)
{
    collector *out = user;
    unsigned char record[168];
    unsigned i;
    ++out->frames;
    if (frame != out->frames || sample != 400+(frame-1)*160 || proposal > 2)
        out->error = 1;
    if (proposal) ++out->events;
    put32(record, 0); put64(record+4, sample);
    for (i = 0; i < 32; ++i) putfloat(record+12+4*i, features[i]);
    for (i = 0; i < 3; ++i) putfloat(record+140+4*i, logits[i]);
    for (i = 0; i < 3; ++i) putfloat(record+152+4*i, probabilities[i]);
    put32(record+164, proposal);
    for (i = 0; i < sizeof(record); ++i) {
        out->signature ^= record[i]; out->signature *= FNV_PRIME;
    }
    if (out->trace && fwrite(record, 1, sizeof(record), out->trace) != sizeof(record))
        out->error = 1;
}

static int nanoseconds(clockid_t clock_id, uint64_t *value)
{
    struct timespec time;
    if (clock_gettime(clock_id, &time) != 0 || time.tv_sec < 0 ||
        time.tv_nsec < 0 || time.tv_nsec >= 1000000000L) return 0;
    *value = (uint64_t)time.tv_sec*UINT64_C(1000000000)+(uint64_t)time.tv_nsec;
    return 1;
}

static int cpu_number(const char *value, unsigned *cpu)
{
    char *end;
    unsigned long result;
    if (!value[0] || strspn(value, "0123456789") != strlen(value)) return 0;
    errno = 0;
    result = strtoul(value, &end, 10);
    if (errno || *end || result >= CPU_SETSIZE) return 0;
    *cpu = (unsigned)result;
    return 1;
}

static int pin_cpu(unsigned cpu)
{
    cpu_set_t desired, actual;
    CPU_ZERO(&desired); CPU_SET(cpu, &desired);
    if (sched_setaffinity(0, sizeof(desired), &desired) != 0 ||
        sched_getaffinity(0, sizeof(actual), &actual) != 0) return 0;
    return CPU_COUNT(&actual) == 1 && CPU_ISSET(cpu, &actual);
}

int main(int argc, char **argv)
{
    const en63_model *model;
    en63_state *state = NULL;
    int16_t *pcm = NULL;
    uint64_t *feed_ns = NULL, cpu_start = 0, cpu_end = 0, wall_start = 0, wall_end = 0;
    struct rusage usage;
    collector out = {NULL, 0, 0, FNV_OFFSET, 0};
    FILE *input = NULL;
    unsigned cpu = 0, i;
    int timed, trace, result = 2;
    /* A fresh small native supervisor removes Python's inherited RSS high-water.
     * wait4 after child exit includes the complete native child's lifetime. */
    if (argc == 7 && strcmp(argv[1], "--supervise") == 0) {
        pid_t child;
        int status;
        struct rusage child_usage;
        if (!cpu_number(argv[6], &cpu) || !pin_cpu(cpu)) return 2;
        child = fork();
        if (child < 0) return 2;
        if (child == 0) {
            char *args[] = {argv[0], argv[2], argv[3], argv[4], argv[5], argv[6], NULL};
            execv(argv[0], args);
            _exit(127);
        }
        while (wait4(child, &status, 0, &child_usage) < 0)
            if (errno != EINTR) return 2;
        printf("{\"schema\":\"english63-pool-child-lifetime-v1\","
               "\"child_exit_code\":%d,\"child_signaled\":%s,"
               "\"process_lifetime_peak_rss_kib\":%ld,"
               "\"child_user_us\":%" PRIu64 ",\"child_system_us\":%" PRIu64 "}\n",
               WIFEXITED(status) ? WEXITSTATUS(status) : -1,
               WIFSIGNALED(status) ? "true" : "false", child_usage.ru_maxrss,
               (uint64_t)child_usage.ru_utime.tv_sec*1000000+(uint64_t)child_usage.ru_utime.tv_usec,
               (uint64_t)child_usage.ru_stime.tv_sec*1000000+(uint64_t)child_usage.ru_stime.tv_usec);
        if (fflush(stdout) != 0 || ferror(stdout)) return 2;
        return WIFEXITED(status) ? WEXITSTATUS(status) : 2;
    }
    if (argc != 6 || (strcmp(argv[1], "gru") && strcmp(argv[1], "dsc_tcn")) ||
        (strcmp(argv[2], "trace") && strcmp(argv[2], "warmup") && strcmp(argv[2], "timed")) ||
        !cpu_number(argv[5], &cpu) || !pin_cpu(cpu)) {
        fprintf(stderr, "usage: en63_pool KIND trace|warmup|timed PCM TRACE_OR_DASH CPU\n");
        return 2;
    }
    timed = strcmp(argv[2], "timed") == 0;
    trace = strcmp(argv[2], "trace") == 0;
    if (trace == (strcmp(argv[4], "-") == 0)) goto cleanup;
    model = strcmp(argv[1], "gru") == 0 ? &en63_gru_model : &en63_dsc_tcn_model;
    input = fopen(argv[3], "rb");
    if (!input || fseek(input, 0, SEEK_END) != 0 || ftell(input) != (long)(SAMPLES*2u) ||
        fseek(input, 0, SEEK_SET) != 0) goto cleanup;
    pcm = malloc(SAMPLES*sizeof(*pcm));
    state = calloc(1, sizeof(*state));
    if (timed) feed_ns = calloc(CALLS, sizeof(*feed_ns));
    if (!pcm || !state || (timed && !feed_ns) ||
        fread(pcm, 2, SAMPLES, input) != SAMPLES || fgetc(input) != EOF || ferror(input)) goto cleanup;
    if (fclose(input) != 0) { input = NULL; goto cleanup; }
    input = NULL;
    for (i = 0; i < SAMPLES; ++i) {
        const unsigned char *bytes = (const unsigned char *)pcm+2*i;
        unsigned value = (unsigned)bytes[0] | ((unsigned)bytes[1] << 8);
        pcm[i] = (int16_t)(value < 32768u ? (int)value : (int)value-65536);
    }
    /* Validation touches the immutable selected model outside the timed region.
     * PCM loading, allocation, init, trace I/O and JSON output are not benchmarked. */
    if (en63_init(state, model) != EN63_OK) goto cleanup;
    if (trace) {
        unsigned char header[16] = {'E','N','6','3','T','R','C','1',0,0,0,0,0,0,0,0};
        put32(header+8, 1); put32(header+12, 168);
        out.trace = fopen(argv[4], "wbx");
        if (!out.trace || fwrite(header, 1, sizeof(header), out.trace) != sizeof(header)) goto cleanup;
    }
    if (timed && (!nanoseconds(CLOCK_PROCESS_CPUTIME_ID, &cpu_start) ||
                  !nanoseconds(CLOCK_MONOTONIC, &wall_start))) goto cleanup;
    for (i = 0; i < CALLS; ++i) {
        uint64_t begin = 0, end = 0;
        if (timed && !nanoseconds(CLOCK_MONOTONIC, &begin)) goto cleanup;
        if (en63_feed(state, pcm+i*160u, 160, collect, &out) != EN63_OK) goto cleanup;
        if (timed) {
            if (!nanoseconds(CLOCK_MONOTONIC, &end) || end <= begin) goto cleanup;
            feed_ns[i] = end-begin;
        }
    }
    if (timed && (!nanoseconds(CLOCK_MONOTONIC, &wall_end) ||
                  !nanoseconds(CLOCK_PROCESS_CPUTIME_ID, &cpu_end) ||
                  wall_end <= wall_start || cpu_end <= cpu_start)) goto cleanup;
    {
        uint64_t frames_before = out.frames, events_before = out.events;
        if (en63_finish(state) != EN63_OK || out.frames != frames_before || out.events != events_before ||
            out.error || out.frames != FRAMES || state->samples != SAMPLES || state->events != out.events ||
            state->frames != FRAMES || state->discontinuities != 0 || state->segment_frames != FRAMES ||
            getrusage(RUSAGE_SELF, &usage) != 0) goto cleanup;
    }
    if (out.trace && fclose(out.trace) != 0) { out.trace = NULL; goto cleanup; }
    out.trace = NULL;
    printf("{\"schema\":\"english63-pool-native-pass-v1\",\"kind\":\"%s\","
           "\"variant\":%d,\"phase\":\"%s\",\"affinity_cpu\":%u,\"affinity_single_cpu\":true,"
           "\"pcm_samples\":%u,\"frames\":%u,\"feed_calls\":%u,\"proposal_count\":%" PRIu64 ","
           "\"signature64\":\"%016" PRIx64 "\",\"state_bytes\":%zu,"
           "\"weight_parameter_bytes\":%zu,\"frontend_table_bytes\":%u,\"input_pcm_bytes\":%u,"
           "\"cpu_ns\":%" PRIu64 ",\"wall_ns\":%" PRIu64 ","
           "\"trace_io_enabled\":%s,\"eof_padding_samples\":0,\"eof_events\":0,"
           "\"inference_end_high_water_rss_kib\":%ld,"
           "\"timed_region\":\"preloaded-pcm-en63_feed-and-constant-signature-callback\",\"feed_ns\":[",
           argv[1], EN63_POOL_TIME_MAJOR, argv[2], cpu, SAMPLES, FRAMES, CALLS, out.events,
           out.signature, en63_state_bytes(), model->parameter_count*sizeof(float),
           EN63_FRONTEND_TABLE_BYTES, SAMPLES*2u, cpu_end-cpu_start, wall_end-wall_start,
           trace ? "true" : "false", usage.ru_maxrss);
    if (timed) for (i = 0; i < CALLS; ++i) printf("%s%" PRIu64, i ? "," : "", feed_ns[i]);
    puts("]}");
    if (fflush(stdout) != 0 || ferror(stdout)) goto cleanup;
    result = 0;
cleanup:
    if (input) fclose(input);
    if (out.trace) fclose(out.trace);
    free(pcm); free(feed_ns); free(state);
    if (result) fprintf(stderr, "EN63_POOL_NATIVE_PASS_FAILED errno=%d\n", errno);
    return result;
}
