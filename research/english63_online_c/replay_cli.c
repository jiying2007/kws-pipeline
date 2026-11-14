#define _POSIX_C_SOURCE 200809L
#include "en63_stream.h"
#include "en63_generated.h"
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    FILE *trace;
    uint32_t stream;
    uint64_t frames, events;
    int error;
} collector;

static int put32(FILE *file, uint32_t value)
{
    unsigned char bytes[4];
    unsigned i;
    for (i = 0; i < 4; ++i) bytes[i] = (unsigned char)(value >> (8*i));
    return fwrite(bytes, 1, 4, file) == 4;
}

static int put64(FILE *file, uint64_t value)
{
    return put32(file, (uint32_t)value) && put32(file, (uint32_t)(value >> 32));
}

static int putfloat(FILE *file, float value)
{
    uint32_t bits;
    memcpy(&bits, &value, 4);
    return put32(file, bits);
}

static void collect(void *user, uint64_t frame, uint64_t sample,
                    const float features[32], const float logits[3],
                    const float probabilities[3], uint32_t proposal)
{
    collector *out = user;
    unsigned i;
    (void)frame;
    ++out->frames;
    if (proposal) ++out->events;
    if (!out->trace) return;
    if (!put32(out->trace, out->stream) || !put64(out->trace, sample)) out->error = 1;
    for (i = 0; i < 32; ++i) if (!putfloat(out->trace, features[i])) out->error = 1;
    for (i = 0; i < 3; ++i) if (!putfloat(out->trace, logits[i])) out->error = 1;
    for (i = 0; i < 3; ++i) if (!putfloat(out->trace, probabilities[i])) out->error = 1;
    if (!put32(out->trace, proposal)) out->error = 1;
}

static double seconds(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return -1;
    return (double)ts.tv_sec+(double)ts.tv_nsec/1e9;
}

static int compare_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y)-(x < y);
}

static size_t plan_size(int mixed, size_t index)
{
    static const size_t pattern[] = {1, 159, 161, 3199, 320, 7};
    return mixed ? pattern[index % 6] : 160;
}

int main(int argc, char **argv)
{
    const en63_model *model;
    en63_state *state = NULL;
    char paths[64][2048];
    size_t samples[64], files = 0, calls = 0, expected_calls = 0, i;
    uint64_t total_samples = 0;
    double *times = NULL, *ordered = NULL, start, elapsed, cpu;
    clock_t cpu_start;
    struct rusage usage;
    FILE *manifest = NULL;
    collector out = {0};
    int mixed, continuous, result = 2;
    /* Start a fresh child from this small native address space, before model
     * allocation/touch. Linux high-water RSS inherited across exec from a large
     * Python caller must not be reported as the native runtime's own peak. */
    if (argc == 7 && strcmp(argv[1], "--supervise") == 0) {
        pid_t child = fork();
        int status;
        if (child < 0) return 2;
        if (child == 0) {
            char *args[] = {argv[0], argv[2], argv[3], argv[4], argv[5], argv[6], NULL};
            execv(argv[0], args);
            _exit(127);
        }
        while (waitpid(child, &status, 0) < 0) if (errno != EINTR) return 2;
        return WIFEXITED(status) ? WEXITSTATUS(status) : 2;
    }
    if (argc != 6 || (strcmp(argv[1], "gru") && strcmp(argv[1], "dsc_tcn")) ||
        (strcmp(argv[2], "isolated") && strcmp(argv[2], "continuous")) ||
        (strcmp(argv[3], "160") && strcmp(argv[3], "mixed"))) {
        fprintf(stderr, "usage: en63_replay gru|dsc_tcn isolated|continuous 160|mixed PCM_MANIFEST TRACE_OR_DASH\n");
        return 2;
    }
    model = strcmp(argv[1], "gru") == 0 ? &en63_gru_model : &en63_dsc_tcn_model;
    mixed = strcmp(argv[3], "mixed") == 0;
    continuous = strcmp(argv[2], "continuous") == 0;
    manifest = fopen(argv[4], "r");
    if (!manifest) goto cleanup;
    while (files < 64 && fgets(paths[files], sizeof(paths[files]), manifest)) {
        FILE *input;
        long bytes;
        size_t offset = 0, index = 0, length = strlen(paths[files]);
        if (!length || paths[files][length-1] != '\n') goto cleanup;
        paths[files][--length] = 0;
        if (!length || strchr(paths[files], '\r')) goto cleanup;
        input = fopen(paths[files], "rb");
        if (!input) goto cleanup;
        if (fseek(input, 0, SEEK_END) != 0) { fclose(input); goto cleanup; }
        bytes = ftell(input);
        fclose(input);
        if (bytes <= 0 || bytes % 2 || bytes > 384000) goto cleanup;
        samples[files] = (size_t)bytes/2;
        total_samples += samples[files];
        while (offset < samples[files]) {
            size_t take = plan_size(mixed, index++);
            if (take > samples[files]-offset) take = samples[files]-offset;
            offset += take; ++expected_calls;
        }
        ++files;
    }
    if (!feof(manifest) || !files || expected_calls > 100000) goto cleanup;
    fclose(manifest); manifest = NULL;
    state = calloc(1, sizeof(*state));
    times = malloc(expected_calls*sizeof(*times));
    ordered = malloc(expected_calls*sizeof(*ordered));
    if (!state || !times || !ordered || en63_init(state, model) != EN63_OK) goto cleanup;
    if (strcmp(argv[5], "-") != 0) {
        out.trace = fopen(argv[5], "wbx");
        if (!out.trace || fwrite("EN63TRC1", 1, 8, out.trace) != 8 ||
            !put32(out.trace, 1) || !put32(out.trace, 168)) goto cleanup;
    }
    cpu_start = clock(); start = seconds();
    if (start < 0 || cpu_start == (clock_t)-1) goto cleanup;
    for (i = 0; i < files; ++i) {
        FILE *input = fopen(paths[i], "rb");
        size_t offset = 0, index = 0;
        unsigned char bytes[EN63_MAX_CHUNK*2];
        int16_t pcm[EN63_MAX_CHUNK];
        if (!input) goto cleanup;
        if (!continuous && i && en63_reset(state) != EN63_OK) { fclose(input); goto cleanup; }
        out.stream = continuous ? 0 : (uint32_t)i;
        while (offset < samples[i]) {
            size_t k, take = plan_size(mixed, index++);
            double begin, end;
            if (take > samples[i]-offset) take = samples[i]-offset;
            if (fread(bytes, 2, take, input) != take) { fclose(input); goto cleanup; }
            for (k = 0; k < take; ++k) {
                uint16_t v = (uint16_t)((uint16_t)bytes[k*2] | ((uint16_t)bytes[k*2+1] << 8));
                int value = v < 32768u ? (int)v : (int)v-65536;
                pcm[k] = (int16_t)value;
            }
            begin = seconds();
            if (en63_feed(state, pcm, take, collect, &out) != EN63_OK) { fclose(input); goto cleanup; }
            end = seconds();
            if (begin < 0 || end < begin || calls >= expected_calls) { fclose(input); goto cleanup; }
            times[calls++] = (end-begin)*1e6;
            offset += take;
        }
        if (fgetc(input) != EOF || ferror(input)) { fclose(input); goto cleanup; }
        fclose(input);
        if (!continuous && en63_finish(state) != EN63_OK) goto cleanup;
    }
    {
        uint64_t old_frames = out.frames, old_events = out.events;
        if (en63_finish(state) != EN63_OK || en63_finish(state) != EN63_OK ||
            old_frames != out.frames || old_events != out.events || out.error || calls != expected_calls) goto cleanup;
    }
    cpu = (double)(clock()-cpu_start)/CLOCKS_PER_SEC;
    elapsed = seconds()-start;
    if (getrusage(RUSAGE_SELF, &usage) != 0 || elapsed < 0 || cpu < 0) goto cleanup;
    if (out.trace && fclose(out.trace) != 0) { out.trace = NULL; goto cleanup; }
    out.trace = NULL;
    memcpy(ordered, times, calls*sizeof(*times));
    qsort(ordered, calls, sizeof(*ordered), compare_double);
    printf("{\"kind\":\"%s\",\"mode\":\"%s\",\"chunk_plan\":\"%s\",", argv[1], argv[2], argv[3]);
    printf("\"pcm_samples\":%" PRIu64 ",\"frames\":%" PRIu64 ",\"proposal_count\":%" PRIu64 ",", total_samples, out.frames, out.events);
    printf("\"state_bytes\":%zu,\"weight_parameter_bytes\":%zu,\"frontend_table_bytes\":%u,", sizeof(*state), model->parameter_count*4, EN63_FRONTEND_TABLE_BYTES);
    printf("\"process_peak_rss_kib\":%ld,\"cpu_seconds\":%.9f,\"wall_seconds\":%.9f,", usage.ru_maxrss, cpu, elapsed);
    printf("\"feed_calls\":%zu,\"feed_p50_us\":%.6f,\"feed_p95_us\":%.6f,\"feed_p99_us\":%.6f,\"feed_max_us\":%.6f,", calls,
           ordered[(calls-1)*50/100], ordered[(calls-1)*95/100], ordered[(calls-1)*99/100], ordered[calls-1]);
    printf("\"trace_io_enabled\":%s,\"eof_padding_samples\":0,\"eof_events\":0,\"stack_bytes\":null,", strcmp(argv[5], "-") ? "true" : "false");
    printf("\"rss_scope\":\"native-C-process-self-linux-kib-not-python-parent\",\"ssc305_measured\":false,\"feed_microseconds\":[");
    for (i = 0; i < calls; ++i) printf("%s%.6f", i ? "," : "", times[i]);
    puts("]}");
    if (ferror(stdout)) goto cleanup;
    result = 0;
cleanup:
    if (manifest) fclose(manifest);
    if (out.trace) fclose(out.trace);
    free(state); free(times); free(ordered);
    if (result) fprintf(stderr, "EN63_NATIVE_REPLAY_FAILED errno=%d\n", errno);
    return result;
}
