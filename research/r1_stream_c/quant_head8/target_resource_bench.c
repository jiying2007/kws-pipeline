/* SPDX-License-Identifier: Apache-2.0
 * Owner-side R1 resource benchmark for a real Linux ARM DUT.
 * The file-based feed is NOT live AFE capture, device qualification, FAR or FRR.
 * This process never publishes original PCM or model weights.
 */
#define _POSIX_C_SOURCE 200809L
#include "fsmn_sdk.h"
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>
#include <sys/utsname.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define CHUNK_BYTES 640u
#define HIST_BUCKETS 32u
#define PCM_LIMIT_BYTES UINT64_C(536870912)
static uint64_t frame_count, event_count, word_events[3];
static uint64_t last_source, last_delivery;
static int callback_failed;

static int on_frame(void *unused, uint64_t frame, uint64_t source, uint64_t delivery,
                    const double *features, const double *logits) {
    (void)unused;
    (void)features;
    if (!logits || frame != frame_count || source > delivery ||
        (frame_count && (source < last_source || delivery < last_delivery))) {
        callback_failed = 1;
        return 1;
    }
    last_source = source;
    last_delivery = delivery;
    frame_count++;
    return 0;
}
static int on_event(void *unused, const fsmn_event *e) {
    (void)unused;
    if (!e || e->abi != FSMN_SDK_ABI || e->keyword_id < 1 ||
        e->keyword_id > 2 || e->dependency_end_sample > e->delivery_end_sample ||
        e->model_frame >= frame_count) {
        callback_failed = 1;
        return 1;
    }
    event_count++;
    word_events[e->keyword_id]++;
    return 0;
}
static int now_ns(clockid_t id, uint64_t *value) {
    struct timespec ts;
    if (clock_gettime(id, &ts) != 0 || ts.tv_sec < 0 || ts.tv_nsec < 0)
        return -1;
    *value = (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
    return 0;
}
static unsigned char *read_exact(const char *name, size_t bytes) {
    FILE *f = fopen(name, "rb");
    unsigned char *data = NULL;
    if (!f) return NULL;
    data = (unsigned char *)malloc(bytes);
    if (!data) { (void)fclose(f); return NULL; }
    int good = fread(data, 1, bytes, f) == bytes;
    int tail = fgetc(f);
    int read_error = ferror(f);
    int close_error = fclose(f);
    if (tail != EOF || read_error || close_error) good = 0;
    if (!good) { free(data); return NULL; }
    return data;
}

/* Only newer borrowed-view SDK headers authorize this explicit option.
 * Frozen source-only reference headers continue to build in copy mode.
 */
#if defined(FSMN_MODEL_VIEW_ALIGNMENT)
#if !defined(O_NOFOLLOW) || !defined(O_CLOEXEC)
#error "Linux secure read-only model mmap requires O_NOFOLLOW and O_CLOEXEC"
#endif
static unsigned char *map_model_exact(const char *name, size_t bytes) {
    int fd = open(name, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) return NULL;
    struct stat st;
    int valid = fstat(fd, &st) == 0 && S_ISREG(st.st_mode) &&
                st.st_size >= 0 && (uintmax_t)st.st_size == (uintmax_t)bytes;
    if (!valid) { (void)close(fd); return NULL; }
    void *view = mmap(NULL, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    int bad_close = close(fd) != 0;
    if (view == MAP_FAILED) return NULL;
    if (bad_close || (uintptr_t)view % FSMN_MODEL_VIEW_ALIGNMENT) {
        (void)munmap(view, bytes);
        return NULL;
    }
    return (unsigned char *)view;
}
#endif
static int release_model(unsigned char *model, int mapped) {
    if (!model) return 0;
#if defined(FSMN_MODEL_VIEW_ALIGNMENT)
    if (mapped) return munmap(model, FSMN_MODEL_BYTES);
#else
    if (mapped) return -1;
#endif
    free(model);
    return 0;
}
static void proc_shared_memory(long *pss, long *private_dirty) {
    FILE *f = fopen("/proc/self/smaps_rollup", "r");
    char line[256];
    *pss = -1;
    *private_dirty = -1;
    if (!f) return;
    while (fgets(line, sizeof(line), f)) {
        long value = -1;
        char unit[3] = {0};
        if (!strncmp(line, "Pss:", 4) &&
            sscanf(line + 4, "%ld %2s", &value, unit) == 2 &&
            value >= 0 && !strcmp(unit, "kB")) *pss = value;
        if (!strncmp(line, "Private_Dirty:", 14) &&
            sscanf(line + 14, "%ld %2s", &value, unit) == 2 &&
            value >= 0 && !strcmp(unit, "kB")) *private_dirty = value;
    }
    int bad = ferror(f);
    if (fclose(f)) bad = 1;
    if (bad) { *pss = -1; *private_dirty = -1; }
}

static unsigned bucket_us(uint64_t value_us) {
    unsigned bucket = 0;
    uint64_t upper = 1;
    while (bucket + 1 < HIST_BUCKETS && value_us > upper) {
        upper *= 2;
        bucket++;
    }
    return bucket;
}
static uint64_t percentile_upper_us(const uint64_t hist[HIST_BUCKETS],
                                     uint64_t samples) {
    uint64_t wanted = samples / 20u * 19u + (samples % 20u * 19u + 19u) / 20u;
    uint64_t seen = 0;
    for (unsigned i = 0; i < HIST_BUCKETS; ++i) {
        seen += hist[i];
        if (seen >= wanted) return UINT64_C(1) << i;
    }
    return 0;
}
int main(int argc, char **argv) {
    int mapped = argc == 5 && !strcmp(argv[4], "--mmap");
    if (!(argc == 4 || mapped)) {
        fputs("usage: r1_target_resource_bench MODEL FRONTEND 16k-mono-pcm16le [--mmap]\n", stderr);
        return 2;
    }
#if !defined(FSMN_MODEL_VIEW_ALIGNMENT)
    if (mapped) {
        fputs("borrowed-view SDK header required for --mmap; no copied fallback\n", stderr);
        return 2;
    }
#endif
#if defined(FSMN_MODEL_VIEW_ALIGNMENT)
    unsigned char *weights = mapped ? map_model_exact(argv[1], FSMN_MODEL_BYTES) :
                                      read_exact(argv[1], FSMN_MODEL_BYTES);
#else
    unsigned char *weights = read_exact(argv[1], FSMN_MODEL_BYTES);
#endif
    unsigned char *frontend = read_exact(argv[2], FSMN_TABLE_BYTES);
    if (!weights || !frontend) {
        fputs("rejected missing, truncated, or altered-size model/frontend\n", stderr);
        (void)release_model(weights, mapped); free(frontend);
        return 2;
    }

    uint64_t init_start=0, init_end=0, run_start=0, run_end=0;
    uint64_t cpu_start=0, cpu_end=0;
    if (now_ns(CLOCK_MONOTONIC, &init_start)) return 2;
    fsmn_sdk *sdk = NULL;
#if defined(FSMN_MODEL_VIEW_ALIGNMENT)
    sdk = mapped ? fsmn_sdk_create_view(weights, FSMN_MODEL_BYTES, frontend,
                                        FSMN_TABLE_BYTES, 0, on_event, on_frame, NULL) :
                   fsmn_sdk_create(weights, FSMN_MODEL_BYTES, frontend,
                                   FSMN_TABLE_BYTES, 0, on_event, on_frame, NULL);
#else
    sdk = fsmn_sdk_create(weights, FSMN_MODEL_BYTES, frontend,
                          FSMN_TABLE_BYTES, 0, on_event, on_frame, NULL);
#endif
    int init_clock_failed = now_ns(CLOCK_MONOTONIC, &init_end);
    free(frontend);
    if (!mapped) { (void)release_model(weights, 0); weights = NULL; }
    if (!sdk || init_clock_failed) {
        if (sdk) (void)fsmn_sdk_destroy(sdk);
        if (release_model(weights, mapped)) return 2;
        fputs("source-bound SDK model/frontend identity/clock rejected\n", stderr);
        return 2;
    }
    FILE *pcm = fopen(argv[3], "rb");
    if (!pcm) {
        (void)fsmn_sdk_destroy(sdk);
        (void)release_model(weights, mapped);
        return 2;
    }

    unsigned char buf[CHUNK_BYTES];
    uint64_t audio_bytes=0, feed_calls=0, feed_max_us=0;
    uint64_t histogram[HIST_BUCKETS]={0};
    int failed = 0;
    if (now_ns(CLOCK_MONOTONIC, &run_start) ||
        now_ns(CLOCK_PROCESS_CPUTIME_ID, &cpu_start)) failed = 1;
    while (!failed) {
        size_t n = fread(buf, 1, sizeof(buf), pcm);
        if (!n) break;
        if (n > PCM_LIMIT_BYTES - audio_bytes) { failed = 1; break; }
        uint64_t t0=0,t1=0;
        size_t consumed=0;
        if (now_ns(CLOCK_MONOTONIC, &t0)) { failed=1; break; }
        int rc = fsmn_sdk_feed(sdk,buf,n,&consumed);
        if (now_ns(CLOCK_MONOTONIC, &t1) || t1 < t0 ||
            rc != FSMN_OK || consumed != n || callback_failed) {
            failed=1; break;
        }
        audio_bytes += n;
        feed_calls++;
        uint64_t us=(t1-t0+999u)/1000u;
        if (us > (UINT64_C(1) << 31)) { failed=1; break; }
        if (us > feed_max_us) feed_max_us=us;
        histogram[bucket_us(us)]++;
    }
    if (ferror(pcm)) failed = 1;
    if (fclose(pcm)) failed = 1;
    if (!audio_bytes || (audio_bytes & 1u)) failed = 1;
    if (!failed && fsmn_sdk_finish(sdk) != FSMN_OK) failed=1;
    if (now_ns(CLOCK_PROCESS_CPUTIME_ID, &cpu_end) ||
        now_ns(CLOCK_MONOTONIC, &run_end) || cpu_end < cpu_start ||
        run_end < run_start) failed=1;

    struct rusage usage;
    struct utsname system;
    long pss=-1, private_dirty=-1;
    proc_shared_memory(&pss, &private_dirty);
    if (getrusage(RUSAGE_SELF, &usage) != 0 || uname(&system) != 0)
        failed=1;
    if (fsmn_sdk_destroy(sdk) != FSMN_OK) failed=1;
    if (release_model(weights, mapped)) failed=1;
    if (failed) {
        fputs("FAILED: incomplete input, SDK/clock error, or invalid measurement\n",stderr);
        return 2;
    }

    double audio_s=(double)audio_bytes / 32000.0;
    double wall_s=(double)(run_end-run_start) / 1e9;
    double cpu_s=(double)(cpu_end-cpu_start) / 1e9;
    printf("{\"schema\":\"r1-strong-fsmn-offline-board-resource-v1\",");
    printf("\"model_kind\":\"source-hash-checked-by-sdk\",");
    printf("\"model_storage\":\"%s\",", mapped ? "read_only_mmap" : "sdk_owned_copy");
    printf("\"model_caller_heap_bytes\":%u,", mapped ? 0u : FSMN_MODEL_BYTES);
    printf("\"device_qualified\":false,\"live_afe\":false,\"independent_far_frr\":false,");
    printf("\"power_mw\":null,\"stack_high_water_bytes\":null,");
    printf("\"arch\":\"%s\",\"kernel\":\"%s\",",system.machine,system.release);
    printf("\"pcm_bytes\":%" PRIu64 ",\"audio_seconds\":%.9f,",audio_bytes,audio_s);
    printf("\"sdk_model_frames\":%" PRIu64 ",\"keyword_events\":%" PRIu64 ",",
           frame_count,event_count);
    printf("\"keyword1_events\":%" PRIu64 ",\"keyword2_events\":%" PRIu64 ",",
           word_events[1],word_events[2]);
    printf("\"feed_chunk_bytes\":%u,\"feed_calls\":%" PRIu64 ",",CHUNK_BYTES,feed_calls);
    printf("\"feed_p95_histogram_upper_us\":%" PRIu64 ",\"feed_max_us\":%" PRIu64 ",",
           percentile_upper_us(histogram,feed_calls),feed_max_us);
    printf("\"model_init_wall_ms\":%.6f,", (double)(init_end-init_start)/1e6);
    printf("\"offline_feed_wall_seconds\":%.9f,\"offline_feed_cpu_seconds\":%.9f,",wall_s,cpu_s);
    printf("\"offline_rtf\":%.9f,\"cpu_seconds_per_audio_second\":%.9f,",wall_s/audio_s,cpu_s/audio_s);
    printf("\"process_peak_rss_kib\":%ld,",usage.ru_maxrss);
    printf("\"proc_self_pss_kib\":");
    if (pss < 0) printf("null"); else printf("%ld", pss);
    printf(",\"proc_self_private_dirty_kib\":");
    if (private_dirty < 0) printf("null"); else printf("%ld", private_dirty);
    printf("}\n");
    if (fflush(stdout) != 0 || ferror(stdout)) {
        fputs("FAILED: unable to write complete resource receipt\n", stderr);
        return 2;
    }
    return 0;
}

