/* SPDX-License-Identifier: Apache-2.0
 * Owner-side R1 resource benchmark for a real Linux ARM DUT.
 * The file-based feed is NOT live AFE capture, device qualification, FAR or FRR.
 * This process never publishes original PCM or model weights.
 */
#define _POSIX_C_SOURCE 200809L
#include "active113_sdk.h"
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

/* Capture-only observer: exactly one real step; no repeated neural forward. */
#if defined(ACTIVE113_CAPTURE)
#include "r1active113.h"
#include <float.h>
_Static_assert(sizeof(double)==8 && DBL_MANT_DIG==53,"binary64 trace required");
static FILE *capture_fe,*capture_scores;
static double capture_trace[R1ACTIVE113_TRACE_FLOATS],capture_hidden[140];
static int capture_hidden_ready;
static void put32(unsigned char *p,uint32_t x){for(unsigned i=0;i<4;i++)p[i]=(unsigned char)(x>>(8*i));}
static void put64(unsigned char *p,uint64_t x){for(unsigned i=0;i<8;i++)p[i]=(unsigned char)(x>>(8*i));}
extern r1active113_status __real_r1active113_step(r1active113_model *,const double *,size_t,double *,size_t,double *,size_t);
r1active113_status __wrap_r1active113_step(r1active113_model *m,const double *x,size_t nx,
        double *y,size_t ny,double *trace,size_t nt){
    if(capture_hidden_ready||trace||nt)return R1ACTIVE113_ARGUMENT;
    r1active113_status status=__real_r1active113_step(m,x,nx,y,ny,capture_trace,R1ACTIVE113_TRACE_FLOATS);
    if(status==R1ACTIVE113_OK){memcpy(capture_hidden,capture_trace+19u*250u,sizeof(capture_hidden));capture_hidden_ready=1;}
    memset(capture_trace,0,sizeof(capture_trace));
    return status;
}
static FILE *exclusive_output(const char *path){
    if(!path||!*path)return NULL;
    int fd=open(path,O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);
    if(fd<0)return NULL;
    FILE *file=fdopen(fd,"wb");if(!file)(void)close(fd);return file;
}
static int open_capture(void){
    capture_fe=exclusive_output(getenv("ACTIVE113_TRACE_PATH"));
    capture_scores=exclusive_output(getenv("ACTIVE113_SCORE_PATH"));
    if(!capture_fe||!capture_scores)return -1;
    unsigned char header[64]={0};memcpy(header,"R1A113H\0",8);
    static const uint32_t fields[12]={1,64,1200,140,6,2601,0,1027,1401,1462,2093,113};
    for(unsigned i=0;i<12;i++)put32(header+8+4*i,fields[i]);
    return fwrite(header,1,sizeof(header),capture_scores)==sizeof(header)?0:-1;
}
static int capture_frame(uint64_t frame,uint64_t source,uint64_t delivery,const double *scores,uint32_t other){
    uint32_t winner=UINT32_MAX;
    if(!capture_hidden_ready||!capture_fe||!capture_scores||
       active113_sdk_pick_original_token(scores,ACTIVE113_SCORE_COUNT,other,&winner)!=ACTIVE113_OK)return -1;
    unsigned char row[1200]={0};put64(row,frame);put64(row+8,source);put64(row+16,delivery);
    put32(row+24,winner);put32(row+28,other);memcpy(row+32,capture_hidden,sizeof(capture_hidden));
    memcpy(row+1152,scores,6u*sizeof(double));
    capture_hidden_ready=0;memset(capture_hidden,0,sizeof(capture_hidden));
    if(fwrite(row,1,sizeof(row),capture_scores)!=sizeof(row)||
       fprintf(capture_fe,"F,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%u\n",frame,source,delivery,winner)<0)return -1;
    return 0;
}
static int capture_event(const active113_event *e){
    return fprintf(capture_fe,"E,%" PRIu64 ",%u,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
        e->segment,e->keyword_id,e->model_frame,e->first_token_frame,e->dependency_end_sample,e->delivery_end_sample)<0?-1:0;
}
static int close_capture(void){
    int bad=capture_hidden_ready;
    FILE *files[2]={capture_fe,capture_scores};
    for(unsigned i=0;i<2;i++){
        if(!files[i]){bad=1;continue;}
        if(fflush(files[i])||ferror(files[i])||fsync(fileno(files[i])))bad=1;
        if(fclose(files[i]))bad=1;
    }
    capture_fe=NULL;capture_scores=NULL;return bad?-1:0;
}
#endif


static int on_frame(void *unused, uint64_t frame, uint64_t source, uint64_t delivery,
                    const double *features, const double *logits, uint32_t other_original_id) {
    (void)unused;
    (void)features;
    (void)other_original_id;
    if (!logits || frame != frame_count || source > delivery ||
        (frame_count && (source < last_source || delivery < last_delivery))) {
        callback_failed = 1;
        return 1;
    }
#if defined(ACTIVE113_CAPTURE)
    if(capture_frame(frame,source,delivery,logits,other_original_id)){callback_failed=1;return 1;}
#endif
    last_source = source;
    last_delivery = delivery;
    frame_count++;
    return 0;
}
static int on_event(void *unused, const active113_event *e) {
    (void)unused;
    if (!e || e->abi != ACTIVE113_SDK_ABI || e->keyword_id < 1 ||
        e->keyword_id > 2 || e->dependency_end_sample > e->delivery_end_sample ||
        e->model_frame >= frame_count) {
        callback_failed = 1;
        return 1;
    }
#if defined(ACTIVE113_CAPTURE)
    if(capture_event(e)){callback_failed=1;return 1;}
#endif
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
#if defined(ACTIVE113_MODEL_VIEW_ALIGNMENT)
#if !defined(O_NOFOLLOW) || !defined(O_CLOEXEC)
#error "Linux secure read-only model mmap requires O_NOFOLLOW and O_CLOEXEC"
#endif
static unsigned char *map_model_exact(const char *name, size_t bytes) {
    /* A FIFO must reach the regular-file check without waiting for a writer. */
    int fd = open(name, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    if (fd < 0) return NULL;
    struct stat st;
    int valid = fstat(fd, &st) == 0 && S_ISREG(st.st_mode) &&
                st.st_size >= 0 && (uintmax_t)st.st_size == (uintmax_t)bytes;
    if (!valid) { (void)close(fd); return NULL; }
    void *view = mmap(NULL, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    int bad_close = close(fd) != 0;
    if (view == MAP_FAILED) return NULL;
    if (bad_close || (uintptr_t)view % ACTIVE113_MODEL_VIEW_ALIGNMENT) {
        (void)munmap(view, bytes);
        return NULL;
    }
    return (unsigned char *)view;
}
#endif
static int release_model(unsigned char *model, int mapped) {
    if (!model) return 0;
#if defined(ACTIVE113_MODEL_VIEW_ALIGNMENT)
    if (mapped) return munmap(model, ACTIVE113_MODEL_BYTES);
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
        fputs("usage: active113_resource_probe MODEL FRONTEND 16k-mono-pcm16le [--mmap]\n", stderr);
        return 2;
    }
#if !defined(ACTIVE113_MODEL_VIEW_ALIGNMENT)
    if (mapped) {
        fputs("borrowed-view SDK header required for --mmap; no copied fallback\n", stderr);
        return 2;
    }
#endif
#if defined(ACTIVE113_MODEL_VIEW_ALIGNMENT)
    unsigned char *weights = mapped ? map_model_exact(argv[1], ACTIVE113_MODEL_BYTES) :
                                      read_exact(argv[1], ACTIVE113_MODEL_BYTES);
#else
    unsigned char *weights = read_exact(argv[1], ACTIVE113_MODEL_BYTES);
#endif
    unsigned char *frontend = read_exact(argv[2], ACTIVE113_TABLE_BYTES);
    if (!weights || !frontend) {
        fputs("rejected missing, truncated, or altered-size model/frontend\n", stderr);
        (void)release_model(weights, mapped); free(frontend);
        return 2;
    }

    uint64_t init_start=0, init_end=0, run_start=0, run_end=0;
    uint64_t cpu_start=0, cpu_end=0;
    if (now_ns(CLOCK_MONOTONIC, &init_start)) return 2;
    active113_sdk *sdk = NULL;
#if defined(ACTIVE113_MODEL_VIEW_ALIGNMENT)
    sdk = mapped ? active113_sdk_create_view(weights, ACTIVE113_MODEL_BYTES, frontend,
                                        ACTIVE113_TABLE_BYTES, 0, on_event, on_frame, NULL) :
                   active113_sdk_create(weights, ACTIVE113_MODEL_BYTES, frontend,
                                   ACTIVE113_TABLE_BYTES, 0, on_event, on_frame, NULL);
#else
    sdk = active113_sdk_create(weights, ACTIVE113_MODEL_BYTES, frontend,
                          ACTIVE113_TABLE_BYTES, 0, on_event, on_frame, NULL);
#endif
    int init_clock_failed = now_ns(CLOCK_MONOTONIC, &init_end);
    free(frontend);
    if (!mapped) { (void)release_model(weights, 0); weights = NULL; }
    if (!sdk || init_clock_failed) {
        if (sdk) (void)active113_sdk_destroy(sdk);
        if (release_model(weights, mapped)) return 2;
        fputs("source-bound SDK model/frontend identity/clock rejected\n", stderr);
        return 2;
    }
#if defined(ACTIVE113_CAPTURE)
    if(open_capture())return 2;
#endif
    FILE *pcm = fopen(argv[3], "rb");
    if (!pcm) {
        (void)active113_sdk_destroy(sdk);
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
        int rc = active113_sdk_feed(sdk,buf,n,&consumed);
        if (now_ns(CLOCK_MONOTONIC, &t1) || t1 < t0 ||
            rc != ACTIVE113_OK || consumed != n || callback_failed) {
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
    if (!failed && active113_sdk_finish(sdk) != ACTIVE113_OK) failed=1;
    if (now_ns(CLOCK_PROCESS_CPUTIME_ID, &cpu_end) ||
        now_ns(CLOCK_MONOTONIC, &run_end) || cpu_end < cpu_start ||
        run_end < run_start) failed=1;

#if defined(ACTIVE113_CAPTURE)
    if(close_capture())failed=1;
#endif
    struct rusage usage;
    struct utsname system;
    long pss=-1, private_dirty=-1;
    proc_shared_memory(&pss, &private_dirty);
    if (getrusage(RUSAGE_SELF, &usage) != 0 || uname(&system) != 0)
        failed=1;
    if (active113_sdk_destroy(sdk) != ACTIVE113_OK) failed=1;
    if (release_model(weights, mapped)) failed=1;
    if (failed) {
        fputs("FAILED: incomplete input, SDK/clock error, or invalid measurement\n",stderr);
        return 2;
    }

    double audio_s=(double)audio_bytes / 32000.0;
    double wall_s=(double)(run_end-run_start) / 1e9;
    double cpu_s=(double)(cpu_end-cpu_start) / 1e9;
    printf("{\"schema\":\"r1-active113-offline-resource-v1\",");
    printf("\"model_sha256\":\"%s\",\"frontend_sha256\":\"%s\",",active113_sdk_model_sha256(),active113_sdk_table_sha256());
    printf("\"selection_sha256\":\"%s\",\"policy_id\":\"%s\",",active113_sdk_selection_sha256(),active113_sdk_policy_id());
    printf("\"sdk_state_bytes\":%zu,\"model_weight_bytes\":%zu,",active113_sdk_state_bytes(),active113_sdk_weight_bytes());
#if defined(ACTIVE113_CAPTURE)
    printf("\"capture_instrumented\":true,");
#else
    printf("\"capture_instrumented\":false,");
#endif
    printf("\"model_kind\":\"source-hash-checked-by-sdk\",");
    printf("\"model_storage\":\"%s\",", mapped ? "read_only_mmap" : "sdk_owned_copy");
    printf("\"model_caller_heap_bytes\":%u,", mapped ? 0u : ACTIVE113_MODEL_BYTES);
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

