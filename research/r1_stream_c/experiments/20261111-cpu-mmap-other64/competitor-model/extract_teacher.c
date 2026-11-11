/* Owner-only fixed teacher extraction. Arithmetic stays in the existing static
 * SDK. Its public per-step trace API supplies stage19; full logits never leave
 * transient memory. This executable is not a runtime performance benchmark. */
#define _POSIX_C_SOURCE 200809L
#include "fsmn_sdk.h"
#include "r1dfsmn.h"
#include <fcntl.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

_Static_assert(sizeof(double)==8 && DBL_MANT_DIG==53 && DBL_MAX_EXP==1024,
               "requires IEEE binary64");
static const unsigned exact_ids[5]={0,1027,1401,1462,2093};
static double step_trace[R1DFSMN_TRACE_FLOATS],hidden[140];
static uint64_t steps,frames,events,word_events[3];
static int pending;
static FILE *binary_file,*trace_file;
static size_t trace_bytes;
extern r1dfsmn_status __real_r1dfsmn_step(r1dfsmn_model *,const double *,size_t,
                                      double *,size_t,double *,size_t);
r1dfsmn_status __wrap_r1dfsmn_step(r1dfsmn_model *model,const double *x,size_t nx,
                                 double *y,size_t ny,double *trace,size_t nt) {
    if (pending || trace || nt || steps>=512u) return R1DFSMN_STATE;
    r1dfsmn_status rc=__real_r1dfsmn_step(model,x,nx,y,ny,step_trace,R1DFSMN_TRACE_FLOATS);
    if (rc!=R1DFSMN_OK) return rc;
    memcpy(hidden,step_trace+19u*250u,sizeof(hidden));
    memset(step_trace,0,sizeof(step_trace));
    pending=1;++steps;
    return rc;
}
static void u32(unsigned char *p,uint32_t value) {
    for (unsigned i=0;i<4;i++) p[i]=(unsigned char)(value>>(8u*i));
}
static void u64(unsigned char *p,uint64_t value) {
    for (unsigned i=0;i<8;i++) p[i]=(unsigned char)(value>>(8u*i));
}
static void f64(unsigned char *p,double value) {
    uint64_t bits;memcpy(&bits,&value,sizeof(bits));u64(p,bits);
}
static int emit_trace(const char *line,int n) {
    if (n<0 || n>=256 || trace_bytes+(size_t)n>65536u ||
        fwrite(line,1,(size_t)n,trace_file)!=(size_t)n) return 1;
    trace_bytes+=(size_t)n;return 0;
}
static int selected(unsigned id) {
    for (unsigned i=0;i<5;i++) if (id==exact_ids[i]) return 1;
    return 0;
}
static int on_frame(void *ctx,uint64_t index,uint64_t source,uint64_t delivery,
                    const double *features,const double *logits) {
    (void)ctx;(void)features;
    if (!pending || !logits || index!=frames || steps!=frames+1u || source>delivery)
        return 1;
    unsigned best=0,other=1;
    for (unsigned i=0;i<FSMN_LOGIT_COUNT;i++) {
        if (!isfinite(logits[i])) return 1;
        if (logits[i]>logits[best]) best=i;
        if (!selected(i) && logits[i]>logits[other]) other=i;
    }
    unsigned char row[1200]={0};
    u64(row,index);u64(row+8,source);u64(row+16,delivery);
    u32(row+24,best);u32(row+28,other);
    for (unsigned i=0;i<140;i++) {
        if (!isfinite(hidden[i])) return 1;
        f64(row+32u+8u*i,hidden[i]);
    }
    for (unsigned i=0;i<5;i++) f64(row+1152u+8u*i,logits[exact_ids[i]]);
    f64(row+1192u,logits[other]);
    if (fwrite(row,1,sizeof(row),binary_file)!=sizeof(row)) return 1;
    char line[256];int n=snprintf(line,sizeof(line),"F,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%u\n",
                                 index,source,delivery,best);
    if (emit_trace(line,n)) return 1;
    pending=0;++frames;return 0;
}
static int on_event(void *ctx,const fsmn_event *e) {
    (void)ctx;
    if (!e || e->abi!=FSMN_SDK_ABI || e->keyword_id<1 || e->keyword_id>2 ||
        e->model_frame>=frames || e->dependency_end_sample>e->delivery_end_sample) return 1;
    char line[256];int n=snprintf(line,sizeof(line),"E,%" PRIu64 ",%u,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
        e->segment,e->keyword_id,e->model_frame,e->first_token_frame,
        e->dependency_end_sample,e->delivery_end_sample);
    if (emit_trace(line,n)) return 1;
    ++events;++word_events[e->keyword_id];return 0;
}
static FILE *regular_input(const char *name,size_t exact,int pcm) {
    int fd=open(name,O_RDONLY|O_CLOEXEC|O_NOFOLLOW|O_NONBLOCK);
    if (fd<0) return NULL;
    struct stat st;
    int good=fstat(fd,&st)==0 && S_ISREG(st.st_mode) && st.st_size>0;
    if (good) good=pcm ? (st.st_size<=266240 && !(st.st_size&1)) :
                          (uintmax_t)st.st_size==(uintmax_t)exact;
    if (!good) { (void)close(fd);return NULL; }
    FILE *f=fdopen(fd,"rb");if (!f) (void)close(fd);return f;
}
static unsigned char *read_exact(const char *name,size_t size) {
    FILE *f=regular_input(name,size,0);if (!f) return NULL;
    unsigned char *data=(unsigned char *)malloc(size);
    if (!data) { (void)fclose(f);return NULL; }
    int bad=fread(data,1,size,f)!=size;
    if (fgetc(f)!=EOF || ferror(f)) bad=1;
    if (fclose(f)) bad=1;
    if (bad) { free(data);return NULL; }return data;
}
static FILE *exclusive_output(const char *name) {
    int fd=open(name,O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW,0600);
    if (fd<0) return NULL;
    FILE *f=fdopen(fd,"wb");if (!f) (void)close(fd);return f;
}
static int finish_output(FILE *f) {
    int bad=fflush(f)!=0 || ferror(f);
    if (fsync(fileno(f))) bad=1;
    if (fclose(f)) bad=1;
    return bad;
}
int main(int argc,char **argv) {
    if (argc!=6) {
        fputs("usage: extract_teacher MODEL FRONTEND PCM HIDDEN.r1h TRACE.txt\n",stderr);return 2;
    }
    unsigned char *model=read_exact(argv[1],FSMN_MODEL_BYTES);
    unsigned char *tables=read_exact(argv[2],FSMN_TABLE_BYTES);
    if (!model || !tables) { free(model);free(tables);return 2; }
    fsmn_sdk *sdk=fsmn_sdk_create_view(model,FSMN_MODEL_BYTES,tables,FSMN_TABLE_BYTES,
                                     0,on_event,on_frame,NULL);
    free(tables);
    if (!sdk) { free(model);return 2; }
    FILE *pcm=regular_input(argv[3],0,1);
    binary_file=exclusive_output(argv[4]);trace_file=exclusive_output(argv[5]);
    int failed=!pcm || !binary_file || !trace_file;
    if (!failed) {
        unsigned char header[64]={0};memcpy(header,"R1HD140",7);
        u32(header+8,1);u32(header+12,64);u32(header+16,1200);u32(header+20,140);
        u32(header+24,5);u32(header+28,2601);
        for (unsigned i=0;i<5;i++)u32(header+32+4u*i,exact_ids[i]);
        u32(header+52,1);
        if (fwrite(header,1,sizeof(header),binary_file)!=sizeof(header))failed=1;
    }
    uint64_t total=0;unsigned char pcm_bytes[640];
    while (!failed) {
        size_t n=fread(pcm_bytes,1,sizeof(pcm_bytes),pcm);if (!n)break;
        if (n>266240u-total) { failed=1;break; }
        size_t used=0;int rc=fsmn_sdk_feed(sdk,pcm_bytes,n,&used);total+=used;
        if (rc!=FSMN_OK || used!=n || pending)failed=1;
    }
    if (pcm) { if (ferror(pcm))failed=1;if (fclose(pcm))failed=1; }
    if (!failed && (!total || (total&1u) || fsmn_sdk_finish(sdk)!=FSMN_OK))failed=1;
    if (pending || steps!=frames || fsmn_sdk_frame_count(sdk)!=frames ||
        fsmn_sdk_event_count(sdk)!=events || fsmn_sdk_consumed_bytes(sdk)!=total)failed=1;
    if (fsmn_sdk_destroy(sdk)!=FSMN_OK)failed=1;
    free(model);
    if (binary_file && finish_output(binary_file))failed=1;
    if (trace_file && finish_output(trace_file))failed=1;
    if (failed) { fputs("FAILED: incomplete teacher extraction\n",stderr);return 2; }
    printf("{\"schema\":\"r1-hidden140-teacher-clip-v1\",\"pcm_bytes\":%" PRIu64 ",\"frames\":%" PRIu64
           ",\"events\":%" PRIu64 ",\"keyword1_events\":%" PRIu64 ",\"keyword2_events\":%" PRIu64
           ",\"teacher_steps\":%" PRIu64 ",\"full_logits_persisted\":false,\"record_bytes\":1200}\n",
           total,frames,events,word_events[1],word_events[2],steps);
    if (fflush(stdout) || ferror(stdout)) return 2;
    return 0;
}
