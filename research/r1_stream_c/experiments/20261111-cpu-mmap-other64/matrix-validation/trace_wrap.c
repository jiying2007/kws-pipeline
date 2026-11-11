/* Owner-local validation only. Link wrapping leaves the actual probe untouched.
 * Record the historical F/E trace while forwarding every original callback.
 * Never use this instrumented binary's timing as performance evidence. */
#define _POSIX_C_SOURCE 200809L
#include "fsmn_sdk.h"
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

extern fsmn_sdk *__real_fsmn_sdk_create(const unsigned char *,size_t,
    const unsigned char *,size_t,uint64_t,fsmn_event_fn,fsmn_frame_fn,void *);
extern fsmn_sdk *__real_fsmn_sdk_create_view(const unsigned char *,size_t,
    const unsigned char *,size_t,uint64_t,fsmn_event_fn,fsmn_frame_fn,void *);
extern int __real_fsmn_sdk_destroy(fsmn_sdk *);
static struct {
    FILE *file;
    fsmn_sdk *sdk;
    fsmn_event_fn event;
    fsmn_frame_fn frame;
    void *user;
    uint64_t frames;
    size_t bytes;
    int failed, used;
} capture;

static int emit(const char *line,int count) {
    if (count<0 || count>=256 || capture.bytes+(size_t)count>65536u ||
        fwrite(line,1,(size_t)count,capture.file)!=(size_t)count) {
        capture.failed=1;
        return 1;
    }
    capture.bytes+=(size_t)count;
    return 0;
}
static int frame(void *unused,uint64_t index,uint64_t source,uint64_t delivery,
                 const double *features,const double *logits) {
    (void)unused;
    if (!logits || index!=capture.frames || source>delivery) return 1;
    unsigned best=0;
    for (unsigned i=0;i<FSMN_LOGIT_COUNT;++i) {
        if (!isfinite(logits[i])) return 1;
        if (logits[i]>logits[best]) best=i;
    }
    char line[256];
    int n=snprintf(line,sizeof(line),"F,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%u\n",
                   index,source,delivery,best);
    if (emit(line,n)) return 1;
    ++capture.frames;
    return capture.frame ? capture.frame(capture.user,index,source,delivery,features,logits) : 0;
}
static int event(void *unused,const fsmn_event *e) {
    (void)unused;
    if (!e || e->abi!=FSMN_SDK_ABI || e->dependency_end_sample>e->delivery_end_sample)
        return 1;
    char line[256];
    int n=snprintf(line,sizeof(line),"E,%" PRIu64 ",%u,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
                   e->segment,e->keyword_id,e->model_frame,e->first_token_frame,
                   e->dependency_end_sample,e->delivery_end_sample);
    if (emit(line,n)) return 1;
    return capture.event ? capture.event(capture.user,e) : 0;
}
static fsmn_sdk *create(int view,const unsigned char *model,size_t bytes,
    const unsigned char *tables,size_t table_bytes,uint64_t origin,
    fsmn_event_fn event_fn,fsmn_frame_fn frame_fn,void *user) {
    const char *path=getenv("R1_TRACE_PATH");
    if (capture.used || !path || !*path) return NULL;
    capture.used=1;
    int fd=open(path,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0600);
    if (fd<0) return NULL;
    capture.file=fdopen(fd,"w");
    if (!capture.file) { (void)close(fd); return NULL; }
    capture.event=event_fn; capture.frame=frame_fn; capture.user=user;
    capture.sdk=view ? __real_fsmn_sdk_create_view(model,bytes,tables,table_bytes,
                    origin,event,frame,NULL) :
                    __real_fsmn_sdk_create(model,bytes,tables,table_bytes,
                    origin,event,frame,NULL);
    if (!capture.sdk) { (void)fclose(capture.file); capture.file=NULL; }
    return capture.sdk;
}
fsmn_sdk *__wrap_fsmn_sdk_create(const unsigned char *m,size_t n,
    const unsigned char *t,size_t tn,uint64_t o,fsmn_event_fn e,fsmn_frame_fn f,void *u) {
    return create(0,m,n,t,tn,o,e,f,u);
}
fsmn_sdk *__wrap_fsmn_sdk_create_view(const unsigned char *m,size_t n,
    const unsigned char *t,size_t tn,uint64_t o,fsmn_event_fn e,fsmn_frame_fn f,void *u) {
    return create(1,m,n,t,tn,o,e,f,u);
}
int __wrap_fsmn_sdk_destroy(fsmn_sdk *sdk) {
    if (!sdk || sdk!=capture.sdk || !capture.file) return FSMN_BAD_ARGUMENT;
    int rc=__real_fsmn_sdk_destroy(sdk);
    if (fflush(capture.file) || ferror(capture.file)) capture.failed=1;
    if (fclose(capture.file)) capture.failed=1;
    capture.file=NULL; capture.sdk=NULL;
    return capture.failed ? FSMN_BACKEND_ERROR : rc;
}
