#define _GNU_SOURCE
#include "en63_stream.h"
#include "en63_generated.h"
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <sched.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#if !defined(EN63_POOL_TIME_MAJOR) || EN63_POOL_TIME_MAJOR != 0
#error "Background comparison freezes the default pooling path"
#endif

typedef struct { FILE *trace; uint64_t frames, events; int error; } observation;
typedef struct { uint64_t samples, frames, calls, events; } segment;

static void put32(unsigned char *p, uint32_t x)
{ unsigned i; for (i=0; i<4; ++i) p[i]=(unsigned char)(x>>(8*i)); }
static void put64(unsigned char *p, uint64_t x)
{ put32(p,(uint32_t)x); put32(p+4,(uint32_t)(x>>32)); }
static void putfloat(unsigned char *p, float x)
{ uint32_t bits; memcpy(&bits,&x,sizeof(bits)); put32(p,bits); }

static void collect(void *user, uint64_t frame, uint64_t sample,
                    const float features[32], const float logits[3],
                    const float probabilities[3], uint32_t proposal)
{
    observation *out=user;
    unsigned char row[32];
    unsigned i;
    (void)features; (void)logits;
    ++out->frames;
    if (frame!=out->frames || sample!=400+(frame-1)*160 || proposal>2) out->error=1;
    if (proposal) ++out->events;
    put64(row,frame); put64(row+8,sample);
    for (i=0; i<3; ++i) {
        if (!isfinite(probabilities[i]) || probabilities[i]<0 || probabilities[i]>1) out->error=1;
        putfloat(row+16+4*i,probabilities[i]);
    }
    put32(row+28,proposal);
    if (fwrite(row,1,sizeof(row),out->trace)!=sizeof(row)) out->error=1;
}

static int number(const char *text, uint64_t maximum, uint64_t *out)
{
    char *end;
    unsigned long long value;
    if (!text[0] || strspn(text,"0123456789")!=strlen(text)) return 0;
    errno=0; value=strtoull(text,&end,10);
    if (errno || *end || value>maximum) return 0;
    *out=(uint64_t)value; return 1;
}

static int pin(unsigned cpu)
{
    cpu_set_t desired,actual;
    CPU_ZERO(&desired); CPU_SET(cpu,&desired);
    return sched_setaffinity(0,sizeof(desired),&desired)==0 &&
           sched_getaffinity(0,sizeof(actual),&actual)==0 && CPU_COUNT(&actual)==1 && CPU_ISSET(cpu,&actual);
}

static int now(clockid_t id, uint64_t *out)
{
    struct timespec t;
    if (clock_gettime(id,&t)!=0 || t.tv_sec<0 || t.tv_nsec<0 || t.tv_nsec>=1000000000L) return 0;
    *out=(uint64_t)t.tv_sec*UINT64_C(1000000000)+(uint64_t)t.tv_nsec; return 1;
}

int main(int argc, char **argv)
{
    uint64_t cpu=0,cpu_start=0,cpu_end=0,wall_start=0,wall_end=0;
    uint64_t parent=0;
    uint64_t total_samples=0,total_frames=0,total_calls=0,total_events=0;
    FILE *manifest=NULL,*input=NULL,*trace=NULL;
    en63_state *state=NULL;
    const en63_model *model;
    segment segments[13];
    unsigned index;
    int result=2;
    if ((argc!=7 && argc!=6) || !number(argv[argc-1],2147483647,&parent) || parent<2 ||
        (uint64_t)getppid()!=parent || prctl(PR_SET_PDEATHSIG,SIGKILL)!=0 ||
        (uint64_t)getppid()!=parent) return 2;
    if (argc==7 && strcmp(argv[1],"--supervise")==0) {
        pid_t child;
        int status;
        struct rusage usage;
        char parent_text[32];
        if (!number(argv[5],CPU_SETSIZE-1,&cpu) || !pin((unsigned)cpu)) return 2;
        if (snprintf(parent_text,sizeof(parent_text),"%ld",(long)getpid())<1) return 2;
        child=fork();
        if (child<0) return 2;
        if (child==0) {
            char *args[]={argv[0],argv[2],argv[3],argv[4],argv[5],parent_text,NULL};
            if (prctl(PR_SET_PDEATHSIG,SIGKILL)!=0 || getppid()!=(pid_t)strtol(parent_text,NULL,10)) _exit(127);
            execv(argv[0],args); _exit(127);
        }
        while (wait4(child,&status,0,&usage)<0) if (errno!=EINTR) return 2;
        printf("{\"schema\":\"english63-background-ab-child-lifetime-v1\",\"child_exit_code\":%d,"
               "\"child_signaled\":%s,\"process_lifetime_peak_rss_kib\":%ld}\n",
               WIFEXITED(status)?WEXITSTATUS(status):-1,WIFSIGNALED(status)?"true":"false",usage.ru_maxrss);
        if (fflush(stdout)!=0 || ferror(stdout)) return 2;
        return WIFEXITED(status)?WEXITSTATUS(status):2;
    }
    if (argc!=6 || (strcmp(argv[1],"gru") && strcmp(argv[1],"dsc_tcn")) ||
        !number(argv[4],CPU_SETSIZE-1,&cpu) || !pin((unsigned)cpu)) {
        fprintf(stderr,"usage: ab_replay KIND MANIFEST TRACE CPU PARENT_PID\n"); return 2;
    }
    model=strcmp(argv[1],"gru")==0?&en63_gru_model:&en63_dsc_tcn_model;
    manifest=fopen(argv[2],"rb");
    trace=fopen(argv[3],"wbx");
    state=calloc(1,sizeof(*state));
    if (!manifest || !trace || !state) goto cleanup;
    {
        unsigned char header[16]={'E','N','6','3','A','B','T','1',0,0,0,0,0,0,0,0};
        put32(header+8,1); put32(header+12,13);
        if (fwrite(header,1,16,trace)!=16) goto cleanup;
    }
    if (!now(CLOCK_PROCESS_CPUTIME_ID,&cpu_start) || !now(CLOCK_MONOTONIC,&wall_start)) goto cleanup;
    for (index=0; index<13; ++index) {
        char path[4096];
        unsigned char header[24];
        uint64_t done=0,calls=0,samples,frames;
        long bytes;
        size_t length;
        observation out={trace,0,0,0};
        if (!fgets(path,sizeof(path),manifest)) goto cleanup;
        length=strlen(path);
        if (length<2 || path[length-1]!='\n' || path[0]!='/') goto cleanup;
        path[length-1]='\0';
        input=fopen(path,"rb");
        if (!input || fseek(input,0,SEEK_END)!=0 || (bytes=ftell(input))<800 ||
            bytes%2 || bytes>117292328L || fseek(input,0,SEEK_SET)!=0) goto cleanup;
        samples=(uint64_t)bytes/2; frames=1+(samples-400)/160;
        put32(header,index); put32(header+4,32); put64(header+8,samples); put64(header+16,frames);
        if (fwrite(header,1,24,trace)!=24 || en63_init(state,model)!=EN63_OK) goto cleanup;
        while (done<samples) {
            unsigned char raw[320];
            int16_t pcm[160];
            size_t i,take=samples-done<160?(size_t)(samples-done):160;
            if (fread(raw,2,take,input)!=take) goto cleanup;
            for (i=0; i<take; ++i) {
                unsigned value=(unsigned)raw[2*i]|((unsigned)raw[2*i+1]<<8);
                pcm[i]=(int16_t)(value<32768u?(int)value:(int)value-65536);
            }
            if (en63_feed(state,pcm,take,collect,&out)!=EN63_OK || out.error) goto cleanup;
            done+=take; ++calls;
        }
        if (fgetc(input)!=EOF || ferror(input) || en63_finish(state)!=EN63_OK || out.error ||
            out.frames!=frames || state->samples!=samples || state->frames!=frames ||
            state->segment_frames!=frames || state->events!=out.events || state->discontinuities!=0 ||
            calls!=(samples+159)/160) goto cleanup;
        if (fclose(input)!=0) { input=NULL; goto cleanup; }
        input=NULL;
        segments[index]=(segment){samples,frames,calls,out.events};
        total_samples+=samples; total_frames+=frames; total_calls+=calls; total_events+=out.events;
    }
    if (fgetc(manifest)!=EOF || ferror(manifest) ||
        !now(CLOCK_MONOTONIC,&wall_end) || !now(CLOCK_PROCESS_CPUTIME_ID,&cpu_end) ||
        wall_end<=wall_start || cpu_end<=cpu_start) goto cleanup;
    if (fclose(trace)!=0) { trace=NULL; goto cleanup; }
    trace=NULL;
    if (fclose(manifest)!=0) { manifest=NULL; goto cleanup; }
    manifest=NULL;
    printf("{\"schema\":\"english63-background-ab-native-v1\",\"kind\":\"%s\","
           "\"pcm_samples\":%" PRIu64 ",\"frames\":%" PRIu64 ",\"feed_calls\":%" PRIu64 ","
           "\"proposal_count\":%" PRIu64 ",\"state_bytes\":%zu,\"weight_parameter_bytes\":%zu,"
           "\"frontend_table_bytes\":%u,\"pool_time_major\":0,\"affinity_cpu\":%" PRIu64 ","
           "\"affinity_single_cpu\":true,\"initializations\":13,\"finish_calls\":13,\"reset_calls\":0,"
           "\"discontinuity_calls\":0,\"eof_padding_samples\":0,\"eof_events\":0,"
           "\"cpu_ns\":%" PRIu64 ",\"wall_ns\":%" PRIu64 ",\"segments\":[",
           argv[1],total_samples,total_frames,total_calls,total_events,en63_state_bytes(),
           model->parameter_count*sizeof(float),EN63_FRONTEND_TABLE_BYTES,cpu,cpu_end-cpu_start,wall_end-wall_start);
    for (index=0; index<13; ++index) {
        segment x=segments[index];
        printf("%s{\"segment_index\":%u,\"samples\":%" PRIu64 ",\"frames\":%" PRIu64
               ",\"feed_calls\":%" PRIu64 ",\"final_feed_samples\":%" PRIu64 ",\"proposal_count\":%" PRIu64 "}",
               index?",":"",index,x.samples,x.frames,x.calls,(x.samples-1)%160+1,x.events);
    }
    printf("]}\n");
    if (fflush(stdout)!=0 || ferror(stdout)) goto cleanup;
    result=0;
cleanup:
    if (input) fclose(input);
    if (manifest) fclose(manifest);
    if (trace) fclose(trace);
    free(state);
    if (result) fprintf(stderr,"EN63_BACKGROUND_NATIVE_FAILED errno=%d\n",errno);
    return result;
}
