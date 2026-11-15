#define _GNU_SOURCE
#include "en63_stream.h"
#include "en63_generated.h"
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#if !defined(EN63_POOL_TIME_MAJOR) || EN63_POOL_TIME_MAJOR != 0
#error "LibriSpeech screening requires the unchanged default pooling path"
#endif

typedef struct { FILE *trace; uint64_t frames, events; int error; } observation;

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
    unsigned long long x;
    if (!text[0] || strspn(text,"0123456789")!=strlen(text)) return 0;
    errno=0; x=strtoull(text,&end,10);
    if (errno || *end || x>maximum) return 0;
    *out=(uint64_t)x; return 1;
}

static int pin(unsigned cpu)
{
    cpu_set_t desired,actual;
    CPU_ZERO(&desired); CPU_SET(cpu,&desired);
    return sched_setaffinity(0,sizeof(desired),&desired)==0 &&
           sched_getaffinity(0,sizeof(actual),&actual)==0 &&
           CPU_COUNT(&actual)==1 && CPU_ISSET(cpu,&actual);
}

static int now(clockid_t id, uint64_t *out)
{
    struct timespec t;
    if (clock_gettime(id,&t)!=0 || t.tv_sec<0 || t.tv_nsec<0 || t.tv_nsec>=1000000000L) return 0;
    *out=(uint64_t)t.tv_sec*UINT64_C(1000000000)+(uint64_t)t.tv_nsec; return 1;
}

int main(int argc, char **argv)
{
    uint64_t cpu=0,samples=0,done=0,calls=0,cpu_start=0,cpu_end=0,wall_start=0,wall_end=0;
    FILE *input=NULL;
    en63_state *state=NULL;
    const en63_model *model;
    observation out={NULL,0,0,0};
    int result=2;
    if (argc==7 && strcmp(argv[1],"--supervise")==0) {
        pid_t child;
        int status;
        struct rusage usage;
        if (!number(argv[5],CPU_SETSIZE-1,&cpu) || !pin((unsigned)cpu)) return 2;
        child=fork();
        if (child<0) return 2;
        if (child==0) {
            char *args[]={argv[0],argv[2],argv[3],argv[4],argv[5],argv[6],NULL};
            execv(argv[0],args); _exit(127);
        }
        while (wait4(child,&status,0,&usage)<0) if (errno!=EINTR) return 2;
        printf("{\"schema\":\"english63-librispeech-child-lifetime-v1\",\"child_exit_code\":%d,"
               "\"child_signaled\":%s,\"process_lifetime_peak_rss_kib\":%ld}\n",
               WIFEXITED(status)?WEXITSTATUS(status):-1,WIFSIGNALED(status)?"true":"false",usage.ru_maxrss);
        if (fflush(stdout)!=0 || ferror(stdout)) return 2;
        return WIFEXITED(status)?WEXITSTATUS(status):2;
    }
    if (argc!=6 || (strcmp(argv[1],"gru") && strcmp(argv[1],"dsc_tcn")) ||
        !number(argv[4],CPU_SETSIZE-1,&cpu) || !pin((unsigned)cpu) ||
        !number(argv[5],58646164,&samples) || samples<400) {
        fprintf(stderr,"usage: en63_librispeech KIND PCM TRACE CPU SAMPLES\n"); return 2;
    }
    model=strcmp(argv[1],"gru")==0?&en63_gru_model:&en63_dsc_tcn_model;
    input=fopen(argv[2],"rb");
    if (!input || fseek(input,0,SEEK_END)!=0 || ftell(input)!=(long)(samples*2) ||
        fseek(input,0,SEEK_SET)!=0) goto cleanup;
    state=calloc(1,sizeof(*state));
    if (!state || en63_init(state,model)!=EN63_OK) goto cleanup;
    out.trace=fopen(argv[3],"wbx");
    if (!out.trace) goto cleanup;
    {
        unsigned char header[16]={'E','N','6','3','L','S','C','1',0,0,0,0,0,0,0,0};
        put32(header+8,1); put32(header+12,32);
        if (fwrite(header,1,sizeof(header),out.trace)!=sizeof(header)) goto cleanup;
    }
    if (!now(CLOCK_PROCESS_CPUTIME_ID,&cpu_start) || !now(CLOCK_MONOTONIC,&wall_start)) goto cleanup;
    while (done<samples) {
        unsigned char bytes[320];
        int16_t pcm[160];
        size_t i,take=samples-done<160?(size_t)(samples-done):160;
        if (fread(bytes,2,take,input)!=take) goto cleanup;
        for (i=0; i<take; ++i) {
            unsigned value=(unsigned)bytes[2*i]|((unsigned)bytes[2*i+1]<<8);
            pcm[i]=(int16_t)(value<32768u?(int)value:(int)value-65536);
        }
        if (en63_feed(state,pcm,take,collect,&out)!=EN63_OK || out.error) goto cleanup;
        done+=take; ++calls;
    }
    if (fgetc(input)!=EOF || ferror(input)) goto cleanup;
    {
        uint64_t frames=out.frames,events=out.events;
        if (en63_finish(state)!=EN63_OK || out.frames!=frames || out.events!=events || out.error ||
            frames!=1+(samples-400)/160 || state->samples!=samples || state->frames!=frames ||
            state->segment_frames!=frames || state->events!=events || state->discontinuities!=0 ||
            calls!=(samples+159)/160) goto cleanup;
    }
    if (!now(CLOCK_MONOTONIC,&wall_end) || !now(CLOCK_PROCESS_CPUTIME_ID,&cpu_end) ||
        wall_end<=wall_start || cpu_end<=cpu_start) goto cleanup;
    if (fclose(out.trace)!=0) { out.trace=NULL; goto cleanup; }
    out.trace=NULL;
    if (fclose(input)!=0) { input=NULL; goto cleanup; }
    input=NULL;
    printf("{\"schema\":\"english63-librispeech-native-pass-v1\",\"kind\":\"%s\","
           "\"pcm_samples\":%" PRIu64 ",\"frames\":%" PRIu64 ",\"feed_calls\":%" PRIu64 ","
           "\"final_feed_samples\":%" PRIu64 ",\"proposal_count\":%" PRIu64 ",\"state_bytes\":%zu,"
           "\"weight_parameter_bytes\":%zu,\"frontend_table_bytes\":%u,\"pool_time_major\":0,"
           "\"affinity_cpu\":%" PRIu64 ",\"affinity_single_cpu\":true,\"initializations\":1,"
           "\"reset_calls\":0,\"discontinuity_calls\":0,\"finish_calls\":1,"
           "\"eof_padding_samples\":0,\"eof_events\":0,\"cpu_ns\":%" PRIu64 ",\"wall_ns\":%" PRIu64 "}\n",
           argv[1],samples,out.frames,calls,(samples-1)%160+1,out.events,en63_state_bytes(),
           model->parameter_count*sizeof(float),EN63_FRONTEND_TABLE_BYTES,cpu,cpu_end-cpu_start,wall_end-wall_start);
    if (fflush(stdout)!=0 || ferror(stdout)) goto cleanup;
    result=0;
cleanup:
    if (input) fclose(input);
    if (out.trace) fclose(out.trace);
    free(state);
    if (result) fprintf(stderr,"EN63_LIBRISPEECH_NATIVE_FAILED errno=%d\n",errno);
    return result;
}
