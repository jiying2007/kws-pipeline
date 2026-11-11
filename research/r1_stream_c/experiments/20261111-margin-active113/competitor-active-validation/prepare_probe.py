#!/usr/bin/env python3
"""Derive one independent-ABI resource probe, without compiling or executing it."""
from pathlib import Path
import hashlib
import os

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'kws-source/research/r1_stream_c/quant_head8/target_resource_bench.c'
EXPECTED='b319f483c65d1a5b8776762cc344737c613f5fb090b8b388df2650589c70f912'
data=BASE.read_bytes()
if hashlib.sha256(data).hexdigest()!=EXPECTED:raise ValueError('original resource probe changed')
source=data.decode().replace('fsmn_sdk','active113_sdk').replace('fsmn_event','active113_event').replace('FSMN_','ACTIVE113_')
source=source.replace('r1_target_resource_bench','active113_resource_probe')

CAPTURE=r'''
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
'''

def replace(old,new):
    global source
    if source.count(old)!=1:raise ValueError('unexpected resource probe anchor: '+old[:70])
    source=source.replace(old,new)

replace('static int callback_failed;','static int callback_failed;\n'+CAPTURE)
replace('const double *features, const double *logits) {','const double *features, const double *logits, uint32_t other_original_id) {')
replace('    (void)features;','    (void)features;\n    (void)other_original_id;')
replace('    last_source = source;','''#if defined(ACTIVE113_CAPTURE)
    if(capture_frame(frame,source,delivery,logits,other_original_id)){callback_failed=1;return 1;}
#endif
    last_source = source;''')
replace('    event_count++;','''#if defined(ACTIVE113_CAPTURE)
    if(capture_event(e)){callback_failed=1;return 1;}
#endif
    event_count++;''')
replace('    FILE *pcm = fopen(argv[3], "rb");','''#if defined(ACTIVE113_CAPTURE)
    if(open_capture())return 2;
#endif
    FILE *pcm = fopen(argv[3], "rb");''')
replace('    struct rusage usage;','''#if defined(ACTIVE113_CAPTURE)
    if(close_capture())failed=1;
#endif
    struct rusage usage;''')
replace('r1-strong-fsmn-offline-board-resource-v1','r1-active113-offline-resource-v1')
replace('    printf("{\\"schema\\":\\"r1-active113-offline-resource-v1\\",");','''    printf("{\\"schema\\":\\"r1-active113-offline-resource-v1\\",");
    printf("\\"model_sha256\\":\\"%s\\",\\"frontend_sha256\\":\\"%s\\",",active113_sdk_model_sha256(),active113_sdk_table_sha256());
    printf("\\"selection_sha256\\":\\"%s\\",\\"policy_id\\":\\"%s\\",",active113_sdk_selection_sha256(),active113_sdk_policy_id());
    printf("\\"sdk_state_bytes\\":%zu,\\"model_weight_bytes\\":%zu,",active113_sdk_state_bytes(),active113_sdk_weight_bytes());
#if defined(ACTIVE113_CAPTURE)
    printf("\\"capture_instrumented\\":true,");
#else
    printf("\\"capture_instrumented\\":false,");
#endif''')
os.umask(0o077)
with (HERE/'probe_active.c').open('x') as f:f.write(source);f.flush();os.fsync(f.fileno())
print('Prepared independent ACTIVE113 probe source; no build or execution')
