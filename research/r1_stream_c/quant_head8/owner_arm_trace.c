/* SPDX-License-Identifier: Apache-2.0
 * Owner-local R1 PCM trace comparator: no private bytes are embedded.
 * Build with the exact baseline/quantized FSMN SDK static library.
 * Emits only model-frame decisions, clock identities, and event identities.
 * This is NOT an independent acoustic-qualification tool.
 */
#define _POSIX_C_SOURCE 200809L
#include "fsmn_sdk.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>
#include <errno.h>

static unsigned long frames = 0, events = 0;
static int on_frame(void *ctx,uint64_t frame,uint64_t source,uint64_t delivery,
                    const double *features,const double *logits) {
    (void)ctx; (void)features;
    if (!logits || frame != (uint64_t)frames || source>delivery) return 1;
    unsigned best = 0;
    for (unsigned i=0;i<FSMN_LOGIT_COUNT;i++) {
        if (!isfinite(logits[i])) return 1;
        if (logits[i]>logits[best]) best=i;
    }
    if (printf("F,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%u\n",
               frame,source,delivery,best)<0) return 1;
    frames++;
    return 0;
}
static int on_event(void *ctx,const fsmn_event *e) {
    (void)ctx;
    if (!e || e->abi != FSMN_SDK_ABI || e->dependency_end_sample>e->delivery_end_sample)
        return 1;
    if (printf("E,%" PRIu64 ",%u,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
               e->segment,e->keyword_id,e->model_frame,e->first_token_frame,
               e->dependency_end_sample,e->delivery_end_sample)<0) return 1;
    events++;
    return 0;
}
static unsigned char *read_exact(const char *path,size_t want){
    FILE *f=fopen(path,"rb"); if (!f) return NULL;
    unsigned char *b=(unsigned char *)malloc(want);
    if(!b){fclose(f);return NULL;}
    int bad=(fread(b,1,want,f)!=want || fgetc(f)!=EOF || ferror(f));
    if(fclose(f))bad=1;
    if(bad){free(b);return NULL;}
    return b;
}
int main(int argc,char **argv) {
    if(argc!=4){fputs("usage: trace_runner MODEL FRONTEND MONO16K.PCM\n",stderr);return 2;}
    unsigned char *w=read_exact(argv[1],FSMN_MODEL_BYTES);
    unsigned char *t=read_exact(argv[2],FSMN_TABLE_BYTES);
    if(!w || !t){free(w);free(t);fputs("Invalid model/frontend bytes\n",stderr);return 2;}
    fsmn_sdk *sdk=fsmn_sdk_create(w,FSMN_MODEL_BYTES,t,FSMN_TABLE_BYTES,0,on_event,on_frame,NULL);
    free(w);free(t);
    if(!sdk){fputs("model identity/host rejected\n",stderr);return 2;}
    FILE *pcm=fopen(argv[3],"rb");if(!pcm){fsmn_sdk_destroy(sdk);return 2;}
    unsigned char buf[640];size_t n=0;uint64_t total=0;int failure=0;
    while((n=fread(buf,1,sizeof(buf),pcm))>0){
        size_t accepted=0;
        int rc=fsmn_sdk_feed(sdk,buf,n,&accepted);
        total+=accepted;
        if(rc!=0 || accepted!=n || total>UINT64_C(20000000)){failure=1;break;}
    }
    if(ferror(pcm) || fclose(pcm)) failure=1;
    if(!failure && fsmn_sdk_finish(sdk)!=FSMN_OK) failure=1;
    if(fflush(stdout))failure=1;
    fprintf(stderr,"{\"ok\":%s,\"pcm_bytes\":%" PRIu64 ",\"frames\":%lu,\"events\":%lu,\"weights\":%zu}\n",
            failure?"false":"true",total,frames,events,fsmn_sdk_weight_bytes());
    if(fsmn_sdk_destroy(sdk))failure=1;
    return failure?2:0;
}
