/* SPDX-License-Identifier: Apache-2.0
 * Fictional SDK callbacks test runner ownership and actual stdio only.
 * Never linked into the generated SDK or a real-model executable.
 */
#include "fsmn_sdk.h"
#include <stdlib.h>
#include <string.h>
struct fsmn_sdk {
    const unsigned char *model;
    unsigned char *owned;
    uint64_t frames,events,bytes;
    fsmn_event_fn event_fn;
    fsmn_frame_fn frame_fn;
    void *user;
};
fsmn_sdk *fsmn_sdk_create_view(const unsigned char *model,size_t n,
        const unsigned char *tables,size_t tn,uint64_t origin,
        fsmn_event_fn event_fn,fsmn_frame_fn frame_fn,void *user) {
    (void)origin;
    if(!model || !tables || n!=FSMN_MODEL_BYTES || tn!=FSMN_TABLE_BYTES)return NULL;
    fsmn_sdk *s=(fsmn_sdk *)calloc(1,sizeof(*s));
    if(!s)return NULL;
    s->model=model;s->event_fn=event_fn;s->frame_fn=frame_fn;s->user=user;
    return s;
}
fsmn_sdk *fsmn_sdk_create(const unsigned char *model,size_t n,
        const unsigned char *tables,size_t tn,uint64_t origin,
        fsmn_event_fn event_fn,fsmn_frame_fn frame_fn,void *user) {
    fsmn_sdk *s=fsmn_sdk_create_view(model,n,tables,tn,origin,event_fn,frame_fn,user);
    if(!s)return NULL;
    s->owned=(unsigned char *)malloc(n);
    if(!s->owned){free(s);return NULL;}
    memcpy(s->owned,model,n);s->model=s->owned;return s;
}
int fsmn_sdk_feed(fsmn_sdk *s,const unsigned char *pcm,size_t n,size_t *used) {
    if(!s || !pcm || !used)return FSMN_BAD_ARGUMENT;
    if(s->model[0]!=0x37 || s->model[FSMN_MODEL_BYTES-1]!=0x37)return FSMN_BACKEND_ERROR;
    *used=n;s->bytes+=n;
    double feature[FSMN_FEATURE_COUNT]={0},logits[FSMN_LOGIT_COUNT]={0};
    uint64_t frame=s->frames++;
    if(s->frame_fn && s->frame_fn(s->user,frame,0,s->bytes/2,feature,logits))return FSMN_CALLBACK_ERROR;
    fsmn_event event={FSMN_SDK_ABI,1,0,frame,0,0,s->bytes/2};
    s->events++;
    if(s->event_fn && s->event_fn(s->user,&event))return FSMN_CALLBACK_ERROR;
    return FSMN_OK;
}
int fsmn_sdk_finish(fsmn_sdk *s){return s?FSMN_OK:FSMN_BAD_ARGUMENT;}
int fsmn_sdk_destroy(fsmn_sdk *s) {
    if(!s)return FSMN_BAD_ARGUMENT;
    int rc=(s->model[0]==0x37 && s->model[FSMN_MODEL_BYTES-1]==0x37)?0:FSMN_BACKEND_ERROR;
    if(getenv("R1_TEST_FAIL_DESTROY"))rc=FSMN_BACKEND_ERROR;
    free(s->owned);free(s);return rc;
}
uint64_t fsmn_sdk_frame_count(const fsmn_sdk *s){return s->frames;}
uint64_t fsmn_sdk_event_count(const fsmn_sdk *s){return s->events;}
uint64_t fsmn_sdk_consumed_bytes(const fsmn_sdk *s){return s->bytes;}
size_t fsmn_sdk_state_bytes(void){return sizeof(fsmn_sdk);}
size_t fsmn_sdk_weight_bytes(void){return FSMN_MODEL_BYTES;}
