/* SPDX-License-Identifier: Apache-2.0 */
#include "r1_sdk.h"
#include "r1d_stream.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define SDK_MAGIC UINT32_C(0x52315344)
struct r1_sdk {
    uint32_t magic;
    int busy,failed,finished;
    uint64_t epoch,base_sample,next_byte,total_proposals;
    r1d_stream *core;
    r1_proposal_decoder decoder;
    r1_proposal_sink sink;
    void *user;
};
void r1_proposal_reset(r1_proposal_decoder *d){if(d)memset(d,0,sizeof(*d));}
int r1_proposal_step(r1_proposal_decoder *d,unsigned token,uint64_t frame,unsigned *keyword){
    if(!d||!keyword||token>=R1_SDK_VOCAB||frame!=d->rows||frame>UINT64_MAX-51u)
        return R1_SDK_ARGUMENT;
    *keyword=0;d->rows++;
    if(!token){d->last=0;return 0;}
    if(token==d->last)return 0;
    d->last=token;
    if(d->used==4){memmove(d->ids,d->ids+1,3*sizeof(unsigned));
        memmove(d->frame,d->frame+1,3*sizeof(uint64_t));d->used=3;}
    d->ids[d->used]=token;d->frame[d->used]=frame;d->used++;
    static const unsigned words[2][4]={{1027,1401,1462,2093},{1462,2093,1462,2093}};
    if(d->used<4||frame-d->frame[0]>150u)return 0;
    for(unsigned k=0;k<2;k++)if(!memcmp(d->ids,words[k],sizeof(d->ids))&&frame>=d->cooldown[k]){
        d->cooldown[k]=frame+50u;d->events++;*keyword=k+1u;break;}
    return 0;
}
static int output(void *user,uint64_t frame,uint64_t source,uint64_t delivery,
                  const double *features,const double *logits){
    r1_sdk *s=(r1_sdk *)user;(void)features;
    if(!s||s->magic!=SDK_MAGIC||!logits||source>delivery||
       source>UINT64_MAX-s->base_sample||delivery>UINT64_MAX-s->base_sample)
        return R1_SDK_STATE;
    unsigned best=0;
    for(unsigned i=0;i<R1D_OUTPUTS;i++){
        if(!isfinite(logits[i]))return R1_SDK_NUMERIC;
        if(logits[i]>logits[best])best=i;
    }
    unsigned keyword=0;int rc=r1_proposal_step(&s->decoder,best,frame,&keyword);
    if(rc||!keyword)return rc;
    if(s->total_proposals==UINT64_MAX)return R1_SDK_STATE;
    s->total_proposals++;
    r1_proposal event={s->epoch,keyword,frame,s->base_sample+source,
                       s->base_sample+delivery,s->total_proposals};
    return s->sink(s->user,&event)?R1_SDK_CALLBACK:0;
}
r1_sdk *r1_sdk_create(const unsigned char *w,size_t wn,const unsigned char *t,size_t tn,
                     r1_proposal_sink sink,void *user){
    if(!sink)return NULL;
    r1_sdk *s=(r1_sdk *)calloc(1,sizeof(*s));if(!s)return NULL;
    s->magic=SDK_MAGIC;s->sink=sink;s->user=user;
    s->core=r1d_create(w,wn,t,tn,output,s);
    if(!s->core){free(s);return NULL;}return s;
}
static int valid(const r1_sdk *s){return s&&s->magic==SDK_MAGIC;}
int r1_sdk_feed(r1_sdk *s,uint64_t offset,const unsigned char *data,size_t n,size_t *used){
    if(!used)return R1_SDK_ARGUMENT;
    *used=0;
    if(!valid(s)||s->busy||s->failed||s->finished)return R1_SDK_STATE;
    if((!data&&n)||n>R1_SDK_MAX_BLOCK_BYTES||n>UINT64_MAX-s->next_byte)return R1_SDK_ARGUMENT;
    if(offset!=s->next_byte)return R1_SDK_OFFSET;
    if(!n)return 0;
    s->busy=1;int rc=r1d_feed(s->core,data,n,used);
    s->next_byte+=*used;s->busy=0;
    if(rc||*used!=n){s->failed=1;return rc?rc:R1_SDK_STATE;}
    return 0;
}
int r1_sdk_gap(r1_sdk *s,uint64_t sample){
    if(!valid(s)||s->busy||s->epoch==UINT64_MAX)return R1_SDK_STATE;
    if(sample>UINT64_MAX/2u||sample*2u<s->next_byte)return R1_SDK_OFFSET;
    s->busy=1;int rc=r1d_reset(s->core);s->busy=0;
    if(rc){s->failed=1;return rc;}
    r1_proposal_reset(&s->decoder);s->epoch++;s->base_sample=sample;
    s->next_byte=sample*2u;s->failed=0;s->finished=0;return 0;
}
int r1_sdk_finish(r1_sdk *s){
    if(!valid(s)||s->busy||s->failed||s->finished)return R1_SDK_STATE;
    s->busy=1;int rc=r1d_finish(s->core);s->busy=0;
    if(rc)s->failed=1;else s->finished=1;return rc;
}
int r1_sdk_destroy(r1_sdk *s){
    if(!valid(s)||s->busy)return R1_SDK_STATE;
    r1d_destroy(s->core);s->magic=0;free(s);return 0;
}
uint64_t r1_sdk_frames(const r1_sdk *s){return valid(s)?r1d_frames(s->core):0;}
uint64_t r1_sdk_proposals(const r1_sdk *s){return valid(s)?s->total_proposals:0;}
size_t r1_sdk_state_bytes(void){return sizeof(r1_sdk)+r1d_state_bytes();}
size_t r1_sdk_weight_bytes(void){return r1d_weight_bytes();}
