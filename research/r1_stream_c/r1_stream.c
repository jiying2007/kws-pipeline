/* SPDX-License-Identifier: Apache-2.0
 * Modified integration 2026-11-07. Reuses the pinned historical arithmetic,
 * but binds a distinct2601 R1 payload and unpadded reference EOF semantics. */
#include "r1_stream.h"
#include "r1_pcm.h"
#include "canonical_fbank.h"
#include "r1fsmn.h"
#include "model_identity.h"
#include "sha256.h"
#include <stdlib.h>
#include <string.h>
#include <fenv.h>
#include <limits.h>
#define R1_MAGIC UINT32_C(0x52315354)
struct r1_stream {
    uint32_t magic;
    int busy,mode;
    r1fsmn_model net;
    kcf_fbank bank;
    kcf_tables tables;
    r1_pcm pcm;
    r1_sink sink;
    void *user;
    uint64_t delivery_end,frames;
    float *weights;
    float table_data[R1_TABLE_FLOATS];
    float logits[R1_OUTPUTS];
};
size_t r1_state_bytes(void){return sizeof(r1_stream);}
size_t r1_weight_bytes(void){return R1FSMN_PAYLOAD_BYTES;}
uint64_t r1_frames(const r1_stream *s){return s&&s->magic==R1_MAGIC?s->frames:0;}
static int bank(void *user,const int16_t *pcm,float *row){
    return kcf_fbank_frame(&((r1_stream *)user)->bank,pcm,row,NULL);
}
static int emit(void *user,int event,uint64_t index,uint64_t center,
                uint64_t source_end,int eof,const float *feature){
    r1_stream *s=(r1_stream *)user;(void)center;
    if(eof)return KCF_STATE;
    if(event==KCF_EVENT_FBANK)return 0;
    if(event!=KCF_EVENT_CONTEXT || index!=s->frames)return KCF_STATE;
    int rc=r1fsmn_step(&s->net,feature,400,s->logits,R1_OUTPUTS,NULL,0);
    if(rc)return rc;
    s->frames++;
    return s->sink(s->user,index,source_end,s->delivery_end,feature,s->logits);
}
r1_stream *r1_create(const unsigned char *payload,size_t bytes,
        const unsigned char *tables,size_t table_bytes,r1_sink sink,void *user){
    if(!payload || bytes!=R1FSMN_PAYLOAD_BYTES || !tables ||
       table_bytes!=R1_TABLE_FLOATS*4u || !sink)return NULL;
    char hash[65];kws_sha256_memory_hex(tables,table_bytes,hash);
    if(memcmp(hash,R1_FRONTEND_SHA,65))return NULL;
    r1_stream *s=(r1_stream *)calloc(1,sizeof(*s));if(!s)return NULL;
    s->weights=(float *)malloc(R1FSMN_PAYLOAD_BYTES);
    if(!s->weights){free(s);return NULL;}
    fenv_t env;
    if(feholdexcept(&env)){free(s->weights);free(s);return NULL;}
    int rc=r1fsmn_load(&s->net,s->weights,R1FSMN_FLOATS,payload,bytes,1,R1_MODEL_SHA);
    memcpy(s->table_data,tables,table_bytes);
    s->tables=(kcf_tables){KCF_TABLE_SCHEMA,400,80*257,s->table_data,s->table_data+400};
    if(!rc)rc=kcf_fbank_init(&s->bank,&s->tables);
    if(!rc)rc=r1_pcm_init(&s->pcm,bank,s,emit,s);
    int restore=fesetenv(&env);
    if(rc || restore){free(s->weights);free(s);return NULL;}
    s->sink=sink;s->user=user;s->magic=R1_MAGIC;return s;
}
int r1_feed(r1_stream *s,const unsigned char *data,size_t bytes,size_t *consumed){
    if(!consumed)return KCF_ARGUMENT;
    *consumed=0;
    if(!s || s->magic!=R1_MAGIC || s->busy || s->mode==2)return KCF_STATE;
    if((!data && bytes) || bytes>UINT64_MAX-s->pcm.input_bytes)return KCF_ARGUMENT;
    if(!r1fsmn_environment_ok())return R1FSMN_ENVIRONMENT;
    fenv_t env;if(feholdexcept(&env))return R1FSMN_ENVIRONMENT;
    s->busy=1;s->mode=1;s->delivery_end=(s->pcm.input_bytes+bytes)/2;
    int rc=r1_pcm_feed(&s->pcm,data,bytes,consumed);
    s->busy=0;if(fesetenv(&env))return R1FSMN_ENVIRONMENT;return rc;
}
int r1_finish(r1_stream *s){
    if(!s || s->magic!=R1_MAGIC || s->busy || s->mode==2)return KCF_STATE;
    s->busy=1;int rc=r1_pcm_finish(&s->pcm);s->busy=0;return rc;
}
int r1_reset(r1_stream *s){
    if(!s || s->magic!=R1_MAGIC || s->busy)return KCF_STATE;
    int rc=r1fsmn_reset(&s->net);if(rc)return rc;
    rc=r1_pcm_init(&s->pcm,bank,s,emit,s);
    if(!rc){s->mode=0;s->frames=0;s->delivery_end=0;}
    return rc;
}
void r1_destroy(r1_stream *s){
    if(!s || s->magic!=R1_MAGIC || s->busy)return;
    s->magic=0;free(s->weights);free(s);
}
int r1_feature(r1_stream *s,const float *feature,float *logits){
    if(!s || s->magic!=R1_MAGIC || s->busy || s->mode==1)return KCF_STATE;
    fenv_t env;if(feholdexcept(&env))return R1FSMN_ENVIRONMENT;
    s->busy=1;s->mode=2;
    int rc=r1fsmn_step(&s->net,feature,400,logits,R1_OUTPUTS,NULL,0);
    if(!rc)s->frames++;
    s->busy=0;if(fesetenv(&env))return R1FSMN_ENVIRONMENT;return rc;
}
