#include "mero43_stream.h"
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* Unlike assert(), this check must execute in Release/-DNDEBUG builds.
 * Decoder and stream tests call the functions being checked as expressions;
 * disabling those expressions would produce a false PASS without testing. */
#define KWS_CHECK(expr) do {                                                \
    if (!(expr)) {                                                          \
        fprintf(stderr, "%s:%d: check failed: %s\n",                     \
                __FILE__, __LINE__, #expr);                                 \
        abort();                                                             \
    }                                                                       \
} while (0)

/* Compact independent synthetic MODEL-PACK/STREAM ABI tests.
 * Real SHA-pinned 43K coefficients and raw PCM vs Torch are validated by
 * the separate, one-shot cross-repository acoustic numeric stage. */
static uint8_t packed[EN_V5_M43_PACK_BYTES];
static en_v5_m43_model model;
static en_v5_m43_stream stream;
static int16_t speech[832];

typedef struct {
    unsigned count;
    uint64_t ends[10], indices[10];
    float logits[10][4];
    unsigned detections;
} capture;

static void put32(uint8_t *p, uint32_t n) {
    p[0]=(uint8_t)n; p[1]=(uint8_t)(n>>8);
    p[2]=(uint8_t)(n>>16);p[3]=(uint8_t)(n>>24);
}
static void value_at(unsigned index, float value) {
    uint32_t bits;
    KWS_CHECK(index<EN_V5_M43_FLOATS);
    memcpy(&bits,&value,4);
    put32(packed+16+4u*index,bits);
}
static void synthetic_model_pack(void) {
    memset(packed,0,sizeof(packed));
    memcpy(packed,"MERO43V1",8);
    put32(packed+8,1u);put32(packed+12,EN_V5_M43_FLOATS);
    /* A valid but not trained 40-bin Mel bank: one shared positive FFT bin,
     * one input coefficient, one output coefficient, no acoustic events. */
    for (unsigned m=0;m<EN_V5_M43_MELS;++m) {
        value_at(EN_V5_M43_PARAMS+10u*EN_V5_M43_MELS+m,0.5f);
        value_at(EN_V5_M43_PARAMS+EN_V5_M43_MEL_WEIGHTS+40u+m,1.0f);
    }
    value_at(0u,1.0f); /* input.weight[0,0] */
    value_at(43488u+96u,1.0f); /* output.weight[1,0] */
}
static void callback(void *user,const float logits[4],
                     uint64_t index,uint64_t end,
                     const en_v5_mero_event *event) {
    capture *out=(capture*)user;
    KWS_CHECK(out!=NULL && logits!=NULL && event!=NULL);
    KWS_CHECK(out->count<10u);
    out->indices[out->count]=index;
    out->ends[out->count]=end;
    memcpy(out->logits[out->count],logits,4*sizeof(float));
    out->detections+=event->detected;
    ++out->count;
}
static void fill_pcm(void) {
    for (unsigned i=0;i<832u;++i)
        speech[i]=(int16_t)(((int)(i%71u)-35)*278);
}
static void model_source_validation(void) {
    KWS_CHECK(EN_V5_M43_PACK_BYTES==216960u);
    KWS_CHECK(en_v5_m43_model_bytes()==sizeof(model));
    KWS_CHECK(en_v5_m43_stream_bytes()==sizeof(stream));
    KWS_CHECK(en_v5_m43_model_bytes()<250000u);
    KWS_CHECK(en_v5_m43_stream_bytes()<24000u);
    KWS_CHECK(en_v5_m43_model_load(NULL,packed,sizeof(packed))==EN_V5_M43_BAD_ARGUMENT);
    KWS_CHECK(en_v5_m43_model_load(&model,NULL,sizeof(packed))==EN_V5_M43_BAD_ARGUMENT);
    KWS_CHECK(en_v5_m43_model_load(&model,packed,sizeof(packed)-1u)==EN_V5_M43_BAD_MODEL);
    packed[0]=0;
    KWS_CHECK(en_v5_m43_model_load(&model,packed,sizeof(packed))==EN_V5_M43_BAD_MODEL);
    packed[0]='M';
    /* Frozen scale must never silently become zero or NaN. */
    value_at(EN_V5_M43_FLOATS-1u,0.0f);
    KWS_CHECK(en_v5_m43_model_load(&model,packed,sizeof(packed))==EN_V5_M43_BAD_MODEL);
    value_at(EN_V5_M43_FLOATS-1u,1.0f);
    value_at(0u,NAN);
    KWS_CHECK(en_v5_m43_model_load(&model,packed,sizeof(packed))==EN_V5_M43_BAD_MODEL);
    value_at(0u,1.0f);
    KWS_CHECK(en_v5_m43_model_load(&model,packed,sizeof(packed))==EN_V5_M43_OK);
}
static void stream_contiguous_and_chunk_independent(void) {
    capture all={0},split={0};
    KWS_CHECK(en_v5_m43_stream_init(&stream,&model,1000000u,0u)==EN_V5_M43_OK);
    KWS_CHECK(en_v5_m43_feed(&stream,speech,511u,0u,callback,&all)==EN_V5_M43_OK);
    KWS_CHECK(all.count==0u);
    KWS_CHECK(en_v5_m43_feed(&stream,speech+511u,321u,511u,callback,&all)==EN_V5_M43_OK);
    KWS_CHECK(all.count==3u);
    for (unsigned j=0;j<3u;++j) {
        KWS_CHECK(all.indices[j]==j);
        KWS_CHECK(all.ends[j]==512u+160u*j);
        for (unsigned k=0;k<4u;++k)
            KWS_CHECK(isfinite(all.logits[j][k]));
    }
    KWS_CHECK(all.detections==0u);
    KWS_CHECK(en_v5_m43_stream_reset(&stream)==EN_V5_M43_OK);
    const unsigned chunks[]={1u,509u,2u,80u,160u,80u};
    unsigned start=0u;
    for (unsigned j=0u;j<sizeof(chunks)/sizeof(chunks[0]);++j) {
        unsigned n=chunks[j];
        KWS_CHECK(en_v5_m43_feed(&stream,speech+start,n,start,
                              callback,&split)==EN_V5_M43_OK);
        start+=n;
    }
    KWS_CHECK(start==832u && split.count==3u);
    for (unsigned j=0u;j<3u;++j) {
        KWS_CHECK(all.ends[j]==split.ends[j]);
        KWS_CHECK(all.indices[j]==split.indices[j]);
        KWS_CHECK(memcmp(all.logits[j],split.logits[j],
                      sizeof(all.logits[j]))==0);
    }
}
static void strict_reset_clock_and_discontinuity(void) {
    capture results={0};
    KWS_CHECK(en_v5_m43_feed(&stream,speech,1u,831u,callback,&results)
           ==EN_V5_M43_BAD_CLOCK);
    KWS_CHECK(en_v5_m43_feed(&stream,speech,0u,832u,callback,&results)
           ==EN_V5_M43_BAD_ARGUMENT);
    KWS_CHECK(en_v5_m43_feed(&stream,speech,16001u,832u,callback,&results)
           ==EN_V5_M43_BAD_ARGUMENT);
    KWS_CHECK(en_v5_m43_discontinuity(&stream,832u)==EN_V5_M43_BAD_CLOCK);
    KWS_CHECK(en_v5_m43_discontinuity(&stream,1000u)==EN_V5_M43_OK);
    KWS_CHECK(en_v5_m43_feed(&stream,speech,512u,1000u,callback,&results)
           ==EN_V5_M43_OK);
    KWS_CHECK(results.count==1u && results.ends[0]==1512u);
    KWS_CHECK(results.indices[0]==0u);
    KWS_CHECK(en_v5_m43_stream_reset(&stream)==EN_V5_M43_OK);
    memset(&results,0,sizeof(results));
    KWS_CHECK(en_v5_m43_feed(&stream,speech,512u,0u,callback,&results)
           ==EN_V5_M43_OK);
    KWS_CHECK(results.count==1u && results.ends[0]==512u);
    KWS_CHECK(en_v5_m43_feed(&stream,speech,1u,0u,callback,&results)
           ==EN_V5_M43_BAD_CLOCK);
}
int main(void) {
    synthetic_model_pack();
    model_source_validation();
    fill_pcm();
    stream_contiguous_and_chunk_independent();
    strict_reset_clock_and_discontinuity();
    puts("ENGLISH_V5_MERO43_PCM_MEL_FSMN_CTC_STREAM_CONTRACT_PASS");
    return 0;
}
