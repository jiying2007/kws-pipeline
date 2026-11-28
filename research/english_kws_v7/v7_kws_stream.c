#include "v7_kws_stream.h"
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __FAST_MATH__
#error "V7 model parity prohibits -ffast-math"
#endif
_Static_assert(sizeof(float)==4 && FLT_RADIX==2 && FLT_MANT_DIG==24,
               "model requires IEEE754 binary32");
_Static_assert(sizeof(int16_t)==2 && CHAR_BIT==8,
               "PCM and model pack require 8-bit bytes and PCM16");

#define V7_MAGIC UINT32_C(0x4b575337)
#define V7_INPUT_BIAS (EN_V7_KWS_MELS*EN_V7_KWS_HIDDEN)
#define V7_BLOCK_START (V7_INPUT_BIAS+EN_V7_KWS_HIDDEN)
#define V7_LINEAR (EN_V7_KWS_HIDDEN*EN_V7_KWS_HIDDEN)
#define V7_MEMORY (EN_V7_KWS_HIDDEN*(EN_V7_KWS_LEFT+1u))
#define V7_BLOCK_STRIDE (V7_LINEAR+EN_V7_KWS_HIDDEN+V7_MEMORY+EN_V7_KWS_HIDDEN)
#define V7_OUTPUT_START (V7_BLOCK_START+EN_V7_KWS_BLOCKS*V7_BLOCK_STRIDE)
_Static_assert(V7_OUTPUT_START+EN_V7_KWS_HIDDEN*4u+4u==EN_V7_KWS_PARAMS,
               "immutable 43876 source-trained parameter geometry");
_Static_assert(EN_V7_KWS_FLOATS==54236u,"model pack geometry");

static uint32_t little32(const uint8_t *p) {
    return ((uint32_t)p[0])|((uint32_t)p[1]<<8)|
           ((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);
}

size_t en_v7_kws_model_bytes(void) {return sizeof(en_v7_kws_model);}
size_t en_v7_kws_stream_bytes(void) {return sizeof(en_v7_kws_stream);}

int en_v7_kws_model_load(en_v7_kws_model *model, const void *bytes,
                         size_t length, uint32_t candidate_id) {
    const uint8_t *src=(const uint8_t *)bytes;
    uint32_t i,positive[EN_V7_KWS_MELS]={0u};
    if (!model || !bytes) return EN_V7_KWS_BAD_ARGUMENT;
    model->magic=0u;
    model->candidate_id=0u;
    if ((candidate_id!=1u && candidate_id!=2u) ||
        length!=EN_V7_KWS_PACK_BYTES ||
        memcmp(src,"KWSV7C01",8)!=0 ||
        little32(src+8)!=1u ||
        little32(src+12)!=EN_V7_KWS_FLOATS)
        return EN_V7_KWS_BAD_MODEL;
    for (i=0u;i<EN_V7_KWS_FLOATS;++i) {
        uint32_t bits=little32(src+16u+4u*i);
        float v;
        memcpy(&v,&bits,sizeof(v));
        if (!isfinite(v)) return EN_V7_KWS_BAD_MODEL;
        if (i<EN_V7_KWS_PARAMS) {
            model->params[i]=v;
        } else if (i<EN_V7_KWS_PARAMS+EN_V7_KWS_MEL_WEIGHTS) {
            uint32_t j=i-EN_V7_KWS_PARAMS;
            if (v<0.0f || v>1.0f) return EN_V7_KWS_BAD_MODEL;
            if (v>0.0f) ++positive[j%EN_V7_KWS_MELS];
            model->mel[j]=v;
        } else if (i<EN_V7_KWS_PARAMS+EN_V7_KWS_MEL_WEIGHTS+40u) {
            model->mean[i-EN_V7_KWS_PARAMS-EN_V7_KWS_MEL_WEIGHTS]=v;
        } else {
            if (v<=0.0f) return EN_V7_KWS_BAD_MODEL;
            model->scale[i-EN_V7_KWS_PARAMS-EN_V7_KWS_MEL_WEIGHTS-40u]=v;
        }
    }
    for (i=0u;i<EN_V7_KWS_MELS;++i)
        if (!positive[i]) return EN_V7_KWS_BAD_MODEL;
    model->candidate_id=candidate_id;
    model->magic=V7_MAGIC;
    return EN_V7_KWS_OK;
}

int en_v7_kws_stream_init(en_v7_kws_stream *s,
                          const en_v7_kws_model *model,
                          uint64_t max_gap_samples,
                          uint64_t cooldown_samples) {
    static const double PI=3.1415926535897932384626433832795;
    unsigned i;
    if (!s || !model || model->magic!=V7_MAGIC || (model->candidate_id!=1u && model->candidate_id!=2u))
        return EN_V7_KWS_BAD_ARGUMENT;
    memset(s,0,sizeof(*s));
    s->model=model;
    if (en_v7_ctc_init(&s->ctc,model->candidate_id,max_gap_samples,cooldown_samples)!=EN_V7_CTC_OK)
        return EN_V7_KWS_BAD_ARGUMENT;
    for (i=0u;i<EN_V7_KWS_FFT/2u;++i) {
        double phase=2.0*PI*(double)i/(double)EN_V7_KWS_FFT;
        s->tw_cos[i]=(float)cos(phase);
        s->tw_sin[i]=(float)-sin(phase);
    }
    for (i=0u;i<EN_V7_KWS_WINDOW;++i)
        s->hann[i]=(float)(0.5-0.5*cos(2.0*PI*(double)i/
                                               (double)EN_V7_KWS_WINDOW));
    s->magic=V7_MAGIC;
    return EN_V7_KWS_OK;
}

int en_v7_kws_stream_reset(en_v7_kws_stream *s) {
    if (!s || s->magic!=V7_MAGIC) return EN_V7_KWS_BAD_STATE;
    if (en_v7_ctc_reset(&s->ctc)!=EN_V7_CTC_OK) return EN_V7_KWS_BAD_DECODER;
    s->buffered=0u;
    s->next_sample=0u;
    s->frame_index=0u;
    memset(s->pending,0,sizeof(s->pending));
    memset(s->memory,0,sizeof(s->memory));
    return EN_V7_KWS_OK;
}

int en_v7_kws_discontinuity(en_v7_kws_stream *s,
                            uint64_t next_sample) {
    if (!s || s->magic!=V7_MAGIC) return EN_V7_KWS_BAD_STATE;
    if (next_sample<=s->next_sample) return EN_V7_KWS_BAD_CLOCK;
    if (en_v7_ctc_discontinuity(&s->ctc,next_sample)!=EN_V7_CTC_OK)
        return EN_V7_KWS_BAD_CLOCK;
    s->next_sample=next_sample;
    s->buffered=0u;
    s->frame_index=0u;
    memset(s->pending,0,sizeof(s->pending));
    memset(s->memory,0,sizeof(s->memory));
    return EN_V7_KWS_OK;
}

static void fft512(en_v7_kws_stream *s) {
    unsigned j=0u;
    for (unsigned i=1u;i<EN_V7_KWS_FFT;++i) {
        unsigned bit=EN_V7_KWS_FFT/2u;
        while (j&bit) {j^=bit;bit>>=1;}
        j^=bit;
        if (i<j) {
            float v=s->re[i];s->re[i]=s->re[j];s->re[j]=v;
            v=s->im[i];s->im[i]=s->im[j];s->im[j]=v;
        }
    }
    for (unsigned size=2u;size<=EN_V7_KWS_FFT;size<<=1u) {
        for (unsigned start=0u;start<EN_V7_KWS_FFT;start+=size) {
            for (unsigned k=0u;k<size/2u;++k) {
                unsigned a=start+k,b=a+size/2u,t=k*(EN_V7_KWS_FFT/size);
                float wr=s->tw_cos[t],wi=s->tw_sin[t];
                float re=s->re[b]*wr-s->im[b]*wi;
                float im=s->re[b]*wi+s->im[b]*wr;
                float ar=s->re[a],ai=s->im[a];
                s->re[a]=ar+re;s->im[a]=ai+im;
                s->re[b]=ar-re;s->im[b]=ai-im;
            }
        }
    }
}

static int frame_logits(en_v7_kws_stream *s, float logits[4]) {
    const en_v7_kws_model *m=s->model;
    float energy[EN_V7_KWS_MELS]={0.0f},mel[EN_V7_KWS_MELS];
    float activ[EN_V7_KWS_HIDDEN],proj[EN_V7_KWS_HIDDEN];
    float next[EN_V7_KWS_HIDDEN];
    unsigned i,j,k,layer;
    for (i=0u;i<EN_V7_KWS_FFT;++i) {
        /* torch.stft(center=False,n_fft=512,win_length=400) centers the
         * periodic Hann400 inside 512 samples, 56 zeros on each side.
         * Input is signed PCM16 divided by 32768 (NO DC or preemphasis). */
        float win=(i>=56u && i<456u)?s->hann[i-56u]:0.0f;
        s->re[i]=((float)s->pending[i]/32768.0f)*win;
        s->im[i]=0.0f;
    }
    fft512(s);
    for (i=0u;i<=EN_V7_KWS_FFT/2u;++i) {
        float power=s->re[i]*s->re[i]+s->im[i]*s->im[i];
        for (j=0u;j<EN_V7_KWS_MELS;++j)
            energy[j]+=power*m->mel[i*EN_V7_KWS_MELS+j];
    }
    for (j=0u;j<EN_V7_KWS_MELS;++j) {
        mel[j]=(log1pf(energy[j])-m->mean[j])/m->scale[j];
        if (!isfinite(mel[j])) return EN_V7_KWS_BAD_NUMERIC;
    }
    /* Exact PyTorch Linear(40,96), four residual depthwise temporal memory
     * blocks (kernel 5, causal zero-padding left 4), Linear(96,4). */
    for (i=0u;i<EN_V7_KWS_HIDDEN;++i) {
        float value=m->params[V7_INPUT_BIAS+i];
        for (j=0u;j<EN_V7_KWS_MELS;++j)
            value+=m->params[i*EN_V7_KWS_MELS+j]*mel[j];
        activ[i]=value>0.0f?value:0.0f;
    }
    for (layer=0u;layer<EN_V7_KWS_BLOCKS;++layer) {
        unsigned begin=V7_BLOCK_START+layer*V7_BLOCK_STRIDE;
        const float *linear=m->params+begin;
        const float *linear_bias=linear+V7_LINEAR;
        const float *memory_weight=linear_bias+EN_V7_KWS_HIDDEN;
        const float *memory_bias=memory_weight+V7_MEMORY;
        for (i=0u;i<EN_V7_KWS_HIDDEN;++i) {
            float value=linear_bias[i];
            for (j=0u;j<EN_V7_KWS_HIDDEN;++j)
                value+=linear[i*EN_V7_KWS_HIDDEN+j]*activ[j];
            proj[i]=value;
        }
        /* Preserve ordered projected history, oldest at slot 0. */
        memmove(s->memory[layer],s->memory[layer]+1,
                EN_V7_KWS_LEFT*EN_V7_KWS_HIDDEN*sizeof(float));
        memcpy(s->memory[layer][EN_V7_KWS_LEFT],proj,
               EN_V7_KWS_HIDDEN*sizeof(float));
        for (i=0u;i<EN_V7_KWS_HIDDEN;++i) {
            float value=activ[i]+memory_bias[i];
            for (k=0u;k<=EN_V7_KWS_LEFT;++k)
                value+=s->memory[layer][k][i]*
                       memory_weight[i*(EN_V7_KWS_LEFT+1u)+k];
            next[i]=value>0.0f?value:0.0f;
        }
        memcpy(activ,next,sizeof(activ));
    }
    for (i=0u;i<4u;++i) {
        float value=m->params[V7_OUTPUT_START+4u*EN_V7_KWS_HIDDEN+i];
        for (j=0u;j<EN_V7_KWS_HIDDEN;++j)
            value+=m->params[V7_OUTPUT_START+i*EN_V7_KWS_HIDDEN+j]*activ[j];
        if (!isfinite(value)) return EN_V7_KWS_BAD_NUMERIC;
        logits[i]=value;
    }
    return EN_V7_KWS_OK;
}

int en_v7_kws_feed(en_v7_kws_stream *s, const int16_t *pcm,
                   size_t count, uint64_t first_sample,
                   en_v7_kws_frame_callback callback, void *user) {
    if (!s || s->magic!=V7_MAGIC || !s->model ||
        s->model->magic!=V7_MAGIC) return EN_V7_KWS_BAD_STATE;
    if (!pcm || !callback || !count || count>EN_V7_KWS_MAX_FEED)
        return EN_V7_KWS_BAD_ARGUMENT;
    if (first_sample!=s->next_sample ||
        (uint64_t)count>UINT64_MAX-s->next_sample)
        return EN_V7_KWS_BAD_CLOCK;
    for (size_t i=0u;i<count;++i) {
        s->pending[s->buffered++]=pcm[i];
        ++s->next_sample;
        if (s->buffered==EN_V7_KWS_FFT) {
            en_v7_ctc_event event;
            float logits[4];
            int status=frame_logits(s,logits);
            if (status!=EN_V7_KWS_OK) {
                (void)en_v7_kws_stream_reset(s);
                return status;
            }
            if (en_v7_ctc_accept(&s->ctc,logits,
                                  s->next_sample,&event)!=EN_V7_CTC_OK) {
                (void)en_v7_kws_stream_reset(s);
                return EN_V7_KWS_BAD_DECODER;
            }
            callback(user,logits,s->frame_index,s->next_sample,&event);
            ++s->frame_index;
            memmove(s->pending,s->pending+EN_V7_KWS_HOP,
                    (EN_V7_KWS_FFT-EN_V7_KWS_HOP)*sizeof(int16_t));
            s->buffered=EN_V7_KWS_FFT-EN_V7_KWS_HOP;
        }
    }
    return EN_V7_KWS_OK;
}
