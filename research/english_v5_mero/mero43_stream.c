#include "mero43_stream.h"
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __FAST_MATH__
#error "Hey Mero V5 parity prohibits -ffast-math"
#endif
_Static_assert(sizeof(float)==4 && FLT_RADIX==2 && FLT_MANT_DIG==24,
               "model requires IEEE754 binary32");
_Static_assert(sizeof(int16_t)==2 && CHAR_BIT==8,
               "PCM and model pack require 8-bit bytes and PCM16");

#define M43_MAGIC UINT32_C(0x4d455243)
#define M43_INPUT_BIAS (EN_V5_M43_MELS*EN_V5_M43_HIDDEN)
#define M43_BLOCK_START (M43_INPUT_BIAS+EN_V5_M43_HIDDEN)
#define M43_LINEAR (EN_V5_M43_HIDDEN*EN_V5_M43_HIDDEN)
#define M43_MEMORY (EN_V5_M43_HIDDEN*(EN_V5_M43_LEFT+1u))
#define M43_BLOCK_STRIDE (M43_LINEAR+EN_V5_M43_HIDDEN+M43_MEMORY+EN_V5_M43_HIDDEN)
#define M43_OUTPUT_START (M43_BLOCK_START+EN_V5_M43_BLOCKS*M43_BLOCK_STRIDE)
_Static_assert(M43_OUTPUT_START+EN_V5_M43_HIDDEN*4u+4u==EN_V5_M43_PARAMS,
               "immutable 43876 source-trained parameter geometry");
_Static_assert(EN_V5_M43_FLOATS==54236u,"model pack geometry");

static uint32_t little32(const uint8_t *p) {
    return ((uint32_t)p[0])|((uint32_t)p[1]<<8)|
           ((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);
}

size_t en_v5_m43_model_bytes(void) {return sizeof(en_v5_m43_model);}
size_t en_v5_m43_stream_bytes(void) {return sizeof(en_v5_m43_stream);}

int en_v5_m43_model_load(en_v5_m43_model *model, const void *bytes,
                         size_t length) {
    const uint8_t *src=(const uint8_t *)bytes;
    uint32_t i,positive[EN_V5_M43_MELS]={0u};
    if (!model || !bytes) return EN_V5_M43_BAD_ARGUMENT;
    model->magic=0u;
    if (length!=EN_V5_M43_PACK_BYTES ||
        memcmp(src,"MERO43V1",8)!=0 ||
        little32(src+8)!=1u ||
        little32(src+12)!=EN_V5_M43_FLOATS)
        return EN_V5_M43_BAD_MODEL;
    for (i=0u;i<EN_V5_M43_FLOATS;++i) {
        uint32_t bits=little32(src+16u+4u*i);
        float v;
        memcpy(&v,&bits,sizeof(v));
        if (!isfinite(v)) return EN_V5_M43_BAD_MODEL;
        if (i<EN_V5_M43_PARAMS) {
            model->params[i]=v;
        } else if (i<EN_V5_M43_PARAMS+EN_V5_M43_MEL_WEIGHTS) {
            uint32_t j=i-EN_V5_M43_PARAMS;
            if (v<0.0f || v>1.0f) return EN_V5_M43_BAD_MODEL;
            if (v>0.0f) ++positive[j%EN_V5_M43_MELS];
            model->mel[j]=v;
        } else if (i<EN_V5_M43_PARAMS+EN_V5_M43_MEL_WEIGHTS+40u) {
            model->mean[i-EN_V5_M43_PARAMS-EN_V5_M43_MEL_WEIGHTS]=v;
        } else {
            if (v<=0.0f) return EN_V5_M43_BAD_MODEL;
            model->scale[i-EN_V5_M43_PARAMS-EN_V5_M43_MEL_WEIGHTS-40u]=v;
        }
    }
    for (i=0u;i<EN_V5_M43_MELS;++i)
        if (!positive[i]) return EN_V5_M43_BAD_MODEL;
    model->magic=M43_MAGIC;
    return EN_V5_M43_OK;
}

int en_v5_m43_stream_init(en_v5_m43_stream *s,
                          const en_v5_m43_model *model,
                          uint64_t max_gap_samples,
                          uint64_t cooldown_samples) {
    static const double PI=3.1415926535897932384626433832795;
    unsigned i;
    if (!s || !model || model->magic!=M43_MAGIC)
        return EN_V5_M43_BAD_ARGUMENT;
    memset(s,0,sizeof(*s));
    s->model=model;
    if (en_v5_mero_init(&s->ctc,max_gap_samples,cooldown_samples)!=EN_V5_OK)
        return EN_V5_M43_BAD_ARGUMENT;
    for (i=0u;i<EN_V5_M43_FFT/2u;++i) {
        double phase=2.0*PI*(double)i/(double)EN_V5_M43_FFT;
        s->tw_cos[i]=(float)cos(phase);
        s->tw_sin[i]=(float)-sin(phase);
    }
    for (i=0u;i<EN_V5_M43_WINDOW;++i)
        s->hann[i]=(float)(0.5-0.5*cos(2.0*PI*(double)i/
                                               (double)EN_V5_M43_WINDOW));
    s->magic=M43_MAGIC;
    return EN_V5_M43_OK;
}

int en_v5_m43_stream_reset(en_v5_m43_stream *s) {
    if (!s || s->magic!=M43_MAGIC) return EN_V5_M43_BAD_STATE;
    if (en_v5_mero_reset(&s->ctc)!=EN_V5_OK) return EN_V5_M43_BAD_DECODER;
    s->buffered=0u;
    s->next_sample=0u;
    s->frame_index=0u;
    memset(s->pending,0,sizeof(s->pending));
    memset(s->memory,0,sizeof(s->memory));
    return EN_V5_M43_OK;
}

int en_v5_m43_discontinuity(en_v5_m43_stream *s,
                            uint64_t next_sample) {
    if (!s || s->magic!=M43_MAGIC) return EN_V5_M43_BAD_STATE;
    if (next_sample<=s->next_sample) return EN_V5_M43_BAD_CLOCK;
    if (en_v5_mero_discontinuity(&s->ctc,next_sample)!=EN_V5_OK)
        return EN_V5_M43_BAD_CLOCK;
    s->next_sample=next_sample;
    s->buffered=0u;
    s->frame_index=0u;
    memset(s->pending,0,sizeof(s->pending));
    memset(s->memory,0,sizeof(s->memory));
    return EN_V5_M43_OK;
}

static void fft512(en_v5_m43_stream *s) {
    unsigned j=0u;
    for (unsigned i=1u;i<EN_V5_M43_FFT;++i) {
        unsigned bit=EN_V5_M43_FFT/2u;
        while (j&bit) {j^=bit;bit>>=1;}
        j^=bit;
        if (i<j) {
            float v=s->re[i];s->re[i]=s->re[j];s->re[j]=v;
            v=s->im[i];s->im[i]=s->im[j];s->im[j]=v;
        }
    }
    for (unsigned size=2u;size<=EN_V5_M43_FFT;size<<=1u) {
        for (unsigned start=0u;start<EN_V5_M43_FFT;start+=size) {
            for (unsigned k=0u;k<size/2u;++k) {
                unsigned a=start+k,b=a+size/2u,t=k*(EN_V5_M43_FFT/size);
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

static int frame_logits(en_v5_m43_stream *s, float logits[4]) {
    const en_v5_m43_model *m=s->model;
    float energy[EN_V5_M43_MELS]={0.0f},mel[EN_V5_M43_MELS];
    float activ[EN_V5_M43_HIDDEN],proj[EN_V5_M43_HIDDEN];
    float next[EN_V5_M43_HIDDEN];
    unsigned i,j,k,layer;
    for (i=0u;i<EN_V5_M43_FFT;++i) {
        /* torch.stft(center=False,n_fft=512,win_length=400) centers the
         * periodic Hann400 inside 512 samples, 56 zeros on each side.
         * Input is signed PCM16 divided by 32768 (NO DC or preemphasis). */
        float win=(i>=56u && i<456u)?s->hann[i-56u]:0.0f;
        s->re[i]=((float)s->pending[i]/32768.0f)*win;
        s->im[i]=0.0f;
    }
    fft512(s);
    for (i=0u;i<=EN_V5_M43_FFT/2u;++i) {
        float power=s->re[i]*s->re[i]+s->im[i]*s->im[i];
        for (j=0u;j<EN_V5_M43_MELS;++j)
            energy[j]+=power*m->mel[i*EN_V5_M43_MELS+j];
    }
    for (j=0u;j<EN_V5_M43_MELS;++j) {
        mel[j]=(log1pf(energy[j])-m->mean[j])/m->scale[j];
        if (!isfinite(mel[j])) return EN_V5_M43_BAD_NUMERIC;
    }
    /* Exact PyTorch Linear(40,96), four residual depthwise temporal memory
     * blocks (kernel 5, causal zero-padding left 4), Linear(96,4). */
    for (i=0u;i<EN_V5_M43_HIDDEN;++i) {
        float value=m->params[M43_INPUT_BIAS+i];
        for (j=0u;j<EN_V5_M43_MELS;++j)
            value+=m->params[i*EN_V5_M43_MELS+j]*mel[j];
        activ[i]=value>0.0f?value:0.0f;
    }
    for (layer=0u;layer<EN_V5_M43_BLOCKS;++layer) {
        unsigned begin=M43_BLOCK_START+layer*M43_BLOCK_STRIDE;
        const float *linear=m->params+begin;
        const float *linear_bias=linear+M43_LINEAR;
        const float *memory_weight=linear_bias+EN_V5_M43_HIDDEN;
        const float *memory_bias=memory_weight+M43_MEMORY;
        for (i=0u;i<EN_V5_M43_HIDDEN;++i) {
            float value=linear_bias[i];
            for (j=0u;j<EN_V5_M43_HIDDEN;++j)
                value+=linear[i*EN_V5_M43_HIDDEN+j]*activ[j];
            proj[i]=value;
        }
        /* Preserve ordered projected history, oldest at slot 0. */
        memmove(s->memory[layer],s->memory[layer]+1,
                EN_V5_M43_LEFT*EN_V5_M43_HIDDEN*sizeof(float));
        memcpy(s->memory[layer][EN_V5_M43_LEFT],proj,
               EN_V5_M43_HIDDEN*sizeof(float));
        for (i=0u;i<EN_V5_M43_HIDDEN;++i) {
            float value=activ[i]+memory_bias[i];
            for (k=0u;k<=EN_V5_M43_LEFT;++k)
                value+=s->memory[layer][k][i]*
                       memory_weight[i*(EN_V5_M43_LEFT+1u)+k];
            next[i]=value>0.0f?value:0.0f;
        }
        memcpy(activ,next,sizeof(activ));
    }
    for (i=0u;i<4u;++i) {
        float value=m->params[M43_OUTPUT_START+4u*EN_V5_M43_HIDDEN+i];
        for (j=0u;j<EN_V5_M43_HIDDEN;++j)
            value+=m->params[M43_OUTPUT_START+i*EN_V5_M43_HIDDEN+j]*activ[j];
        if (!isfinite(value)) return EN_V5_M43_BAD_NUMERIC;
        logits[i]=value;
    }
    return EN_V5_M43_OK;
}

int en_v5_m43_feed(en_v5_m43_stream *s, const int16_t *pcm,
                   size_t count, uint64_t first_sample,
                   en_v5_m43_frame_callback callback, void *user) {
    if (!s || s->magic!=M43_MAGIC || !s->model ||
        s->model->magic!=M43_MAGIC) return EN_V5_M43_BAD_STATE;
    if (!pcm || !callback || !count || count>EN_V5_M43_MAX_FEED)
        return EN_V5_M43_BAD_ARGUMENT;
    if (first_sample!=s->next_sample ||
        (uint64_t)count>UINT64_MAX-s->next_sample)
        return EN_V5_M43_BAD_CLOCK;
    for (size_t i=0u;i<count;++i) {
        s->pending[s->buffered++]=pcm[i];
        ++s->next_sample;
        if (s->buffered==EN_V5_M43_FFT) {
            en_v5_mero_event event;
            float logits[4];
            int status=frame_logits(s,logits);
            if (status!=EN_V5_M43_OK) {
                (void)en_v5_m43_stream_reset(s);
                return status;
            }
            if (en_v5_mero_accept(&s->ctc,logits,
                                  s->next_sample,&event)!=EN_V5_OK) {
                (void)en_v5_m43_stream_reset(s);
                return EN_V5_M43_BAD_DECODER;
            }
            callback(user,logits,s->frame_index,s->next_sample,&event);
            ++s->frame_index;
            memmove(s->pending,s->pending+EN_V5_M43_HOP,
                    (EN_V5_M43_FFT-EN_V5_M43_HOP)*sizeof(int16_t));
            s->buffered=EN_V5_M43_FFT-EN_V5_M43_HOP;
        }
    }
    return EN_V5_M43_OK;
}
