#ifdef KWS_B_PROFILE
#include "b_profile.h"
#endif
/* Research candidate, NOT numerically validated. FFT adapted from the isolated
 * donor_fbank.c radix2 skeleton, Apache-2.0. Algorithm semantics audited against
 * pinned Torchaudio compliance/kaldi.py, BSD-2-Clause. See NOTICE.md. */
#include "canonical_fbank.h"
#include <float.h>
#include <math.h>
#include <string.h>
#include "vendor/fft_twiddles.h"
#define KCF_BANK_MAGIC UINT32_C(0x4b464231)
_Static_assert(sizeof(float)==4 && FLT_RADIX==2 && FLT_MANT_DIG==24,
               "IEEE float32 required");
_Static_assert(sizeof(kcf_fbank)<=4608, "bounded FFT workspace");
static void fft512(kcf_fbank *s) {
    unsigned j=0;
    for(unsigned i=1;i<512;i++) {
        unsigned bit=256;
        while((j&bit)!=0) {j^=bit; bit>>=1;}
        j^=bit;
        if(i<j) {
            float v=s->re[i]; s->re[i]=s->re[j]; s->re[j]=v;
            v=s->im[i]; s->im[i]=s->im[j]; s->im[j]=v;
        }
    }
    for(unsigned len=2;len<=512;len<<=1)
        for(unsigned i=0;i<512;i+=len)
            for(unsigned k=0;k<len/2;k++) {
                unsigned a=i+k, b=a+len/2, t=k*(512/len);
                float wr=donor_fft_cos[t], wi=donor_fft_sin[t];
                float vr=s->re[b]*wr-s->im[b]*wi;
                float vi=s->re[b]*wi+s->im[b]*wr;
                float ur=s->re[a], ui=s->im[a];
                s->re[a]=ur+vr; s->im[a]=ui+vi;
                s->re[b]=ur-vr; s->im[b]=ui-vi;
            }
}
int kcf_fbank_init(kcf_fbank *s, const kcf_tables *t) {
    if (!s) return KCF_FBANK_ARGUMENT;
    memset(s,0,sizeof(*s));
    if (!t || t->schema!=KCF_TABLE_SCHEMA || t->povey_count!=400 ||
        t->mel_count!=80*257 || !t->povey || !t->mel) return KCF_FBANK_TABLE;
    for (unsigned i=0;i<400;i++)
        if (!isfinite(t->povey[i]) || t->povey[i]<0 || t->povey[i]>1)
            return KCF_FBANK_TABLE;
    if (t->povey[0]!=0 || t->povey[399]!=0) return KCF_FBANK_TABLE;
    for (unsigned m=0;m<80;m++) {
        int positive=0;
        for (unsigned k=0;k<257;k++) {
            float w=t->mel[m*257+k];
            if (!isfinite(w) || w<0 || w>1) return KCF_FBANK_TABLE;
            if (w>0) positive=1;
        }
        if (!positive || t->mel[m*257+256]!=0) return KCF_FBANK_TABLE;
    }
    s->tables=t; s->magic=KCF_BANK_MAGIC;
    return KCF_FBANK_OK;
}
#ifdef KWS_B_PROFILE
static int kcf_fbank_frame_profile_impl(kcf_fbank *s, const int16_t pcm[400], float out[80], kcf_trace *t)
#else
int kcf_fbank_frame(kcf_fbank *s, const int16_t pcm[400], float out[80], kcf_trace *t)
#endif
{
    int32_t sum=0;
    float row[80];
    if (!s || !pcm || !out) return KCF_FBANK_ARGUMENT;
    if (s->magic!=KCF_BANK_MAGIC) return KCF_FBANK_STATE;
    /* Raw integer amplitudes: all 400-sample partial sums fit exactly into
     * binary32 (abs <= 13,107,200). Mean division/backend still must be tested. */
    for (unsigned i=0;i<400;i++) sum+=pcm[i];
    float mean=(float)sum/400.0f;
    for (unsigned i=0;i<400;i++) s->re[i]=(float)pcm[i]-mean;
    if(t) memcpy(t->dc,s->re,sizeof(t->dc));
    /* Descending order preserves original previous sample. Separate multiply
     * and subtract; future build must disable FMA contraction/fast-math. */
    for(unsigned i=399;i>0;i--) s->re[i]=s->re[i]-0.97f*s->re[i-1];
    s->re[0]=s->re[0]-0.97f*s->re[0];
    if(t) memcpy(t->preemphasis,s->re,sizeof(t->preemphasis));
    for(unsigned i=0;i<400;i++) s->re[i]*=s->tables->povey[i];
    memset(s->re+400,0,112*sizeof(float)); memset(s->im,0,sizeof(s->im));
    if(t) memcpy(t->windowed,s->re,sizeof(t->windowed));
    fft512(s);
    for(unsigned i=0;i<257;i++) {
        /* Mirror abs().pow(2) operation order, NOT donor re^2+im^2 shortcut.
         * hypotf and Torch complex abs remain a measured numerical boundary. */
        float magnitude=hypotf(s->re[i],s->im[i]);
        s->re[i]=magnitude*magnitude;
        if(!isfinite(s->re[i])) return KCF_FBANK_NONFINITE;
        if(t) t->magnitude[i]=magnitude;
    }
    if(t) memcpy(t->power,s->re,sizeof(t->power));
    for(unsigned m=0;m<80;m++) {
        float energy=0.0f;
        /* Dense increasing-bin scalar sum is explicit, not claimed equivalent
         * to the frozen torch.mm backend's reduction grouping. */
        for(unsigned k=0;k<257;k++) energy+=s->re[k]*s->tables->mel[m*257+k];
        if (!isfinite(energy)) return KCF_FBANK_NONFINITE;
        if(t) t->mel[m]=energy;
        row[m]=logf(fmaxf(energy,FLT_EPSILON));
        if(!isfinite(row[m])) return KCF_FBANK_NONFINITE;
    }
    if(t) memcpy(t->logfbank,row,sizeof(row));
    memcpy(out,row,sizeof(row));
    return KCF_FBANK_OK;
}
int kcf_fbank_callback(void *s, const int16_t pcm[400], float row[80]) {
    return kcf_fbank_frame((kcf_fbank *)s,pcm,row,NULL);
}

#ifdef KWS_B_PROFILE
int kcf_fbank_frame(kcf_fbank *s, const int16_t pcm[400], float out[80], kcf_trace *t) {
 b_profile_stamp stamp=b_profile_begin();
 int rc=kcf_fbank_frame_profile_impl(s,pcm,out,t);
 b_profile_end(B_PROFILE_FRONTEND,stamp,rc==KCF_FBANK_OK);
 return rc;
}
#endif
