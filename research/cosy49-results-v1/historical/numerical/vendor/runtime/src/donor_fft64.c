#include "donor_fft64.h"
#include <float.h>
#include <fenv.h>
#include <math.h>
#include <string.h>
#include "../baseline/native/donor_fbank/frontend_tables.h"
#include "fft64_twiddles.h"
#define DONOR_MAGIC UINT32_C(0x46423634)
_Static_assert(sizeof(float)==4 && FLT_RADIX==2 && FLT_MANT_DIG==24 && FLT_MAX_EXP==128 && FLT_MIN_EXP==-125,"IEEE float32 required");
_Static_assert(sizeof(double)==8 && DBL_MANT_DIG==53 && DBL_MAX_EXP==1024 && DBL_MIN_EXP==-1021,"IEEE float64 required");
_Static_assert(FLT_EVAL_METHOD==0,"No excess intermediate precision allowed");
/* No arithmetic mode is changed. Reject rounding or subnormal modes that break
 * the frozen FP32 boundaries or binary64 butterfly semantics. Volatile operands
 * plus -frounding-math force runtime operations, including DAZ/FTZ detection. */
int donor_fft64_environment_ok(void) {
 if(fegetround()!=FE_TONEAREST)return 0;
 volatile float fmin=FLT_MIN, fhalf=0.5f, ftiny=FLT_TRUE_MIN, fone=1.0f;
 volatile double dmin=DBL_MIN, dhalf=0.5, dtiny=DBL_TRUE_MIN, done=1.0;
 float a=fmin*fhalf,b=ftiny*fone;double c=dmin*dhalf,d=dtiny*done;
 uint32_t ua,ub;uint64_t uc,ud;
 memcpy(&ua,&a,sizeof(ua));memcpy(&ub,&b,sizeof(ub));
 memcpy(&uc,&c,sizeof(uc));memcpy(&ud,&d,sizeof(ud));
 return ua==UINT32_C(0x00400000)&&ub==1&&uc==UINT64_C(0x0008000000000000)&&ud==1;
}

static float interface_rne(double x) { return (float)x; }

/* Isolated adaptation of the repository's Apache2 src/frontend.c fft512
 * radix2 skeleton. Product math is neither linked nor modified. */
static void fft512(donor_fft64_state *s) {
 for(unsigned i=0;i<512;i++){s->fft_re[i]=(double)s->re[i];s->fft_im[i]=(double)s->im[i];}
 unsigned j=0;
 for(unsigned i=1;i<512;i++){
  unsigned bit=256;
  while((j&bit)!=0){j^=bit;bit>>=1;}
  j^=bit;
  if(i<j){double v=s->fft_re[i];s->fft_re[i]=s->fft_re[j];s->fft_re[j]=v;v=s->fft_im[i];s->fft_im[i]=s->fft_im[j];s->fft_im[j]=v;}
 }
 for(unsigned len=2;len<=512;len<<=1){
  for(unsigned i=0;i<512;i+=len){
   for(unsigned k=0;k<len/2;k++){
    unsigned a=i+k,b=a+len/2;
    unsigned t=k*(512/len);double wr=donor_fft64_cos[t],wi=donor_fft64_sin[t];
    double vr=s->fft_re[b]*wr-s->fft_im[b]*wi,vi=s->fft_re[b]*wi+s->fft_im[b]*wr;
    double ur=s->fft_re[a],ui=s->fft_im[a];
    s->fft_re[a]=ur+vr;s->fft_im[a]=ui+vi;s->fft_re[b]=ur-vr;s->fft_im[b]=ui-vi;

   }
  }
 } /* Explicit binary32 complex interface, RNE verified at public entry. */
 for(unsigned i=0;i<512;i++){s->re[i]=interface_rne(s->fft_re[i]);s->im[i]=interface_rne(s->fft_im[i]);}
}
int donor_fft64_fft_power(donor_fft64_state*s,const float input[512],float output[257]){
 if(!s||!input||!output)return DONOR_FFT64_ARGUMENT;
 if(s->initialized!=DONOR_MAGIC||!donor_fft64_environment_ok())return DONOR_FFT64_STATE;
 for(unsigned i=0;i<512;i++)if(!isfinite(input[i])||fabsf(input[i])>65536.0f)return DONOR_FFT64_ARGUMENT;
 memcpy(s->re,input,512*sizeof(float));memset(s->im,0,sizeof(s->im));fft512(s);
 for(unsigned i=0;i<257;i++)output[i]=s->re[i]*s->re[i]+s->im[i]*s->im[i];
 return DONOR_FFT64_OK;
}
static void analyze(donor_fft64_state*s,const int16_t*pcm,float output[80],donor_fft64_trace*t){
 int32_t sum=0;
 for(unsigned i=0;i<400;i++)sum+=pcm[i];
 float mean=(float)sum/400.0f;
 for(unsigned i=0;i<400;i++)s->re[i]=(float)pcm[i]-mean;
 if(t)memcpy(t->dc,s->re,sizeof(t->dc));
 for(unsigned i=399;i>0;i--)s->re[i]=s->re[i]-0.97f*s->re[i-1];
 s->re[0]=s->re[0]-0.97f*s->re[0];
 if(t)memcpy(t->preemphasis,s->re,sizeof(t->preemphasis));
 for(unsigned i=0;i<400;i++)s->re[i]*=donor_hamming[i];
 memset(s->re+400,0,112*sizeof(float));memset(s->im,0,sizeof(s->im));
 if(t)memcpy(t->windowed,s->re,sizeof(t->windowed));
 fft512(s);
 /* Reuse real scratch for power once every complex output is complete. */
 for(unsigned i=0;i<257;i++)s->re[i]=s->re[i]*s->re[i]+s->im[i]*s->im[i];
 if(t)memcpy(t->power,s->re,sizeof(t->power));
 for(unsigned m=0;m<80;m++){
  float energy=0.0f;
  for(unsigned k=donor_mel_offsets[m];k<donor_mel_offsets[m+1];k++)energy+=s->re[donor_mel_bins[k]]*donor_mel_weights[k];
  if(t)t->mel[m]=energy;
  output[m]=logf(fmaxf(energy,FLT_EPSILON));
 }
 if(t)memcpy(t->logfbank,output,sizeof(t->logfbank));
}
size_t donor_fft64_state_bytes(void){return sizeof(donor_fft64_state);}
int donor_fft64_init(donor_fft64_state*s){
 if(!s)return DONOR_FFT64_ARGUMENT;
 if(!donor_fft64_environment_ok())return DONOR_FFT64_STATE;
 memset(s,0,sizeof(*s));s->initialized=DONOR_MAGIC;
 return DONOR_FFT64_OK;
}
int donor_fft64_reset(donor_fft64_state*s){return donor_fft64_init(s);}
int donor_fft64_feed(donor_fft64_state*s,const int16_t*pcm,size_t count,donor_fft64_callback cb,void*user){
 if(!s||(!pcm&&count)||!cb||count>DONOR_FFT64_MAX_FEED)return DONOR_FFT64_ARGUMENT;
 if(s->initialized!=DONOR_MAGIC||s->finished||!donor_fft64_environment_ok())return DONOR_FFT64_STATE;
 if((uint64_t)count>UINT64_MAX-s->total_samples)return DONOR_FFT64_ARGUMENT;
 for(size_t i=0;i<count;i++){
  s->pcm[s->used++]=pcm[i];s->total_samples++;
  if(s->used==400){
   float row[80];analyze(s,s->pcm,row,NULL);
   cb(user,row,s->frame_index,s->total_samples);s->frame_index++;
   memmove(s->pcm,s->pcm+160,240*sizeof(int16_t));s->used=240;
  }
 }
 return DONOR_FFT64_OK;
}
int donor_fft64_finish(donor_fft64_state*s){
 if(!s)return DONOR_FFT64_ARGUMENT;
 if(s->initialized!=DONOR_MAGIC||s->finished||!donor_fft64_environment_ok())return DONOR_FFT64_STATE;
 s->used=0;s->finished=1;return DONOR_FFT64_OK;
}
int donor_fft64_analyze_frame(donor_fft64_state*s,const int16_t pcm[400],donor_fft64_trace*t){
 if(!s||!pcm||!t)return DONOR_FFT64_ARGUMENT;
 if(s->initialized!=DONOR_MAGIC||!donor_fft64_environment_ok())return DONOR_FFT64_STATE;
 float row[80];analyze(s,pcm,row,t);return DONOR_FFT64_OK;
}
static int overlaps400(const float*a,const float*b){
 uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;
 return (x<=y?y-x:x-y)<400*sizeof(float);
}
int donor_fft64_cmvn400(const float x[400],const float mean[400],const float istd[400],float out[400]){
 if(!x||!mean||!istd||!out)return DONOR_FFT64_ARGUMENT;
 if(!donor_fft64_environment_ok())return DONOR_FFT64_STATE;
 if((x!=out&&overlaps400(x,out))||overlaps400(mean,out)||overlaps400(istd,out))return DONOR_FFT64_ARGUMENT;
 for(unsigned i=0;i<400;i++)if(!isfinite(x[i])||!isfinite(mean[i])||!isfinite(istd[i])||istd[i]<=0.0f)return DONOR_FFT64_ARGUMENT;
 for(unsigned i=0;i<400;i++)if(!isfinite((x[i]-mean[i])*istd[i]))return DONOR_FFT64_ARGUMENT;
 for(unsigned i=0;i<400;i++)out[i]=(x[i]-mean[i])*istd[i];
 return DONOR_FFT64_OK;
}
