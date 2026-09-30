#include "donor_fbank.h"
#include <float.h>
#include <math.h>
#include <string.h>
#include "frontend_tables.h"
#include "fft_twiddles.h"
#define DONOR_MAGIC UINT32_C(0x46423830)
_Static_assert(sizeof(float)==4 && FLT_RADIX==2 && FLT_MANT_DIG==24,"IEEE float32 required");

/* Isolated adaptation of the repository's Apache2 src/frontend.c fft512
 * radix2 skeleton. Product math is neither linked nor modified. */
static void fft512(donor_fbank_state *s) {
 unsigned j=0;
 for(unsigned i=1;i<512;i++){
  unsigned bit=256;
  while((j&bit)!=0){j^=bit;bit>>=1;}
  j^=bit;
  if(i<j){float v=s->re[i];s->re[i]=s->re[j];s->re[j]=v;v=s->im[i];s->im[i]=s->im[j];s->im[j]=v;}
 }
 for(unsigned len=2;len<=512;len<<=1){
  for(unsigned i=0;i<512;i+=len){
   for(unsigned k=0;k<len/2;k++){
    unsigned a=i+k,b=a+len/2;
    unsigned t=k*(512/len);float wr=donor_fft_cos[t],wi=donor_fft_sin[t];
    float vr=s->re[b]*wr-s->im[b]*wi,vi=s->re[b]*wi+s->im[b]*wr;
    float ur=s->re[a],ui=s->im[a];
    s->re[a]=ur+vr;s->im[a]=ui+vi;s->re[b]=ur-vr;s->im[b]=ui-vi;

   }
  }
 }
}
int donor_fbank_fft_power(donor_fbank_state*s,const float input[512],float output[257]){
 if(!s||!input||!output)return DONOR_FBANK_ARGUMENT;
 if(s->initialized!=DONOR_MAGIC)return DONOR_FBANK_STATE;
 for(unsigned i=0;i<512;i++)if(!isfinite(input[i])||fabsf(input[i])>65536.0f)return DONOR_FBANK_ARGUMENT;
 memcpy(s->re,input,512*sizeof(float));memset(s->im,0,sizeof(s->im));fft512(s);
 for(unsigned i=0;i<257;i++)output[i]=s->re[i]*s->re[i]+s->im[i]*s->im[i];
 return DONOR_FBANK_OK;
}
static void analyze(donor_fbank_state*s,const int16_t*pcm,float output[80],donor_fbank_trace*t){
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
size_t donor_fbank_state_bytes(void){return sizeof(donor_fbank_state);}
int donor_fbank_init(donor_fbank_state*s){
 if(!s)return DONOR_FBANK_ARGUMENT;
 memset(s,0,sizeof(*s));s->initialized=DONOR_MAGIC;
 return DONOR_FBANK_OK;
}
int donor_fbank_reset(donor_fbank_state*s){return donor_fbank_init(s);}
int donor_fbank_feed(donor_fbank_state*s,const int16_t*pcm,size_t count,donor_fbank_callback cb,void*user){
 if(!s||(!pcm&&count)||!cb||count>DONOR_FBANK_MAX_FEED)return DONOR_FBANK_ARGUMENT;
 if(s->initialized!=DONOR_MAGIC||s->finished)return DONOR_FBANK_STATE;
 if((uint64_t)count>UINT64_MAX-s->total_samples)return DONOR_FBANK_ARGUMENT;
 for(size_t i=0;i<count;i++){
  s->pcm[s->used++]=pcm[i];s->total_samples++;
  if(s->used==400){
   float row[80];analyze(s,s->pcm,row,NULL);
   cb(user,row,s->frame_index,s->total_samples);s->frame_index++;
   memmove(s->pcm,s->pcm+160,240*sizeof(int16_t));s->used=240;
  }
 }
 return DONOR_FBANK_OK;
}
int donor_fbank_finish(donor_fbank_state*s){
 if(!s)return DONOR_FBANK_ARGUMENT;
 if(s->initialized!=DONOR_MAGIC||s->finished)return DONOR_FBANK_STATE;
 s->used=0;s->finished=1;return DONOR_FBANK_OK;
}
int donor_fbank_analyze_frame(donor_fbank_state*s,const int16_t pcm[400],donor_fbank_trace*t){
 if(!s||!pcm||!t)return DONOR_FBANK_ARGUMENT;
 if(s->initialized!=DONOR_MAGIC)return DONOR_FBANK_STATE;
 float row[80];analyze(s,pcm,row,t);return DONOR_FBANK_OK;
}
static int overlaps400(const float*a,const float*b){
 uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;
 return (x<=y?y-x:x-y)<400*sizeof(float);
}
int donor_cmvn400(const float x[400],const float mean[400],const float istd[400],float out[400]){
 if(!x||!mean||!istd||!out)return DONOR_FBANK_ARGUMENT;
 if((x!=out&&overlaps400(x,out))||overlaps400(mean,out)||overlaps400(istd,out))return DONOR_FBANK_ARGUMENT;
 for(unsigned i=0;i<400;i++)if(!isfinite(x[i])||!isfinite(mean[i])||!isfinite(istd[i])||istd[i]<=0.0f)return DONOR_FBANK_ARGUMENT;
 for(unsigned i=0;i<400;i++)if(!isfinite((x[i]-mean[i])*istd[i]))return DONOR_FBANK_ARGUMENT;
 for(unsigned i=0;i<400;i++)out[i]=(x[i]-mean[i])*istd[i];
 return DONOR_FBANK_OK;
}
