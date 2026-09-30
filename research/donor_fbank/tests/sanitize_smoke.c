#include "donor_fbank.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <string.h>
static unsigned frames;
static void receive(void *u,const float row[80],uint64_t index,uint64_t end){
 (void)u;assert(index==frames);assert(end==400+160*index);frames++;
 for(unsigned i=0;i<80;i++)assert(isfinite(row[i]));
}
int main(void){
 donor_fbank_state state,before;int16_t pcm[16000];
 for(unsigned i=0;i<16000;i++)pcm[i]=(int16_t)((int)((i*977u)%65536u)-32768);
 assert(donor_fbank_init(&state)==0);before=state;
 assert(donor_fbank_feed(&state,pcm,16001,receive,0)==DONOR_FBANK_ARGUMENT);
 assert(!memcmp(&state,&before,sizeof(state)));
 for(unsigned pos=0;pos<16000;){unsigned n=pos%641+1;if(n>16000-pos)n=16000-pos;assert(donor_fbank_feed(&state,pcm+pos,n,receive,0)==0);pos+=n;}
 assert(frames==98);assert(donor_fbank_finish(&state)==0);assert(donor_fbank_finish(&state)==DONOR_FBANK_STATE);
 assert(donor_fbank_reset(&state)==0);frames=0;assert(donor_fbank_feed(&state,pcm,400,receive,0)==0);assert(frames==1);
 donor_fbank_trace trace;assert(donor_fbank_analyze_frame(&state,pcm,&trace)==0);
 float impulse[512]={0},power[257];impulse[0]=1;assert(donor_fbank_fft_power(&state,impulse,power)==0);for(unsigned i=0;i<257;i++)assert(power[i]==1);
 float x[400]={0},mean[400]={0},std[400],out[400];for(unsigned i=0;i<400;i++){std[i]=1;out[i]=123;}
 x[399]=FLT_MAX;mean[399]=-FLT_MAX;assert(donor_cmvn400(x,mean,std,out)==DONOR_FBANK_ARGUMENT);for(unsigned i=0;i<400;i++)assert(out[i]==123);
 return 0;
}
