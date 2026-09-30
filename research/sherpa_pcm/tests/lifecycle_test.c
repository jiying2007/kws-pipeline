#include "pcm_kws.h"
#include <assert.h>
#include <string.h>
static int hits;static uint64_t last;
static void event(void*u,const PcmKwsEvent*e){(void)u;assert(!strcmp(e->keyword,"你好小窝"));assert(e->keyword_id==1);assert(e->eof_flush==0);hits++;last=e->available_samples;}
int main(void){
 PcmKwsFiles f={"e","d","j","t","k"};PcmKws*x=NULL;char err[128];int16_t pcm[640]={0};
 assert(pcm_kws_create(NULL,event,NULL,&x,err,sizeof(err))==PCM_KWS_ARGUMENT);assert(!x);pcm_kws_destroy(NULL);
 f.encoder="fail";assert(pcm_kws_create(&f,event,NULL,&x,err,sizeof(err))==PCM_KWS_RUNTIME);assert(!x);f.encoder="e";
 for(int cycle=0;cycle<3;cycle++){
  assert(!pcm_kws_create(&f,event,NULL,&x,err,sizeof(err)));hits=0;
  assert(pcm_kws_feed(x,NULL,1,err,sizeof(err))==PCM_KWS_ARGUMENT);assert(!pcm_kws_feed(x,NULL,0,err,sizeof(err)));
  pcm[0]=-32768;assert(pcm_kws_feed(x,pcm,320,err,sizeof(err))==PCM_KWS_RUNTIME);assert(pcm_kws_feed(x,pcm,1,err,sizeof(err))==PCM_KWS_STATE);assert(!pcm_kws_reset(x,err,sizeof(err)));pcm[0]=0;
  assert(!pcm_kws_feed(x,pcm,1,err,sizeof(err)));assert(!pcm_kws_feed(x,pcm,318,err,sizeof(err)));assert(!hits);
  assert(!pcm_kws_feed(x,pcm,321,err,sizeof(err)));assert(hits==1&&last==640);
  assert(!pcm_kws_finish(x,err,sizeof(err)));assert(pcm_kws_finish(x,err,sizeof(err))==PCM_KWS_STATE);assert(pcm_kws_feed(x,pcm,1,err,sizeof(err))==PCM_KWS_STATE);
  assert(!pcm_kws_reset(x,err,sizeof(err)));hits=0;assert(!pcm_kws_feed(x,pcm,169,err,sizeof(err)));assert(!pcm_kws_reset(x,err,sizeof(err)));
  assert(!pcm_kws_feed(x,pcm,639,err,sizeof(err)));assert(!hits);assert(!pcm_kws_finish(x,err,sizeof(err)));assert(!hits);
  assert(!pcm_kws_reset(x,err,sizeof(err)));assert(!pcm_kws_feed(x,pcm,640,err,sizeof(err)));assert(hits==1&&last==640);pcm_kws_destroy(x);x=NULL;
 }
 return 0;
}
