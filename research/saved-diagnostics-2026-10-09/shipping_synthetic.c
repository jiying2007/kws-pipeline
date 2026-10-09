/* Invented logits, not reproduction of archived audio/model or old FA cause. */
#include "decoder.h"
#include <assert.h>
#include <stdio.h>
static kws_decoder_t d;
static void setup(void){
 static const uint16_t tokens[]={3,4,3,4};kws_keyword_t k={0};k.id=2;k.tokens=tokens;k.num_tokens=4;k.threshold=.55f;k.prefix_policy=KWS_PREFIX_IMMEDIATE;
 kws_decoder_init(&d,1.5f,.94f);assert(kws_decoder_set_keywords(&d,&k,1,6)==KWS_OK);
}
static int step(int token,int speech){float logits[6];for(int j=0;j<6;j++)logits[j]=j==token?8.0f:-8.0f;uint32_t id;float confidence;return kws_decoder_step(&d,logits,6,speech,&id,&confidence);}
static int seq(const int *p,int n){int hits=0;for(int i=0;i<n;i++){hits+=step(p[i],1);if(i<n-1)hits+=step(0,1);}return hits;}
int main(void){
 int a[]={3,3,3,4},b[]={3,4,3,1},prior[]={3,2,3,4};setup();int a_hits=seq(a,4);setup();int b_hits=seq(b,4);
 setup();int prior_hits=seq(prior,4);for(int i=0;i<19;i++)step(0,0);int inactive_hits=seq(b,4);
 setup();seq(prior,4);for(int i=0;i<19;i++)step(0,1);int active_hits=seq(b,4);
 assert(a_hits==0&&b_hits==0&&prior_hits==0&&inactive_hits==0);
 printf("{\"invented_logits_only\":true,\"negative_3334_hits\":%d,\"confusable_3431_hits\":%d,\"prior_3234_hits\":%d,\"19_inactive_blank_gap_then_3431_hits\":%d,\"19_active_blank_gap_then_3431_hits\":%d,\"original_FA_reproduced\":false}\n",a_hits,b_hits,prior_hits,inactive_hits,active_hits);return 0;
}
