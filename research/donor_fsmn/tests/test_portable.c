/* No donor assets: exact-power-of-two synthetic weights and PCM silence. */
#include "../fsmn.c"
#include "../pcm.h"
#include <assert.h>
#include <float.h>
#include <stdlib.h>
typedef struct { unsigned calls; size_t rows; uint64_t center; } observed;
static void batch(void*u,const donor_pcm_batch*b){
 observed*o=u;assert(b->call_index==o->calls);assert(b->available_samples==(uint64_t)(o->calls+1)*4800);
 assert(b->selected_rows==(o->calls==0?9u:10u));assert(b->wave_samples==320);assert(b->feature_count==4);assert(b->offset==1);
 for(size_t i=0;i<b->selected_rows;i++){assert(b->centers[i]==o->center);o->center+=3;for(size_t j=0;j<400;j++)assert(isfinite(b->rows[i*400+j]));}
 o->calls++;o->rows+=b->selected_rows;
}
int main(void){
 float*w=calloc(DF_FLOATS,sizeof(float));assert(w);for(int i=400;i<800;i++)w[i]=1;
 /* Two successive negative affines must NOT acquire an extra ReLU. */
 w[800]=-1;w[56940]=-1;
 for(int l=0;l<4;l++){size_t off=92190u+(size_t)l*65786u;w[off]=1;w[off+33536]=1;}
 w[355334]=-1;w[390474]=-1;
 df_model a,b;assert(df_init(&a,w,DF_FLOATS)==0);assert(df_init(&b,w,DF_FLOATS)==0);
 for(int t=0;t<25;t++){
  float x[400]={0},y[2599],z[2599];x[0]=(float)(t+1);assert(df_step(&a,x,y,NULL)==0);assert(df_step(&b,x,z,NULL)==0);
  assert(y[0]==(t>=8?(float)(t-7):0));for(int i=1;i<2599;i++)assert(y[i]==0);assert(memcmp(y,z,sizeof(y))==0);
 }
 df_reset(&a);assert(a.fault==0);for(size_t i=0;i<DF_CACHE_FLOATS;i++)assert(a.cache[i]==0);
 assert(df_init(&a,w,DF_FLOATS-1)==-1);assert(a.weights==NULL);
 w[0]=NAN;assert(df_init(&a,w,DF_FLOATS)==-2);w[0]=0;
 /* Exact finite-memory tap order and chronological cache. */
 memset(w,0,DF_FLOATS*sizeof(float));assert(df_init(&a,w,DF_FLOATS)==0);
 for(int k=0;k<10;k++)w[124190+k]=(float)(k+1)/32;
 w[125470]=.25f;w[125471]=.5f;
 for(int t=0;t<17;t++){
  float p[128]={0},out[128];p[0]=(float)(t+1)/8;float left=0;
  for(int k=0;k<10;k++){int idx=t-11+k;left+=w[124190+k]*(idx>=0?(float)(idx+1)/8:0);}
  float right=.25f*(t>=1?(float)t/8:0)+.5f*p[0];float expected=(t>=2?(float)(t-1)/8:0)+left+right;
  memory(&a,0,92190,p,out);assert(out[0]==expected);
  for(int k=0;k<11;k++){int idx=t-10+k;assert(a.cache[k*4]==(idx>=0?(float)(idx+1)/8:0));}
 }
 /* Canonical grouping regardless of application ingress partitions. */
 donor_pcm_state *pcm=malloc(sizeof(*pcm));assert(pcm);assert(donor_pcm_init(pcm)==0);observed o={0};int16_t zeros[9600]={0};
 assert(donor_pcm_feed(pcm,zeros,17,batch,&o)==0);assert(o.calls==0);
 assert(donor_pcm_feed(pcm,zeros+17,9583,batch,&o)==0);assert(o.calls==2&&o.rows==19);
 assert(donor_pcm_finish(pcm,batch,&o)==0);assert(o.calls==2);assert(donor_pcm_finish(pcm,batch,&o)==DONOR_PCM_STATE);
 assert(donor_pcm_reset(pcm)==0);o=(observed){0};assert(donor_pcm_feed(pcm,zeros,4800,batch,&o)==0);assert(o.calls==1);
 free(pcm);free(w);return 0;
}
