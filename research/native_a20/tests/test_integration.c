/* Pure invented coefficients only. Test-only identity substitution isolates
 * integration geometry; production a20fft64s_init remains SHA-verified. */
#include "../src/a20_stream_fft64.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
static int synthetic_init(a20_model*m,const float*w,size_t n){return a20_init(m,w,n);}
#define a20_init_verified synthetic_init
#include "../src/a20_stream_fft64.c"
#undef a20_init_verified
typedef struct {size_t calls,rows;uint64_t samples;float logits[128][6];} observed;
static void callback(void*user,const a20fft64s_batch*b){
 observed*o=user;assert(b->frontend->call_index==o->calls);o->calls++;
 assert(b->result->state==0&&b->result->keyword==0);assert(b->frontend->selected_rows<=11);
 assert(o->rows+b->frontend->selected_rows<=128);
 for(size_t i=0;i<b->frontend->selected_rows;i++){
  assert(b->logits[i*6]==8.0f);for(size_t j=1;j<6;j++)assert(b->logits[i*6+j]==0.0f);
  memcpy(o->logits[o->rows++],b->logits+i*6,6*sizeof(float));
 }
 o->samples=b->frontend->available_samples;
}
typedef struct {size_t rows;a20fft64s_state*state;} trace_observed;
static void trace_callback(void*u,uint64_t call,size_t row,const float*x,const float*t,const float*cache){
 trace_observed*o=u;(void)call;(void)row;(void)x;assert(t[20*250]==8.0f);
 for(size_t i=0;i<A20_CACHE_FLOATS;i++)assert(cache[i]==0.0f);
 assert(a20fft64s_reset(o->state)==-1);o->rows++;
}
int main(void){
 float*w=calloc(A20_FLOATS,sizeof(float));a20fft64s_state*a=malloc(sizeof(*a));a20fft64s_state*b=malloc(sizeof(*b));int16_t*p=calloc(10500,sizeof(*p));assert(w&&a&&b&&p);
 for(size_t i=400;i<800;i++)w[i]=1;
 w[391314]=8; /* constant blank output; self-created, no checkpoint */
 assert(a20fft64s_init(a,w,A20_FLOATS)==0&&a20fft64s_init(b,w,A20_FLOATS)==0);
 float trace[21*250]={0};trace_observed trace_count={0,a};
 assert(a20fft64s_set_trace(a,trace,NULL,&trace_count)==-1);
 assert(a20fft64s_set_trace(a,trace,trace_callback,&trace_count)==0);
 observed x={0},y={0};assert(a20fft64s_feed(a,p,10500,callback,&x)==0);assert(x.calls==2&&x.rows==19);assert(a20fft64s_finish(a,callback,&x)==0);
 size_t pos=0;const size_t parts[]={1,13,799,477,4801,3,4406};for(size_t i=0;i<7;i++){assert(a20fft64s_feed(b,p+pos,parts[i],callback,&y)==0);pos+=parts[i];}assert(pos==10500);assert(a20fft64s_finish(b,callback,&y)==0);
 assert(trace_count.rows==x.rows);assert(x.calls==y.calls&&x.rows==y.rows&&x.samples==10500&&y.samples==10500);assert(memcmp(x.logits,y.logits,sizeof(x.logits))==0);assert(memcmp(a->model.cache,b->model.cache,sizeof(a->model.cache))==0);
 assert(a20fft64s_finish(a,callback,&x)!=0);assert(a20fft64s_feed(a,p,1,callback,&x)!=0);assert(a20fft64s_reset(a)==0);observed z={0};assert(a20fft64s_feed(a,p,799,callback,&z)==0);assert(a20fft64s_finish(a,callback,&z)==0);assert(z.calls==1&&z.rows==0);assert(a20fft64s_reset(a)==0);
 assert(a->decoder.total_frames==0&&a->decoder.last_active_pos==-1);for(size_t i=0;i<A20_CACHE_FLOATS;i++)assert(a->model.cache[i]==0);
 printf("PASS invented-coefficient integrated geometry, reset, no-pad tail and partition; state_bytes=%zu\n",sizeof(*a));free(p);free(a);free(b);free(w);return 0;
}
