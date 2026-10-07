#include "a20_stream.h"
#include <string.h>
#define A20S_MAGIC UINT32_C(0x41323053)
size_t a20s_state_bytes(void){return sizeof(a20s_state);}
int a20s_init(a20s_state*s,const float*w,size_t n){
 if(!s)return -1;
 memset(s,0,sizeof(*s));s->fault=1;
 int rc=a20_init_verified(&s->model,w,n);if(rc)return rc;
 if(donor_pcm_init(&s->pcm)||a20d_init(&s->decoder))return -2;
 s->initialized=A20S_MAGIC;s->fault=0;return 0;
}
int a20s_reset(a20s_state*s){
 if(!s||s->initialized!=A20S_MAGIC||s->busy)return -1;
 a20_reset(&s->model);
 if(donor_pcm_reset(&s->pcm)||a20d_reset_all(&s->decoder)){s->fault=1;return -2;}
 memset(s->logits,0,sizeof(s->logits));s->fault=0;return 0;
}
int a20s_set_trace(a20s_state*s,float*t,a20s_trace_callback cb,void*u){
 if(!s||s->initialized!=A20S_MAGIC||s->busy||((t==NULL)!=(cb==NULL)))return -1;
 s->trace_buffer=t;s->trace_callback=cb;s->trace_user=u;return 0;
}
typedef struct {a20s_state*s;a20s_callback cb;void*user;} context;
static void accept_batch(void*u,const donor_pcm_batch*b){
 context*c=u;a20s_state*s=c->s;if(s->fault)return;
 if(b->selected_rows>DONOR_PCM_MAX_SELECTED_ROWS){s->fault=1;return;}
 for(size_t i=0;i<b->selected_rows;i++){
  if(a20_step(&s->model,b->rows+i*400,s->logits[i],s->trace_buffer)){s->fault=1;return;}
  if(s->trace_callback)s->trace_callback(s->trace_user,b->call_index,i,b->rows+i*400,s->trace_buffer,s->model.cache);
 }
 a20d_result result;
 if(a20d_process_logits(&s->decoder,&s->decoder_work,&s->logits[0][0],b->selected_rows,&result)){s->fault=1;return;}
 a20s_batch out={b,&s->logits[0][0],s->model.cache,&result};c->cb(c->user,&out);
}
int a20s_feed(a20s_state*s,const int16_t*pcm,size_t n,a20s_callback cb,void*u){
 if(!s||s->initialized!=A20S_MAGIC||s->fault||s->busy||!cb)return -1;
 s->busy=1;context c={s,cb,u};int rc=donor_pcm_feed(&s->pcm,pcm,n,accept_batch,&c);s->busy=0;
 if(rc||s->fault){s->fault=1;return -2;}return 0;
}
int a20s_finish(a20s_state*s,a20s_callback cb,void*u){
 if(!s||s->initialized!=A20S_MAGIC||s->fault||s->busy||!cb)return -1;
 s->busy=1;context c={s,cb,u};int rc=donor_pcm_finish(&s->pcm,accept_batch,&c);s->busy=0;
 if(rc||s->fault){s->fault=1;return -2;}return 0;
}
