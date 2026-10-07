#define _POSIX_C_SOURCE 200809L
#include "../runtime/src/a20_stream_fft64.h"
#include "inputs.h"
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>
#include <unistd.h>
#include <dlfcn.h>
/* One complete N1 leading-plus-tail-context stream; original A20 only, verified loader unchanged. */
static int (*run_init)(a20fft64s_state*,const float*,size_t);
static int (*run_reset)(a20fft64s_state*);
static int (*run_feed)(a20fft64s_state*,const int16_t*,size_t,a20fft64s_callback,void*);
static int (*run_finish)(a20fft64s_state*,a20fft64s_callback,void*);
static size_t (*run_state_bytes)(void);
static uint64_t ns(clockid_t id){struct timespec t;if(clock_gettime(id,&t))exit(90);return (uint64_t)t.tv_sec*1000000000+(uint64_t)t.tv_nsec;}
typedef struct {a20fft64s_state*s;const clip_spec*c;size_t cb,fb,mr,dr,ev;int bad;const char*phase;uint64_t startwall,startcpu;} context;
static int valid_batch(const a20fft64s_batch*b){
 if(!b||!b->frontend||!b->result)return 0;
 const donor_fft64_pcm_batch*f=b->frontend;const a20d_result*r=b->result;
 if(f->selected_rows>10||f->fbank_rows>DONOR_FFT64_PCM_MAX_FBANK_ROWS||f->call_samples>4800||f->call_samples==0||f->waveform_samples>DONOR_FFT64_PCM_MAX_WAVE_SAMPLES)return 0;
 if(f->selected_rows&&(!b->logits||!f->centers))return 0;
 if(r->valid!=(f->selected_rows!=0)||r->state<0||r->state>1||!isfinite(r->score)||r->rows_decoded>f->selected_rows)return 0;
 if(!f->selected_rows&&(r->state||r->keyword||r->start_frame||r->end_frame||r->score||r->rows_decoded))return 0;
 if(f->selected_rows&&r->rows_decoded==0)return 0;
 if(r->state){if(r->keyword<1||r->keyword>2||r->start_frame<0||r->end_frame<r->start_frame)return 0;}
 else if(f->selected_rows&&(r->keyword||r->start_frame!=-1||r->end_frame!=-1||r->rows_decoded!=f->selected_rows))return 0;
 for(size_t i=0;i<f->selected_rows*6;i++)if(!isfinite(b->logits[i]))return 0;
 return 1;
}
static void report(void*u,const a20fft64s_batch*b){
 context*c=u;uint64_t entry_wall=ns(CLOCK_MONOTONIC)-c->startwall,entry_cpu=ns(CLOCK_PROCESS_CPUTIME_ID)-c->startcpu;if(c->bad)return;if(!valid_batch(b)){c->bad=1;return;}
 const donor_fft64_pcm_batch*f=b->frontend;const a20d_result*r=b->result;
 /* Fixed expected geometry generated without runtime/audio execution. */
 if(c->cb>=c->c->callback_count){c->bad=1;return;}
 const callback_spec*p=&c->c->plan[c->cb];
 if(strcmp(c->phase,p->phase)||f->call_index!=p->call_index||
    f->available_samples!=p->available_samples||f->call_samples!=p->call_samples||
    f->waveform_samples!=p->waveform_samples||f->fbank_rows!=p->fbank_rows||
    f->splice_rows!=p->splice_rows||f->selected_rows!=p->selected_rows||
    f->is_final_short!=p->is_final_short||f->wave_samples!=p->wave_samples||
    f->feature_count!=p->feature_count||f->offset!=p->offset||
    c->s->decoder.total_frames!=(int64_t)p->decoder_total_frames||
    (r->state&&(r->end_frame>=c->s->decoder.total_frames||r->start_frame>=c->s->decoder.total_frames))) {c->bad=1;return;}
 for(size_t i=0;i<f->selected_rows;i++)if(f->centers[i]!=p->centers[i]){c->bad=1;return;}
 printf("{\"kind\":\"callback\",\"recording\":\"%s\",\"phase\":\"%s\",\"call_index\":%"PRIu64",\"available_samples\":%"PRIu64",\"call_samples\":%zu,\"waveform_samples\":%zu,\"fbank_rows\":%zu,\"splice_rows\":%zu,\"selected_rows\":%zu,\"is_final_short\":%u,\"wave_samples\":%u,\"feature_count\":%u,\"offset\":%u,\"centers\":[",c->c->id,c->phase,f->call_index,f->available_samples,f->call_samples,f->waveform_samples,f->fbank_rows,f->splice_rows,f->selected_rows,f->is_final_short,f->wave_samples,f->feature_count,f->offset);
 for(size_t i=0;i<f->selected_rows;i++)printf("%s%"PRIu64,i?",":"",f->centers[i]);
 printf("],\"valid\":%d,\"state\":%d,\"keyword\":%d,\"start_frame\":%"PRId64",\"end_frame\":%"PRId64",\"score\":%.17g,\"decoder_rows_decoded\":%zu,\"decoder_total_frames\":%"PRId64",\"callback_entry_wall_ns_since_run_start\":%"PRIu64",\"callback_entry_cpu_ns_since_run_start\":%"PRIu64",\"logits\":[",r->valid,r->state,r->keyword,r->start_frame,r->end_frame,r->score,r->rows_decoded,c->s->decoder.total_frames,entry_wall,entry_cpu);
 for(size_t i=0;i<f->selected_rows;i++){printf("%s[",i?",":"");for(size_t j=0;j<6;j++)printf("%s%.9g",j?",":"",b->logits[i*6+j]);printf("]");}
 printf("]}\n"); c->cb++;c->fb+=f->fbank_rows;c->mr+=f->selected_rows;c->dr+=r->rows_decoded;c->ev+=(size_t)r->state;
 if(fflush(stdout)||ferror(stdout))c->bad=1;
}
static int digest(const char*s){if(strlen(s)!=64||strspn(s,"0")==64)return 0;for(size_t i=0;i<64;i++)if(!((s[i]>='0'&&s[i]<='9')||(s[i]>='a'&&s[i]<='f')))return 0;return 1;}
int main(int argc,char**argv){
 /* argc gate executes before dlopen, input reads, model or DSP. */
 if(argc!=12||strcmp(argv[1],"--execute-after-reviewed-release")){fputs("DISARMED: exact reviewed one-stream N1 context release required\n",stderr);return 77;}
 for(int i=2;i<8;i++)if(!digest(argv[i]))return 2;
 if(strcmp(argv[11],"original_A20"))return 2;
 void*lib=dlopen(argv[9],RTLD_NOW|RTLD_LOCAL);if(!lib)return 91;
 *(void**)(&run_init)=dlsym(lib,"a20fft64s_init");
 *(void**)(&run_reset)=dlsym(lib,"a20fft64s_reset");
 *(void**)(&run_feed)=dlsym(lib,"a20fft64s_feed");
 *(void**)(&run_finish)=dlsym(lib,"a20fft64s_finish");
 *(void**)(&run_state_bytes)=dlsym(lib,"a20fft64s_state_bytes");
 if(!run_init||!run_reset||!run_feed||!run_finish||!run_state_bytes||run_state_bytes()!=sizeof(a20fft64s_state))return 92;
 /* Start barrier is a supervisor-owned pipe; no model/audio work before GO. */
 char go=0;if(read(atoi(argv[10]),&go,1)!=1||go!='G')return 93;
 uint64_t startwall=ns(CLOCK_MONOTONIC),startcpu=ns(CLOCK_PROCESS_CPUTIME_ID);
 printf("{\"kind\":\"run_start\",\"schema\":\"n1-wholeclip-leading-a20-raw-v1\",\"protocol_sha256\":\"%s\",\"manifest_sha256\":\"%s\",\"geometry_sha256\":\"%s\",\"decoder_config_sha256\":\"%s\",\"model_sha256\":\"%s\",\"library_sha256\":\"%s\",\"pid\":%ld}\n",argv[2],argv[3],argv[4],argv[5],argv[6],argv[7],(long)getpid());fflush(stdout);
 float*w=malloc(A20_FLOATS*sizeof(float));a20fft64s_state*s=malloc(sizeof(*s));if(!w||!s)return 3;
 FILE*mf=fopen(argv[8],"rb");if(!mf)return 4;
 size_t wn=fread(w,sizeof(float),A20_FLOATS,mf);int extra=fgetc(mf),err=ferror(mf);fclose(mf);
 if(wn!=A20_FLOATS||extra!=EOF||err||run_init(s,w,wn))return 5;
 printf("{\"kind\":\"model_loaded\",\"wall_ns_since_run_start\":%"PRIu64",\"cpu_ns_since_run_start\":%"PRIu64",\"state_bytes\":%zu,\"weights_bytes\":%zu}\n",ns(CLOCK_MONOTONIC)-startwall,ns(CLOCK_PROCESS_CPUTIME_ID)-startcpu,sizeof(*s),(size_t)A20_FLOATS*sizeof(float));
 size_t feeds=0,callbacks=0,fb=0,mr=0,dr=0,ev=0;
 for(size_t k=0;k<CLIP_COUNT;k++){
  const clip_spec*c=&clips[k];if(run_reset(s))return 6;
  printf("{\"kind\":\"clip_start\",\"recording\":\"%s\",\"frames\":%zu,\"wav_sha256\":\"%s\",\"pcm_sha256\":\"%s\"}\n",c->id,c->frames,c->wav_sha,c->pcm_sha);fflush(stdout);
  FILE*f=fopen(c->pcm,"rb");if(!f||fseek(f,(long)c->data_offset,SEEK_SET))return 7;
  context ctx={.s=s,.c=c,.phase="feed",.startwall=startwall,.startcpu=startcpu};
  unsigned char raw[9600];int16_t pcm[4800];size_t got=0,feed=0;
  while(got<c->frames){
   size_t count=c->frames-got;if(count>4800)count=4800;
   if(fread(raw,1,count*2,f)!=count*2)return 8;
   for(size_t i=0;i<count;i++)pcm[i]=(int16_t)((uint16_t)raw[2*i]|((uint16_t)raw[2*i+1]<<8));
   size_t oldcb=ctx.cb;uint64_t tw=ns(CLOCK_MONOTONIC),tc=ns(CLOCK_PROCESS_CPUTIME_ID);
   int rc=run_feed(s,pcm,count,report,&ctx);uint64_t dc=ns(CLOCK_PROCESS_CPUTIME_ID)-tc,dw=ns(CLOCK_MONOTONIC)-tw;
   if(rc||ctx.bad)return 9;
   got+=count;
   printf("{\"kind\":\"feed\",\"recording\":\"%s\",\"feed_index\":%zu,\"input_samples\":%zu,\"cumulative_samples\":%zu,\"callbacks_delta\":%zu,\"service_wall_ns_including_callback_output\":%"PRIu64",\"service_cpu_ns_including_callback_output\":%"PRIu64"}\n",c->id,feed++,count,got,ctx.cb-oldcb,dw,dc);fflush(stdout);
  }
  if(fgetc(f)!=EOF||ferror(f))return 10;
  fclose(f);
  ctx.phase="finish";size_t oldcb=ctx.cb;uint64_t tw=ns(CLOCK_MONOTONIC),tc=ns(CLOCK_PROCESS_CPUTIME_ID);
  int rc=run_finish(s,report,&ctx);uint64_t dc=ns(CLOCK_PROCESS_CPUTIME_ID)-tc,dw=ns(CLOCK_MONOTONIC)-tw;
  if(rc||ctx.bad||ctx.cb!=c->callback_count)return 11;
  printf("{\"kind\":\"finish\",\"recording\":\"%s\",\"finish_calls\":1,\"callbacks_delta\":%zu,\"service_wall_ns_including_callback_output\":%"PRIu64",\"service_cpu_ns_including_callback_output\":%"PRIu64"}\n",c->id,ctx.cb-oldcb,dw,dc);
  printf("{\"kind\":\"clip_end\",\"recording\":\"%s\",\"frames\":%zu,\"wav_sha256\":\"%s\",\"pcm_sha256\":\"%s\",\"feed_calls\":%zu,\"finish_calls\":1,\"callbacks\":%zu,\"fbank_rows\":%zu,\"model_rows\":%zu,\"decoder_rows_decoded\":%zu,\"event_count\":%zu,\"complete\":true}\n",c->id,got,c->wav_sha,c->pcm_sha,feed,ctx.cb,ctx.fb,ctx.mr,ctx.dr,ctx.ev);fflush(stdout);
  feeds+=feed;callbacks+=ctx.cb;fb+=ctx.fb;mr+=ctx.mr;dr+=ctx.dr;ev+=ctx.ev;
 }
 struct rusage ru;if(getrusage(RUSAGE_SELF,&ru))return 12;
 printf("{\"kind\":\"run_end\",\"complete\":true,\"clips\":%d,\"frames\":%d,\"feed_calls\":%zu,\"finish_calls\":%d,\"callbacks\":%zu,\"fbank_rows\":%zu,\"model_rows\":%zu,\"decoder_rows_decoded\":%zu,\"event_count\":%zu,\"wall_ns\":%"PRIu64",\"process_cpu_ns\":%"PRIu64",\"maxrss_kib\":%ld,\"minor_faults\":%ld,\"major_faults\":%ld,\"block_input_ops\":%ld,\"block_output_ops\":%ld}\n",CLIP_COUNT,TOTAL_SAMPLES,feeds,CLIP_COUNT,callbacks,fb,mr,dr,ev,ns(CLOCK_MONOTONIC)-startwall,ns(CLOCK_PROCESS_CPUTIME_ID)-startcpu,ru.ru_maxrss,ru.ru_minflt,ru.ru_majflt,ru.ru_inblock,ru.ru_oublock);
 free(s);free(w);dlclose(lib);return fflush(stdout)||ferror(stdout)?13:0;
}
