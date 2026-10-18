/* SPDX-License-Identifier: Apache-2.0
 * Public-layout candidate: historical private-layout fixtures passed;
 * this rearranged build requires its own verification.
 * Synthetic glue/transport fixtures, not acoustic or numerical qualification.
 * Build later ONLY under separately approved fixed execution budget:
 * this TU + canonical_pcm.c, never production numerical objects.
 * Including adapter.c exposes private state ONLY in this isolated test TU.
 */
#include <assert.h>
#include <fenv.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "b_stream_adapter.h"
#include "canonical_pcm.h"
#include "canonical_fbank.h"
#include "fsmn2599.h"
#include "softmax2599.h"

#define CHECK(x) do { if (!(x)) { fprintf(stderr,"fixture:%d: %s\n",__LINE__,#x); abort(); } } while(0)
enum { MAX_HANDLES=40, MAX_GENERATIONS=48, MAX_API=512,
       MAX_FBANK=1024, MAX_CONTEXT=256, MAX_CALLBACK=256,
       MAX_FEED_BYTES=131072, MAX_ZERO_SAMPLES=96000 };
static struct {
 unsigned handles, generations, api, banks, network, softmax, decoder, callbacks;
 uint64_t feed_bytes, zero_samples;
} budget;
static void api_charge(void) { CHECK(++budget.api<=MAX_API); }
/* These controls are test-TU private. None is a production API symbol. */
enum { FAIL_NONE, FAIL_BANK, FAIL_NETWORK, FAIL_SOFTMAX, FAIL_DECODER, FAIL_CALLBACK };
static int fault, fail_at=1, bank_local, network_local, softmax_local, decoder_local;
static int env_bad, hold_fail, restore_fail, reenter, callback_round_change;
static int check_pcm, diagnostic_support_false;
static uint64_t digest, callbacks_local, delivered_local;
static bsa_handle *active;
static bsa_row last_row;
static double passed_threshold;
static const int16_t edge_values[4]={0,32767,-32768,-1};
static void clear_controls(void) {
 fault=FAIL_NONE; fail_at=1; bank_local=network_local=softmax_local=decoder_local=0;
 env_bad=hold_fail=restore_fail=reenter=callback_round_change=check_pcm=0;
 diagnostic_support_false=0;
 digest=UINT64_C(1469598103934665603); callbacks_local=delivered_local=0;
 memset(&last_row,0,sizeof(last_row));
}
static int fixture_hold(fenv_t *e) { if(hold_fail) return 1; return feholdexcept(e); }
static int fixture_restore(const fenv_t *e) {
 int r=fesetenv(e); return restore_fail ? 1 : r;
}
#define feholdexcept fixture_hold
#define fesetenv fixture_restore
#include "b_stream_adapter.c"
#undef feholdexcept
#undef fesetenv

/* All production entrypoints below are replaced by link-time test backends.
 * Transport is the actual canonical_pcm.c. No weights/audio are read. */
int fsmn2599_environment_ok(void) { return !env_bad && fegetround()==FE_TONEAREST; }
fsmn2599_status fsmn2599_load(fsmn2599_model *m,float *w,size_t n,
 const unsigned char *p,size_t b,unsigned order,const char sha[65]) {
 CHECK(m && w && p && n==FSMN2599_FLOATS && b==FSMN2599_PAYLOAD_BYTES);
 CHECK(order==FSMN2599_PAYLOAD_LITTLE_ENDIAN);
 CHECK(strcmp(sha,bsa_model()->payload_sha256)==0);
 memset(m,0,sizeof(*m)); m->weights=w; return FSMN2599_OK;
}
fsmn2599_status fsmn2599_reset(fsmn2599_model *m) {
 memset(m->cache,0,sizeof(m->cache)); return FSMN2599_OK;
}
fsmn2599_status fsmn2599_step(fsmn2599_model *m,const float *x,size_t n,
 float *y,size_t c,float *trace,size_t tn) {
 (void)trace; (void)tn; CHECK(n==400 && c==2599);
 CHECK(++budget.network<=MAX_CONTEXT); ++network_local;
 if(fault==FAIL_NETWORK && network_local==fail_at) return FSMN2599_NONFINITE;
 memset(y,0,c*sizeof(*y)); y[0]=x[0]; m->cache[0]+=1.0f; return FSMN2599_OK;
}
int kcf_fbank_init(kcf_fbank *s,const kcf_tables *t) {
 memset(s,0,sizeof(*s)); s->tables=t; return KCF_FBANK_OK;
}
int kcf_fbank_frame(kcf_fbank *s,const int16_t pcm[400],float row[80],kcf_trace *t) {
 (void)s; (void)t; CHECK(++budget.banks<=MAX_FBANK); ++bank_local;
 if(fault==FAIL_BANK && bank_local==fail_at) return KCF_FBANK_NONFINITE;
 if(check_pcm) for(unsigned i=0;i<400;i++) CHECK(pcm[i]==edge_values[i%4]);
 for(unsigned i=0;i<80;i++) { row[i]=(float)pcm[i]; }
 return KCF_FBANK_OK;
}
int kcf_fbank_callback(void *s,const int16_t pcm[400],float row[80]) {
 return kcf_fbank_frame((kcf_fbank *)s,pcm,row,NULL);
}
ksx_status ksx_softmax2599(const float *x,size_t n,float *p,size_t c,ksx_workspace *w) {
 (void)w; CHECK(n==2599 && c==2599); CHECK(++budget.softmax<=MAX_CONTEXT); ++softmax_local;
 if(fault==FAIL_SOFTMAX && softmax_local==fail_at) return KSX_NONFINITE;
 /* Deliberately NOT a softmax: deterministic transport marker only. */
 memset(p,0,c*sizeof(*p)); p[0]=x[0]; p[1]=1.0f; return KSX_OK;
}
struct kws3_decoder { double threshold; uint64_t rows; int finished; char id[256]; };
kws3_status kws3_create_with_threshold(const char *id,double t,kws3_decoder **out) {
 CHECK(isfinite(t) && t>=0 && t<=1); passed_threshold=t;
 *out=calloc(1,sizeof(**out)); if(!*out) return KWS3_ALLOCATION;
 (*out)->threshold=t==0.0?0.0:t; return kws3_reset(*out,id);
}
void kws3_destroy(kws3_decoder *d) { free(d); }
size_t kws3_storage_bytes(void) { return sizeof(kws3_decoder); }
kws3_status kws3_reset(kws3_decoder *d,const char *id) {
 CHECK(id && *id && strlen(id)<256); d->rows=0; d->finished=0;
 strcpy(d->id,id); return KWS3_OK;
}
kws3_status kws3_get_threshold_status(const kws3_decoder *d,double *t,uint64_t *r) {
 *t=d->threshold; *r=0; return KWS3_OK;
}
kws3_status kws3_step_f32(kws3_decoder *d,const float *p,size_t n,kws3_event *ev,size_t cap,size_t *count) {
 (void)p; CHECK(n==2599 && cap>=3); CHECK(++budget.decoder<=MAX_CONTEXT); ++decoder_local;
 *count=0; if(fault==FAIL_DECODER && decoder_local==fail_at) return KWS3_INVARIANT;
 for(unsigned k=0;k<3;k++) { memset(&ev[k],0,sizeof(ev[k]));
  ev[k].event_index=d->rows*3+k; ev[k].decision_row=(int64_t)d->rows;
  ev[k].score=d->threshold; ev[k].keyword_id=k; strcpy(ev[k].recording_id,d->id);
 }
 ++d->rows; *count=3; return KWS3_OK;
}
kws3_status kws3_finish(kws3_decoder *d) { CHECK(!d->finished); d->finished=1; return KWS3_OK; }
kws3_status kws3_get_summary(const kws3_decoder *d,kws3_summary *s) {
 memset(s,0,sizeof(*s)); s->rows_decoded=d->rows; s->events_emitted=d->rows*3;
 s->finished=d->finished; s->valid=1; s->numerical_support_ok=!diagnostic_support_false;
 if(diagnostic_support_false) { s->empty_beam_window_rows=1; s->zero_mass_window_rows=2; }
 strcpy(s->recording_id,d->id); return KWS3_OK;
}

/* Charge all public calls made by the harness after adapter implementation. */
#define bsa_create(...) (api_charge(),bsa_create(__VA_ARGS__))
#define bsa_begin(...) (api_charge(),bsa_begin(__VA_ARGS__))
#define bsa_reset(...) (api_charge(),bsa_reset(__VA_ARGS__))
static void feed_charge(size_t n) { budget.feed_bytes+=n; CHECK(budget.feed_bytes<=MAX_FEED_BYTES); }
#define bsa_feed(h,p,n,c) (api_charge(),feed_charge(n),bsa_feed(h,p,n,c))
#define bsa_finish(...) (api_charge(),bsa_finish(__VA_ARGS__))
#define bsa_get(...) (api_charge(),bsa_get(__VA_ARGS__))
#define bsa_destroy(...) (api_charge(),bsa_destroy(__VA_ARGS__))
static uint8_t synthetic_payload[FSMN2599_PAYLOAD_BYTES];
static uint8_t pcm_bytes[2400];
static int receive(void *u,const bsa_row *r,const float *p,size_t n,const kws3_event *ev,size_t count) {
 (void)u; CHECK(++budget.callbacks<=MAX_CALLBACK); CHECK(n==2599 && count==3);
 CHECK(r->valid_so_far && r->row==callbacks_local);
 CHECK(r->required_padded_sample==r->row*480+720);
 CHECK(r->eof_derived || r->available_real_sample>=r->required_padded_sample);
 for(size_t i=0;i<n;i++) { uint32_t bits; memcpy(&bits,p+i,4); digest=(digest^bits)*UINT64_C(1099511628211); }
 for(size_t i=0;i<count;i++) CHECK(ev[i].decision_row==(int64_t)r->row);
 last_row=*r; ++callbacks_local; delivered_local+=count;
 if(reenter) { size_t used=99; bsa_status s; kws3_summary summary; uint64_t rejected;
  CHECK(bsa_feed(active,NULL,0,&used)==BSA_STATE && used==0);
  CHECK(bsa_finish(active)==BSA_STATE);
  CHECK(bsa_reset(active,"reentry",receive,NULL)==BSA_STATE);
  CHECK(bsa_get(active,&s,&summary,&rejected)==BSA_STATE);
 }
 if(callback_round_change) CHECK(fesetround(FE_DOWNWARD)==0);
 return fault==FAIL_CALLBACK ? 77 : 0;
}
static bsa_handle *make(double t) {
 bsa_handle *h=NULL; CHECK(++budget.handles<=MAX_HANDLES);
 CHECK(bsa_create(bsa_model(),synthetic_payload,sizeof(synthetic_payload),t,&h)==BSA_OK && h);
 CHECK(++budget.generations<=MAX_GENERATIONS);
 CHECK(bsa_begin(h,"fixture",receive,NULL)==BSA_OK); active=h; return h;
}
static void restart(bsa_handle *h) {
 CHECK(++budget.generations<=MAX_GENERATIONS); clear_controls();
 CHECK(bsa_reset(h,"reset",receive,NULL)==BSA_OK); active=h;
 CHECK(h->pcm.have_low==0 && h->pcm.used==0 && h->decoder->rows==0);
 CHECK(h->status.network_completed==0 && h->status.events_delivered==0);
 for(unsigned i=0;i<FSMN2599_CACHE_FLOATS;i++) CHECK(h->model.cache[i]==0.0f);
 for(unsigned i=0;i<5;i++) for(unsigned j=0;j<80;j++) CHECK(h->pcm.history[i][j]==0.0f);
}
static void feed_ok(bsa_handle *h,size_t offset,size_t bytes) {
 size_t used=99; CHECK(offset+bytes<=sizeof(pcm_bytes));
 CHECK(bsa_feed(h,bytes?pcm_bytes+offset:NULL,bytes,&used)==BSA_OK && used==bytes);
}
static bsa_status status(bsa_handle *h) {
 bsa_status s; kws3_summary summary; uint64_t rejects;
 CHECK(bsa_get(h,&s,&summary,&rejects)==BSA_OK); return s;
}
static void finish_ok(bsa_handle *h) {
 budget.zero_samples+=4800; CHECK(budget.zero_samples<=MAX_ZERO_SAMPLES);
 CHECK(bsa_finish(h)==BSA_OK); bsa_status s=status(h);
 CHECK(s.phase==BSA_CLOSED && s.valid_so_far && s.zero_samples==4800);
 uint64_t f=1+(s.real_samples+4800-400)/160, r=1+(f-3)/3;
 CHECK(s.fbank_completed==f && s.network_completed==r && s.softmax_completed==r);
 CHECK(s.decoder_completed==r && s.callbacks_completed==r);
 CHECK(s.real_samples*2==s.processed_bytes && s.padded_samples==s.real_samples+4800);
}
static void drop(bsa_handle *h) { CHECK(bsa_destroy(h)==BSA_OK); active=NULL; }
static void group1(void) {
 bsa_handle *h=(bsa_handle *)(uintptr_t)1; const bsa_descriptor *d=bsa_model();
 bsa_descriptor wrong=*d;
 /* Even exact descriptor copies are rejected: caller metadata is not trust. */
 CHECK(bsa_create(&wrong,synthetic_payload,sizeof(synthetic_payload),.7,&h)==BSA_ARGUMENT && h==NULL);
 for(unsigned i=0;i<8;i++) { wrong=*d;
  switch(i) { case 0: wrong.abi++; break; case 1: wrong.inputs++; break;
   case 2: wrong.outputs++; break; case 3: wrong.byte_order++; break;
   case 4: wrong.payload_bytes++; break; case 5: wrong.id="A240"; break;
   case 6: wrong.payload_sha256="untrusted"; break; default: wrong.sample_rate++; }
  h=(bsa_handle *)(uintptr_t)1;
  CHECK(bsa_create(&wrong,synthetic_payload,sizeof(synthetic_payload),.7,&h)==BSA_ARGUMENT && h==NULL);
 }
 const double bad[4]={NAN,INFINITY,-0.01,1.01};
 for(unsigned i=0;i<4;i++) CHECK(bsa_create(d,synthetic_payload,sizeof(synthetic_payload),bad[i],&h)==BSA_ARGUMENT && h==NULL);
 CHECK(bsa_create(d,NULL,sizeof(synthetic_payload),.7,&h)==BSA_ARGUMENT && h==NULL);
 CHECK(bsa_create(d,synthetic_payload,1,.7,&h)==BSA_ARGUMENT && h==NULL);
 CHECK(budget.banks==0 && budget.network==0 && budget.decoder==0);
}
static void group2(void) {
 const size_t chunks[5][4]={{1440,0,0,0},{1,1439,0,0},{639,1,800,0},
                           {2000,0,0,0},{1440,1,559,0}};
 uint64_t reference=0;
 for(unsigned i=0;i<5;i++) { clear_controls(); bsa_handle *h=make(.7); size_t off=0;
  for(unsigned j=0;j<4;j++) { feed_ok(h,off,chunks[i][j]); off+=chunks[i][j]; }
  CHECK(callbacks_local==1 && last_row.processed_real_sample==720);
  CHECK(last_row.available_real_sample==(i==3?1000:720));
  if(i==0) reference=digest; else CHECK(reference==digest);
  CHECK(status(h).real_samples==off/2); drop(h);
 }
}
static void group3(void) {
 clear_controls(); check_pcm=1; bsa_handle *h=make(.7); size_t used=9;
 CHECK(bsa_feed(h,NULL,1,&used)==BSA_ARGUMENT && used==0);
 CHECK(bsa_feed(h,pcm_bytes,2,NULL)==BSA_ARGUMENT);
 feed_ok(h,0,1); feed_ok(h,1,798); feed_ok(h,799,641);
 CHECK(status(h).real_samples==720 && callbacks_local==1); drop(h);
}
static void group4(void) {
 for(unsigned i=0;i<3;i++) { clear_controls(); bsa_handle *h=make(.7);
  feed_ok(h,0,i); if(i==1) {
   CHECK(bsa_finish(h)!=BSA_OK); bsa_status s=status(h);
   CHECK(s.phase==BSA_FAILED && !s.valid_so_far && s.zero_samples==0);
   CHECK(callbacks_local==0 && bsa_finish(h)!=BSA_OK);
  } else {
   /* Isolated summary injection proves separation, not real reachability. */
   diagnostic_support_false=(i==2);
   finish_ok(h); bsa_status before=status(h); CHECK(callbacks_local==9 && last_row.eof_derived);
   if(i==2) {
    bsa_status st;kws3_summary summary;uint64_t rejected;
    CHECK(bsa_get(h,&st,&summary,&rejected)==BSA_OK);
    CHECK(st.valid_so_far && st.phase==BSA_CLOSED && summary.valid && summary.finished);
    CHECK(!summary.numerical_support_ok && summary.empty_beam_window_rows==1 && summary.zero_mass_window_rows==2);
   }
   CHECK(last_row.available_real_sample==i/2);
   CHECK(bsa_finish(h)==BSA_STATE); size_t used=1;
   CHECK(bsa_feed(h,NULL,0,&used)==BSA_STATE && used==0);
   bsa_status after=status(h); CHECK(before.network_completed==after.network_completed);
   CHECK(before.zero_samples==after.zero_samples);
  } drop(h);
 }
}
static void group5(void) {
 clear_controls(); bsa_handle *h=make(.7); feed_ok(h,0,1); restart(h);
 CHECK(status(h).real_samples==0 && status(h).threshold==.7);
 feed_ok(h,0,1440); uint64_t ref=digest; restart(h);
 CHECK(status(h).network_completed==0); feed_ok(h,0,1440); CHECK(digest==ref);
 fault=FAIL_NETWORK; fail_at=2; size_t used=0;
 CHECK(bsa_feed(h,pcm_bytes+1440,960,&used)!=BSA_OK); restart(h);
 feed_ok(h,0,1440); CHECK(digest==ref); finish_ok(h); restart(h);
 feed_ok(h,0,1440); CHECK(digest==ref && status(h).threshold==.7); drop(h);
 clear_controls(); h=make(.7); feed_ok(h,0,1440); CHECK(digest==ref); drop(h);
}
static void group6(void) {
 for(int f=FAIL_BANK;f<=FAIL_CALLBACK;f++) {
  clear_controls(); bsa_handle *h=make(.7); fault=f; size_t used=0;
  CHECK(bsa_feed(h,pcm_bytes,1440,&used)!=BSA_OK);
  CHECK(used==(f==FAIL_BANK?800u:1440u)); bsa_status s=status(h);
  CHECK(s.phase==BSA_FAILED && !s.valid_so_far);
  CHECK(s.fbank_reserved==s.fbank_completed+(f==FAIL_BANK));
  CHECK(s.network_reserved==s.network_completed+(f==FAIL_NETWORK));
  CHECK(s.softmax_reserved==s.softmax_completed+(f==FAIL_SOFTMAX));
  CHECK(s.decoder_reserved==s.decoder_completed+(f==FAIL_DECODER));
  unsigned before=budget.network; used=9;
  CHECK(bsa_feed(h,pcm_bytes,2,&used)!=BSA_OK && used==0);
  CHECK(bsa_finish(h)!=BSA_OK && budget.network==before); drop(h);
 }
 clear_controls(); bsa_handle *h=make(.7); reenter=1; feed_ok(h,0,1440);
 CHECK(delivered_local==3 && status(h).valid_so_far); reenter=0;
 fault=FAIL_NETWORK; fail_at=2; size_t used=0;
 CHECK(bsa_feed(h,pcm_bytes+1440,960,&used)!=BSA_OK);
 CHECK(delivered_local==3 && !status(h).valid_so_far); drop(h);
}
/* Private counter mutations are isolated to this TU, never public hooks. */
static void group7(void) {
 for(unsigned which=0;which<13;which++) {
  clear_controls(); bsa_handle *h=make(.7); size_t used=99; int rc;
  switch(which) {
   case 0: h->status.supplied_bytes=UINT64_MAX; break;
   case 1: h->pcm.input_bytes=UINT64_MAX; break;
   case 2: h->status.events_delivered=UINT64_MAX-1; break;
   case 3: h->status.generation=UINT64_MAX; break;
   case 4: h->status.aborted_generations=UINT64_MAX; break;
   case 5: h->status.fbank_reserved=UINT64_MAX; break;
   case 6: h->status.fbank_completed=UINT64_MAX; break;
   case 7: h->status.softmax_reserved=UINT64_MAX; break;
   case 8: h->status.softmax_completed=UINT64_MAX; break;
   case 9: h->status.decoder_reserved=UINT64_MAX; break;
   case 10: h->status.decoder_completed=UINT64_MAX; break;
   case 11: h->status.callbacks_completed=UINT64_MAX; break;
   case 12: h->status.finish_calls=UINT64_MAX; break;
  }
  if(which==3 || which==4) rc=bsa_reset(h,"overflow",receive,NULL);
  else if(which==12) rc=bsa_finish(h);
  else rc=bsa_feed(h,pcm_bytes,1440,&used);
  CHECK(rc==BSA_OVERFLOW && !status(h).valid_so_far); drop(h);
 }
 for(unsigned which=0;which<6;which++) {
  clear_controls(); bsa_handle *h=make(.7); size_t used=0;
  fenv_t original; CHECK(fegetenv(&original)==0);
  CHECK(feclearexcept(FE_ALL_EXCEPT)==0 && feraiseexcept(FE_INVALID)==0);
  int expected_flags=fetestexcept(FE_ALL_EXCEPT), expected_round=fegetround(), rc;
  if(which==0) {
   /* Successful call and later stage failure both restore caller's flags. */
   feed_ok(h,0,1440); CHECK(fetestexcept(FE_ALL_EXCEPT)==expected_flags);
   fault=FAIL_NETWORK; fail_at=2;
   rc=bsa_feed(h,pcm_bytes+1440,960,&used); CHECK(rc==BSA_COMPONENT);
  } else {
   if(which==1) env_bad=1;
   if(which==2) { CHECK(fesetround(FE_DOWNWARD)==0); expected_round=FE_DOWNWARD; }
   if(which==3) hold_fail=1;
   if(which==4) restore_fail=1;
   if(which==5) callback_round_change=1;
   rc=bsa_feed(h,pcm_bytes,1440,&used);
   CHECK(rc==(which==4?BSA_RESTORE:BSA_ENVIRONMENT));
  }
  CHECK(fegetround()==expected_round && fetestexcept(FE_ALL_EXCEPT)==expected_flags);
  CHECK(!status(h).valid_so_far);
  /* The restore-failure mock restores then reports failure. Real failed
   * restoration is explicitly indeterminate; this checks error propagation. */
  env_bad=hold_fail=restore_fail=callback_round_change=0;
  CHECK(fesetenv(&original)==0); drop(h);
 }
}
static void group8(void) {
 const double theta[4]={0.0,0.1,0.7,-0.0};
 for(unsigned i=0;i<4;i++) {
  clear_controls(); bsa_handle *h=make(theta[i]);
  CHECK(memcmp(&passed_threshold,&theta[i],sizeof(double))==0);
  double expected=theta[i]==0.0?0.0:theta[i];
  bsa_status s=status(h); CHECK(memcmp(&s.threshold,&expected,sizeof(double))==0);
  feed_ok(h,0,1440); restart(h); s=status(h);
  CHECK(memcmp(&s.threshold,&expected,sizeof(double))==0);
  feed_ok(h,0,1440); drop(h);
 }
}
int main(void) {
 /* This writes synthetic bytes only if/when separately authorized to run. */
 for(unsigned i=0;i<sizeof(pcm_bytes)/2;i++) {
  uint16_t word=(uint16_t)edge_values[i%4]; pcm_bytes[2*i]=(uint8_t)word;
  pcm_bytes[2*i+1]=(uint8_t)(word>>8);
 }
 group1(); group2(); group3(); group4(); group5(); group6(); group7(); group8();
 CHECK(budget.handles==40 && budget.generations==48);
 CHECK(budget.banks==197 && budget.network==64 && budget.softmax==58);
 CHECK(budget.decoder==55 && budget.callbacks==52 && budget.zero_samples==14400);
 puts("eight synthetic glue groups completed; no acoustic qualification"); return 0;
}
