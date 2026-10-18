/* SPDX-License-Identifier: Apache-2.0. Thin research glue; frozen math unchanged. */
#include "b_stream_adapter.h"
#include "canonical_pcm.h"
#include "canonical_fbank.h"
#include "canonical_tables.h"
#include "fsmn2599.h"
#include "softmax2599.h"
#include <fenv.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#define BSA_MAGIC UINT32_C(0x42534131)
static const bsa_descriptor registered_b={BSA_ABI,400,2599,16000,1,16,1,
 FSMN2599_PAYLOAD_BYTES,BSA_MODEL_ID,
 "a366d9f5ebe7027a496e3f119debcf1e19d2f7a79226c96b31a4ddb3a2757f77",
 "2175df1a7f8621a6b6566a9729b63f021c007d93f2c8c17cb54e7c0f496bbfad",
 "b8fd2af3bb76f1f5d116c41d8f0fb97adeafa987a1e0ed43bb690f65ccd14679"};
struct bsa_handle {
 uint32_t magic; int busy, decoder_ready;
 float *weights; fsmn2599_model model; kcf_pcm pcm; kcf_fbank bank;
 ksx_workspace softmax; float logits[2599],posterior[2599];
 kws3_decoder *decoder; bsa_callback callback; void *user; bsa_status status;
};
const bsa_descriptor *bsa_model(void) {return &registered_b;}
size_t bsa_abi_size(unsigned k) {
 switch(k){case 1:return sizeof(bsa_row);case 2:return sizeof(bsa_status);
 case 3:return sizeof(kws3_event);case 4:return sizeof(kws3_summary);
 case 5:return sizeof(bsa_descriptor);default:return 0;}
}
size_t bsa_storage_bytes(void) {return sizeof(bsa_handle)+FSMN2599_PAYLOAD_BYTES+kws3_storage_bytes();}
static int valid(const bsa_handle *h) {return h && h->magic==BSA_MAGIC && !h->busy;}
static int bad(bsa_handle *h,int code,int component,int detail) {
 h->status.phase=BSA_FAILED;h->status.valid_so_far=0;
 h->status.error=code;h->status.component=component;h->status.component_error=detail;
 return code;
}
#define BUMP(field) do { \
 if(h->status.field==UINT64_MAX)return bad(h,BSA_OVERFLOW,0,0); \
 h->status.field++; \
} while(0)
static int component_fail(bsa_handle *h,int component,int rc) {
 return bad(h,BSA_COMPONENT,component,rc);
}
static int fp_enter(bsa_handle *h,fenv_t *saved) {
 if(feholdexcept(saved)) return bad(h,BSA_ENVIRONMENT,0,0);
 return BSA_OK;
}
static int fp_leave(bsa_handle *h,const fenv_t *saved,int rc) {
 if(fesetenv(saved)) return bad(h,BSA_RESTORE,0,0);
 return rc;
}
static void counters(bsa_handle *h) {
 h->status.processed_bytes=h->pcm.input_bytes;
 h->status.real_samples=h->pcm.real_samples;h->status.padded_samples=h->pcm.padded_samples;
 h->status.zero_samples=h->pcm.zero_samples;
}
static int bank_row(void *user,const int16_t *pcm,float *out) {
 bsa_handle *h=user;int rc;
 if(!fsmn2599_environment_ok()) return bad(h,BSA_ENVIRONMENT,1,0);
 if(h->status.fbank_reserved==UINT64_MAX || h->status.fbank_completed==UINT64_MAX)
  return bad(h,BSA_OVERFLOW,0,0);
 BUMP(fbank_reserved);
 rc=kcf_fbank_frame(&h->bank,pcm,out,NULL);
 if(rc) return component_fail(h,1,rc);
 BUMP(fbank_completed);return BSA_OK;
}
static int emit_row(void *user,int kind,uint64_t index,uint64_t center,
 uint64_t end,int eof,const float *context) {
 bsa_handle *h=user;int rc;size_t n=0;kws3_event events[3];bsa_row row;
 if(kind==KCF_EVENT_FBANK) return BSA_OK;
 if(kind!=KCF_EVENT_CONTEXT || index!=h->status.network_reserved ||
    index>(UINT64_MAX-720)/480 || center!=3*index || end!=480*index+720 ||
    index>=(uint64_t)KWS3_MAX_ROWS || eof!=(h->pcm.phase==KCF_PHASE_EOF))
  return bad(h,BSA_INVARIANT,0,0);
 if(h->status.network_reserved==UINT64_MAX || h->status.network_completed==UINT64_MAX)
  return bad(h,BSA_OVERFLOW,0,0);
 BUMP(network_reserved);
 rc=fsmn2599_step(&h->model,context,400,h->logits,2599,NULL,0);
 if(rc) return component_fail(h,2,rc);
 BUMP(network_completed);
 if(h->status.softmax_reserved==UINT64_MAX || h->status.softmax_completed==UINT64_MAX)
  return bad(h,BSA_OVERFLOW,0,0);
 BUMP(softmax_reserved);
 rc=ksx_softmax2599(h->logits,2599,h->posterior,2599,&h->softmax);
 if(rc) return component_fail(h,3,rc);
 BUMP(softmax_completed);
 if(h->status.decoder_reserved==UINT64_MAX || h->status.decoder_completed==UINT64_MAX)
  return bad(h,BSA_OVERFLOW,0,0);
 BUMP(decoder_reserved);
 rc=kws3_step_f32(h->decoder,h->posterior,2599,events,3,&n);
 if(rc) return component_fail(h,4,rc);
 BUMP(decoder_completed);
 if(n>3 || UINT64_MAX-h->status.events_delivered<n) return bad(h,BSA_OVERFLOW,0,0);
 row=(bsa_row){h->status.generation,index,end,h->status.supplied_bytes/2,
                h->pcm.real_samples,1,eof};
 if(!eof && row.available_real_sample<end) return bad(h,BSA_INVARIANT,0,0);
 if(h->status.callbacks_completed==UINT64_MAX)return bad(h,BSA_OVERFLOW,0,0);
 /* Delivery is immediate, including live streams which never reach EOF. */
 h->status.events_delivered+=n;
 rc=h->callback(h->user,&row,h->posterior,2599,events,n);
 if(rc) return bad(h,BSA_CALLBACK,0,rc);
 BUMP(callbacks_completed);
 if(!fsmn2599_environment_ok()) return bad(h,BSA_ENVIRONMENT,0,0);
 return BSA_OK;
}
int bsa_create(const bsa_descriptor *d,const uint8_t *payload,size_t bytes,
 double threshold,bsa_handle **out) {
 bsa_handle *h;fenv_t saved;int rc;
 if(!out) return BSA_ARGUMENT;
 *out=NULL;
 if(d!=&registered_b || !payload || bytes!=FSMN2599_PAYLOAD_BYTES)
  return BSA_ARGUMENT;
 h=calloc(1,sizeof(*h));if(!h) return BSA_ALLOCATION;
 h->magic=BSA_MAGIC;h->status.phase=BSA_NEW;
 if((rc=fp_enter(h,&saved))!=0){free(h);return rc;}
 if(!fsmn2599_environment_ok()) {rc=BSA_ENVIRONMENT;goto done;}
 if(!isfinite(threshold) || threshold<0. || threshold>1.) {rc=BSA_ARGUMENT;goto done;}
 h->status.threshold=threshold==0.?0.:threshold;
 h->weights=malloc(FSMN2599_PAYLOAD_BYTES);
 if(!h->weights){rc=BSA_ALLOCATION;goto done;}
 rc=fsmn2599_load(&h->model,h->weights,FSMN2599_FLOATS,payload,bytes,
                  FSMN2599_PAYLOAD_LITTLE_ENDIAN,d->payload_sha256);
 if(rc){rc=component_fail(h,2,rc);goto done;}
 rc=kws3_create_with_threshold("unstarted",threshold,&h->decoder);
 if(rc) rc=component_fail(h,4,rc);
done:
 rc=fp_leave(h,&saved,rc);
 if(rc){if(h->decoder)kws3_destroy(h->decoder);free(h->weights);free(h);return rc;}
 *out=h;return BSA_OK;
}
static int good_id(const char *id) {
 if(!id || !id[0])return 0;
 for(size_t n=1;n<KWS3_ID_BYTES;n++)if(!id[n])return 1;
 return 0;
}
static int start(bsa_handle *h,const char *id,bsa_callback callback,void *user) {
 fenv_t saved;uint64_t gen,aborted;double threshold;int rc;
 if(!valid(h))return BSA_STATE;
 if(!good_id(id) || !callback)return BSA_ARGUMENT;
 if(h->status.generation==UINT64_MAX)return bad(h,BSA_OVERFLOW,0,0);
 gen=h->status.generation+1;aborted=h->status.aborted_generations;
 if(h->status.phase==BSA_OPEN || h->status.phase==BSA_FAILED){
  if(aborted==UINT64_MAX)return bad(h,BSA_OVERFLOW,0,0);
  aborted++;
 }
 threshold=h->status.threshold;h->busy=1;
 rc=fp_enter(h,&saved);if(rc){h->busy=0;return rc;}
 if(!fsmn2599_environment_ok()){rc=bad(h,BSA_ENVIRONMENT,0,0);goto done;}
 memset(&h->status,0,sizeof(h->status));h->status.generation=gen;
 h->status.aborted_generations=aborted;h->status.threshold=threshold;
 h->callback=callback;h->user=user;h->decoder_ready=0;
 rc=fsmn2599_reset(&h->model);if(rc){rc=component_fail(h,2,rc);goto done;}
 rc=kcf_fbank_init(&h->bank,&kcf_exported_tables);if(rc){rc=component_fail(h,1,rc);goto done;}
 rc=kws3_reset(h->decoder,id);if(rc){rc=component_fail(h,4,rc);goto done;}
 h->decoder_ready=1;
 rc=kcf_pcm_init(&h->pcm,bank_row,h,emit_row,h);if(rc){rc=component_fail(h,5,rc);goto done;}
 h->status.phase=BSA_OPEN;h->status.valid_so_far=1;
done:rc=fp_leave(h,&saved,rc);h->busy=0;return rc;
}
int bsa_begin(bsa_handle *h,const char *id,bsa_callback cb,void *user) {
 if(!valid(h) || h->status.phase!=BSA_NEW)return BSA_STATE;
 return start(h,id,cb,user);
}
int bsa_reset(bsa_handle *h,const char *id,bsa_callback cb,void *user) {
 return start(h,id,cb,user);
}
int bsa_feed(bsa_handle *h,const uint8_t *data,size_t bytes,size_t *consumed) {
 fenv_t saved;int rc;
 if(!consumed)return BSA_ARGUMENT;
 *consumed=0;
 if(!valid(h))return BSA_STATE;
 if(h->status.phase==BSA_FAILED)return h->status.error;
 if(h->status.phase!=BSA_OPEN)return BSA_STATE;
 if(!data && bytes)return BSA_ARGUMENT;
 if(h->status.supplied_bytes>UINT64_MAX-UINT64_C(9600) ||
    h->pcm.input_bytes>UINT64_MAX-UINT64_C(9600))
  return bad(h,BSA_OVERFLOW,0,0);
 if(bytes>UINT64_MAX-UINT64_C(9600)-h->status.supplied_bytes)
  return bad(h,BSA_OVERFLOW,0,0);
 if(!bytes)return BSA_OK;
 h->busy=1;rc=fp_enter(h,&saved);if(rc){h->busy=0;return rc;}
 if(!fsmn2599_environment_ok()){rc=bad(h,BSA_ENVIRONMENT,0,0);goto done;}
 h->status.supplied_bytes+=(uint64_t)bytes;
 rc=kcf_pcm_feed(&h->pcm,data,bytes,consumed);counters(h);
 if(rc)rc=h->status.error?h->status.error:component_fail(h,5,rc);
 else if(*consumed!=bytes)rc=bad(h,BSA_INVARIANT,0,0);
done:rc=fp_leave(h,&saved,rc);h->busy=0;return rc;
}
int bsa_finish(bsa_handle *h) {
 fenv_t saved;kws3_summary summary;uint64_t frames,rows;int rc;
 if(!valid(h))return BSA_STATE;
 if(h->status.phase==BSA_FAILED)return h->status.error;
 if(h->status.phase!=BSA_OPEN)return BSA_STATE;
 if(h->status.finish_calls==UINT64_MAX)return bad(h,BSA_OVERFLOW,0,0);
 h->busy=1;rc=fp_enter(h,&saved);if(rc){h->busy=0;return rc;}
 if(!fsmn2599_environment_ok()){rc=bad(h,BSA_ENVIRONMENT,0,0);goto done;}
 h->status.finish_calls++;
 rc=kcf_pcm_finish(&h->pcm);counters(h);
 if(rc){rc=h->status.error?h->status.error:component_fail(h,5,rc);goto done;}
 rc=kws3_finish(h->decoder);if(rc){rc=component_fail(h,4,rc);goto done;}
 rc=kws3_get_summary(h->decoder,&summary);if(rc){rc=component_fail(h,4,rc);goto done;}
 frames=1+(h->status.real_samples+4800-400)/160;
 rows=frames<3?0:1+(frames-3)/3;
 if(h->status.processed_bytes!=h->status.supplied_bytes ||
    h->status.zero_samples!=4800 || h->status.padded_samples!=h->status.real_samples+4800 ||
    h->status.fbank_completed!=frames || h->status.network_completed!=rows ||
    h->status.softmax_completed!=rows || h->status.decoder_completed!=rows ||
    h->status.callbacks_completed!=rows || summary.rows_decoded!=rows ||
    !summary.finished || !summary.valid){
  rc=bad(h,BSA_INVARIANT,0,0);goto done;
 }
 /* Support counters are diagnostics, not decoder validity or lifecycle errors.
  * bsa_get preserves them for callers; no qualification is implied. */
 h->status.phase=BSA_CLOSED;
done:rc=fp_leave(h,&saved,rc);h->busy=0;return rc;
}
int bsa_get(bsa_handle *h,bsa_status *status,kws3_summary *summary,uint64_t *rejections) {
 double threshold;int rc;
 if(!valid(h))return BSA_STATE;
 if(!status || !summary || !rejections)return BSA_ARGUMENT;
 *status=h->status;memset(summary,0,sizeof(*summary));*rejections=0;
 if(h->decoder_ready){
  rc=kws3_get_summary(h->decoder,summary);if(rc)return BSA_COMPONENT;
 }
 rc=kws3_get_threshold_status(h->decoder,&threshold,rejections);
 if(rc || memcmp(&threshold,&h->status.threshold,sizeof(double)))return BSA_INVARIANT;
 if(!h->decoder_ready)*rejections=0;
 return BSA_OK;
}
int bsa_destroy(bsa_handle *h) {
 if(!h)return BSA_OK;
 if(!valid(h))return BSA_STATE;
 kws3_destroy(h->decoder);free(h->weights);h->magic=0;free(h);return BSA_OK;
}
