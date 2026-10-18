/* SPDX-License-Identifier: Apache-2.0. Explicit user-input research example.
 * Usage: example_pcm PAYLOAD_F32LE RAW_PCM16LE RECORDING_ID THRESHOLD
 * No download, WAV parsing, resampling, model export or product qualification.
 */
#include "b_stream_adapter.h"
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
static int on_row(void *user,const bsa_row *r,const float *posterior,size_t classes,
                  const kws3_event *events,size_t count) {
 (void)user;(void)posterior;(void)classes;
 for(size_t i=0;i<count;i++) {
  const kws3_event *e=&events[i];
  if(printf("generation=%" PRIu64 " keyword=%u score=%a row=%" PRIu64
            " required=%" PRIu64 " available=%" PRIu64 " eof=%d valid_so_far=%d\n",
            r->generation,e->keyword_id,e->score,r->row,r->required_padded_sample,
            r->available_real_sample,r->eof_derived,r->valid_so_far)<0)return 1;
 }
 return count && fflush(stdout) ? 1 : 0;
}
int main(int argc,char **argv) {
 const bsa_descriptor *model=bsa_model();bsa_handle *handle=NULL;
 FILE *weights=NULL,*pcm=NULL;uint8_t *payload=NULL,bytes[641];
 char *end=NULL;double threshold;int rc=1,api=0;size_t n,used;
 if(argc!=5) {
  fprintf(stderr,"Usage: %s PAYLOAD_F32LE RAW_PCM16LE RECORDING_ID THRESHOLD\n",argv[0]);
  return 2;
 }
 errno=0;threshold=strtod(argv[4],&end);
 if(errno || end==argv[4] || *end || !isfinite(threshold) || threshold<0. || threshold>1.) {fputs("Invalid explicit threshold\n",stderr);return 2;}
 if(model->payload_bytes!=UINT64_C(3027732) || model->payload_bytes>SIZE_MAX) {fputs("Unsupported payload size\n",stderr);return 2;}
 weights=fopen(argv[1],"rb");if(!weights){perror("payload");goto done;}
 payload=malloc((size_t)model->payload_bytes);if(!payload){perror("malloc");goto done;}
 n=fread(payload,1,(size_t)model->payload_bytes,weights);
 if(n!=model->payload_bytes || fgetc(weights)!=EOF || ferror(weights)) {
  fputs("Payload must match the exact registered byte length\n",stderr);goto done;
 }
 if(fclose(weights)){weights=NULL;perror("payload close");goto done;}weights=NULL;
 api=bsa_create(model,payload,n,threshold,&handle);
 free(payload);payload=NULL;
 if(api){fprintf(stderr,"create failed: %d (identity/environment/threshold)\n",api);goto done;}
 pcm=fopen(argv[2],"rb");if(!pcm){perror("PCM");goto done;}
 api=bsa_begin(handle,argv[3],on_row,NULL);if(api)goto failed;
 while((n=fread(bytes,1,sizeof(bytes),pcm))!=0) {
  used=0;api=bsa_feed(handle,bytes,n,&used);
  if(api || used!=n)goto failed;
 }
 if(ferror(pcm)){perror("PCM read");goto failed;}
 /* Only actual file EOF requests the fixed synthetic4800 tail. */
 api=bsa_finish(handle);if(api)goto failed;
 {
  bsa_status status;kws3_summary summary;uint64_t rejected;
  api=bsa_get(handle,&status,&summary,&rejected);if(api)goto failed;
  if(!summary.numerical_support_ok &&
     fprintf(stderr,"Warning: decoder support diagnostic; empty=%" PRIu64 " zero_mass=%" PRIu64
                    "; valid recording retained, no qualification implied\n",
             summary.empty_beam_window_rows,summary.zero_mass_window_rows)<0)goto failed;
 }
 rc=0;goto done;
failed:
 {
  bsa_status status;kws3_summary summary;uint64_t rejected;
  if(!bsa_get(handle,&status,&summary,&rejected))
   fprintf(stderr,"generation=%" PRIu64 " phase=%d error=%d component=%d detail=%d\n",
           status.generation,status.phase,status.error,status.component,status.component_error);
  fprintf(stderr,"Input failed (API %d); no replay. Already delivered events cannot be retracted.\n",api);
 }
done:
 if(weights)fclose(weights);
 if(pcm && fclose(pcm))rc=1;
 free(payload);
 if(bsa_destroy(handle))rc=1;
 return rc;
}
