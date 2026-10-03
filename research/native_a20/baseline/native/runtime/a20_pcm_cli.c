/* Research-only runner. Executing on audio is separately gated by protocol.
 * Usage: a20_pcm_cli exact-a20.f32 mono16k-pcm16le.raw
 * JSON lines stdout. All allocation/file I/O is outside a20s core operations. */
#include "a20_stream.h"
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
static void report(void*u,const a20s_batch*b){
 (void)u;const donor_pcm_batch*f=b->frontend;const a20d_result*r=b->result;
 printf("{\"call\":%"PRIu64",\"available_samples\":%"PRIu64",\"model_rows\":%zu,\"keyword_id\":%d,\"start_frame\":%"PRId64",\"end_frame\":%"PRId64",\"score\":%.17g}\n",f->call_index,f->available_samples,f->selected_rows,r->keyword,r->start_frame,r->end_frame,r->score);
}
int main(int argc,char**argv){
 if(argc!=3){fprintf(stderr,"usage: %s exact-a20.f32 mono16k-pcm16le.raw\n",argv[0]);return 2;}
 float*w=malloc(A20_FLOATS*sizeof(float));a20s_state*s=malloc(sizeof(*s));if(!w||!s){free(w);free(s);return 3;}
 FILE*f=fopen(argv[1],"rb");if(!f){free(w);free(s);return 3;}
 size_t n=fread(w,sizeof(float),A20_FLOATS,f);int extra=fgetc(f);int error=ferror(f);fclose(f);
 if(n!=A20_FLOATS||extra!=EOF||error||a20s_init(s,w,n)){free(w);free(s);return 4;}
 f=fopen(argv[2],"rb");if(!f){free(w);free(s);return 3;}
 unsigned char raw[9600];int16_t pcm[4800];int rc=0;
 while((n=fread(raw,1,sizeof(raw),f))!=0){
  if(n%2){rc=5;break;}
  for(size_t i=0;i<n/2;i++)pcm[i]=(int16_t)((uint16_t)raw[2*i]|((uint16_t)raw[2*i+1]<<8));
  if(a20s_feed(s,pcm,n/2,report,NULL)){rc=6;break;}
 }
 if(ferror(f))rc=5;
 fclose(f);if(!rc&&a20s_finish(s,report,NULL))rc=6;
 if(ferror(stdout))rc=7;
 free(w);free(s);return rc;
}
