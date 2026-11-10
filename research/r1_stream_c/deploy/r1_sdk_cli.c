/* SPDX-License-Identifier: Apache-2.0; raw PCM file/stdin research adapter. */
#include "r1_sdk.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
static unsigned char *load(const char *path,size_t n){
    FILE *f=fopen(path,"rb");if(!f)return NULL;
    unsigned char *p=(unsigned char *)malloc(n);if(!p){fclose(f);return NULL;}
    size_t got=fread(p,1,n,f);int tail=fgetc(f),bad=ferror(f);fclose(f);
    if(got!=n||tail!=EOF||bad){free(p);return NULL;}return p;
}
static int sink(void *user,const r1_proposal *p){
    (void)user;
    int n=printf("{\"kind\":\"uncalibrated_greedy_proposal\",\"epoch\":%" PRIu64 ",\"keyword_id\":%" PRIu64 ",\"model_frame\":%" PRIu64 ",\"required_end_sample\":%" PRIu64 ",\"delivery_end_sample\":%" PRIu64 ",\"word_end_annotation\":null}\n",
      p->epoch,p->keyword_id,p->model_frame,p->required_end_sample,p->delivery_end_sample);
    return n<0||fflush(stdout)?1:0;
}
int main(int argc,char **argv){
    if(argc!=5||strcmp(argv[1],"--research")){
        fputs("usage: r1_sdk_pcm --research model.f32le frontend.f32le input.pcm-or--\n"
              "mono 16kHz PCM16LE only; diagnostic proposals, not calibrated wake decisions\n",stderr);return 2;}
    unsigned char *w=load(argv[2],3028860),*t=load(argv[3],83840);
    if(!w||!t){free(w);free(t);return 2;}
    r1_sdk *s=r1_sdk_create(w,3028860,t,83840,sink,NULL);free(w);free(t);if(!s)return 2;
    FILE *f=!strcmp(argv[4],"-")?stdin:fopen(argv[4],"rb");if(!f){r1_sdk_destroy(s);return 2;}
    unsigned char b[640];uint64_t at=0;size_t n,used;int rc=0;
    while((n=fread(b,1,sizeof(b),f))){rc=r1_sdk_feed(s,at,b,n,&used);at+=used;if(rc)break;}
    if(ferror(f))rc=1;
    if(f!=stdin)fclose(f);
    if(!rc)rc=r1_sdk_finish(s);
    fprintf(stderr,"{\"status\":\"%s\",\"pcm_bytes\":%" PRIu64 ",\"frames\":%" PRIu64 ",\"proposals\":%" PRIu64 ",\"state_bytes\":%zu,\"weight_bytes\":%zu,\"synthetic_eof_samples\":0,\"physical_ssc305\":false}\n",
      rc?"failed":"complete",at,r1_sdk_frames(s),r1_sdk_proposals(s),r1_sdk_state_bytes(),r1_sdk_weight_bytes());
    r1_sdk_destroy(s);return rc?2:0;
}
