/* SPDX-License-Identifier: Apache-2.0
 * Research PCM16LE streaming CLI. Prints diagnostic greedy phrase proposals,
 * not calibrated wake decisions or keyword-end annotations. No Python needed. */
#include "r1_stream.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <errno.h>
#include <inttypes.h>
typedef struct {unsigned last,ids[4];uint64_t frames[4],cooldown[2],rows,events;unsigned used;} decode;
static int emit(void *user,uint64_t frame,uint64_t source,uint64_t delivery,const float *feat,const float *logits){
    decode *d=(decode *)user;(void)feat;
    unsigned best=0;
    for(unsigned i=1;i<R1_OUTPUTS;i++)if(logits[i]>logits[best])best=i;
    d->rows++;
    if(best==0){d->last=0;return 0;}
    if(best==d->last)return 0;
    d->last=best;
    if(d->used==4){memmove(d->ids,d->ids+1,3*sizeof(unsigned));memmove(d->frames,d->frames+1,3*sizeof(uint64_t));d->used=3;}
    d->ids[d->used]=best;d->frames[d->used]=frame;d->used++;
    static const unsigned words[2][4]={{1027,1401,1462,2093},{1462,2093,1462,2093}};
    if(d->used<4 || frame-d->frames[0]>150)return 0;
    for(unsigned k=0;k<2;k++)if(!memcmp(d->ids,words[k],sizeof(d->ids)) && frame>=d->cooldown[k]){
        d->events++;d->cooldown[k]=frame+50;
        if(printf("{\"kind\":\"uncalibrated_greedy_proposal\",\"keyword_id\":%u,\"model_frame\":%" PRIu64 ",\"required_end_sample\":%" PRIu64 ",\"delivery_end_sample\":%" PRIu64 ",\"synthetic_eof\":false,\"word_end_annotation\":null}\n",k+1,frame,source,delivery)<0)return 1;
        if(fflush(stdout))return 1;
    }
    return 0;
}
static unsigned char *read_exact(const char *name,size_t size){
    FILE *f=fopen(name,"rb");if(!f)return NULL;
    unsigned char *p=(unsigned char *)malloc(size);
    if(!p){fclose(f);return NULL;}
    size_t got=fread(p,1,size,f);int tail=fgetc(f);int bad=ferror(f);fclose(f);
    if(got!=size || tail!=EOF || bad){free(p);return NULL;}return p;
}
int main(int argc,char **argv){
    if(argc!=5 || strcmp(argv[1],"--research")){
        fprintf(stderr,"Usage: %s --research model.f32le frontend.f32le input.pcm-or--\nInput: mono16000Hz signed16 little-endian raw PCM, NOT WAV. Output proposals are uncalibrated.\n",argv[0]);return 2;
    }
    unsigned char *w=read_exact(argv[2],3028860),*t=read_exact(argv[3],83840);
    if(!w||!t){free(w);free(t);fprintf(stderr,"invalid model/table file\n");return 2;}
    decode d;memset(&d,0,sizeof(d));
    r1_stream *s=r1_create(w,3028860,t,83840,emit,&d);free(w);free(t);
    if(!s){fprintf(stderr,"model/table identity or platform rejected\n");return 2;}
    FILE *input=!strcmp(argv[4],"-")?stdin:fopen(argv[4],"rb");
    if(!input){r1_destroy(s);fprintf(stderr,"cannot open PCM\n");return 2;}
    unsigned char buffer[640];size_t n,consumed;int rc=0;uint64_t bytes=0;
    while((n=fread(buffer,1,sizeof(buffer),input))>0){
        rc=r1_feed(s,buffer,n,&consumed);bytes+=consumed;
        if(rc || consumed!=n){rc=rc?rc:1;break;}
    }
    if(ferror(input))rc=1;
    if(input!=stdin)fclose(input);
    if(!rc)rc=r1_finish(s);
    fprintf(stderr,"{\"status\":\"%s\",\"pcm_bytes\":%" PRIu64 ",\"frames\":%" PRIu64 ",\"diagnostic_proposals\":%" PRIu64 ",\"eof_padding_samples\":0,\"ssc305_measured\":false}\n",rc?"failed":"complete",bytes,d.rows,d.events);
    r1_destroy(s);return rc?2:0;
}
