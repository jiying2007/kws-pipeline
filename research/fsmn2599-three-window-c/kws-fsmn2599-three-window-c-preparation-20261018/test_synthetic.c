/* SPDX-License-Identifier: Apache-2.0
 * Small, synthetic, white-box preparation. NOT RUN. No model/audio/file inputs.
 * Compile this one TU; do NOT also link kws3.c (it is included here).
 */
#define KWS3_TESTING 1
#include "kws3.c"
#include <stdio.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); return 1; } } while (0)
static void row(double p[2599], unsigned token,double prob) {
    memset(p,0,2599*sizeof(double)); p[token]=prob; p[2]=1.-prob;
}
struct collected { kws3_event e[8]; size_t n; int overflow; };
static void collect(void *arg,const kws3_event *e) {
    struct collected *c=arg;
    if (c->n==8) c->overflow=1;
    else c->e[c->n++]=*e;
}
static void fill_hyp(kws3_test_hyp *h,const uint16_t *tokens,const int64_t *frames,size_t n) {
    size_t i; memset(h,0,sizeof(*h)); h->length=n; h->pb=1.;
    for (i=0;i<n;++i) { h->tokens[i]=tokens[i]; h->nodes[i]=(kws3_test_node){frames[i],.8}; }
}
int main(void) {
    kws3_decoder *d=NULL;
    double p[2599],probs[3]; uint16_t ids[3];
    kws3_event e[3]; size_t n,i,consumed;
    kws3_summary s;
    static double chunk[25][2599];
    static kws3_test_hyp in[20],before_h[20],out[80];
    struct collected got={0};
    CHECK(kws3_create("synthetic",&d)==KWS3_OK);
    /* 1: C chunk consumes both events and all 25 rows. */
    for (i=0;i<25;++i) row(chunk[i],i<4?keywords[0][i]:i>=21?keywords[1][i-21]:0,.9);
    CHECK(kws3_process(d,&chunk[0][0],25,2599,collect,&got,&consumed)==KWS3_OK);
    CHECK(consumed==25 && got.n==2 && !got.overflow);
    CHECK(got.e[0].keyword_id==1 && got.e[1].keyword_id==2 && got.e[1].end_frame==72);
    CHECK(kws3_get_summary(d,&s)==KWS3_OK && s.rows_decoded==25);
    /* 2: full alphabet, deterministic ties, exclusive .05 and float promotion. */
    CHECK(kws3_reset(d,"top3")==KWS3_OK);
    memset(p,0,sizeof(p)); p[0]=p[1027]=p[1401]=p[1462]=.25;
    CHECK(kws3_test_top3(d,p,2599,ids,probs)==KWS3_OK);
    CHECK(ids[0]==0 && ids[1]==1027 && ids[2]==1401);
    row(p,1027,.05); CHECK(kws3_step(d,p,2599,e,3,&n)==KWS3_OK);
    CHECK(d->committed.windows[0].hyps[0].length==0);
    { float pf[2599]={0}; pf[1027]=.05f; pf[2]=.95f;
      CHECK(kws3_reset(d,"float-promotion")==KWS3_OK);
      CHECK(kws3_step_f32(d,pf,2599,e,3,&n)==KWS3_OK);
      CHECK(d->committed.windows[0].hyps[0].nodes[0].prob==(double).05f); }
    /* 3: invalid input latch and explicit reset; no partial row commit. */
    CHECK(kws3_reset(d,"invalid")==KWS3_OK);
    row(p,0,.9); CHECK(kws3_step(d,p,6,e,3,&n)==KWS3_POSTERIOR && n==0);
    CHECK(d->committed.metrics.rows_decoded==0);
    CHECK(kws3_step(d,p,2599,e,3,&n)==KWS3_INVALID);
    CHECK(kws3_reset(d,"capacity")==KWS3_OK);
    CHECK(kws3_step(d,p,2599,e,2,&n)==KWS3_CAPACITY && n==0);
    CHECK(d->committed.metrics.rows_decoded==0 && d->committed.window_count==0);
    { char longid[257]; memset(longid,'x',256); longid[256]=0;
      CHECK(kws3_reset(d,longid)==KWS3_ARGUMENT);
      CHECK(d->committed.metrics.invalid_reason==KWS3_CAPACITY); }
    /* 4: tiny positive branches, input isolation, retained blank-last-writer. */
    { uint16_t t[]={1027}; int64_t f[]={0};
      fill_hyp(&in[0],t,f,1); in[0].pb=4e-10; in[0].pnb=2e-10;
      memcpy(before_h,in,sizeof(in)); row(p,1027,.6);
      CHECK(kws3_test_beam(d,p,in,1,6,out,80,&n)==KWS3_OK);
      CHECK(!memcmp(before_h,in,sizeof(in)) && n==2);
      CHECK(out[0].pb+out[0].pnb>0 && out[1].pb+out[1].pnb>0);
      memset(p,0,sizeof(p)); p[1027]=.6; p[0]=.4;
      CHECK(kws3_test_beam(d,p,in,1,6,out,80,&n)==KWS3_OK);
      for(i=0;i<n;++i) if(out[i].length==1) CHECK(out[i].nodes[0].frame==0); }
    /* 5: inclusive 5..250 duration; first invalid match blocks later valid one. */
    { int64_t f[]={0,1,2,5},last; int active;
      fill_hyp(&in[0],keywords[0],f,4);
      CHECK(kws3_test_detect_seed(in,1,-1,100,e,&active,&last)==KWS3_OK && active);
      in[0].nodes[3].frame=250;
      CHECK(kws3_test_detect_seed(in,1,-1,100,e,&active,&last)==KWS3_OK && active);
      in[0].nodes[3].frame=251;
      CHECK(kws3_test_detect_seed(in,1,-1,100,e,&active,&last)==KWS3_OK && !active);
      in[1]=in[0]; in[1].nodes[3].frame=9;
      CHECK(kws3_test_detect_seed(in,2,-1,100,e,&active,&last)==KWS3_OK && !active);
      in[0].nodes[3].frame=4;
      CHECK(kws3_test_detect_seed(in,1,-1,100,e,&active,&last)==KWS3_OK && !active);
      { double score=e[0].score;
        CHECK(fabs(score-.64)<1e-15);
        CHECK(kws3_test_detect_seed(in,1,-1,100,e,&active,&last)==KWS3_OK && !active);
        CHECK(e[0].score==score); }
      for (i=49;i<=50;++i) {
        kws3_event c={0}; size_t j;
        CHECK(kws3_reset(d,"gap-boundary")==KWS3_OK);
        d->committed.last_accepted_end=0;
        c.keyword_id=2; c.score=.8; c.start_frame=(int64_t)i-9; c.end_frame=(int64_t)i;
        for(j=0;j<4;++j) c.node_frames[j]=(int64_t)i-9+(int64_t)(3*j);
        CHECK(kws3_test_merge(d,&c,1,100,e,3,&n)==KWS3_OK);
        CHECK(n==(i==50?1U:0U));
      } }
    /* 6: prefix and clock limits fail closed. */
    CHECK(kws3_reset(d,"prefix-capacity")==KWS3_OK);
    d->committed.metrics.rows_decoded=125; d->committed.window_count=1;
    d->committed.windows[0].start=0; d->committed.windows[0].stop=126;
    d->committed.windows[0].count=1;
    d->committed.windows[0].hyps[0].length=126;
    d->committed.windows[0].hyps[0].pb=1.;
    for(i=0;i<126;++i) d->committed.windows[0].hyps[0].tokens[i]=1027;
    row(p,1462,.9); CHECK(kws3_step(d,p,2599,e,3,&n)==KWS3_CAPACITY);
    CHECK(d->committed.metrics.rows_decoded==125 && d->committed.windows[0].hyps[0].length==126);
    CHECK(kws3_reset(d,"clock")==KWS3_OK);
    d->committed.metrics.rows_decoded=(uint64_t)KWS3_MAX_ROWS;
    CHECK(kws3_step(d,p,2599,e,3,&n)==KWS3_CLOCK);
    /* 7: EOF is idempotent and emits no padding; later row fails. */
    CHECK(kws3_reset(d,"eof")==KWS3_OK);
    row(p,1027,.9); CHECK(kws3_step(d,p,2599,e,3,&n)==KWS3_OK);
    CHECK(kws3_finish(d)==KWS3_OK && kws3_finish(d)==KWS3_OK);
    CHECK(kws3_get_summary(d,&s)==KWS3_OK && s.rows_decoded==1 && s.eof_windows_closed==1);
    CHECK(kws3_step(d,p,2599,e,3,&n)==KWS3_FINISHED);
    /* 8: partial invalid chunk reports exactly the committed prefix. */
    CHECK(kws3_reset(d,"chunk-error")==KWS3_OK);
    row(chunk[0],0,.9); memset(chunk[1],0,sizeof(chunk[1])); row(chunk[2],0,.9);
    CHECK(kws3_process(d,&chunk[0][0],3,2599,NULL,NULL,&consumed)==KWS3_POSTERIOR);
    CHECK(consumed==1 && d->committed.metrics.rows_decoded==1);
    printf("synthetic groups=8 storage_bytes=%zu; no acoustic admission\n",kws3_storage_bytes());
    kws3_destroy(d); return 0;
}
