/* SPDX-License-Identifier: Apache-2.0
 * Preparation only. This source has NOT been compiled or executed.
 * Fixed synthetic posterior fixtures; no model, audio, NPZ or Python decoder.
 * Compile this one translation unit, which includes kws3.c, without separately
 * linking kws3.c. KWS3_TESTING is never part of the production build.
 * Exact successful scope: 569 rows, 572 step attempts, 949 committed beam
 * updates, 21 windows started, and 11 test-only direct merge calls.
 */
#define KWS3_TESTING 1
#include "kws3.c"
#include <inttypes.h>
#include <stdio.h>

typedef struct {
    uint64_t completed_rows, step_attempts, beam_updates, windows;
    uint64_t direct_merge_calls;
} budget;
static const budget limit = {569,572,949,21,11};
static const budget fixture_limit[8] = {
    {268,268,552,8,0}, {28,28,28,2,0}, {264,264,360,8,0},
    {0,0,0,0,3}, {0,0,0,0,6}, {8,9,8,2,0},
    {0,0,0,0,2}, {1,3,1,1,0}
};
static budget used, fixture_start, reserved;
static unsigned fixture_id, fixtures_passed;
static const char *operation = "initialization";
static uint64_t operation_row;
static double operation_threshold;

static void json_string(const char *s) {
    const unsigned char *p = (const unsigned char *)s;
    putchar('"');
    for (; *p; ++p) {
        if (*p == '"' || *p == '\\') { putchar('\\'); putchar(*p); }
        else if (*p < 32) printf("\\u%04x", (unsigned)*p);
        else putchar(*p);
    }
    putchar('"');
}
static void print_counts(const char *status, int line, const char *reason) {
    printf("{\"status\":"); json_string(status);
    printf(",\"fixture\":%u,\"fixtures_passed\":%u,\"operation\":",
           fixture_id, fixtures_passed);
    json_string(operation);
    printf(",\"operation_row\":%" PRIu64 ",\"threshold\":%.17g"
           ",\"completed_rows\":%" PRIu64 ",\"step_attempts\":%" PRIu64
           ",\"beam_updates\":%" PRIu64 ",\"windows\":%" PRIu64
           ",\"direct_merge_calls\":%" PRIu64
           ",\"reserved_completed_rows\":%" PRIu64
           ",\"reserved_step_attempts\":%" PRIu64
           ",\"reserved_beam_updates\":%" PRIu64
           ",\"reserved_windows\":%" PRIu64
           ",\"reserved_direct_merge_calls\":%" PRIu64,
           operation_row, operation_threshold, used.completed_rows,
           used.step_attempts, used.beam_updates, used.windows,
           used.direct_merge_calls, reserved.completed_rows,
           reserved.step_attempts, reserved.beam_updates, reserved.windows,
           reserved.direct_merge_calls);
    if (line) { printf(",\"line\":%d,\"check\":",line); json_string(reason); }
    printf("}\n");
    fflush(stdout);
}
static void failed(int line, const char *check) {
    print_counts("FAIL",line,check);
    exit(EXIT_FAILURE);
}
#define CHECK(x) do { if (!(x)) failed(__LINE__,#x); } while (0)

static int budget_equal(budget a, budget b) {
    return a.completed_rows==b.completed_rows && a.step_attempts==b.step_attempts &&
           a.beam_updates==b.beam_updates && a.windows==b.windows &&
           a.direct_merge_calls==b.direct_merge_calls;
}
static budget budget_delta(budget a, budget b) {
    budget c;
    CHECK(a.completed_rows>=b.completed_rows && a.step_attempts>=b.step_attempts &&
          a.beam_updates>=b.beam_updates && a.windows>=b.windows &&
          a.direct_merge_calls>=b.direct_merge_calls);
    c.completed_rows=a.completed_rows-b.completed_rows;
    c.step_attempts=a.step_attempts-b.step_attempts;
    c.beam_updates=a.beam_updates-b.beam_updates;
    c.windows=a.windows-b.windows;
    c.direct_merge_calls=a.direct_merge_calls-b.direct_merge_calls;
    return c;
}
static void reserve_operation(const char *name, const kws3_decoder *d,
                              uint64_t attempts, uint64_t successful_rows,
                              uint64_t merges) {
    uint64_t i, rejections;
    kws3_summary s;
    CHECK(!reserved.completed_rows && !reserved.step_attempts &&
          !reserved.beam_updates && !reserved.windows && !reserved.direct_merge_calls);
    CHECK(kws3_get_summary(d,&s)==KWS3_OK);
    CHECK(kws3_get_threshold_status(d,&operation_threshold,&rejections)==KWS3_OK);
    operation=name; operation_row=s.rows_decoded;
    reserved.completed_rows=successful_rows;
    reserved.step_attempts=attempts;
    reserved.direct_merge_calls=merges;
    for (i=0;i<successful_rows;++i) {
        uint64_t r=s.rows_decoded+i, live=r/KWS3_HOP_ROWS+1;
        reserved.beam_updates+=live>KWS3_WINDOWS ? KWS3_WINDOWS : live;
        if (r%KWS3_HOP_ROWS==0) ++reserved.windows;
    }
    CHECK(used.completed_rows+reserved.completed_rows<=limit.completed_rows);
    CHECK(used.step_attempts+reserved.step_attempts<=limit.step_attempts);
    CHECK(used.beam_updates+reserved.beam_updates<=limit.beam_updates);
    CHECK(used.windows+reserved.windows<=limit.windows);
    CHECK(used.direct_merge_calls+reserved.direct_merge_calls<=limit.direct_merge_calls);
    /* The flushed reservation remains visible if the operation is interrupted. */
    print_counts("RESERVED",0,NULL);
}
static void account_operation(const kws3_summary *before, const kws3_decoder *d,
                              uint64_t attempts, uint64_t merges) {
    kws3_summary after;
    CHECK(kws3_get_summary(d,&after)==KWS3_OK);
    CHECK(after.rows_decoded>=before->rows_decoded &&
          after.beam_updates>=before->beam_updates &&
          after.windows_started>=before->windows_started);
    used.completed_rows+=after.rows_decoded-before->rows_decoded;
    used.beam_updates+=after.beam_updates-before->beam_updates;
    used.windows+=after.windows_started-before->windows_started;
    used.step_attempts+=attempts;
    used.direct_merge_calls+=merges;
    /* Count the actual result before assertions, including unexpected failure. */
    memset(&reserved,0,sizeof(reserved));
    print_counts("CONSUMED",0,NULL);
    CHECK(used.completed_rows<=limit.completed_rows && used.step_attempts<=limit.step_attempts &&
          used.beam_updates<=limit.beam_updates && used.windows<=limit.windows &&
          used.direct_merge_calls<=limit.direct_merge_calls);
}
static kws3_status counted_step(kws3_decoder *d, const double *p,
                                size_t classes, kws3_event out[3], size_t *n,
                                int expected_success) {
    kws3_summary before;
    kws3_status rc;
    CHECK(kws3_get_summary(d,&before)==KWS3_OK);
    reserve_operation("step",d,1,expected_success ? 1 : 0,0);
    rc=kws3_step(d,p,classes,out,3,n);
    account_operation(&before,d,1,0);
    return rc;
}
struct collected { kws3_event e[16]; size_t n; int overflow; };
static void collect(void *arg,const kws3_event *e) {
    struct collected *c=(struct collected *)arg;
    if (c->n==sizeof(c->e)/sizeof(c->e[0])) c->overflow=1;
    else c->e[c->n++]=*e;
}
static kws3_status counted_process(kws3_decoder *d, const double *p,
                                   size_t rows, struct collected *events,
                                   size_t *consumed) {
    kws3_summary before;
    kws3_status rc;
    uint64_t attempts;
    CHECK(rows>0 && p && events && consumed);
    CHECK(kws3_get_summary(d,&before)==KWS3_OK);
    CHECK(before.valid && !before.finished);
    reserve_operation("process",d,(uint64_t)rows,(uint64_t)rows,0);
    rc=kws3_process(d,p,rows,KWS3_CLASSES,collect,events,consumed);
    /* All process arguments are valid. Each committed prefix row called step;
     * an unexpected failure adds exactly the first unsuccessful step call. */
    attempts=(uint64_t)*consumed;
    if (rc!=KWS3_OK && *consumed<rows) ++attempts;
    account_operation(&before,d,attempts,0);
    CHECK(*consumed<=rows);
    return rc;
}
static kws3_status counted_merge(kws3_decoder *d, const kws3_event *in,
                                 size_t count, int64_t decision,
                                 kws3_event out[3], size_t *n) {
    kws3_summary before;
    kws3_status rc;
    CHECK(kws3_get_summary(d,&before)==KWS3_OK);
    reserve_operation("test_merge",d,0,0,1);
    rc=kws3_test_merge(d,in,count,decision,out,3,n);
    account_operation(&before,d,0,1);
    CHECK(d->committed.metrics.rows_decoded==before.rows_decoded &&
          d->committed.metrics.beam_updates==before.beam_updates &&
          d->committed.metrics.windows_started==before.windows_started);
    return rc;
}
static void begin_fixture(unsigned id) {
    CHECK(id==fixtures_passed+1 && id<=8);
    fixture_id=id; fixture_start=used; operation="fixture_start";
    operation_row=0; operation_threshold=0.;
    print_counts("FIXTURE_START",0,NULL);
}
static void end_fixture(void) {
    CHECK(budget_equal(budget_delta(used,fixture_start),fixture_limit[fixture_id-1]));
    ++fixtures_passed; operation="fixture_complete";
    print_counts("FIXTURE_PASS",0,NULL);
}
static void posterior(double p[KWS3_CLASSES], unsigned token, double probability) {
    CHECK(token<KWS3_CLASSES && token!=2);
    memset(p,0,KWS3_CLASSES*sizeof(*p));
    p[token]=probability; p[2]=1.-probability;
}
static double expected_score(double probability) {
    double x=1.;
    unsigned i;
    for (i=0;i<4;++i) x*=probability;
    return sqrt(x);
}
static void threshold_is(const kws3_decoder *d, double threshold, uint64_t rejections) {
    double got=-1.;
    uint64_t count=UINT64_MAX;
    CHECK(kws3_get_threshold_status(d,&got,&count)==KWS3_OK);
    CHECK(memcmp(&got,&threshold,sizeof(got))==0 && count==rejections);
}
static int events_equal(const kws3_event *a,const kws3_event *b,size_t n) {
    return !memcmp(a,b,n*sizeof(*a));
}
static void states_equal(const kws3_decoder *a,const kws3_decoder *b) {
    kws3_summary sa,sb;
    double ta,tb;
    uint64_t ra,rb;
    CHECK(kws3_get_summary(a,&sa)==KWS3_OK && kws3_get_summary(b,&sb)==KWS3_OK);
    CHECK(!memcmp(&sa,&sb,sizeof(sa)));
    CHECK(kws3_get_threshold_status(a,&ta,&ra)==KWS3_OK &&
          kws3_get_threshold_status(b,&tb,&rb)==KWS3_OK);
    CHECK(!memcmp(&ta,&tb,sizeof(ta)) && ra==rb);
    /* State objects originate in zeroed allocations/resets. Compare complete
     * committed and transactional state, including all fixed-capacity slots. */
    CHECK(!memcmp(&a->committed,&b->committed,sizeof(state)));
    CHECK(!memcmp(&a->trial,&b->trial,sizeof(state)));
}
static void local_reset_is_visible(const kws3_decoder *d, int64_t end) {
    static kws3_test_hyp h[KWS3_PATHS];
    int64_t start,last;
    size_t n;
    CHECK(kws3_test_snapshot(d,0,&start,&last,h,KWS3_PATHS,&n)==KWS3_OK);
    CHECK(start==0 && last==end && n==1 && h[0].length==0 &&
          h[0].pb==1. && h[0].pnb==0.);
}
static kws3_event candidate(uint32_t keyword,int64_t first,double score,
                            int64_t decision) {
    kws3_event e;
    size_t j;
    memset(&e,0,sizeof(e));
    e.keyword_id=keyword; e.score=score;
    for (j=0;j<4;++j) e.node_frames[j]=first+(int64_t)(3*j);
    e.start_frame=first; e.end_frame=first+9;
    e.decision_row=decision; e.decision_frame=3*decision;
    e.diagnostic_candidate=1; e.grid_seconds=.01;
    return e;
}

static void fixture_one(void) {
    kws3_decoder *a=NULL,*b=NULL;
    double p[KWS3_CLASSES];
    kws3_event ea[3],eb[3];
    size_t i,na,nb,total=0;
    begin_fixture(1);
    CHECK(kws3_create("synthetic-zero",&a)==KWS3_OK);
    CHECK(kws3_create_with_threshold("synthetic-zero",0.,&b)==KWS3_OK);
    states_equal(a,b);
    for (i=0;i<134;++i) {
        unsigned token=i==0 ? 1401 : i>=81 && i<85 ? keywords[0][i-81] :
                       i>=130 ? keywords[1][i-130] : 0;
        posterior(p,token,.9);
        CHECK(counted_step(a,p,KWS3_CLASSES,ea,&na,1)==KWS3_OK);
        CHECK(counted_step(b,p,KWS3_CLASSES,eb,&nb,1)==KWS3_OK);
        CHECK(na==nb && events_equal(ea,eb,na));
        states_equal(a,b);
        threshold_is(a,0.,0); threshold_is(b,0.,0);
        total+=na;
    }
    CHECK(total==2 && a->committed.metrics.rows_decoded==134 &&
          a->committed.metrics.beam_updates==276 &&
          a->committed.metrics.windows_started==4);
    CHECK(kws3_finish(a)==KWS3_OK && kws3_finish(b)==KWS3_OK);
    states_equal(a,b);
    kws3_destroy(a); kws3_destroy(b);
    end_fixture();
}
static void fixture_two(void) {
    kws3_decoder *a=NULL,*b=NULL;
    double p[KWS3_CLASSES];
    kws3_event ea[3],eb[3];
    struct collected old={0},high={0};
    size_t i,j,na,nb;
    begin_fixture(2);
    CHECK(kws3_create_with_threshold("synthetic-low-high",0.,&a)==KWS3_OK);
    CHECK(kws3_create_with_threshold("synthetic-low-high",.7,&b)==KWS3_OK);
    for (i=0;i<14;++i) {
        unsigned token=i<4 ? keywords[0][i] : i>=10 ? keywords[1][i-10] : 0;
        double probability=i<4 ? .65 : i>=10 ? .95 : .9;
        posterior(p,token,probability);
        CHECK(counted_step(a,p,KWS3_CLASSES,ea,&na,1)==KWS3_OK);
        CHECK(counted_step(b,p,KWS3_CLASSES,eb,&nb,1)==KWS3_OK);
        for (j=0;j<na;++j) collect(&old,&ea[j]);
        for (j=0;j<nb;++j) collect(&high,&eb[j]);
        if (i==3) {
            CHECK(na==1 && nb==0 && b->committed.seen_count==1 &&
                  b->committed.last_accepted_end==-1 && a->committed.last_accepted_end==9);
            threshold_is(b,.7,1);
            /* Detection resets locally even when threshold rejects globally. */
            local_reset_is_visible(a,9); local_reset_is_visible(b,9);
        }
    }
    CHECK(!old.overflow && !high.overflow && old.n==1 && high.n==1);
    CHECK(old.e[0].keyword_id==1 && old.e[0].score==expected_score(.65) && old.e[0].score<.7);
    CHECK(high.e[0].keyword_id==2 && high.e[0].score==expected_score(.95) &&
          high.e[0].start_frame==30 && high.e[0].end_frame==39 && high.e[0].event_index==0);
    CHECK(a->committed.metrics.refractory_rejections==1 &&
          b->committed.metrics.refractory_rejections==0 && b->committed.last_accepted_end==39);
    local_reset_is_visible(a,39); local_reset_is_visible(b,39);
    threshold_is(a,0.,0); threshold_is(b,.7,1);
    CHECK(kws3_finish(a)==KWS3_OK && kws3_finish(b)==KWS3_OK);
    kws3_destroy(a); kws3_destroy(b);
    end_fixture();
}
static void fixture_three(void) {
    static double rows[66][KWS3_CLASSES];
    static state reference_committed,reference_trial;
    static kws3_summary reference_summary;
    static const size_t chunks[4]={1,7,42,126};
    struct collected reference={0};
    size_t i,partition;
    begin_fixture(3);
    for (i=0;i<66;++i) {
        unsigned token=i>=40 && i<44 ? keywords[0][i-40] :
                       i>=62 ? keywords[1][i-62] : 0;
        posterior(rows[i],token,.9);
    }
    for (partition=0;partition<4;++partition) {
        kws3_decoder *d=NULL;
        kws3_summary summary;
        struct collected got={0};
        CHECK(kws3_create_with_threshold("synthetic-chunks",.5,&d)==KWS3_OK);
        for (i=0;i<66;i+=chunks[partition]) {
            size_t n=66-i<chunks[partition] ? 66-i : chunks[partition],consumed=0;
            CHECK(counted_process(d,&rows[i][0],n,&got,&consumed)==KWS3_OK);
            CHECK(consumed==n && !got.overflow);
        }
        CHECK(got.n==2 && got.e[0].keyword_id==1 && got.e[1].keyword_id==2);
        CHECK(d->committed.metrics.rows_decoded==66 &&
              d->committed.metrics.beam_updates==90 &&
              d->committed.metrics.windows_started==2);
        threshold_is(d,.5,0);
        CHECK(kws3_finish(d)==KWS3_OK && kws3_get_summary(d,&summary)==KWS3_OK);
        if (partition==0) {
            reference=got; reference_committed=d->committed;
            reference_trial=d->trial; reference_summary=summary;
        } else {
            CHECK(got.n==reference.n && events_equal(got.e,reference.e,got.n));
            CHECK(!memcmp(&summary,&reference_summary,sizeof(summary)));
            CHECK(!memcmp(&d->committed,&reference_committed,sizeof(state)));
            CHECK(!memcmp(&d->trial,&reference_trial,sizeof(state)));
        }
        kws3_destroy(d);
    }
    end_fixture();
}
static void fixture_four(void) {
    kws3_decoder *d=NULL;
    kws3_event e,out[3];
    size_t n;
    begin_fixture(4);
    CHECK(kws3_create_with_threshold("synthetic-dedup",.7,&d)==KWS3_OK);
    e=candidate(1,0,.6,3);
    CHECK(counted_merge(d,&e,1,3,out,&n)==KWS3_OK && n==0);
    threshold_is(d,.7,1);
    CHECK(d->committed.seen_count==1 && d->committed.last_accepted_end==-1 &&
          d->committed.metrics.refractory_rejections==0);
    e.score=.9; e.decision_row=4; e.decision_frame=12;
    CHECK(counted_merge(d,&e,1,4,out,&n)==KWS3_OK && n==0);
    threshold_is(d,.7,1);
    CHECK(d->committed.metrics.duplicate_candidates==1 && d->committed.seen_count==1 &&
          d->committed.last_accepted_end==-1);
    e=candidate(2,30,.7,13);
    CHECK(counted_merge(d,&e,1,13,out,&n)==KWS3_OK && n==1);
    CHECK(out[0].score==.7 && out[0].keyword_id==2 && out[0].event_index==0 &&
          d->committed.last_accepted_end==39 && d->committed.seen_count==2);
    threshold_is(d,.7,1);
    kws3_destroy(d);
    end_fixture();
}
static void fixture_five(void) {
    static const unsigned permutations[6][3]={{0,1,2},{0,2,1},{1,0,2},
                                             {1,2,0},{2,0,1},{2,1,0}};
    static state reference_committed,reference_trial;
    kws3_event base[3],in[3],out[3],reference[3];
    size_t i,j,n,reference_n=0;
    begin_fixture(5);
    base[0]=candidate(1,0,.9,100); base[1]=candidate(2,60,.8,100); base[2]=base[0];
    for (i=0;i<6;++i) {
        kws3_decoder *d=NULL;
        CHECK(kws3_create_with_threshold("synthetic-permutations",.7,&d)==KWS3_OK);
        for (j=0;j<3;++j) in[j]=base[permutations[i][j]];
        CHECK(counted_merge(d,in,3,100,out,&n)==KWS3_OK && n==2);
        CHECK(out[0].keyword_id==1 && out[1].keyword_id==2 &&
              out[0].event_index==0 && out[1].event_index==1 &&
              d->committed.metrics.duplicate_candidates==1 &&
              d->committed.metrics.refractory_rejections==0 &&
              d->committed.last_accepted_end==69 && d->committed.seen_count==2);
        threshold_is(d,.7,0);
        if (i==0) {
            memcpy(reference,out,n*sizeof(*out)); reference_n=n;
            reference_committed=d->committed; reference_trial=d->trial;
        } else {
            CHECK(n==reference_n && events_equal(out,reference,n));
            CHECK(!memcmp(&d->committed,&reference_committed,sizeof(state)));
            CHECK(!memcmp(&d->trial,&reference_trial,sizeof(state)));
        }
        kws3_destroy(d);
    }
    end_fixture();
}
static void fixture_six(void) {
    static state before,trial_before;
    kws3_decoder *d=NULL;
    kws3_event out[3];
    kws3_summary s;
    double p[KWS3_CLASSES];
    size_t pass,i,n;
    begin_fixture(6);
    CHECK(kws3_create_with_threshold("synthetic-finish",.9,&d)==KWS3_OK);
    for (pass=0;pass<2;++pass) {
        for (i=0;i<4;++i) {
            posterior(p,keywords[0][i],.9);
            CHECK(counted_step(d,p,KWS3_CLASSES,out,&n,1)==KWS3_OK && n==0);
        }
        threshold_is(d,.9,1); local_reset_is_visible(d,9);
        CHECK(d->committed.last_accepted_end==-1 && d->committed.seen_count==1);
        CHECK(kws3_finish(d)==KWS3_OK && kws3_get_summary(d,&s)==KWS3_OK);
        CHECK(s.rows_decoded==4 && s.finished && s.valid && s.eof_windows_closed==1 &&
              s.live_windows==0 && s.dedup_keys==0 && s.events_emitted==0);
        if (pass==0) {
            before=d->committed; trial_before=d->trial;
            CHECK(kws3_finish(d)==KWS3_OK);
            CHECK(!memcmp(&before,&d->committed,sizeof(state)) &&
                  !memcmp(&trial_before,&d->trial,sizeof(state)));
            posterior(p,0,.9);
            CHECK(counted_step(d,p,KWS3_CLASSES,out,&n,0)==KWS3_FINISHED && n==0);
            before.metrics.invalid_reason=KWS3_FINISHED; before.metrics.valid=0;
            CHECK(!memcmp(&before,&d->committed,sizeof(state)) &&
                  !memcmp(&trial_before,&d->trial,sizeof(state)));
            threshold_is(d,.9,1);
            CHECK(kws3_reset(d,"synthetic-reset")==KWS3_OK);
            threshold_is(d,.9,0);
            CHECK(kws3_get_summary(d,&s)==KWS3_OK && s.rows_decoded==0 &&
                  s.beam_updates==0 && s.windows_started==0 && s.events_emitted==0 &&
                  s.valid && !s.finished && s.invalid_reason==KWS3_OK &&
                  d->committed.last_accepted_end==-1 && !d->committed.seen_count &&
                  !d->committed.window_count && !strcmp(s.recording_id,"synthetic-reset"));
        }
    }
    kws3_destroy(d);
    end_fixture();
}
static void fixture_seven(void) {
    static state before,trial_before;
    static const double invalid[]={-.1,1.1,NAN,INFINITY,-INFINITY};
    static const double endpoints[]={0.,-0.,1.};
    kws3_decoder *d=NULL,*out_handle=NULL;
    kws3_event e,out[3];
    char long_id[KWS3_ID_BYTES+1];
    size_t i,n;
    double threshold;
    uint64_t rejections;
    begin_fixture(7);
    for (i=0;i<sizeof(endpoints)/sizeof(endpoints[0]);++i) {
        CHECK(kws3_create_with_threshold("synthetic-endpoint",endpoints[i],&d)==KWS3_OK);
        threshold_is(d,endpoints[i]==0. ? 0. : 1.,0);
        if (i+1<sizeof(endpoints)/sizeof(endpoints[0])) { kws3_destroy(d); d=NULL; }
    }
    CHECK(d!=NULL); before=d->committed; trial_before=d->trial;
    for (i=0;i<sizeof(invalid)/sizeof(invalid[0]);++i) {
        out_handle=d;
        CHECK(kws3_create_with_threshold("synthetic-invalid",invalid[i],&out_handle)==KWS3_ARGUMENT);
        CHECK(out_handle==NULL);
    }
    memset(long_id,'x',KWS3_ID_BYTES); long_id[KWS3_ID_BYTES]='\0';
    out_handle=d;
    CHECK(kws3_create_with_threshold(NULL,.7,&out_handle)==KWS3_ARGUMENT && out_handle==NULL);
    out_handle=d;
    CHECK(kws3_create_with_threshold("",.7,&out_handle)==KWS3_ARGUMENT && out_handle==NULL);
    out_handle=d;
    CHECK(kws3_create_with_threshold(long_id,.7,&out_handle)==KWS3_ARGUMENT && out_handle==NULL);
    CHECK(kws3_create_with_threshold("synthetic-null-out",.7,NULL)==KWS3_ARGUMENT);
    CHECK(kws3_create_with_threshold("synthetic-null-out",NAN,NULL)==KWS3_ARGUMENT);
    CHECK(kws3_create("synthetic-null-out",NULL)==KWS3_ARGUMENT);
    threshold=-2.; rejections=UINT64_C(123);
    CHECK(kws3_get_threshold_status(NULL,&threshold,&rejections)==KWS3_ARGUMENT);
    CHECK(threshold==-2. && rejections==UINT64_C(123));
    CHECK(kws3_get_threshold_status(d,NULL,&rejections)==KWS3_ARGUMENT && rejections==UINT64_C(123));
    CHECK(kws3_get_threshold_status(d,&threshold,NULL)==KWS3_ARGUMENT && threshold==-2.);
    CHECK(kws3_get_threshold_status(d,NULL,NULL)==KWS3_ARGUMENT);
    CHECK(kws3_reset(d,NULL)==KWS3_ARGUMENT && kws3_reset(d,"")==KWS3_ARGUMENT &&
          kws3_reset(d,long_id)==KWS3_ARGUMENT && kws3_reset(NULL,"synthetic")==KWS3_ARGUMENT);
    CHECK(!memcmp(&before,&d->committed,sizeof(state)) &&
          !memcmp(&trial_before,&d->trial,sizeof(state)));
    threshold_is(d,1.,0);
    e=candidate(1,0,0x1.fffffffffffffp-1,3);
    CHECK(counted_merge(d,&e,1,3,out,&n)==KWS3_OK && n==0);
    threshold_is(d,1.,1);
    CHECK(d->committed.last_accepted_end==-1 && d->committed.seen_count==1);
    e=candidate(2,30,1.,13);
    CHECK(counted_merge(d,&e,1,13,out,&n)==KWS3_OK && n==1);
    CHECK(out[0].score==1. && out[0].keyword_id==2 && out[0].event_index==0 &&
          d->committed.last_accepted_end==39 && d->committed.metrics.refractory_rejections==0);
    threshold_is(d,1.,1);
    kws3_destroy(d);
    end_fixture();
}
static void fixture_eight(void) {
    static state before,trial_before;
    kws3_decoder *d=NULL;
    kws3_event out[3];
    kws3_summary s;
    double p[KWS3_CLASSES]={0};
    size_t n;
    begin_fixture(8);
    CHECK(kws3_create_with_threshold("synthetic-malformed",.7,&d)==KWS3_OK);
    before=d->committed; trial_before=d->trial;
    CHECK(counted_step(d,p,KWS3_CLASSES,out,&n,0)==KWS3_POSTERIOR && n==0);
    before.metrics.invalid_reason=KWS3_POSTERIOR; before.metrics.valid=0;
    CHECK(!memcmp(&before,&d->committed,sizeof(state)) &&
          !memcmp(&trial_before,&d->trial,sizeof(state)));
    threshold_is(d,.7,0);
    posterior(p,0,.9);
    CHECK(counted_step(d,p,KWS3_CLASSES,out,&n,0)==KWS3_INVALID && n==0);
    CHECK(!memcmp(&before,&d->committed,sizeof(state)) &&
          !memcmp(&trial_before,&d->trial,sizeof(state)));
    CHECK(kws3_reset(d,"synthetic-recovered")==KWS3_OK);
    threshold_is(d,.7,0);
    CHECK(counted_step(d,p,KWS3_CLASSES,out,&n,1)==KWS3_OK && n==0);
    CHECK(kws3_get_summary(d,&s)==KWS3_OK && s.valid && !s.finished &&
          s.invalid_reason==KWS3_OK && s.rows_decoded==1 && s.beam_updates==1 &&
          s.windows_started==1 && s.live_windows==1 && s.events_emitted==0);
    CHECK(kws3_finish(d)==KWS3_OK);
    threshold_is(d,.7,0);
    kws3_destroy(d);
    end_fixture();
}
int main(void) {
    print_counts("START",0,NULL);
    fixture_one(); fixture_two(); fixture_three(); fixture_four();
    fixture_five(); fixture_six(); fixture_seven(); fixture_eight();
    CHECK(fixtures_passed==8 && budget_equal(used,limit));
    operation="synthetic_complete";
    print_counts("PASS_DECODER_THRESHOLD_SYNTHETIC_ONLY",0,NULL);
    return EXIT_SUCCESS;
}
