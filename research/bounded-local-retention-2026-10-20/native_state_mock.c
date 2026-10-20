/* No-model native state mock. No input files, posterior arrays, frontend,
 * model, kws3_step/process or scientific replay are used. Synthetic hypotheses
 * and synthetic top-three scratch state exercise the actual C policy/advance.
 */
#include <assert.h>
#include <stdio.h>
#include "../source/candidate/kws3.c"
static unsigned checks, cases;
#define CHECK(x) do { ++checks; if (!(x)) { fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); exit(2); } } while(0)
#define CASE(name) do { ++cases; printf("%s%s",cases==1?"":",",name); } while(0)
static kws3_decoder *fresh(void) {
    kws3_decoder *d=NULL;
    CHECK(kws3_create_with_threshold("synthetic-state-only",.7,&d)==KWS3_OK);
    return d;
}
static window *win(state *s,size_t index,int64_t start,int64_t stop,int64_t last) {
    window *w=&s->windows[index];
    memset(w,0,sizeof(*w));w->start=start;w->stop=stop;w->last_active_pos=last;
    retention_clear(w);local_reset(w);
    if (s->window_count<=index) s->window_count=index+1;
    return w;
}
static void hyp(hypothesis *h,unsigned keyword,int64_t start,int64_t end,double score) {
    size_t i;memset(h,0,sizeof(*h));h->length=4;h->pb=1.;
    memcpy(h->tokens,keywords[keyword-1],sizeof(keywords[0]));
    for(i=0;i<4;++i)h->nodes[i]=(node){start+(int64_t)i*(end-start)/3,1.};
    h->nodes[3].frame=end;h->nodes[0].prob=score*score;
}
static void seed(window *w,unsigned keyword,int64_t start,int64_t end,double score) {
    memset(w->hyps,0,sizeof(w->hyps));w->count=1;hyp(w->hyps,keyword,start,end,score);
}
static kws3_status policy(state *s,window *w,int64_t row,kws3_event *e,int *activated) {
    kws3_status rc=retention_clock(w,row);
    if(rc!=KWS3_OK)return rc;
    *activated=detect(w,row,e);
    return retention_after_detect(s,w,row,e,*activated);
}
static void opened(state *s,window *w,int64_t row) {
    kws3_event e;int active;
    seed(w,1,3*row-15,3*row-6,.5);
    CHECK(policy(s,w,row,&e,&active)==KWS3_OK&&active);
}
static void blank_scratch(kws3_decoder *d) {
    d->work.top_ids[0]=0;d->work.top_probs[0]=1.;
    d->work.top_ids[1]=1;d->work.top_probs[1]=0.;
    d->work.top_ids[2]=2;d->work.top_probs[2]=0.;
}
/* Same transaction boundary as the production step, but bypasses all posterior
 * ingestion. Only hand-constructed internal hypotheses/top-three are inputs. */
static kws3_status mock_row(kws3_decoder *d,kws3_event out[3],size_t *n) {
    kws3_status rc;
    *n=0;
    if(d->committed.metrics.invalid_reason)return KWS3_INVALID;
    blank_scratch(d);d->trial=d->committed;
    rc=advance(d,out,n);
    if(rc!=KWS3_OK){*n=0;return fail(d,rc);}
    d->committed=d->trial;return KWS3_OK;
}
static kws3_status merge_one(state *s,kws3_event *e,int64_t row,kws3_event out[3],size_t *n) {
    kws3_event candidates[3];kws3_status rc;candidates[0]=*e;
    rc=merge(s,candidates,1,row,out,n);*e=candidates[0];return rc;
}
static void fill_seen(state *s,int64_t end) {
    size_t i;s->seen_count=KWS3_DEDUP_KEYS;
    for(i=0;i<KWS3_DEDUP_KEYS;++i){s->seen[i].keyword_id=2;s->seen[i].frames[0]=(int64_t)i;
        s->seen[i].frames[1]=1;s->seen[i].frames[2]=2;s->seen[i].frames[3]=end;}
}
int main(void) {
    kws3_decoder *d,*other;window *w;state *s,*saved;
    kws3_event e,out[3],copy;size_t n;int active;int64_t last;
    kws3_retention_summary snap,a,b;
    printf("{\"status\":\"PASS_NATIVE_SYNTHETIC_STATE_ONLY\",\"cases\":[");

    CASE("\"new_window_sentinel\"");d=fresh();CHECK(mock_row(d,out,&n)==KWS3_OK);
    w=&d->committed.windows[0];CHECK(w->anchor_row==-1&&w->deadline_row==-1);kws3_destroy(d);

    CASE("\"first_low_anchors\"");d=fresh();s=&d->committed;w=win(s,0,0,126,-1);opened(s,w,10);
    CHECK(w->anchor_row==10&&w->deadline_row==15&&w->last_active_pos==-1);
    CHECK(s->retention.episodes_started==1&&s->retention.active_episodes==1);
    CASE("\"keyword_node_score_change_never_renews\"");seed(w,2,25,33,.69);
    CHECK(policy(s,w,11,&e,&active)==KWS3_OK&&active);CHECK(w->anchor_row==10&&w->deadline_row==15);
    CHECK(s->retention.low_repeated==1);
    CASE("\"no_suffix_before_deadline\"");local_reset(w);CHECK(policy(s,w,14,&e,&active)==KWS3_OK&&!active);
    CHECK(w->anchor_row==10);
    CASE("\"no_suffix_expiry_preserves_last_active\"");w->last_active_pos=42;
    CHECK(policy(s,w,15,&e,&active)==KWS3_OK&&!active);
    CHECK(w->anchor_row==-1&&w->deadline_row==-1&&w->last_active_pos==42);
    CHECK(w->count==1&&w->hyps[0].length==0&&w->hyps[0].pb==1.&&s->retention.expired==1);
    CASE("\"new_episode_after_expiry\"");seed(w,1,43,48,.5);
    CHECK(policy(s,w,16,&e,&active)==KWS3_OK&&active);CHECK(w->anchor_row==16&&w->deadline_row==21);
    CHECK(s->retention.episodes_started==2);kws3_destroy(d);

    CASE("\"deadline_low_copied_then_expired_then_merge\"");d=fresh();s=&d->committed;w=win(s,0,0,126,-1);opened(s,w,10);
    seed(w,2,30,42,.6);CHECK(policy(s,w,15,&e,&active)==KWS3_OK&&active);copy=e;
    CHECK(w->count==1&&w->hyps[0].length==0&&w->anchor_row==-1&&e.end_frame==42);
    CHECK(merge_one(s,&e,15,out,&n)==KWS3_OK&&n==0&&s->threshold_rejections==1);
    CHECK(!memcmp(&e,&copy,sizeof(e))&&s->retention.expired==1);kws3_destroy(d);

    CASE("\"deadline_exact_threshold_strong_single_reset\"");d=fresh();s=&d->committed;w=win(s,0,0,126,-1);opened(s,w,10);
    seed(w,1,30,42,.7);CHECK(policy(s,w,15,&e,&active)==KWS3_OK&&active);CHECK(e.score==.7);
    CHECK(w->last_active_pos==42&&w->anchor_row==-1&&w->hyps[0].length==0);
    CHECK(s->retention.strong_closed==1&&s->retention.expired==0);kws3_destroy(d);

    CASE("\"strong_reset_before_duplicate_rejection\"");d=fresh();s=&d->committed;w=win(s,0,0,126,-1);opened(s,w,10);
    seed(w,1,24,39,.8);CHECK(policy(s,w,13,&e,&active)==KWS3_OK&&active);
    s->seen_count=1;s->seen[0].keyword_id=e.keyword_id;memcpy(s->seen[0].frames,e.node_frames,sizeof(e.node_frames));
    CHECK(merge_one(s,&e,13,out,&n)==KWS3_OK&&n==0&&s->metrics.duplicate_candidates==1);
    CHECK(w->last_active_pos==39&&w->anchor_row==-1&&s->retention.strong_closed==1);kws3_destroy(d);

    CASE("\"strong_reset_before_refractory_rejection\"");d=fresh();s=&d->committed;w=win(s,0,0,126,-1);opened(s,w,10);
    seed(w,1,24,39,.8);s->last_accepted_end=30;CHECK(policy(s,w,13,&e,&active)==KWS3_OK&&active);
    CHECK(merge_one(s,&e,13,out,&n)==KWS3_OK&&n==0&&s->metrics.refractory_rejections==1);
    CHECK(w->last_active_pos==39&&w->anchor_row==-1);kws3_destroy(d);

    CASE("\"illegal_short_high_does_not_block_expiry\"");d=fresh();s=&d->committed;w=win(s,0,0,126,-1);opened(s,w,10);
    seed(w,1,40,44,.9);CHECK(policy(s,w,15,&e,&active)==KWS3_OK&&!active);
    CHECK(w->last_active_pos==-1&&s->retention.expired==1&&s->retention.strong_closed==0);kws3_destroy(d);

    CASE("\"illegal_long_high_does_not_block_expiry\"");d=fresh();s=&d->committed;w=win(s,0,0,126,-1);opened(s,w,100);
    seed(w,1,10,270,.9);CHECK(policy(s,w,105,&e,&active)==KWS3_OK&&!active);
    CHECK(w->last_active_pos==-1&&s->retention.expired==1);kws3_destroy(d);

    CASE("\"illegal_last_active_high_does_not_block_expiry\"");d=fresh();s=&d->committed;w=win(s,0,0,126,-1);opened(s,w,10);
    w->last_active_pos=44;seed(w,1,30,42,.9);CHECK(policy(s,w,15,&e,&active)==KWS3_OK&&!active);
    CHECK(w->last_active_pos==44&&s->retention.expired==1);kws3_destroy(d);

    CASE("\"first_illegal_suffix_blocks_later_high\"");d=fresh();s=&d->committed;w=win(s,0,0,126,-1);opened(s,w,10);
    seed(w,1,40,44,.9);hyp(&w->hyps[1],2,30,42,.9);w->count=2;
    CHECK(policy(s,w,15,&e,&active)==KWS3_OK&&!active);CHECK(w->last_active_pos==-1&&s->retention.expired==1);kws3_destroy(d);

    CASE("\"three_windows_independent_clocks\"");d=fresh();s=&d->committed;
    for(size_t i=0;i<3;++i){w=win(s,i,(int64_t)i*42,(int64_t)i*42+126,-1);opened(s,w,95+(int64_t)i);local_reset(w);}
    s->metrics.rows_decoded=100;CHECK(mock_row(d,out,&n)==KWS3_OK);
    CHECK(d->committed.windows[0].anchor_row==-1&&d->committed.windows[1].deadline_row==101&&d->committed.windows[2].deadline_row==102);
    CHECK(d->committed.retention.expired==1&&d->committed.retention.active_episodes==2&&d->committed.retention.peak_active_episodes==3);kws3_destroy(d);

    CASE("\"window_retirement_censors_and_compacts_identity\"");d=fresh();s=&d->committed;
    w=win(s,0,0,126,-1);opened(s,w,124);local_reset(w);
    w=win(s,1,42,168,-1);opened(s,w,123);local_reset(w);
    s->metrics.rows_decoded=125;CHECK(mock_row(d,out,&n)==KWS3_OK);
    CHECK(d->committed.window_count==1&&d->committed.windows[0].start==42&&d->committed.windows[0].anchor_row==123);
    CHECK(d->committed.retention.window_censored==1&&d->committed.retention.active_episodes==1);kws3_destroy(d);

    CASE("\"deadline_on_final_window_row_expires_not_censors\"");d=fresh();s=&d->committed;
    w=win(s,0,0,126,-1);opened(s,w,120);local_reset(w);s->metrics.rows_decoded=125;
    CHECK(mock_row(d,out,&n)==KWS3_OK);CHECK(d->committed.retention.expired==1&&d->committed.retention.window_censored==0&&d->committed.window_count==0);kws3_destroy(d);

    CASE("\"readonly_eof_snapshot_never_finishes_or_resets\"");d=fresh();s=&d->committed;w=win(s,0,0,126,-1);opened(s,w,10);
    saved=malloc(sizeof(*saved));CHECK(saved!=NULL);*saved=*s;
    CHECK(kws3_get_retention_summary(d,&a)==KWS3_OK&&kws3_get_retention_summary(d,&b)==KWS3_OK);
    CHECK(!memcmp(&a,&b,sizeof(a))&&!memcmp(saved,s,sizeof(*s))&&a.active_episodes==1&&a.windows[0].deadline_row==15);free(saved);
    CASE("\"explicit_reset_clears_policy_and_preserves_threshold\"");CHECK(kws3_reset(d,"synthetic-reset")==KWS3_OK);
    CHECK(kws3_get_retention_summary(d,&snap)==KWS3_OK&&snap.episodes_started==0&&snap.active_episodes==0&&d->committed.threshold==.7);kws3_destroy(d);

    CASE("\"row_after_deadline_rejected_before_high_can_rescue\"");d=fresh();s=&d->committed;w=win(s,0,0,126,-1);opened(s,w,10);
    seed(w,1,30,42,.9);s->metrics.rows_decoded=16;
    CHECK(mock_row(d,out,&n)==KWS3_INVARIANT&&n==0);CHECK(d->committed.windows[0].anchor_row==10&&d->committed.windows[0].last_active_pos==-1);
    CHECK(mock_row(d,out,&n)==KWS3_INVALID);kws3_destroy(d);

    CASE("\"checked_anchor_plus_five_overflow\"");d=fresh();s=&d->committed;w=win(s,0,0,INT64_MAX,-1);e.score=.5;
    CHECK(retention_after_detect(s,w,INT64_MAX-4,&e,1)==KWS3_CLOCK);
    CHECK(w->anchor_row==-1&&s->retention.episodes_started==0);kws3_destroy(d);

    CASE("\"legal_max_row_deadline_representable\"");d=fresh();s=&d->committed;w=win(s,0,0,INT64_MAX,-1);e.score=.5;
    CHECK(retention_after_detect(s,w,KWS3_MAX_ROWS-1,&e,1)==KWS3_OK);
    CHECK(w->deadline_row==KWS3_MAX_ROWS+4);kws3_destroy(d);

    CASE("\"inconsistent_sentinel_and_deadline_rejected\"");d=fresh();s=&d->committed;w=win(s,0,0,126,-1);
    w->deadline_row=15;CHECK(retention_clock(w,10)==KWS3_INVARIANT);
    w->anchor_row=10;w->deadline_row=16;CHECK(retention_clock(w,10)==KWS3_INVARIANT);kws3_destroy(d);

    CASE("\"counter_overflow_aborts_trial\"");d=fresh();s=&d->committed;w=win(s,0,0,126,-1);seed(w,1,15,24,.5);
    s->metrics.rows_decoded=10;s->retention.episodes_started=UINT64_MAX;
    CHECK(mock_row(d,out,&n)==KWS3_CLOCK&&n==0&&s->windows[0].anchor_row==-1&&s->retention.episodes_started==UINT64_MAX);kws3_destroy(d);

    CASE("\"failure_after_open_rolls_back_clocks_and_counters\"");d=fresh();s=&d->committed;w=win(s,0,0,126,-1);seed(w,1,15,24,.5);
    s->metrics.rows_decoded=10;fill_seen(s,30);CHECK(mock_row(d,out,&n)==KWS3_CAPACITY&&n==0);
    CHECK(d->trial.retention.episodes_started==1&&s->retention.episodes_started==0&&s->windows[0].anchor_row==-1&&s->metrics.rows_decoded==10);
    CHECK(s->metrics.invalid_reason==KWS3_CAPACITY&&!s->metrics.valid);kws3_destroy(d);

    CASE("\"failure_after_expiry_rolls_back_reset_and_counters\"");d=fresh();s=&d->committed;w=win(s,0,0,126,-1);opened(s,w,10);
    seed(w,2,30,42,.5);s->metrics.rows_decoded=15;fill_seen(s,45);last=w->last_active_pos;
    CHECK(mock_row(d,out,&n)==KWS3_CAPACITY&&n==0);CHECK(d->trial.retention.expired==1&&s->retention.expired==0);
    CHECK(s->windows[0].anchor_row==10&&s->windows[0].hyps[0].length==4&&s->windows[0].last_active_pos==last&&s->retention.low_repeated==0);kws3_destroy(d);

    CASE("\"diagnostic_overflow_never_changes_computation\"");d=fresh();other=fresh();s=&d->committed;w=win(s,0,0,126,-1);opened(s,w,10);
    seed(w,2,30,42,.5);s->metrics.rows_decoded=15;other->committed=*s;
    diag_ledger.active=0;CHECK(mock_row(d,out,&n)==KWS3_OK);
    kws3_ledger_begin();diag_ledger.used=KWS3_LEDGER_CAPACITY;
    CHECK(mock_row(other,out,&n)==KWS3_OK&&!kws3_ledger_valid()&&kws3_ledger_dropped()>0);
    CHECK(!memcmp(&d->committed,&other->committed,sizeof(state)));kws3_destroy(d);kws3_destroy(other);

    CASE("\"ledger_kinds_and_attempt_commit_separation\"");kws3_ledger_begin();d=fresh();s=&d->committed;w=win(s,0,0,126,-1);seed(w,1,15,24,.5);
    s->metrics.rows_decoded=10;fill_seen(s,30);CHECK(mock_row(d,out,&n)==KWS3_CAPACITY);
    {int saw_start=0,saw_fail=0;const kws3_ledger_record *r=kws3_ledger_records();
      for(size_t i=0;i<kws3_ledger_size();++i){if(!strcmp(r[i].kind,"retention_started"))saw_start=1;
        if(!strcmp(r[i].kind,"decoder_failure")){saw_fail=1;}
        CHECK(strcmp(r[i].kind,"row_committed")!=0);}
      CHECK(saw_start&&saw_fail&&s->retention.episodes_started==0);}
    kws3_destroy(d);

    CASE("\"finished_snapshot_explicitly_unsupported\"");d=fresh();d->committed.metrics.finished=1;
    CHECK(kws3_get_retention_summary(d,&snap)==KWS3_FINISHED);kws3_destroy(d);

    CASE("\"deadline_then_next_row_cannot_reuse_old_suffix\"");d=fresh();s=&d->committed;w=win(s,0,0,126,-1);opened(s,w,10);
    seed(w,1,30,42,.5);s->metrics.rows_decoded=15;CHECK(mock_row(d,out,&n)==KWS3_OK);
    CHECK(d->committed.metrics.candidates_seen==1&&d->committed.retention.expired==1);
    CHECK(mock_row(d,out,&n)==KWS3_OK&&d->committed.metrics.candidates_seen==1&&d->committed.windows[0].hyps[0].length==0);kws3_destroy(d);

    CASE("\"three_candidate_handoff_and_original_merge_capacity\"");d=fresh();s=&d->committed;
    for(size_t i=0;i<3;++i){w=win(s,i,(int64_t)i*42,(int64_t)i*42+126,-1);opened(s,w,95);seed(w,1,270,285,.8);}
    s->metrics.rows_decoded=100;CHECK(mock_row(d,out,&n)==KWS3_OK&&n==1);
    CHECK(d->committed.metrics.candidates_seen==3&&d->committed.metrics.duplicate_candidates==2&&d->committed.retention.strong_closed==3&&d->committed.retention.expired==0);kws3_destroy(d);

    CASE("\"retired_slot_new_window_starts_with_sentinel\"");d=fresh();s=&d->committed;
    w=win(s,0,42,168,-1);opened(s,w,125);local_reset(w);
    w=win(s,1,84,210,-1);opened(s,w,125);local_reset(w);
    s->metrics.rows_decoded=126;CHECK(mock_row(d,out,&n)==KWS3_OK);
    CHECK(d->committed.window_count==3&&d->committed.windows[2].start==126&&d->committed.windows[2].anchor_row==-1&&d->committed.windows[2].deadline_row==-1&&d->committed.retention.active_episodes==2);kws3_destroy(d);

    CASE("\"original_event_summary_and_new_api_sizes\"");CHECK(sizeof(kws3_retention_window)==48&&sizeof(kws3_retention_summary)==216);
    CHECK(kws3_retention_summary_bytes()==216&&kws3_retention_window_bytes()==48);
    printf("],\"checks\":%u,\"case_count\":%u,\"sizeof\":{\"event\":%zu,\"summary\":%zu,\"ledger_record\":%zu,\"window\":%zu,\"state\":%zu,\"decoder\":%zu,\"retention_summary\":%zu,\"retention_window\":%zu},\"scientific_posterior_reads\":0,\"model_forward_calls\":0,\"frontend_calls\":0,\"kws3_step_calls\":0,\"kws3_process_calls\":0}\n",
       checks,cases,sizeof(kws3_event),sizeof(kws3_summary),sizeof(kws3_ledger_record),sizeof(window),sizeof(state),sizeof(kws3_decoder),sizeof(kws3_retention_summary),sizeof(kws3_retention_window));
    return 0;
}
