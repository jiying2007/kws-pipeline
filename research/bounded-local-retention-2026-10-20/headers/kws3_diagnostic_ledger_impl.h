#include "kws3_diagnostic_ledger.h"
static _Thread_local struct {
    kws3_ledger_record records[KWS3_LEDGER_CAPACITY];
    size_t used, delivered;
    uint64_t sequence, dropped;
    int valid, active, draining;
} diag_ledger;
void kws3_ledger_begin(void) {
    if (diag_ledger.draining) { diag_ledger.valid=0; return; }
    memset(&diag_ledger,0,sizeof(diag_ledger));
    diag_ledger.valid=1; diag_ledger.active=1;
}
int kws3_ledger_valid(void) { return diag_ledger.active && diag_ledger.valid; }
size_t kws3_ledger_size(void) { return diag_ledger.used-diag_ledger.delivered; }
size_t kws3_ledger_record_size(void) { return sizeof(kws3_ledger_record); }
uint64_t kws3_ledger_dropped(void) { return diag_ledger.dropped; }
const kws3_ledger_record *kws3_ledger_records(void) { return diag_ledger.records+diag_ledger.delivered; }
static void diag_append(kws3_ledger_record *r) {
    if (!diag_ledger.active) return;
    if (diag_ledger.draining || diag_ledger.used==KWS3_LEDGER_CAPACITY ||
        diag_ledger.sequence==UINT64_MAX) { diag_ledger.valid=0;
        if (diag_ledger.dropped!=UINT64_MAX) ++diag_ledger.dropped;
        return; }
    r->sequence=diag_ledger.sequence++;
    diag_ledger.records[diag_ledger.used++]=*r;
}
int kws3_ledger_drain(kws3_ledger_sink sink, void *user) {
    fenv_t env;
    if (!diag_ledger.active || !diag_ledger.valid || diag_ledger.draining || !sink) {
        diag_ledger.valid=0; return 0;
    }
    if (fegetenv(&env)!=0) { diag_ledger.valid=0; return 0; }
    diag_ledger.draining=1;
    while (diag_ledger.delivered<diag_ledger.used) {
        kws3_ledger_record copy=diag_ledger.records[diag_ledger.delivered];
        if (sink(user,&copy)!=0) { diag_ledger.valid=0; break; }
        ++diag_ledger.delivered;
    }
    diag_ledger.draining=0;
    if (fesetenv(&env)!=0) diag_ledger.valid=0;
    /* Preserve unacknowledged tail on failure; do not retry an uncertain append. */
    if (diag_ledger.delivered==diag_ledger.used) diag_ledger.used=diag_ledger.delivered=0;
    return diag_ledger.valid;
}
static void diag_detect(const char *kind,const window *w,int64_t row,
                        const hypothesis *h,size_t path,uint32_t key,int off,double score) {
    kws3_ledger_record r={0}; size_t n;
    r.kind=kind; r.row=row; r.window_start=w->start;
    r.last_active_or_accepted=w->last_active_pos; r.keyword_id=key;
    r.path_index=path; r.node_offset=off; r.score=score;
    r.has_nodes=1; r.has_probabilities=1;
    for(n=0;n<4;++n) {
        r.node_frames[n]=h->nodes[(size_t)off+n].frame;
        r.node_probabilities[n]=h->nodes[(size_t)off+n].prob;
    }
    r.duration_too_short=r.node_frames[3]-r.node_frames[0]<5;
    r.duration_too_long=r.node_frames[3]-r.node_frames[0]>250;
    r.before_last_active=w->last_active_pos!=-1 && r.node_frames[3]<w->last_active_pos;
    diag_append(&r);
}
static void diag_event(const char *kind,const state *s,const kws3_event *e) {
    kws3_ledger_record r={0};
    r.kind=kind; r.row=e->decision_row; r.window_start=e->window_start;
    r.last_active_or_accepted=s->last_accepted_end; r.keyword_id=e->keyword_id;
    r.path_index=SIZE_MAX; r.node_offset=-1; r.has_nodes=1;
    memcpy(r.node_frames,e->node_frames,sizeof(r.node_frames));
    r.score=e->score; r.threshold=s->threshold; r.has_threshold=1;
    diag_append(&r);
}
static void diag_window(const char *kind,const window *w,int64_t row) {
    kws3_ledger_record r={0};
    r.kind=kind; r.row=row; r.window_start=w->start;
    r.last_active_or_accepted=w->last_active_pos; r.hypothesis_count=w->count;
    r.path_index=SIZE_MAX; r.node_offset=-1; diag_append(&r);
}
static void diag_row(const char *kind,int64_t row,int status) {
    kws3_ledger_record r={0};
    r.kind=kind; r.row=row; r.window_start=-1; r.node_offset=-1;
    r.path_index=SIZE_MAX; r.decoder_status=status; diag_append(&r);
}
#define KWS_DIAG_WINDOW(kind) diag_window(kind,w,row)
#define KWS_DIAG_DETECT(kind) diag_detect(kind,w,row,h,i,key,off,score)
#define KWS_DIAG_EVENT(kind) diag_event(kind,s,e)
#define KWS_DIAG_ROW(kind,row,status) diag_row(kind,row,status)
