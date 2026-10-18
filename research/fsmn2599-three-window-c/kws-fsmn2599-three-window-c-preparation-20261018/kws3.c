/* SPDX-License-Identifier: Apache-2.0
 * Derived from source-pinned WeKws stream_kws_ctc.py, Copyright (c) 2023
 * Jing Du (thuduj12@163.com), with the reviewed wrapper/three-site patches.
 * See NOTICE. Preparation only: this version has NOT been compiled or tested.
 */
#include "kws3.h"
#include <float.h>
#include <fenv.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#if FLT_RADIX != 2 || DBL_MANT_DIG != 53 || DBL_MAX_EXP != 1024
#error "kws3 requires IEEE-754 binary64 double"
#endif
#if FLT_MANT_DIG != 24 || FLT_MAX_EXP != 128
#error "kws3_step_f32 requires IEEE-754 binary32 float"
#endif
#if FLT_EVAL_METHOD != 0
#error "kws3 requires operations evaluated in their declared precision"
#endif
#ifdef __FAST_MATH__
#error "kws3 forbids fast-math; build with -fno-fast-math -ffp-contract=off"
#endif
_Static_assert(sizeof(double) == 8, "binary64 storage required");
_Static_assert(sizeof(float) == 4, "binary32 storage required");

typedef struct { int64_t frame; double prob; } node;
typedef struct {
    double pb, pnb;
    size_t length;
    uint16_t tokens[KWS3_WINDOW_ROWS];
    node nodes[KWS3_WINDOW_ROWS];
} hypothesis;
typedef struct {
    int64_t start, stop, last_active_pos;
    size_t count;
    hypothesis hyps[KWS3_PATHS];
} window;
typedef struct { uint32_t keyword_id; int64_t frames[4]; } dedup_key;
typedef struct {
    kws3_summary metrics;
    int64_t last_accepted_end;
    size_t window_count, seen_count;
    window windows[KWS3_WINDOWS];
    dedup_key seen[KWS3_DEDUP_KEYS];
} state;
typedef struct {
    hypothesis next[KWS3_NEXT_PATHS];
    size_t order[KWS3_NEXT_PATHS], next_count;
    /* Bounded expansion, one partial per original class in the worst case. */
    double partials[KWS3_CLASSES], promoted[KWS3_CLASSES];
    uint16_t top_ids[3];
    double top_probs[3];
} scratch;
struct kws3_decoder { state committed, trial; scratch work; };
static const uint16_t keywords[2][4] = {{1027,1401,1462,2093},{1462,2093,1462,2093}};

/* The caller must mask FP traps. Portable C cannot verify that trap mask.
 * Probes do not change rounding/FTZ/DAZ settings, but may set FP status flags.
 * Volatile operands prevent folding away the arithmetic/conversion checks. */
static int environment_ok(void) {
    volatile double normal=DBL_MIN, half=.5, smallest=DBL_TRUE_MIN, one=1.;
    volatile float float_smallest=FLT_TRUE_MIN;
    double gradual, preserved, promoted;
    if (fegetround()!=FE_TONEAREST) return 0;
    gradual=normal*half;
    preserved=smallest*one;
    promoted=(double)float_smallest;
    return gradual>0. && gradual==0x1p-1023 && preserved>0. &&
           preserved==DBL_TRUE_MIN && promoted==0x1p-149;
}

static kws3_status fail(kws3_decoder *d, kws3_status why) {
    if (d && d->committed.metrics.invalid_reason == KWS3_OK) {
        d->committed.metrics.invalid_reason = why;
        d->committed.metrics.valid = 0;
    }
    return why;
}
static int identity_valid(const char *s, size_t *length) {
    size_t n;
    if (!s) return 0;
    for (n=0; n<KWS3_ID_BYTES; ++n) if (!s[n]) {
        *length=n; return n!=0;
    }
    return 0;
}
static void local_reset(window *w) {
    /* last_active_pos and the fixed lifetime deliberately survive. */
    memset(w->hyps, 0, sizeof(w->hyps));
    w->count=1; w->hyps[0].pb=1.;
}
size_t kws3_storage_bytes(void) { return sizeof(kws3_decoder); }
kws3_status kws3_reset(kws3_decoder *d, const char *id) {
    size_t n;
    char copy[KWS3_ID_BYTES];
    if (!d || !identity_valid(id,&n)) return KWS3_ARGUMENT;
    /* Permit resetting from the current summary's identity without alias loss. */
    memcpy(copy,id,n+1);
    memset(&d->committed,0,sizeof(d->committed));
    memcpy(d->committed.metrics.recording_id,copy,n+1);
    d->committed.metrics.valid=1;
    d->committed.metrics.numerical_support_ok=1;
    d->committed.last_accepted_end=-1;
    return KWS3_OK;
}
kws3_status kws3_create(const char *id, kws3_decoder **out) {
    kws3_decoder *d;
    size_t n;
    if (!out) return KWS3_ARGUMENT;
    *out=NULL;
    if (!identity_valid(id,&n)) return KWS3_ARGUMENT;
    if (!environment_ok()) return KWS3_ENVIRONMENT;
    d=(kws3_decoder *)calloc(1,sizeof(*d));
    if (!d) return KWS3_ALLOCATION;
    (void)kws3_reset(d,id); *out=d; return KWS3_OK;
}
void kws3_destroy(kws3_decoder *d) { free(d); }

/* Positive-input expansion summation with an exact-add residual and final
 * halfway correction. Mirrors the arithmetic contract of math.fsum rather
 * than using ordinary sum or platform-dependent long double. Required build:
 * no reassociation/contraction, binary64 intermediates, round-to-nearest-even.
 * Boundary equivalence still requires the prepared differential tests. */
static kws3_status full_top3(scratch *w, const double *p, size_t classes) {
    size_t n=0,i,j,k;
    double total=0.;
    if (!p || classes!=KWS3_CLASSES) return KWS3_POSTERIOR;
    for (k=0;k<3;++k) { w->top_ids[k]=UINT16_MAX; w->top_probs[k]=-1.; }
    for (i=0;i<classes;++i) {
        double x=p[i];
        size_t write=0;
        if (!isfinite(x) || x<0. || x>1.) return KWS3_POSTERIOR;
        /* Ascending traversal makes exact probability ties token-ID ascending. */
        for (k=0;k<3;++k) if (x>w->top_probs[k]) {
            for (j=2;j>k;--j) {
                w->top_probs[j]=w->top_probs[j-1];
                w->top_ids[j]=w->top_ids[j-1];
            }
            w->top_probs[k]=x; w->top_ids[k]=(uint16_t)i; break;
        }
        for (j=0;j<n;++j) {
            double y=w->partials[j],hi,lo,yr;
            if (fabs(x)<fabs(y)) { double swap=x; x=y; y=swap; }
            hi=x+y; yr=hi-x; lo=y-yr;
            if (lo!=0.) w->partials[write++]=lo;
            x=hi;
        }
        if (x!=0.) {
            if (write>=KWS3_CLASSES) return KWS3_CAPACITY;
            w->partials[write++]=x;
        }
        n=write;
    }
    if (n) {
        double lo=0.;
        total=w->partials[--n];
        while (n) {
            double x=total,y=w->partials[--n],yr;
            total=x+y; yr=total-x; lo=y-yr;
            if (lo!=0.) break;
        }
        if (n && ((lo<0. && w->partials[n-1]<0.) ||
                  (lo>0. && w->partials[n-1]>0.))) {
            double y=lo*2.,x=total+y,yr=x-total;
            if (y==yr) total=x;
        }
    }
    if (fabs(total-1.)>1e-5) return KWS3_POSTERIOR;
    return KWS3_OK;
}
static int retained(uint16_t t) {
    return t==0 || t==1027 || t==1401 || t==1462 || t==2093;
}
/* Dictionary insertion order is represented by next[] position. */
static kws3_status entry(scratch *s, const hypothesis *h, int append,
                         uint16_t token, hypothesis **out) {
    size_t i,n=h->length+(append!=0);
    if (n>KWS3_WINDOW_ROWS) return KWS3_CAPACITY;
    for (i=0;i<s->next_count;++i) {
        hypothesis *a=&s->next[i];
        if (a->length==n && !memcmp(a->tokens,h->tokens,h->length*sizeof(uint16_t)) &&
            (!append || a->tokens[n-1]==token)) { *out=a; return KWS3_OK; }
    }
    if (s->next_count>=KWS3_NEXT_PATHS) return KWS3_CAPACITY;
    *out=&s->next[s->next_count++];
    memset(*out,0,sizeof(**out));
    (*out)->length=n;
    memcpy((*out)->tokens,h->tokens,h->length*sizeof(uint16_t));
    if (append) (*out)->tokens[n-1]=token;
    return KWS3_OK;
}
static void copy_nodes(hypothesis *to, const hypothesis *from) {
    memcpy(to->nodes,from->nodes,from->length*sizeof(node));
}
static kws3_status beam(scratch *s, const hypothesis *cur, size_t count,
                        int64_t frame, size_t max_length) {
    size_t k,i,j,filtered=0;
    kws3_status rc;
    s->next_count=0;
    if (count>KWS3_PATHS || max_length>KWS3_WINDOW_ROWS) return KWS3_CAPACITY;
    for (k=0;k<3;++k) if (s->top_probs[k]>.05 && retained(s->top_ids[k])) ++filtered;
    if (!filtered) {
        memcpy(s->next,cur,count*sizeof(*cur)); s->next_count=count;
        for (i=0;i<count;++i) s->order[i]=i;
        return KWS3_OK;
    }
    /* Ordering intentionally token outer / old hypothesis inner. */
    for (k=0;k<3;++k) {
        const uint16_t token=s->top_ids[k];
        const double ps=s->top_probs[k];
        if (!(ps>.05) || !retained(token)) continue;
        for (i=0;i<count;++i) {
            const hypothesis *h=&cur[i];
            hypothesis *a;
            if (h->length>KWS3_WINDOW_ROWS) return KWS3_CAPACITY;
            if (token==0) {
                if ((rc=entry(s,h,0,token,&a))!=KWS3_OK) return rc;
                a->pb=a->pb+h->pb*ps;
                a->pb=a->pb+h->pnb*ps;
                /* Preserve upstream blank-last-writer metadata policy. */
                copy_nodes(a,h);
            } else if (h->length && token==h->tokens[h->length-1]) {
                if (h->pnb>0.) {
                    if ((rc=entry(s,h,0,token,&a))!=KWS3_OK) return rc;
                    a->pnb=a->pnb+h->pnb*ps; copy_nodes(a,h);
                    /* Value-owned nodes provide the patched copy-on-write isolation. */
                    if (ps>a->nodes[a->length-1].prob) {
                        a->nodes[a->length-1].prob=ps;
                        a->nodes[a->length-1].frame=frame;
                    }
                }
                if (h->pb>0.) {
                    if ((rc=entry(s,h,1,token,&a))!=KWS3_OK) return rc;
                    a->pnb=a->pnb+h->pb*ps; copy_nodes(a,h);
                    a->nodes[a->length-1]=(node){frame,ps};
                }
            } else {
                size_t before=s->next_count;
                if ((rc=entry(s,h,1,token,&a))!=KWS3_OK) return rc;
                /* A freshly created entry has no node list yet. An existing
                 * entry preserves prior nodes except a strictly better peak. */
                if (s->next_count!=before) {
                    copy_nodes(a,h); a->nodes[a->length-1]=(node){frame,ps};
                } else if (ps>a->nodes[a->length-1].prob) {
                    a->nodes[a->length-1]=(node){frame,ps};
                }
                a->pnb=a->pnb+h->pb*ps;
                a->pnb=a->pnb+h->pnb*ps;
            }
        }
    }
    for (i=0;i<s->next_count;++i) {
        hypothesis *h=&s->next[i];
        if (h->length>max_length) return KWS3_CAPACITY;
        if (!isfinite(h->pb) || !isfinite(h->pnb) || h->pb<0. || h->pnb<0.)
            return KWS3_INVARIANT;
        s->order[i]=i;
    }
    /* Stable insertion sort of indices, never lexicographic path tie breaking. */
    for (i=1;i<s->next_count;++i) {
        size_t idx=s->order[i];
        double mass=s->next[idx].pb+s->next[idx].pnb;
        j=i;
        while (j && s->next[s->order[j-1]].pb+s->next[s->order[j-1]].pnb<mass) {
            s->order[j]=s->order[j-1]; --j;
        }
        s->order[j]=idx;
    }
    return KWS3_OK;
}
static int suffix(const hypothesis *h, uint32_t keyword) {
    size_t i;
    if (h->length<4) return -1;
    for (i=0;i<=h->length-4;++i)
        if (!memcmp(h->tokens+i,keywords[keyword-1],4*sizeof(uint16_t))) return (int)i;
    return -1;
}
static int detect(window *w, int64_t row, kws3_event *e) {
    size_t i,j;
    uint32_t key;
    for (i=0;i<w->count;++i) for (key=1;key<=2;++key) {
        const hypothesis *h=&w->hyps[i];
        int off=suffix(h,key);
        if (off>=0) {
            double score=1.;
            int64_t start=h->nodes[off].frame,end=h->nodes[off+3].frame;
            for (j=0;j<4;++j) score*=h->nodes[(size_t)off+j].prob;
            score=sqrt(score);
            /* Test-only detect_seed can inspect a rejected attempt's score;
             * this slot is never emitted unless detect returns one. */
            e->score=score;
            /* The first path/keyword hit blocks further search even when its
             * duration is invalid. Local interval=0 still preserves last_active. */
            if (end-start<5 || end-start>250 ||
                (w->last_active_pos!=-1 && end<w->last_active_pos)) return 0;
            memset(e,0,sizeof(*e)); e->keyword_id=key; e->score=score;
            for (j=0;j<4;++j) e->node_frames[j]=h->nodes[(size_t)off+j].frame;
            e->start_frame=start; e->end_frame=end;
            e->window_start=w->start; e->decision_row=row; e->decision_frame=3*row;
            e->diagnostic_candidate=1; e->grid_seconds=.01;
            w->last_active_pos=end;
            local_reset(w); /* Before deduplication/refractory, unconditionally. */
            return 1;
        }
    }
    return 0;
}
static int before(const kws3_event *a, const kws3_event *b) {
    size_t j;
    if (a->score!=b->score) return a->score>b->score;
    if (a->end_frame!=b->end_frame) return a->end_frame<b->end_frame;
    if (a->keyword_id!=b->keyword_id) return a->keyword_id<b->keyword_id;
    if (a->window_start!=b->window_start) return a->window_start<b->window_start;
    for (j=0;j<4;++j) if (a->node_frames[j]!=b->node_frames[j])
        return a->node_frames[j]<b->node_frames[j];
    return 0;
}
static int same_key(const dedup_key *key, const kws3_event *e) {
    return key->keyword_id==e->keyword_id &&
           !memcmp(key->frames,e->node_frames,sizeof(key->frames));
}
static kws3_status merge(state *s, kws3_event candidates[3], size_t count,
                         int64_t row, kws3_event out[3], size_t *emitted) {
    size_t i,j,keep=0;
    int64_t cutoff=3*row-378;
    *emitted=0;
    if (count>3) return KWS3_CAPACITY;
    /* Linear fixed-size set replaces heap+set without changing expiry/key semantics. */
    for (i=0;i<s->seen_count;++i) if (s->seen[i].frames[3]>cutoff)
        s->seen[keep++]=s->seen[i];
    s->seen_count=keep;
    for (i=1;i<count;++i) {
        kws3_event e=candidates[i]; j=i;
        while (j && before(&e,&candidates[j-1])) { candidates[j]=candidates[j-1]; --j; }
        candidates[j]=e;
    }
    for (i=0;i<count;++i) {
        kws3_event *e=&candidates[i];
        for (j=0;j<s->seen_count;++j) if (same_key(&s->seen[j],e)) break;
        if (j<s->seen_count) { ++s->metrics.duplicate_candidates; continue; }
        if (s->seen_count>=KWS3_DEDUP_KEYS) return KWS3_CAPACITY;
        s->seen[s->seen_count].keyword_id=e->keyword_id;
        memcpy(s->seen[s->seen_count++].frames,e->node_frames,sizeof(e->node_frames));
        if (s->last_accepted_end!=-1 && e->end_frame-s->last_accepted_end<50) {
            ++s->metrics.refractory_rejections; continue;
        }
        s->last_accepted_end=e->end_frame;
        e->event_index=s->metrics.events_emitted++;
        memcpy(e->recording_id,s->metrics.recording_id,KWS3_ID_BYTES);
        out[(*emitted)++]=*e;
    }
    if (s->seen_count>s->metrics.max_dedup_keys) s->metrics.max_dedup_keys=s->seen_count;
    return KWS3_OK;
}
static kws3_status advance(kws3_decoder *d, kws3_event out[3], size_t *emitted) {
    state *s=&d->trial;
    scratch *work=&d->work;
    kws3_event candidates[3];
    size_t i,j,count=0,refs=0,keep=0;
    const int64_t row=(int64_t)s->metrics.rows_decoded;
    kws3_status rc;
    if (row%KWS3_HOP_ROWS==0) {
        window *w;
        if (s->window_count>=KWS3_WINDOWS) return KWS3_CAPACITY;
        w=&s->windows[s->window_count++]; memset(w,0,sizeof(*w));
        w->start=row; w->stop=row+126; w->last_active_pos=-1; local_reset(w);
        ++s->metrics.windows_started;
    }
    if (s->window_count>s->metrics.max_live_windows) s->metrics.max_live_windows=s->window_count;
    for (i=0;i<s->window_count;++i) {
        window *w=&s->windows[i];
        size_t maxlen=0;
        int positive=0;
        if (!(w->start<=row && row<w->stop)) return KWS3_INVARIANT;
        rc=beam(work,w->hyps,w->count,3*row,(size_t)(row-w->start+1));
        if (rc!=KWS3_OK) return rc;
        ++s->metrics.beam_updates;
        if (work->next_count>s->metrics.max_next_hypotheses)
            s->metrics.max_next_hypotheses=work->next_count;
        w->count=work->next_count<KWS3_PATHS ? work->next_count : KWS3_PATHS;
        for (j=0;j<w->count;++j) {
            w->hyps[j]=work->next[work->order[j]];
            if (w->hyps[j].length>maxlen) maxlen=w->hyps[j].length;
            if (w->hyps[j].pb+w->hyps[j].pnb>0.) positive=1;
        }
        if (!w->count) ++s->metrics.empty_beam_window_rows;
        if (w->count && !positive) ++s->metrics.zero_mass_window_rows;
        if (maxlen>s->metrics.max_prefix_seen) s->metrics.max_prefix_seen=maxlen;
        if (detect(w,row,&candidates[count])) ++count;
    }
    ++s->metrics.topk_computations;
    for (i=0;i<s->window_count;++i) for (j=0;j<s->windows[i].count;++j)
        refs+=s->windows[i].hyps[j].length;
    if (refs>s->metrics.max_live_node_references) s->metrics.max_live_node_references=refs;
    s->metrics.candidates_seen+=count;
    rc=merge(s,candidates,count,row,out,emitted);
    if (rc!=KWS3_OK) return rc;
    ++s->metrics.rows_decoded;
    for (i=0;i<s->window_count;++i) {
        if (s->windows[i].stop>(int64_t)s->metrics.rows_decoded) {
            if (keep!=i) s->windows[keep]=s->windows[i];
            ++keep;
        } else ++s->metrics.windows_retired;
    }
    s->window_count=keep;
    return KWS3_OK;
}
kws3_status kws3_step(kws3_decoder *d, const double *p, size_t classes,
                     kws3_event *events, size_t cap, size_t *count) {
    kws3_status rc;
    kws3_event output[3];
    size_t n=0;
    if (count) *count=0;
    if (!d) return KWS3_ARGUMENT;
    if (d->committed.metrics.invalid_reason!=KWS3_OK) return KWS3_INVALID;
    if (!count || !events) return fail(d,KWS3_ARGUMENT);
    if (cap<3) return fail(d,KWS3_CAPACITY);
    if (d->committed.metrics.finished) return fail(d,KWS3_FINISHED);
    if (!environment_ok()) return fail(d,KWS3_ENVIRONMENT);
    if (d->committed.metrics.rows_decoded>=(uint64_t)KWS3_MAX_ROWS) return fail(d,KWS3_CLOCK);
    if ((rc=full_top3(&d->work,p,classes))!=KWS3_OK) return fail(d,rc);
    d->trial=d->committed;
    if ((rc=advance(d,output,&n))!=KWS3_OK) return fail(d,rc);
    d->committed=d->trial;
    memcpy(events,output,n*sizeof(*events)); *count=n;
    return KWS3_OK;
}
kws3_status kws3_step_f32(kws3_decoder *d, const float *p, size_t classes,
                         kws3_event *events, size_t cap, size_t *count) {
    size_t i;
    if (count) *count=0;
    if (!d) return KWS3_ARGUMENT;
    if (d->committed.metrics.invalid_reason!=KWS3_OK) return KWS3_INVALID;
    if (!p || classes!=KWS3_CLASSES) return fail(d,KWS3_POSTERIOR);
    if (!environment_ok()) return fail(d,KWS3_ENVIRONMENT);
    for (i=0;i<classes;++i) d->work.promoted[i]=(double)p[i];
    return kws3_step(d,d->work.promoted,classes,events,cap,count);
}
kws3_status kws3_process(kws3_decoder *d, const double *rows, size_t num_rows,
                        size_t stride, kws3_event_sink sink, void *user,
                        size_t *consumed) {
    size_t i,j,n;
    kws3_status rc;
    kws3_event events[3];
    if (consumed) *consumed=0;
    if (!d) return KWS3_ARGUMENT;
    if (d->committed.metrics.invalid_reason!=KWS3_OK) return KWS3_INVALID;
    if (!consumed) return fail(d,KWS3_ARGUMENT);
    if (!num_rows) return KWS3_OK;
    if (!rows || stride<KWS3_CLASSES) return fail(d,KWS3_ARGUMENT);
    /* Bound both element and byte arithmetic before pointer stepping. */
    if (num_rows-1>(SIZE_MAX/sizeof(double)-KWS3_CLASSES)/stride)
        return fail(d,KWS3_CAPACITY);
    for (i=0;i<num_rows;++i) {
        rc=kws3_step(d,rows+i*stride,KWS3_CLASSES,events,3,&n);
        if (rc!=KWS3_OK) return rc;
        ++*consumed;
        if (sink) for (j=0;j<n;++j) sink(user,&events[j]);
    }
    return KWS3_OK;
}
kws3_status kws3_finish(kws3_decoder *d) {
    if (!d) return KWS3_ARGUMENT;
    if (d->committed.metrics.invalid_reason!=KWS3_OK) return KWS3_INVALID;
    if (!d->committed.metrics.finished) {
        d->committed.metrics.eof_windows_closed+=d->committed.window_count;
        d->committed.window_count=0; d->committed.seen_count=0;
        d->committed.metrics.finished=1;
    }
    return KWS3_OK;
}
kws3_status kws3_get_summary(const kws3_decoder *d, kws3_summary *out) {
    if (!d || !out) return KWS3_ARGUMENT;
    *out=d->committed.metrics;
    out->live_windows=d->committed.window_count; out->dedup_keys=d->committed.seen_count;
    out->valid=out->invalid_reason==KWS3_OK;
    out->numerical_support_ok=!out->zero_mass_window_rows && !out->empty_beam_window_rows;
    return KWS3_OK;
}
const char *kws3_status_string(kws3_status s) {
    static const char *const names[]={"ok","invalid argument/identity","invalid full posterior",
      "explicit capacity exceeded","clock limit exceeded","posterior after EOF",
      "recording already invalid","unsupported floating-point environment","allocation failed",
      "decoder invariant failed"};
    return (unsigned)s<sizeof(names)/sizeof(names[0]) ? names[(unsigned)s] : "unknown status";
}
const char *kws3_keyword(uint32_t id) {
    return id==1 ? "你好小窝" : id==2 ? "小窝小窝" : NULL;
}

#ifdef KWS3_TESTING
#include "kws3_test.h"
_Static_assert(sizeof(kws3_test_hyp)==sizeof(hypothesis),"test ABI layout");
_Static_assert(offsetof(kws3_test_hyp,nodes)==offsetof(hypothesis,nodes),"test ABI nodes");
static int test_hyp_valid(const kws3_test_hyp *h, int64_t frame) {
    size_t i;
    if (h->length>126 || !isfinite(h->pb) || !isfinite(h->pnb) || h->pb<0 || h->pnb<0)
        return 0;
    for (i=0;i<h->length;++i) if (!retained(h->tokens[i]) || h->tokens[i]==0 ||
        h->nodes[i].frame<0 || h->nodes[i].frame>frame ||
        !isfinite(h->nodes[i].prob) || h->nodes[i].prob<0 || h->nodes[i].prob>1) return 0;
    return 1;
}
kws3_status kws3_test_top3(kws3_decoder *d,const double *p,size_t n,
                         uint16_t ids[3],double probabilities[3]) {
    kws3_status rc;
    if (!d || !ids || !probabilities) return KWS3_ARGUMENT;
    if ((rc=full_top3(&d->work,p,n))!=KWS3_OK) return rc;
    memcpy(ids,d->work.top_ids,sizeof(d->work.top_ids));
    memcpy(probabilities,d->work.top_probs,sizeof(d->work.top_probs));
    return KWS3_OK;
}
kws3_status kws3_test_beam(kws3_decoder *d,const double *p,
                         const kws3_test_hyp *input,size_t n,int64_t frame,
                         kws3_test_hyp *out,size_t cap,size_t *count) {
    hypothesis cur[20];
    kws3_status rc;
    size_t i;
    if (count) *count=0;
    if (!d || !input || !out || !count || frame<0 || frame>3*KWS3_MAX_ROWS)
        return KWS3_ARGUMENT;
    if (n>20 || cap<80) return KWS3_CAPACITY;
    for (i=0;i<n;++i) if (!test_hyp_valid(&input[i],frame)) return KWS3_ARGUMENT;
    memcpy(cur,input,n*sizeof(*cur));
    if ((rc=full_top3(&d->work,p,2599))!=KWS3_OK) return rc;
    if ((rc=beam(&d->work,cur,n,frame,126))!=KWS3_OK) return rc;
    for (i=0;i<d->work.next_count;++i)
        memcpy(&out[i],&d->work.next[d->work.order[i]],sizeof(out[i]));
    *count=d->work.next_count; return KWS3_OK;
}
kws3_status kws3_test_snapshot(const kws3_decoder *d,size_t index,
                         int64_t *start,int64_t *last,kws3_test_hyp *out,
                         size_t cap,size_t *count) {
    const window *w;
    if (!d || !out || !count || !start || !last || index>=d->committed.window_count)
        return KWS3_ARGUMENT;
    w=&d->committed.windows[index];
    if (cap<w->count) return KWS3_CAPACITY;
    memcpy(out,w->hyps,w->count*sizeof(*out)); *count=w->count;
    *start=w->start; *last=w->last_active_pos; return KWS3_OK;
}
kws3_status kws3_test_detect_seed(const kws3_test_hyp *input,size_t n,
                         int64_t last,int64_t row,kws3_event *out,int *activated,
                         int64_t *new_last) {
    window w;
    size_t i;
    if (!input || !out || !activated || !new_last || n>20 || row<0 ||
        row>KWS3_MAX_ROWS || last< -1 || last>3*row) return KWS3_ARGUMENT;
    for (i=0;i<n;++i) if (!test_hyp_valid(&input[i],3*row)) return KWS3_ARGUMENT;
    memset(&w,0,sizeof(w)); memset(out,0,sizeof(*out));
    w.last_active_pos=last; w.count=n;
    memcpy(w.hyps,input,n*sizeof(*input));
    *activated=detect(&w,row,out); *new_last=w.last_active_pos; return KWS3_OK;
}
kws3_status kws3_test_merge(kws3_decoder *d,const kws3_event *input,size_t n,
                         int64_t row,kws3_event *out,size_t cap,size_t *count) {
    kws3_event c[3],events[3];
    kws3_status rc;
    size_t i,j,emitted=0;
    if (count) *count=0;
    if (!d || !input || !out || !count || row<0 || row>KWS3_MAX_ROWS)
        return KWS3_ARGUMENT;
    if (d->committed.metrics.invalid_reason) return KWS3_INVALID;
    if (n>3 || cap<3) return KWS3_CAPACITY;
    for (i=0;i<n;++i) {
        const kws3_event *e=&input[i];
        if (e->keyword_id<1 || e->keyword_id>2 || !isfinite(e->score) ||
            e->score<0 || e->score>1 || e->end_frame!=e->node_frames[3] ||
            e->start_frame!=e->node_frames[0]) return KWS3_ARGUMENT;
        for (j=0;j<4;++j) if (e->node_frames[j]<0 || e->node_frames[j]>3*row)
            return KWS3_ARGUMENT;
    }
    memcpy(c,input,n*sizeof(*c)); d->trial=d->committed;
    rc=merge(&d->trial,c,n,row,events,&emitted);
    if (rc!=KWS3_OK) return fail(d,rc);
    d->committed=d->trial; memcpy(out,events,emitted*sizeof(*out));
    *count=emitted; return KWS3_OK;
}
#endif
