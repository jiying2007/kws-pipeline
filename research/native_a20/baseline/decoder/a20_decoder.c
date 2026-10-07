/* Copyright (c) 2023 Jing Du(thuduj12@163.com), original WeKws algorithm.
 * C11 research port, Apache-2.0. See NOTICE and README.md. */
#include "a20_decoder.h"
#include <float.h>
#include <limits.h>
#include <math.h>
#include <string.h>
#define A20D_MAGIC UINT32_C(0x41323044)
_Static_assert(A20D_NODE_CAP < UINT16_MAX, "node ids must fit uint16_t");
_Static_assert(sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24,
               "requires IEEE-like binary32 float");
_Static_assert(sizeof(double) == 8 && DBL_MANT_DIG == 53,
               "requires binary64 double");

size_t a20d_decoder_bytes(void) { return sizeof(a20d_decoder); }
size_t a20d_workspace_bytes(void) { return sizeof(a20d_workspace); }
static int a20d_valid(const a20d_decoder *d) { return d && d->magic == A20D_MAGIC; }
static void a20d_reset_hyps(a20d_decoder *d) {
    d->hyp_count = 1; d->node_count = 0; d->hit_score = 1.0;
    memset(&d->hyps[0], 0, sizeof(d->hyps[0])); d->hyps[0].pb = 1.0;
}
a20d_status a20d_init(a20d_decoder *d) {
    if (!d) return A20D_INVALID;
    memset(d, 0, sizeof(*d)); d->magic = A20D_MAGIC; d->last_active_pos = -1;
    a20d_reset_hyps(d); return A20D_OK;
}
a20d_status a20d_reset(a20d_decoder *d) {
    if (!a20d_valid(d)) return A20D_INVALID;
    a20d_reset_hyps(d); return A20D_OK;
}
a20d_status a20d_reset_all(a20d_decoder *d) { return a20d_init(d); }

a20d_status a20d_softmax6(const float logits[A20D_CLASSES], float probs[A20D_CLASSES]) {
    double ex[A20D_CLASSES], sum = 0.0, max;
    if (!logits || !probs) return A20D_INVALID;
    for (int i=0; i<A20D_CLASSES; ++i) if (!isfinite(logits[i])) return A20D_INVALID;
    max = logits[0];
    for (int i=1; i<A20D_CLASSES; ++i) if (logits[i] > max) max = logits[i];
    for (int i=0; i<A20D_CLASSES; ++i) { ex[i]=exp((double)logits[i]-max); sum+=ex[i]; }
    for (int i=0; i<A20D_CLASSES; ++i) probs[i]=(float)(ex[i]/sum);
    return A20D_OK;
}
/* Port of libstdc++ small-range nth_element (nth=2) + sort([0,2)).
 * Strict value comparator intentionally has no token-id tie breaker. */
static void a20d_swap(int32_t *a, int32_t *b) { int32_t t=*a; *a=*b; *b=t; }
static int a20d_gt(const float *p, int32_t a, int32_t b) { return p[a] > p[b]; }
a20d_status a20d_top3(const float p[A20D_CLASSES], int32_t out[A20D_SCORE_BEAM]) {
    int32_t q[A20D_CLASSES]={0,1,2,3,4,5}; int lo=0,hi=A20D_CLASSES;
    if (!p || !out) return A20D_INVALID;
    for(int i=0;i<A20D_CLASSES;++i) if(!isfinite(p[i])) return A20D_INVALID;
    while(hi-lo>3) {
        int a=lo+1,b=lo+(hi-lo)/2,c=hi-1, median;
        if(a20d_gt(p,q[a],q[b])) {
            median=a20d_gt(p,q[b],q[c]) ? b : (a20d_gt(p,q[a],q[c]) ? c : a);
        } else {
            median=a20d_gt(p,q[a],q[c]) ? a : (a20d_gt(p,q[b],q[c]) ? c : b);
        }
        a20d_swap(&q[lo],&q[median]);
        int first=lo+1,last=hi; int32_t pivot=q[lo];
        for(;;) {
            while(a20d_gt(p,q[first],pivot)) ++first;
            --last; while(a20d_gt(p,pivot,q[last])) --last;
            if(first>=last) break;
            a20d_swap(&q[first],&q[last]); ++first;
        }
        if(first<=2) lo=first; else hi=first;
    }
    for(int i=lo+1;i<hi;++i) {
        int32_t v=q[i]; int j=i;
        while(j>lo && a20d_gt(p,v,q[j-1])) { q[j]=q[j-1]; --j; }
        q[j]=v;
    }
    if(a20d_gt(p,q[1],q[0])) a20d_swap(&q[1],&q[0]);
    memcpy(out,q,A20D_SCORE_BEAM*sizeof(*out)); return A20D_OK;
}
static int a20d_same(const a20d_decoder *d,const a20d_hyp *a,
                     const a20d_hyp *b,int append,int token) {
    size_t n=(size_t)b->len+(size_t)append;
    if(a->len!=n) return 0;
    for(size_t j=0;j<b->len;++j)
        if(d->nodes[a->node[j]].token!=d->nodes[b->node[j]].token) return 0;
    return !append || d->nodes[a->node[b->len]].token==token;
}
static a20d_status a20d_find(a20d_decoder *d,a20d_workspace *w,size_t *count,
                            const a20d_hyp *cur,int append,int token,a20d_hyp **out) {
    if((size_t)cur->len+(size_t)append>A20D_PREFIX_CAP) return A20D_CAPACITY;
    for(size_t i=0;i<*count;++i) if(a20d_same(d,&w->next[i],cur,append,token)) {
        *out=&w->next[i]; return A20D_OK;
    }
    if(*count>=A20D_NEXT_CAP) return A20D_CAPACITY;
    *out=&w->next[(*count)++]; memset(*out,0,sizeof(**out));
    /* A new dictionary entry's nodes start empty. The branch writes them. */
    return A20D_OK;
}
static a20d_status a20d_newnode(a20d_decoder *d,int token,int64_t frame,double p,uint16_t *id) {
    if(d->node_count>=A20D_NODE_CAP) return A20D_CAPACITY;
    *id=d->node_count++; d->nodes[*id]=(a20d_node){.frame=frame,.prob=p,.token=token};
    return A20D_OK;
}
static a20d_status a20d_copyappend(a20d_decoder *d,a20d_hyp *next,const a20d_hyp *cur,
                                   int token,int64_t frame,double p) {
    if(cur->len>=A20D_PREFIX_CAP) return A20D_CAPACITY;
    uint16_t id; a20d_status st=a20d_newnode(d,token,frame,p,&id); if(st) return st;
    next->len=cur->len+1; memcpy(next->node,cur->node,cur->len*sizeof(cur->node[0]));
    next->node[cur->len]=id; return A20D_OK;
}
static a20d_status a20d_frame(a20d_decoder *d,a20d_workspace *w,int64_t t,const float *p) {
    int32_t top[A20D_SCORE_BEAM]; size_t count=0; int eligible=0;
    a20d_status st=a20d_top3(p,top); if(st) return st;
    for(int k=0;k<A20D_SCORE_BEAM;++k) {
        int s=top[k]; double ps=(double)p[s];
        if(!(ps>0.05)) continue; /* All six retained by exact A20 contract. */
        eligible=1;
        for(size_t h=0;h<d->hyp_count;++h) {
            const a20d_hyp *cur=&d->hyps[h]; a20d_hyp *next;
            int last=cur->len ? d->nodes[cur->node[cur->len-1]].token : -1;
            if(s==0) {
                st=a20d_find(d,w,&count,cur,0,s,&next); if(st) return st;
                next->pb=(next->pb+cur->pb*ps)+cur->pnb*ps;
                next->len=cur->len; memcpy(next->node,cur->node,cur->len*sizeof(cur->node[0]));
            } else if(s==last) {
                if(fabs(cur->pnb)>0.000001) {
                    st=a20d_find(d,w,&count,cur,0,s,&next); if(st) return st;
                    next->pnb=next->pnb+cur->pnb*ps;
                    next->len=cur->len; memcpy(next->node,cur->node,cur->len*sizeof(cur->node[0]));
                    a20d_node *n=&d->nodes[next->node[next->len-1]];
                    /* Faithful shallow-copy alias: mutates every shared dict. */
                    if(ps>n->prob) { n->prob=ps; n->frame=t; }
                }
                if(fabs(cur->pb)>0.000001) {
                    st=a20d_find(d,w,&count,cur,1,s,&next); if(st) return st;
                    next->pnb=next->pnb+cur->pb*ps;
                    st=a20d_copyappend(d,next,cur,s,t,ps); if(st) return st;
                }
            } else {
                st=a20d_find(d,w,&count,cur,1,s,&next); if(st) return st;
                if(next->len) {
                    if(ps>d->nodes[next->node[next->len-1]].prob) {
                        uint16_t id; st=a20d_newnode(d,s,t,ps,&id); if(st) return st;
                        next->node[next->len-1]=id;
                    }
                } else { st=a20d_copyappend(d,next,cur,s,t,ps); if(st) return st; }
                next->pnb=(next->pnb+cur->pb*ps)+cur->pnb*ps;
            }
        }
    }
    if(!eligible) return A20D_OK;
    /* Python sorted is stable: insertion order resolves equal path scores. */
    for(size_t i=1;i<count;++i) {
        a20d_hyp v=w->next[i]; size_t j=i;
        while(j && (v.pb+v.pnb)>(w->next[j-1].pb+w->next[j-1].pnb)) {
            w->next[j]=w->next[j-1]; --j;
        }
        w->next[j]=v;
    }
    size_t kept=count<A20D_PATH_BEAM?count:A20D_PATH_BEAM, nn=0;
    for(size_t i=0;i<d->node_count;++i) w->remap[i]=UINT16_MAX;
    for(size_t h=0;h<kept;++h) {
        d->hyps[h]=w->next[h];
        for(size_t j=0;j<d->hyps[h].len;++j) {
            uint16_t id=d->hyps[h].node[j];
            if(w->remap[id]==UINT16_MAX) {
                if(nn>=A20D_LIVE_NODES) return A20D_CAPACITY;
                w->remap[id]=(uint16_t)nn; w->compact[nn++]=d->nodes[id];
            }
            d->hyps[h].node[j]=w->remap[id];
        }
    }
    memcpy(d->nodes,w->compact,nn*sizeof(d->nodes[0]));
    d->node_count=(uint16_t)nn; d->hyp_count=(uint16_t)kept; return A20D_OK;
}
static int a20d_sublist(const a20d_decoder *d,const a20d_hyp *h,const int *keyword) {
    if(h->len<4) return -1;
    for(int i=0;i<=(int)h->len-4;++i) { /* Reviewed final-start correction. */
        int j=0; while(j<4 && d->nodes[h->node[i+j]].token==keyword[j]) ++j;
        if(j==4) return i;
    }
    return -1;
}
static void a20d_detect(a20d_decoder *d) {
    static const int words[2][4]={{1,2,3,4},{3,4,3,4}};
    int word=0; int64_t start=0,end=0;
    for(size_t h=0;h<d->hyp_count && !word;++h) {
        for(int k=0;k<2;++k) {
            int offset=a20d_sublist(d,&d->hyps[h],words[k]);
            if(offset>=0) {
                word=k+1; start=d->nodes[d->hyps[h].node[offset]].frame;
                end=d->nodes[d->hyps[h].node[offset+3]].frame;
                /* Upstream hit_score intentionally persists after rejection. */
                for(int j=0;j<4;++j) d->hit_score*=d->nodes[d->hyps[h].node[offset+j]].prob;
                d->hit_score=sqrt(d->hit_score); break;
            }
        }
    }
    d->result=(a20d_result){.valid=1,.start_frame=-1,.end_frame=-1};
    int64_t duration=end-start;
    if(word && d->hit_score>=0.0 && duration>=5 && duration<=250 &&
       (d->last_active_pos==-1 || end-d->last_active_pos>=50)) {
        d->last_active_pos=end;
        d->result.state=1; d->result.keyword=word; d->result.start_frame=start;
        d->result.end_frame=end; d->result.score=d->hit_score;
    }
}
static a20d_status a20d_process(a20d_decoder *d,a20d_workspace *w,const float *input,
                                 size_t rows,a20d_result *result,int logits) {
    if(!a20d_valid(d) || !w || !result || (rows && !input)) return A20D_INVALID;
    if(rows>SIZE_MAX/(A20D_CLASSES*sizeof(float))) return A20D_OVERFLOW;
    if(d->total_frames<0 || rows>(size_t)((INT64_MAX-d->total_frames)/A20D_DOWNSAMPLE))
        return A20D_OVERFLOW;
    for(size_t r=0;r<rows;++r) {
        double sum=0;
        for(int k=0;k<A20D_CLASSES;++k) {
            float x=input[r*A20D_CLASSES+(size_t)k];
            if(!isfinite(x) || (!logits && (x<0.0f || x>1.0f))) return A20D_INVALID;
            sum+=(double)x;
        }
        if(!logits && fabs(sum-1.0)>0.00001) return A20D_INVALID;
    }
    if(!rows) { *result=(a20d_result){0}; return A20D_OK; }
    memcpy(&w->work,d,sizeof(*d)); a20d_decoder *work=&w->work;
    size_t decoded=0;
    for(size_t r=0;r<rows;++r) {
        float probs[A20D_CLASSES]; const float *p=input+r*A20D_CLASSES;
        a20d_status st=A20D_OK;
        if(logits) { st=a20d_softmax6(p,probs); p=probs; }
        if(st) return st;
        st=a20d_frame(work,w,work->total_frames+(int64_t)r*A20D_DOWNSAMPLE,p);
        if(st) return st;
        a20d_detect(work); decoded=r+1;
        if(work->result.state) { a20d_reset_hyps(work); break; }
    }
    work->total_frames+=(int64_t)rows*A20D_DOWNSAMPLE;
    if(work->hyp_count && work->hyps[0].len &&
       work->total_frames-work->nodes[work->hyps[0].node[0]].frame>250)
        a20d_reset_hyps(work);
    work->result.rows_decoded=decoded;
    memcpy(d,work,sizeof(*d)); *result=d->result; return A20D_OK;
}
a20d_status a20d_process_probs(a20d_decoder *d,a20d_workspace *w,const float *p,size_t n,a20d_result *r) {
    return a20d_process(d,w,p,n,r,0);
}
a20d_status a20d_process_logits(a20d_decoder *d,a20d_workspace *w,const float *p,size_t n,a20d_result *r) {
    return a20d_process(d,w,p,n,r,1);
}
a20d_status a20d_get_hyp(const a20d_decoder *d,size_t index,a20d_hyp_view *v) {
    if(!a20d_valid(d) || !v || index>=d->hyp_count) return A20D_INVALID;
    const a20d_hyp *h=&d->hyps[index]; memset(v,0,sizeof(*v));
    v->len=h->len; v->pb=h->pb; v->pnb=h->pnb;
    for(size_t j=0;j<h->len;++j) {
        const a20d_node *n=&d->nodes[h->node[j]];
        v->token[j]=n->token; v->frame[j]=n->frame; v->prob[j]=n->prob;
    }
    return A20D_OK;
}
const char *a20d_status_string(a20d_status st) {
    switch(st) { case A20D_OK:return "ok"; case A20D_INVALID:return "invalid input";
    case A20D_CAPACITY:return "fixed capacity exceeded"; case A20D_OVERFLOW:return "size or frame overflow"; }
    return "unknown status";
}
