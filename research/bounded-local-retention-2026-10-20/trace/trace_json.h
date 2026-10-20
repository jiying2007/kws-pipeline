/* Research harness adapter: bounded row serialization, no header/footer fabrication.
 * Output is provisional until controller flush/close/exit and parser admission.
 * On any write/trace failure preserve partial output and never retry appends. */
#ifndef KT_TRACE_JSON_H
#define KT_TRACE_JSON_H
#include "trace_mode.h"
#include <stdio.h>
#include <stdarg.h>
#include <inttypes.h>
#include <string.h>
typedef struct { FILE *file; uint64_t bytes,limit; int valid,first; } kt_json_writer;
static int kt_json_put(kt_json_writer *w,const char *fmt,...) {
 char b[2048];int n;va_list ap;
 if(!w || !w->valid || !w->file) return 1;
 va_start(ap,fmt);n=vsnprintf(b,sizeof(b),fmt,ap);va_end(ap);
 if(n<0 || (size_t)n>=sizeof(b) || w->bytes>w->limit || (uint64_t)n>w->limit-w->bytes) {w->valid=0;return 1;}
 if(fwrite(b,1,(size_t)n,w->file)!=(size_t)n) {w->valid=0;return 1;}
 w->bytes+=(uint64_t)n;return 0;
}
static uint64_t kt_score_bits(double v) {uint64_t b;memcpy(&b,&v,sizeof(b));return b;}
static int kt_json_record(void *u,const kws3_ledger_record *r) {
 kt_json_writer *w=u;
 /* kind is an immutable internal hook constant, never external text. */
 int rc=kt_json_put(w,"%s{\"sequence\":%" PRIu64 ",\"kind\":\"%s\",\"row\":%" PRId64 ",\"window_start\":%" PRId64 ",\"keyword_id\":%u,\"node_frames\":[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "],\"score_bits\":\"%016" PRIx64 "\",\"decoder_status\":%d}",w->first?"":",",r->sequence,r->kind,r->row,r->window_start,r->keyword_id,r->node_frames[0],r->node_frames[1],r->node_frames[2],r->node_frames[3],kt_score_bits(r->score),r->decoder_status);
 w->first=0;return rc;
}
static int kt_json_row(kt_json_writer *w,kws3_decoder *d,int64_t row,kws3_status rc,const kws3_event *e,size_t count) {
 size_t i;kt_status_v1 s;
 if(!w || !w->valid || rc!=KWS3_OK || count>3 || (count&&!e) || row<0 ||
    kt_status(d,&s)!=KWS3_OK || !s.valid || !s.active || s.ended || s.rows_committed!=(uint64_t)row+1) {if(w)w->valid=0;return 1;}
 if(kt_json_put(w,"{\"type\":\"row\",\"row\":%" PRId64 ",\"decoder_status\":0,\"records\":[",row)) return 1;
 w->first=1;
 if(kt_drain(d,kt_json_record,w)!=KWS3_OK) {w->valid=0;return 1;}
 if(kt_json_put(w,"],\"events\":["))return 1;
 for(i=0;i<count;i++) {
  if(kt_json_put(w,"%s{\"event_index\":%" PRIu64 ",\"row\":%" PRId64 ",\"window_start\":%" PRId64 ",\"keyword_id\":%u,\"node_frames\":[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "],\"score_bits\":\"%016" PRIx64 "\"}",i?",":"",e[i].event_index,e[i].decision_row,e[i].window_start,e[i].keyword_id,e[i].node_frames[0],e[i].node_frames[1],e[i].node_frames[2],e[i].node_frames[3],kt_score_bits(e[i].score))) return 1;
 }
 return kt_json_put(w,"]}\n");
}
#endif
