/* Invented posterior rows only. Explicit checks remain under NDEBUG. */
#include "trace_mode.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#define CHECK(x) do {if(!(x)){fprintf(stderr,"check failed line %d: %s\n",__LINE__,#x);return 1;}}while(0)
static int records, commits, reenter;
static kws3_decoder *captured;
static int sink(void *u,const kws3_ledger_record *r) {
 (void)u; records++; if(!strcmp(r->kind,"row_committed")) commits++;
 if(reenter) {kt_status_v1 s; kt_status(captured,&s);} return 0;
}
static int fail_sink(void *u,const kws3_ledger_record *r) {(void)u;(void)r;return 1;}
static void *foreign(void *d) {kt_status_v1 s; return (void *)(uintptr_t)kt_status(d,&s);}
int main(void) {
 kws3_decoder *d=NULL,*other=NULL; float row[KWS3_CLASSES]={0}; kws3_event e[3]; size_t n;
 kws3_summary plain,traced; kws3_retention_summary rp,rt; kt_status_v1 s; int i; pthread_t th;void *ret;
 row[0]=1;
 CHECK(kt_create("wrong","invented",&d)==KWS3_ARGUMENT && !d);
 CHECK(kt_create("bounded-retention-trace-v1","invented",&d)==KWS3_OK);
 for(i=0;i<130;i++) CHECK(kt_step_f32(d,row,KWS3_CLASSES,e,3,&n)==KWS3_OK && !n);
 CHECK(kt_summary(d,&plain)==KWS3_OK); CHECK(kt_retention(d,&rp)==KWS3_OK);CHECK(kt_destroy(d)==KWS3_OK);
 CHECK(kt_create("bounded-retention-trace-v1","invented",&d)==KWS3_OK);captured=d;
 CHECK(kt_begin(d)==KWS3_OK);
 for(i=0;i<130;i++) {CHECK(kt_step_f32(d,row,KWS3_CLASSES,e,3,&n)==KWS3_OK&&!n);CHECK(kt_drain(d,sink,NULL)==KWS3_OK);}
 CHECK(commits==130 && records>130);CHECK(kt_summary(d,&traced)==KWS3_OK);CHECK(kt_retention(d,&rt)==KWS3_OK);
 CHECK(!memcmp(&plain,&traced,sizeof(plain)));CHECK(!memcmp(&rp,&rt,sizeof(rp)));
 CHECK(kt_end(d)==KWS3_OK);CHECK(kt_status(d,&s)==KWS3_OK && s.valid && s.ended && !s.pending && !s.dropped);
 CHECK(kt_destroy(d)==KWS3_OK);
 /* second begin invalidates */
 CHECK(kt_create("bounded-retention-trace-v1","invented",&d)==KWS3_OK);CHECK(kt_begin(d)==KWS3_OK);
 CHECK(kt_begin(d)==KWS3_INVALID);CHECK(kt_destroy(d)==KWS3_INVALID);
 /* unfinished destroy */
 CHECK(kt_create("bounded-retention-trace-v1","invented",&d)==KWS3_OK);CHECK(kt_begin(d)==KWS3_OK);CHECK(kt_destroy(d)==KWS3_INVALID);
 /* late begin */
 CHECK(kt_create("bounded-retention-trace-v1","invented",&d)==KWS3_OK);CHECK(kt_step_f32(d,row,KWS3_CLASSES,e,3,&n)==KWS3_OK);CHECK(kt_begin(d)==KWS3_INVALID);CHECK(kt_destroy(d)==KWS3_INVALID);
 /* wrong handle and foreign thread */
 CHECK(kt_create("bounded-retention-trace-v1","invented",&d)==KWS3_OK);CHECK(kt_status((kws3_decoder *)(uintptr_t)1,&s)==KWS3_INVALID);CHECK(kt_destroy(d)==KWS3_INVALID);
 CHECK(kt_create("bounded-retention-trace-v1","invented",&d)==KWS3_OK);CHECK(!pthread_create(&th,NULL,foreign,d));CHECK(!pthread_join(th,&ret));CHECK((uintptr_t)ret==KWS3_INVALID);CHECK(kt_destroy(d)==KWS3_INVALID);
 /* pending drain, sink failure, reentry */
 for(i=0;i<3;i++) {CHECK(kt_create("bounded-retention-trace-v1","invented",&d)==KWS3_OK);captured=d;CHECK(kt_begin(d)==KWS3_OK);CHECK(kt_step_f32(d,row,KWS3_CLASSES,e,3,&n)==KWS3_OK);
 if(i==0) CHECK(kt_end(d)==KWS3_INVALID);
 if(i==1) CHECK(kt_drain(d,fail_sink,NULL)==KWS3_INVALID);
 if(i==2) {reenter=1;CHECK(kt_drain(d,sink,NULL)==KWS3_INVALID);reenter=0;}
 CHECK(kt_destroy(d)==KWS3_INVALID);}
 /* second owner and invalid row */
 CHECK(kt_create("bounded-retention-trace-v1","invented",&d)==KWS3_OK);CHECK(kt_create("bounded-retention-trace-v1","other",&other)==KWS3_INVALID&&!other);CHECK(kt_destroy(d)==KWS3_INVALID);
 CHECK(kt_create("bounded-retention-trace-v1","invented",&d)==KWS3_OK);CHECK(kt_begin(d)==KWS3_OK);row[1]=NAN;CHECK(kt_step_f32(d,row,KWS3_CLASSES,e,3,&n)==KWS3_POSTERIOR&&!n);CHECK(kt_end(d)==KWS3_INVALID);CHECK(kt_destroy(d)==KWS3_INVALID);
 puts("PASS: native synthetic capture isolation, lifecycle, trace equivalence; no model/audio inputs");return 0;
}
