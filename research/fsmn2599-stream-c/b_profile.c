/* SPDX-License-Identifier: Apache-2.0. Integer-only bounded diagnostics. */
#define _POSIX_C_SOURCE 200809L
#include "b_profile.h"
#include <errno.h>
#include <fenv.h>
#include <string.h>
#include <time.h>
static _Thread_local b_profile_snapshot current;
static void add(uint64_t *x,uint64_t amount) {
 if(*x>UINT64_MAX-amount) {*x=UINT64_MAX;current.saturated=1;}
 else *x+=amount;
}
static int now_ns(uint64_t *value) {
#ifdef B_PROFILE_TEST_CLOCK
 extern int b_profile_test_now(uint64_t *);
 return b_profile_test_now(value);
#else
 struct timespec t;
 int saved_errno=errno, rc;
 fenv_t environment;
 /* Clock implementations are outside our arithmetic contract. Preserve both
  * errno and complete FP environment; failures invalidate timing only. */
 if(fegetenv(&environment)) {errno=saved_errno;return 0;}
 rc=clock_gettime(CLOCK_MONOTONIC,&t);
 if(fesetenv(&environment)) rc=-1;
 errno=saved_errno;
 if(rc || t.tv_sec<0 || t.tv_nsec<0 || t.tv_nsec>=1000000000L ||
    (uint64_t)t.tv_sec>(UINT64_MAX-(uint64_t)t.tv_nsec)/UINT64_C(1000000000)) return 0;
 *value=(uint64_t)t.tv_sec*UINT64_C(1000000000)+(uint64_t)t.tv_nsec;
 return 1;
#endif
}
void b_profile_reset(void) {
 memset(&current,0,sizeof(current));current.abi=B_PROFILE_ABI;
 current.stages=B_PROFILE_STAGES;current.buckets=B_PROFILE_BUCKETS;
}
size_t b_profile_size(void) {return sizeof(b_profile_snapshot);}
int b_profile_get(b_profile_snapshot *out,size_t bytes) {
 if(!out || bytes!=sizeof(*out)) return 1;
 *out=current;out->abi=B_PROFILE_ABI;out->stages=B_PROFILE_STAGES;
 out->buckets=B_PROFILE_BUCKETS;return 0;
}
b_profile_stamp b_profile_begin(void) {
 b_profile_stamp stamp={0,0};stamp.valid=now_ns(&stamp.ns);
 if(!stamp.valid) add(&current.timer_errors,1);
 return stamp;
}
void b_profile_end(unsigned stage,b_profile_stamp start,int success) {
 b_profile_stage *s;uint64_t end,elapsed,limit=1000;unsigned bucket=0;
 if(stage>=B_PROFILE_STAGES) {add(&current.timer_errors,1);return;}
 s=&current.stage[stage];add(&s->calls,1);
 if(!success)add(&s->failed_calls,1);
 if(!start.valid)return;
 if(!now_ns(&end) || end<start.ns) {add(&current.timer_errors,1);return;}
 elapsed=end-start.ns;
 if(!s->timed_calls || elapsed<s->min_ns)s->min_ns=elapsed;
 if(elapsed>s->max_ns)s->max_ns=elapsed;
 add(&s->timed_calls,1);add(&s->total_ns,elapsed);
 while(bucket+1<B_PROFILE_BUCKETS && elapsed>limit){bucket++;limit*=2;}
 add(&s->histogram[bucket],1);
}
