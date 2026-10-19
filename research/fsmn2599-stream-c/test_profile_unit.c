/* SPDX-License-Identifier: Apache-2.0. Test-private synthetic clock/state. */
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#define B_PROFILE_TEST_CLOCK
#include "b_profile.c"
static uint64_t fake;
static int fail_clock;
int b_profile_test_now(uint64_t *out) {if(fail_clock)return 0;*out=fake;return 1;}
int main(void) {
 b_profile_snapshot s;
 b_profile_reset();
 for(unsigned i=0;i<B_PROFILE_STAGES;i++) {
  fake=100; b_profile_stamp t=b_profile_begin();fake=2100;
  b_profile_end(i,t,i!=2);
 }
 assert(!b_profile_get(&s,sizeof(s)) && s.abi==1);
 for(unsigned i=0;i<B_PROFILE_STAGES;i++) {
  assert(s.stage[i].calls==1 && s.stage[i].timed_calls==1);
  assert(s.stage[i].failed_calls==(i==2));
  assert(s.stage[i].total_ns==2000 && s.stage[i].min_ns==2000 && s.stage[i].max_ns==2000);
  assert(s.stage[i].histogram[1]==1);
 }
 fake=30;b_profile_stamp t=b_profile_begin();fake=29;b_profile_end(0,t,1);
 fail_clock=1;t=b_profile_begin();b_profile_end(0,t,0);fail_clock=0;
 assert(!b_profile_get(&s,sizeof(s)) && s.timer_errors==2);
 assert(s.stage[0].calls==3 && s.stage[0].timed_calls==1 && s.stage[0].failed_calls==1);
 /* Saturation is explicit, never unsigned wrap. */
 current.stage[1].calls=UINT64_MAX;current.stage[1].total_ns=UINT64_MAX-1;
 current.stage[1].histogram[1]=UINT64_MAX;
 fake=1;t=b_profile_begin();fake=2001;b_profile_end(1,t,1);
 assert(!b_profile_get(&s,sizeof(s)) && s.saturated);
 assert(s.stage[1].calls==UINT64_MAX && s.stage[1].total_ns==UINT64_MAX && s.stage[1].histogram[1]==UINT64_MAX);
 s.abi=99;assert(b_profile_get(&s,1)==1 && s.abi==99);
 b_profile_reset();assert(!b_profile_get(&s,sizeof(s)) && !s.timer_errors && !s.saturated && !s.stage[0].calls);
 puts("profile unit: stages/timer errors/backwards clock/saturation/reset PASS");return 0;
}
