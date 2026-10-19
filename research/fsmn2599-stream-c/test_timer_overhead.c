/* SPDX-License-Identifier: Apache-2.0. Timer-only, 1000 empty scopes, no model. */
#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include "b_profile.h"
int main(void){
 b_profile_reset();
 for(unsigned i=0;i<1000;i++){b_profile_stamp t=b_profile_begin();b_profile_end(0,t,1);}
 b_profile_snapshot s;assert(!b_profile_get(&s,sizeof(s)) && !s.timer_errors && s.stage[0].timed_calls==1000);
 printf("{\"empty_scopes\":1000,\"total_ns\":%" PRIu64 ",\"min_ns\":%" PRIu64 ",\"max_ns\":%" PRIu64 ",\"subtract_from_operator_times\":false}\n",s.stage[0].total_ns,s.stage[0].min_ns,s.stage[0].max_ns);
 return 0;
}
