/* SPDX-License-Identifier: Apache-2.0. Invalid-argument paths only; no model. */
#include <assert.h>
#include <errno.h>
#include <fenv.h>
#include <stdio.h>
#include "canonical_fbank.h"
#include "fsmn2599.h"
#include "softmax2599.h"
#include "kws3.h"
#ifdef KWS_B_PROFILE
#include "b_profile.h"
#endif
int main(void) {
 fenv_t saved;assert(!fegetenv(&saved));
 assert(!feclearexcept(FE_ALL_EXCEPT));assert(!feraiseexcept(FE_INVALID));
 assert(!fesetround(FE_DOWNWARD));errno=EDOM;
#ifdef KWS_B_PROFILE
 b_profile_reset();
#endif
 int a=kcf_fbank_frame(NULL,NULL,NULL,NULL);
 int b=ksx_softmax2599(NULL,0,NULL,0,NULL);
 int c=kws3_step(NULL,NULL,0,NULL,0,NULL);
 int d=fsmn2599_step(NULL,NULL,0,NULL,0,NULL,0);
 assert(fegetround()==FE_DOWNWARD && fetestexcept(FE_ALL_EXCEPT)==FE_INVALID && errno==EDOM);
#ifdef KWS_B_PROFILE
 b_profile_snapshot s;assert(b_profile_size()==sizeof(s));assert(!b_profile_get(&s,sizeof(s)));
 assert(!s.timer_errors && !s.saturated);
 assert(s.stage[B_PROFILE_FRONTEND].calls==1 && s.stage[B_PROFILE_FRONTEND].failed_calls==1);
 assert(s.stage[B_PROFILE_SOFTMAX].calls==1 && s.stage[B_PROFILE_SOFTMAX].failed_calls==1);
 assert(s.stage[B_PROFILE_DECODER].calls==1 && s.stage[B_PROFILE_DECODER].failed_calls==1);
 assert(s.stage[B_PROFILE_BODY].calls==0 && s.stage[B_PROFILE_HEAD].calls==0);
#endif
 assert(!fesetenv(&saved));
 printf("invalid argument rc=%d,%d,%d,%d; fenv/errno unchanged\n",a,b,c,d);return 0;
}
