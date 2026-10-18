/* SPDX-License-Identifier: Apache-2.0
 * New fixed arithmetic candidate, SOURCE PREPARED ONLY; no execution yet. */
#include "softmax2599.h"
#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#if defined(__FAST_MATH__)
#error "fast-math is forbidden"
#endif
_Static_assert(sizeof(float)==4 && FLT_RADIX==2 && FLT_MANT_DIG==24,
               "binary32 required");
_Static_assert(sizeof(double)==8 && DBL_MANT_DIG==53 && DBL_MAX_EXP==1024,
               "binary64 required");
_Static_assert(FLT_EVAL_METHOD==0,"excess precision forbidden");
static int overlap(const void *a,size_t na,const void *b,size_t nb) {
    uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;
    if(x>UINTPTR_MAX-na || y>UINTPTR_MAX-nb) return 1;
    return x<y+nb && y<x+na;
}
static int environment_ok(void) {
    volatile float f=FLT_MIN,hf=.5f,subf=f*hf,smallf=FLT_TRUE_MIN;
    volatile double d=DBL_MIN,hd=.5,subd=d*hd,promoted=smallf;
    return fegetround()==FE_TONEAREST && subf>0 && subf+subf==FLT_MIN &&
           subd>0 && subd+subd==DBL_MIN && promoted==0x1p-149;
}
ksx_status ksx_softmax2599(const float *x,size_t nx,float *out,size_t no,
                         ksx_workspace *w) {
    float peak;
    double total=0.;
    size_t bytes=KSX_CLASSES*sizeof(float);
    if(!x || !out || !w || nx!=KSX_CLASSES || no!=KSX_CLASSES) return KSX_ARGUMENT;
    if(overlap(x,bytes,out,bytes) || overlap(x,bytes,w,sizeof(*w)) ||
       overlap(out,bytes,w,sizeof(*w))) return KSX_OVERLAP;
    if(!environment_ok()) return KSX_ENVIRONMENT;
    peak=x[0];
    for(size_t i=0;i<KSX_CLASSES;i++) {
        if(!isfinite(x[i])) return KSX_NONFINITE;
        if(x[i]>peak) peak=x[i];
    }
    for(size_t i=0;i<KSX_CLASSES;i++) {
        float e;
        /* A finite pair may have a difference below -FLT_MAX. Avoid float
         * subtraction overflow. Such exp is far below binary32 half-minsub;
         * its correctly rounded result is zero. This is not a tunable cutoff. */
        if((double)x[i]-(double)peak < -(double)FLT_MAX) e=0.f;
        else {
            float shifted=x[i]-peak;
            e=expf(shifted);
        }
        if(!isfinite(e) || e<0.f || e>1.f) return KSX_ARITHMETIC;
        w->exponentials[i]=e;
        total+=(double)e;
    }
    if(!isfinite(total) || total<1. || total>(double)KSX_CLASSES) return KSX_ARITHMETIC;
    for(size_t i=0;i<KSX_CLASSES;i++) {
        float p=(float)((double)w->exponentials[i]/total);
        if(!isfinite(p) || p<0.f || p>1.f) return KSX_ARITHMETIC;
        w->probabilities[i]=p;
    }
    memcpy(out,w->probabilities,bytes);
    return KSX_OK;
}
