/* SPDX-License-Identifier: Apache-2.0
 * Standalone source-only candidate. Caller supplies original authenticated
 * full qhead8 arrays; no model file, PCM, scheduler, decoder or allocation.
 */
#include "i16_certified_head.h"
#include <float.h>
#include <fenv.h>
#include <math.h>
#include <string.h>
#include <limits.h>

_Static_assert(140 * 32767 * 127 < INT32_MAX, "int32 dot cannot overflow");
static const uint32_t precise[5]={0u,1027u,1401u,1462u,2093u};
static int is_precise(uint32_t id) {
    for (size_t i=0;i<5u;i++) if (precise[i]==id) return 1;
    return 0;
}
static int environment(void) {
    const uint16_t one=1u;
    volatile double lo=DBL_MIN,half=0.5,sub=lo*half;
    return sizeof(float)==4u && FLT_MANT_DIG==24 &&
           sizeof(double)==8u && DBL_MANT_DIG==53 &&
           FLT_RADIX==2 && FLT_EVAL_METHOD==0 &&
           *(const unsigned char *)&one==1u &&
           fegetround()==FE_TONEAREST && sub>0.0;
}

/* The unchanged original 4x35 double accumulators, then scale and bias. */
static double exact_par4(const r1_i16_head *h,uint32_t id,const double *x) {
    const int8_t *w=h->quant_rows+(size_t)id*R1_I16_HIDDEN;
    double a=0.0,b=0.0,c=0.0,d=0.0;
    for(size_t k=0u;k<R1_I16_HIDDEN;k+=4u) {
        a+=x[k]*(double)w[k];
        b+=x[k+1u]*(double)w[k+1u];
        c+=x[k+2u]*(double)w[k+2u];
        d+=x[k+3u]*(double)w[k+3u];
    }
    double sum=(a+b)+(c+d);
    sum*=h->scales[id];
    return sum+(double)h->biases[id];
}

int r1_i16_init(r1_i16_head *h,const int8_t *rows,
                const double *scales,const float *biases) {
    if(!h||!rows||!scales||!biases) return R1_I16_BAD_ARGUMENT;
    if(!environment()) return R1_I16_BAD_ENVIRONMENT;
    uint16_t sums[R1_I16_CLASSES]={0};
    for(uint32_t id=0u;id<R1_I16_CLASSES;id++) {
        if(is_precise(id)) continue;
        if(!isfinite(scales[id])||scales[id]<=0.0||
           !isfinite((double)biases[id])) return R1_I16_BAD_MODEL;
        const int8_t *w=rows+(size_t)id*R1_I16_HIDDEN;
        unsigned sum=0u;
        for(size_t k=0u;k<R1_I16_HIDDEN;k++) {
            if(w[k]==INT8_MIN) return R1_I16_BAD_MODEL;
            int value=(int)w[k];
            sum+=(unsigned)(value<0?-value:value);
        }
        if(sum>UINT16_MAX) return R1_I16_BAD_MODEL;
        sums[id]=(uint16_t)sum;
    }
    h->quant_rows=rows;h->scales=scales;h->biases=biases;
    memcpy(h->abs_weight_sum,sums,sizeof(sums));
    return R1_I16_OK;
}

/* Exact original fallback when the screening arithmetic cannot be bounded.
 * Return only after all rows have been verified finite. */
static int scan_all(const r1_i16_head *h,const double *x,
                    r1_i16_result *out) {
    double best=-INFINITY;uint32_t winner=UINT32_MAX;
    for(uint32_t id=0u;id<R1_I16_CLASSES;id++) {
        if(is_precise(id)) continue;
        double v=exact_par4(h,id,x);
        if(!isfinite(v)) return R1_I16_NONFINITE;
        if(v>best||(v==best&&id<winner)){best=v;winner=id;}
    }
    r1_i16_result value={best,winner,0u,R1_I16_OTHER,1u};
    *out=value;return R1_I16_OK;
}

int r1_i16_step(r1_i16_head *h,const double hidden[R1_I16_HIDDEN],
                r1_i16_result *result) {
    if(!h||!h->quant_rows||!h->scales||!h->biases||!hidden||!result)
        return R1_I16_BAD_ARGUMENT;
    if(!environment()) return R1_I16_BAD_ENVIRONMENT;
    double maxabs=0.0;
    for(size_t k=0;k<R1_I16_HIDDEN;k++) {
        if(!isfinite(hidden[k])) return R1_I16_NONFINITE;
        double mag=fabs(hidden[k]);
        if(mag>maxabs) maxabs=mag;
    }
    /* Deltas are computed against the reconstructed int16 hidden, not just
     * a presumed half-step. This keeps the error bound valid under rounding
     * at ties and for the zero vector. Overflow/underflow -> original scan. */
    double unit=maxabs==0.0?1.0:maxabs/32767.0;
    if(!isfinite(unit)||unit<=0.0) return scan_all(h,hidden,result);
    int16_t qx[R1_I16_HIDDEN];
    double worst=0.0;
    for(size_t k=0;k<R1_I16_HIDDEN;k++) {
        double ratio=hidden[k]/unit;
        if(!isfinite(ratio)||ratio>32767.5||ratio< -32767.5)
            return scan_all(h,hidden,result);
        double rounded=nearbyint(ratio);
        if(!isfinite(rounded)||rounded>32767.0||rounded< -32767.0)
            return scan_all(h,hidden,result);
        qx[k]=(int16_t)rounded;
        double reconstructed=(double)qx[k]*unit;
        double e=fabs(hidden[k]-reconstructed);
        if(!isfinite(e)) return scan_all(h,hidden,result);
        if(e>worst)worst=e;
    }
    /* Protect against rounding in unit, reconstruction and max-error.
     * A fixed 1e-6 plus relative 1e-8 guard is additionally retained
     * from the preregistered owner saved-hidden prototype. */
    double delta=worst+(64.0*DBL_EPSILON)*(1.0+maxabs+fabs(unit)*32767.0);
    if(!isfinite(delta)) return scan_all(h,hidden,result);
    delta=nextafter(delta,INFINITY);

    double best_lower=-INFINITY;
    for(uint32_t id=0u;id<R1_I16_CLASSES;id++) {
        if(is_precise(id))continue;
        const int8_t *w=h->quant_rows+(size_t)id*R1_I16_HIDDEN;
        int32_t dot=0;
        for(size_t k=0u;k<R1_I16_HIDDEN;k++)
            dot+=(int32_t)qx[k]*(int32_t)w[k];
        /* Each term is exactly representable in binary64; |dot| <= 5.83e8. */
        double approx=((double)dot*unit)*h->scales[id]+(double)h->biases[id];
        if(!isfinite(approx))return scan_all(h,hidden,result);
        double mass=maxabs*h->scales[id]*(double)h->abs_weight_sum[id];
        double error=delta*h->scales[id]*(double)h->abs_weight_sum[id];
        if(!isfinite(mass)||!isfinite(error))return scan_all(h,hidden,result);
        error+=(8192.0*DBL_EPSILON)*
            (1.0+mass+fabs(approx)+fabs((double)h->biases[id]));
        error+=1e-6+1e-8*fabs(approx);
        if(!isfinite(error))return scan_all(h,hidden,result);
        error=nextafter(error,INFINITY);
        double lower=nextafter(approx-error,-INFINITY);
        double upper=nextafter(approx+error,INFINITY);
        if(!isfinite(lower)||!isfinite(upper)||lower>upper)
            return scan_all(h,hidden,result);
        h->candidate_upper[id]=upper;
        if(lower>best_lower)best_lower=lower;
    }
    double best=-INFINITY;uint32_t winner=UINT32_MAX;uint32_t exact=0u;
    for(uint32_t id=0u;id<R1_I16_CLASSES;id++) {
        if(is_precise(id)||h->candidate_upper[id]<best_lower)continue;
        double v=exact_par4(h,id,hidden);
        if(!isfinite(v))return R1_I16_NONFINITE;
        exact++;
        if(v>best||(v==best&&id<winner)){best=v;winner=id;}
    }
    /* At least the row with highest lower endpoint must be evaluated. */
    if(!exact) return R1_I16_NONFINITE;
    r1_i16_result value={best,winner,R1_I16_OTHER,exact,0u};
    *result=value;
    return R1_I16_OK;
}
