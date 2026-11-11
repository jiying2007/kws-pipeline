/* SPDX-License-Identifier: Apache-2.0
 * Invented full-vocabulary original par4 oracle. No real weights/audio/PCM.
 */
#include "i16_certified_head.h"
#include <fenv.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int8_t rows[R1_I16_CLASSES * R1_I16_HIDDEN];
static double scales[R1_I16_CLASSES];
static float biases[R1_I16_CLASSES];
static r1_i16_head state;
static uint32_t checked=0u,small=0u,fallbacks=0u;
#define NEED(x) do {if(!(x)){fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x);return 1;}}while(0)

static int precise(uint32_t id) {
    return id==0u||id==1027u||id==1401u||id==1462u||id==2093u;
}
static double full_par4(uint32_t id,const double *x) {
    const int8_t *w=rows+(size_t)id*R1_I16_HIDDEN;
    double a=0.0,b=0.0,c=0.0,d=0.0;
    for(size_t i=0u;i<R1_I16_HIDDEN;i+=4u) {
        a+=x[i]*(double)w[i];b+=x[i+1u]*(double)w[i+1u];
        c+=x[i+2u]*(double)w[i+2u];d+=x[i+3u]*(double)w[i+3u];
    }
    double z=(a+b)+(c+d);z*=scales[id];return z+(double)biases[id];
}
static int compare(const double x[R1_I16_HIDDEN]) {
    double best=-INFINITY;uint32_t winner=UINT32_MAX;
    for(uint32_t id=0u;id<R1_I16_CLASSES;id++) {
        if(precise(id)) continue;
        double z=full_par4(id,x);
        NEED(isfinite(z));
        if(z>best||(z==best&&id<winner)) {best=z;winner=id;}
    }
    r1_i16_result got={0};
    NEED(r1_i16_step(&state,x,&got)==R1_I16_OK);
    NEED(!memcmp(&best,&got.exact_other_max,sizeof(double)));
    NEED(got.other_original_id==winner);
    NEED(got.numeric_fallback<=1u);
    NEED(got.approx_rows==0u||got.approx_rows==R1_I16_OTHER);
    NEED(got.exact_rows>=1u&&got.exact_rows<=R1_I16_OTHER);
    checked++;
    small+=(unsigned)(got.exact_rows<=10u);
    fallbacks+=got.numeric_fallback;
    return 0;
}
static void constant_fixture(void) {
    memset(rows,0,sizeof(rows));
    for(uint32_t id=0;id<R1_I16_CLASSES;id++) {
        scales[id]=0.1;biases[id]=-10.0f;
    }
    biases[1]=100.0f;
    rows[R1_I16_HIDDEN]=1;
}
static int test_constant(void) {
    constant_fixture();
    NEED(r1_i16_init(&state,rows,scales,biases)==R1_I16_OK);
    double x[R1_I16_HIDDEN]={0};
    NEED(compare(x)==0);
    x[0]=0.25;
    NEED(compare(x)==0);
    x[0]=-15.75;
    NEED(compare(x)==0);
    x[0]=32767.0;x[1]=-0.1;
    NEED(compare(x)==0);
    x[0]=0.0;x[1]=0.0;
    /* Tie: original class ID 1 beats every higher equal other ID. */
    biases[1]=-10.0f;
    NEED(r1_i16_init(&state,rows,scales,biases)==R1_I16_OK);
    NEED(compare(x)==0);
    return 0;
}
static uint32_t seed=UINT32_C(0xBEEFCACE);
static uint32_t rnd(void) {
    uint32_t x=seed;x^=x<<13;x^=x>>17;x^=x<<5;seed=x;return x;
}
static int test_random(void) {
    for(uint32_t id=0u;id<R1_I16_CLASSES;id++) {
        scales[id]=(double)(id%23u+1u)/500.0;
        biases[id]=(float)((int)(id%17u)-8)*0.25f;
        for(size_t j=0u;j<R1_I16_HIDDEN;j++)
            rows[(size_t)id*R1_I16_HIDDEN+j]=(int8_t)((int)(rnd()%63u)-31);
    }
    NEED(r1_i16_init(&state,rows,scales,biases)==R1_I16_OK);
    double x[R1_I16_HIDDEN]={0};
    for(unsigned step=0;step<130u;step++) {
        for(size_t j=0u;j<R1_I16_HIDDEN;j++) {
            double v=((int)(rnd()%10001u)-5000)/250.0;
            if(step%11u==0u)v*=0.0000001;
            if(step%17u==0u)v*=100000.0;
            x[j]=v;
        }
        NEED(compare(x)==0);
    }
    return 0;
}
static int test_rejections(void) {
    constant_fixture();
    r1_i16_head dest;
    memset(&dest,0,sizeof(dest));
    scales[1]=0.0;
    NEED(r1_i16_init(&dest,rows,scales,biases)==R1_I16_BAD_MODEL);
    NEED(dest.quant_rows==NULL);
    scales[1]=0.1;
    rows[R1_I16_HIDDEN]=INT8_MIN;
    NEED(r1_i16_init(&dest,rows,scales,biases)==R1_I16_BAD_MODEL);
    NEED(dest.quant_rows==NULL);
    rows[R1_I16_HIDDEN]=0;
    NEED(r1_i16_init(&dest,rows,scales,biases)==R1_I16_OK);
    r1_i16_result untouched={3.5,16u,8u,9u,1u}, snapshot=untouched;
    double x[R1_I16_HIDDEN]={0};x[5]=NAN;
    NEED(r1_i16_step(&dest,x,&untouched)==R1_I16_NONFINITE);
    NEED(!memcmp(&untouched,&snapshot,sizeof(snapshot)));
    x[5]=0.0;
    NEED(fesetround(FE_DOWNWARD)==0);
    NEED(r1_i16_step(&dest,x,&untouched)==R1_I16_BAD_ENVIRONMENT);
    NEED(fesetround(FE_TONEAREST)==0);
    NEED(!memcmp(&untouched,&snapshot,sizeof(snapshot)));
    return 0;
}
int main(void) {
    if(test_constant()||test_random()||test_rejections())return 1;
    NEED(checked==135u);
    NEED(small>0u);
    printf("{\"status\":\"PASS_INVENTED_I16_CERTIFIED_OTHER\","
           "\"frames\":%u,\"ten_or_fewer_exact_rows\":%u,"
           "\"numeric_fallbacks\":%u,\"model_calls\":0,"
           "\"pcm_calls\":0,\"decoder_calls\":0}\n",
           checked,small,fallbacks);
    return 0;
}
