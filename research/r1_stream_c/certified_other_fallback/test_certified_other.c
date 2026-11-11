/* SPDX-License-Identifier: Apache-2.0
 * Invented 2,601-row fixtures only. No pretrained weights, audio, model or
 * decoder calls. Compare *bitwise* against a separate full par4 reference.
 */
#include "certified_other.h"
#include "../experiments/20261111-margin-active113/competitor-active-v1/selected_rows.h"
#include <fenv.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int8_t rows[R1_CERT_CLASSES * R1_CERT_HIDDEN];
static double scales[R1_CERT_CLASSES];
static float biases[R1_CERT_CLASSES];
static const uint32_t precise[5] = {0u,1027u,1401u,1462u,2093u};
static uint32_t checked = 0u, skipped = 0u, fallback = 0u;

#define REQUIRE(c) do { if (!(c)) { \
    fprintf(stderr,"FAIL test line %d: %s\n",__LINE__,#c); return 1; \
} } while (0)

static int precise_id(uint32_t id) {
    for (size_t k = 0u; k < 5u; ++k)
        if (precise[k] == id) return 1;
    return 0;
}
static double reference_row(uint32_t id, const double *x) {
    const int8_t *q=rows+(size_t)id*R1_CERT_HIDDEN;
    double s0=0.0,s1=0.0,s2=0.0,s3=0.0;
    for (size_t i=0u;i<R1_CERT_HIDDEN;i+=4u) {
        s0+=x[i]*(double)q[i];
        s1+=x[i+1u]*(double)q[i+1u];
        s2+=x[i+2u]*(double)q[i+2u];
        s3+=x[i+3u]*(double)q[i+3u];
    }
    double sum=(s0+s1)+(s2+s3);
    sum*=scales[id];
    return sum+(double)biases[id];
}
static int compare_one(r1_cert_head *gate,const double *x,int desired_path) {
    double full=-INFINITY;uint32_t full_id=UINT32_MAX;
    for (uint32_t id=0u;id<R1_CERT_CLASSES;++id) {
        if (precise_id(id)) continue;
        double v=reference_row(id,x);
        REQUIRE(isfinite(v));
        if (v>full || (v==full && id<full_id)) {
            full=v;full_id=id;
        }
    }
    r1_cert_result got={0};
    REQUIRE(r1_cert_step(gate,x,&got)==R1_CERT_OK);
    REQUIRE(!memcmp(&full,&got.other_max,sizeof(double)));
    REQUIRE(got.other_original_id==full_id);
    REQUIRE(got.certified_this_frame==0 || got.certified_this_frame==1);
    REQUIRE(desired_path<0 || got.certified_this_frame==desired_path);
    REQUIRE(got.evaluated_other_rows==
            (got.certified_this_frame?ACTIVE_OTHER_COUNT:R1_CERT_CLASSES-5u));
    checked++;
    skipped+=(unsigned)got.certified_this_frame;
    fallback+=(unsigned)!got.certified_this_frame;
    return 0;
}
static void trivial_fixture(void) {
    memset(rows,0,sizeof(rows));
    for (size_t i=0u;i<R1_CERT_CLASSES;++i) {
        scales[i]=1.0;biases[i]=-2.0f;
    }
    for (size_t i=0u;i<ACTIVE_OTHER_COUNT;++i)
        biases[active_other_ids[i]]=10.0f;
    rows[R1_CERT_HIDDEN]=80;
}
static int test_trivial(void) {
    r1_cert_head gate;
    trivial_fixture();
    REQUIRE(r1_cert_init(&gate,rows,scales,biases,active_other_ids,
                         ACTIVE_OTHER_COUNT)==R1_CERT_OK);
    double x[R1_CERT_HIDDEN]={0};
    REQUIRE(compare_one(&gate,x,0)==0);
    REQUIRE(compare_one(&gate,x,1)==0); /* Exact previous residual bound. */
    x[0]=0.001;
    REQUIRE(compare_one(&gate,x,1)==0); /* Bounded temporal drift. */
    x[0]=1.0;
    REQUIRE(compare_one(&gate,x,0)==0); /* Residual class 1 now wins. */
    REQUIRE(compare_one(&gate,x,0)==0); /* No unsound certificate. */
    x[0]=0.0;
    REQUIRE(compare_one(&gate,x,0)==0);
    REQUIRE(compare_one(&gate,x,1)==0);
    REQUIRE(r1_cert_reset(&gate)==R1_CERT_OK);
    REQUIRE(compare_one(&gate,x,0)==0); /* Discontinuity invalidates bound. */

    r1_cert_result original={123.0,55u,77u,0u,0u,0};
    r1_cert_result copy=original;
    x[1]=NAN;
    REQUIRE(r1_cert_step(&gate,x,&original)==R1_CERT_NONFINITE);
    REQUIRE(!memcmp(&original,&copy,sizeof(original)));
    x[1]=0.0;
    REQUIRE(compare_one(&gate,x,1)==0);
    REQUIRE(fesetround(FE_UPWARD)==0);
    REQUIRE(r1_cert_step(&gate,x,&original)==R1_CERT_BAD_ENVIRONMENT);
    REQUIRE(fesetround(FE_TONEAREST)==0);
    REQUIRE(!memcmp(&original,&copy,sizeof(original)));
    REQUIRE(compare_one(&gate,x,1)==0);
    return 0;
}
static int test_ties(void) {
    trivial_fixture();biases[1]=10.0f;
    r1_cert_head gate;
    REQUIRE(r1_cert_init(&gate,rows,scales,biases,active_other_ids,
                         ACTIVE_OTHER_COUNT)==R1_CERT_OK);
    double x[R1_CERT_HIDDEN]={0};
    REQUIRE(compare_one(&gate,x,0)==0);
    REQUIRE(compare_one(&gate,x,0)==0); /* Strict bound forbids tie pruning. */
    x[0]=-1.0;
    REQUIRE(compare_one(&gate,x,0)==0);
    return 0;
}
static uint32_t rng_state=UINT32_C(0x1234ABCD);
static uint32_t pseudo_random(void) {
    uint32_t v=rng_state;
    v^=v<<13;v^=v>>17;v^=v<<5;rng_state=v;
    return v;
}
static int test_mixed(void) {
    for (uint32_t id=0u;id<R1_CERT_CLASSES;++id) {
        scales[id]=(double)(1u+id%17u)/500.0;
        biases[id]=(float)((int)(id%11u)-5)*0.125f;
        for (size_t i=0u;i<R1_CERT_HIDDEN;++i)
            rows[(size_t)id*R1_CERT_HIDDEN+i]=
                (int8_t)((int)(pseudo_random()%15u)-7);
    }
    r1_cert_head gate;
    REQUIRE(r1_cert_init(&gate,rows,scales,biases,active_other_ids,
                         ACTIVE_OTHER_COUNT)==R1_CERT_OK);
    double x[R1_CERT_HIDDEN]={0};
    for (unsigned step=0u;step<220u;++step) {
        for (size_t i=0u;i<R1_CERT_HIDDEN;++i) {
            if ((step%9u)==0u)
                x[i]=((int)(pseudo_random()%2001u)-1000)/500.0;
            else
                x[i]+=((int)(pseudo_random()%21u)-10)/100000.0;
        }
        REQUIRE(compare_one(&gate,x,-1)==0);
    }
    return 0;
}
static int test_invalid(void) {
    trivial_fixture();
    r1_cert_head state,unchanged;
    memset(&state,0,sizeof(state));unchanged=state;
    uint32_t wrong[ACTIVE_OTHER_COUNT];
    memcpy(wrong,active_other_ids,sizeof(wrong));
    wrong[1]=wrong[0];
    REQUIRE(r1_cert_init(&state,rows,scales,biases,wrong,
                         ACTIVE_OTHER_COUNT)==R1_CERT_BAD_MODEL);
    REQUIRE(!memcmp(&state,&unchanged,sizeof(state)));
    rows[R1_CERT_HIDDEN]=INT8_MIN;
    REQUIRE(r1_cert_init(&state,rows,scales,biases,active_other_ids,
                         ACTIVE_OTHER_COUNT)==R1_CERT_BAD_MODEL);
    REQUIRE(!memcmp(&state,&unchanged,sizeof(state)));
    rows[R1_CERT_HIDDEN]=0;
    scales[1]=0.0;
    REQUIRE(r1_cert_init(&state,rows,scales,biases,active_other_ids,
                         ACTIVE_OTHER_COUNT)==R1_CERT_BAD_MODEL);
    REQUIRE(!memcmp(&state,&unchanged,sizeof(state)));
    return 0;
}
int main(void) {
    if (test_trivial() || test_ties() || test_mixed() || test_invalid())
        return 1;
    REQUIRE(checked==233u);
    REQUIRE(skipped>0u && fallback>0u);
    printf("{\"status\":\"PASS_INVENTED_EXACT_OTHER\","
           "\"frames\":%u,\"certified_skips\":%u,"
           "\"exact_fallbacks\":%u,\"owner_model_calls\":0,"
           "\"pcm_calls\":0,\"decoder_calls\":0}\n",
           checked,skipped,fallback);
    return 0;
}
