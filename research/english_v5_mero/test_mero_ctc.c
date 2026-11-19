#include "mero_ctc.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int feed(en_v5_mero_state *s, unsigned tok, uint64_t t,
                en_v5_mero_event *e) {
    float logits[4]={-10.f,-10.f,-10.f,-10.f};
    assert(tok<4u);
    logits[tok]=3.f;
    return en_v5_mero_accept(s,logits,t,e);
}

static void full_sequence_and_no_second_keyword(void) {
    en_v5_mero_state s;
    en_v5_mero_event e;
    assert(en_v5_mero_init(&s,2500,1600)==0);
    assert(feed(&s,1,1000,&e)==0 && !e.detected);
    assert(feed(&s,1,1160,&e)==0 && !e.detected);
    assert(feed(&s,0,1320,&e)==0 && !e.detected);
    assert(feed(&s,2,1480,&e)==0 && !e.detected);
    assert(feed(&s,0,1640,&e)==0 && !e.detected);
    assert(feed(&s,3,1800,&e)==0 && e.detected);
    assert(e.keyword_id==1u && e.available_after_sample==1800);
    assert(s.event_count==1u);
    /* Old second keyword has no class, token path or event ID in V5. */
    assert(feed(&s,1,1960,&e)==0 && !e.detected);
    assert(feed(&s,2,2120,&e)==0 && !e.detected);
    assert(feed(&s,3,2280,&e)==0 && !e.detected);
    assert(s.event_count==1u);
    assert(feed(&s,0,3500,&e)==0 && !e.detected);
    assert(feed(&s,1,3660,&e)==0 && !e.detected);
    assert(feed(&s,2,3820,&e)==0 && !e.detected);
    assert(feed(&s,3,3980,&e)==0 && e.detected);
    assert(s.event_count==2u);
}

static void no_partial_no_cross_gap(void) {
    en_v5_mero_state s;
    en_v5_mero_event e;
    assert(en_v5_mero_init(&s,1000,0)==0);
    assert(feed(&s,2,1000,&e)==0 && !e.detected);
    assert(feed(&s,3,1160,&e)==0 && !e.detected);
    assert(feed(&s,1,1320,&e)==0 && !e.detected);
    assert(feed(&s,3,1480,&e)==0 && !e.detected);
    assert(feed(&s,2,1640,&e)==0 && !e.detected);
    assert(feed(&s,3,1800,&e)==0 && !e.detected);
    assert(feed(&s,1,1960,&e)==0 && !e.detected);
    assert(feed(&s,0,2120,&e)==0 && !e.detected);
    assert(feed(&s,2,4000,&e)==0 && !e.detected);
    assert(feed(&s,0,4160,&e)==0 && !e.detected);
    assert(feed(&s,3,4320,&e)==0 && !e.detected);
    assert(s.event_count==0u);
}

static void reset_and_discontinuity(void) {
    en_v5_mero_state s;
    en_v5_mero_event e;
    assert(en_v5_mero_init(&s,3000,0)==0);
    assert(feed(&s,1,1000,&e)==0 && !e.detected);
    assert(feed(&s,2,2000,&e)==0 && !e.detected);
    assert(en_v5_mero_reset(&s)==0);
    assert(feed(&s,3,1000,&e)==0 && !e.detected);
    assert(feed(&s,1,2000,&e)==0 && !e.detected);
    assert(en_v5_mero_discontinuity(&s,5000)==0);
    assert(en_v5_mero_discontinuity(&s,4000)==EN_V5_CLOCK);
    assert(feed(&s,2,6000,&e)==0 && !e.detected);
    assert(feed(&s,3,7000,&e)==0 && !e.detected);
    assert(s.event_count==0u);
    assert(feed(&s,1,8000,&e)==0 && !e.detected);
    assert(feed(&s,2,9000,&e)==0 && !e.detected);
    assert(feed(&s,3,10000,&e)==0 && e.detected);
    assert(en_v5_mero_reset(&s)==0);
    assert(s.event_count==1u && !s.clock_initialized);
}

static void fail_closed(void) {
    en_v5_mero_state s;
    en_v5_mero_event e={42,43,44};
    float bad[4]={0,1,NAN,3};
    float scores[4]={0,3,-2,-4};
    assert(en_v5_mero_init(0,100,0)==EN_V5_INVALID);
    assert(en_v5_mero_init(&s,0,0)==EN_V5_INVALID);
    assert(en_v5_mero_init(&s,2500,0)==0);
    assert(en_v5_mero_accept(&s,bad,1000,&e)==EN_V5_NONFINITE);
    assert(e.detected==42u && !s.clock_initialized);
    assert(en_v5_mero_accept(&s,scores,1000,&e)==0 && !e.detected);
    assert(en_v5_mero_accept(&s,scores,1000,&e)==EN_V5_CLOCK);
    assert(en_v5_mero_accept(&s,0,2000,&e)==EN_V5_INVALID);
    assert(en_v5_mero_accept(&s,scores,2000,0)==EN_V5_INVALID);
}

int main(void) {
    full_sequence_and_no_second_keyword();
    no_partial_no_cross_gap();
    reset_and_discontinuity();
    fail_closed();
    puts("ENGLISH_V5_ONE_MERO_CTC_DECODER_PASS");
    return 0;
}
