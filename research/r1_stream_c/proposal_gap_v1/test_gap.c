/* SPDX-License-Identifier: Apache-2.0
 * Fictional tokens and the existing mock acoustic core only. */
#define main legacy_sdk_fixture_main
#include "candidate_sdk/test_sdk.c"
#undef main
#include <stddef.h>

/* Preserve the caller-allocated original decoder ABI. */
typedef struct {
    uint64_t rows,events,frame[4],cooldown[2];
    unsigned last,used,ids[4];
} original_decoder_layout;
_Static_assert(sizeof(original_decoder_layout)==sizeof(r1_proposal_decoder),"original decoder size");
_Static_assert(offsetof(original_decoder_layout,ids)==offsetof(r1_proposal_decoder,ids),"original decoder offsets");

static void token(r1_proposal_decoder_v2 *d,unsigned t,unsigned *word){
    CHECK(!r1_proposal_step_v2(d,t,d->original.rows,word));
}
static void zeros(r1_proposal_decoder_v2 *d,unsigned n,unsigned *word){
    for(unsigned i=0;i<n;i++)token(d,0,word);
}
static void candidate_start(unsigned limit){
    count=0;callback_fail=reenter=fail_feed=fail_reset=0;
    active=r1_sdk_create_with_blank_gap(&seven,1,&seven,1,limit,sink,NULL);
    CHECK(active&&r1_sdk_blank_gap_limit(active)==limit);
}
int main(void){
    CHECK(!legacy_sdk_fixture_main());
    unsigned new_groups=0,k=99;
    r1_proposal_decoder_v2 d,before;
    const unsigned *words[]={k1,k2};
    for(unsigned w=0;w<2;w++)for(unsigned n=19;n<=21;n++){
        CHECK(!r1_proposal_init_v2(&d,20));
        token(&d,words[w][0],&k);token(&d,words[w][1],&k);
        zeros(&d,n,&k);token(&d,words[w][2],&k);token(&d,words[w][3],&k);
        CHECK(d.original.events==(n==19?1u:0u));
        CHECK(d.prefix_expirations==(n==19?0u:1u));
    }new_groups++;
    CHECK(!r1_proposal_init_v2(&d,20));
    for(unsigned i=0;i<101;i++)token(&d,1027,&k);
    for(unsigned i=1;i<4;i++)token(&d,k1[i],&k);
    CHECK(k==1&&d.original.events==1&&d.prefix_expirations==0);new_groups++;
    CHECK(!r1_proposal_init_v2(&d,20));
    for(unsigned i=0;i<4;i++){token(&d,k1[i],&k);if(i<3)zeros(&d,10,&k);}
    CHECK(k==1&&d.prefix_expirations==0);new_groups++;
    for(unsigned n=148;n<=149;n++){
        CHECK(!r1_proposal_init_v2(&d,20));
        for(unsigned i=0;i<n;i++)token(&d,1027,&k);
        for(unsigned i=1;i<4;i++)token(&d,k1[i],&k);
        CHECK(d.original.events==(n==148?1u:0u));
    }new_groups++;
    CHECK(!r1_proposal_init_v2(&d,20));
    for(unsigned i=0;i<4;i++)token(&d,k2[i],&k);
    zeros(&d,20,&k);CHECK(d.original.cooldown[1]==53);
    for(unsigned i=0;i<4;i++)token(&d,k2[i],&k);
    CHECK(d.original.events==1&&d.original.cooldown[1]==53);
    zeros(&d,22,&k);for(unsigned i=0;i<4;i++)token(&d,k2[i],&k);
    CHECK(d.original.events==2&&k==2&&d.original.rows==54);new_groups++;
    CHECK(!r1_proposal_init_v2(&d,20));
    for(unsigned i=0;i<4;i++)token(&d,k1[i],&k);
    zeros(&d,20,&k);for(unsigned i=0;i<4;i++)token(&d,k2[i],&k);
    CHECK(d.original.events==2&&k==2);new_groups++;
    CHECK(!r1_proposal_init_v2(&d,20));token(&d,1027,&k);zeros(&d,1000,&k);
    CHECK(d.prefix_expirations==1&&d.consecutive_blanks==20&&d.original.rows==1001);
    CHECK(!d.original.used&&!d.original.last);new_groups++;
    before=d;CHECK(r1_proposal_init_v2(&d,151)==R1_SDK_ARGUMENT);
    CHECK(!memcmp(&before,&d,sizeof(d)));
    CHECK(r1_proposal_step_v2(&d,2601,d.original.rows,&k)==R1_SDK_ARGUMENT);
    CHECK(!memcmp(&before,&d,sizeof(d)));
    CHECK(r1_proposal_step_v2(&d,0,d.original.rows+1,&k)==R1_SDK_ARGUMENT);
    CHECK(!memcmp(&before,&d,sizeof(d)));
    CHECK(r1_proposal_step_v2(&d,0,d.original.rows,NULL)==R1_SDK_ARGUMENT);
    CHECK(!memcmp(&before,&d,sizeof(d)));
    CHECK(r1_proposal_init_v2(NULL,20)==R1_SDK_ARGUMENT);
    CHECK(r1_proposal_step_v2(NULL,0,0,&k)==R1_SDK_ARGUMENT);
    d.original.rows=UINT64_MAX-50u;before=d;
    CHECK(r1_proposal_step_v2(&d,0,d.original.rows,&k)==R1_SDK_ARGUMENT);
    CHECK(!memcmp(&before,&d,sizeof(d)));new_groups++;
    r1_proposal_decoder old;
    r1_proposal_reset(&old);CHECK(!r1_proposal_init_v2(&d,0));
    for(unsigned i=0;i<1000;i++){
        unsigned value=(i%11<4?k1[i%11]:(i%11<8?k2[i%11-4]:0));
        unsigned a=0,b=0;CHECK(!r1_proposal_step(&old,value,i,&a));
        CHECK(!r1_proposal_step_v2(&d,value,i,&b));
        CHECK(a==b&&!memcmp(&old,&d.original,sizeof(old)));
    }CHECK(!d.prefix_expirations&&!d.consecutive_blanks);new_groups++;
    CHECK(!r1_sdk_create_with_blank_gap(&seven,1,&seven,1,151,sink,NULL));
    CHECK(!r1_sdk_create_with_blank_gap(&seven,1,&seven,1,20,NULL,NULL));new_groups++;
    candidate_start(20);unsigned zero[20]={0};unsigned initial_allocs=allocs,initial_frees=frees;
    CHECK(!push(0,k1,2));CHECK(!push(4,zero,20));CHECK(!push(44,k1+2,2));
    CHECK(!count&&r1_sdk_frames(active)==24);CHECK(allocs==initial_allocs&&frees==initial_frees);
    CHECK(!r1_sdk_gap(active,100));CHECK(r1_sdk_blank_gap_limit(active)==20);
    CHECK(!push(200,k1,4));CHECK(count==1&&last.epoch==1&&last.model_frame==3);
    CHECK(last.required_end_sample==2260&&last.delivery_end_sample==2260);
    CHECK(allocs==initial_allocs&&frees==initial_frees);
    CHECK(!r1_sdk_finish(active));CHECK(!r1_sdk_destroy(active));new_groups++;
    candidate_start(0);CHECK(!push(0,k1,2));CHECK(!push(4,zero,20));CHECK(!push(44,k1+2,2));
    CHECK(count==1&&last.keyword_id==1);CHECK(!r1_sdk_destroy(active));new_groups++;
    candidate_start(20);reenter=1;CHECK(!push(0,k1,4));CHECK(count==1);
    CHECK(!r1_sdk_destroy(active));new_groups++;
    candidate_start(20);callback_fail=1;CHECK(push(0,k1,4)!=0);
    CHECK(push(8,k1,4)==R1_SDK_STATE);callback_fail=0;CHECK(!r1_sdk_gap(active,100));
    CHECK(r1_sdk_blank_gap_limit(active)==20);CHECK(!push(200,k1,4));CHECK(count==2);
    CHECK(!r1_sdk_destroy(active));new_groups++;
    candidate_start(20);unsigned bad=65535u;CHECK(push(0,&bad,1)==R1_SDK_NUMERIC);
    CHECK(push(2,k1,4)==R1_SDK_STATE);CHECK(!r1_sdk_destroy(active));new_groups++;
    printf("{\"status\":\"PASS\",\"new_groups\":%u,\"legacy_groups\":21,\"fixture_only\":true,\"model_forwards\":0,\"max_blank_gap\":20,\"original_decoder_bytes\":%zu,\"v2_decoder_bytes\":%zu}\n",new_groups,sizeof(r1_proposal_decoder),sizeof(r1_proposal_decoder_v2));
    return 0;
}
