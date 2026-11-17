/* SPDX-License-Identifier: Apache-2.0
 * Invented token transport/mock core only. No acoustic model or real PCM. */
#include "r1_sdk.h"
#include "r1d_stream.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL:%d:%s\n",__LINE__,#x);exit(1);}}while(0)
struct r1d_stream {r1d_sink sink;void *user;uint64_t rows;unsigned low;int have;};
static unsigned allocs,frees,core_calls;static int fail_feed,fail_reset;
void *__real_malloc(size_t);void *__real_calloc(size_t,size_t);void __real_free(void *);
void *__wrap_malloc(size_t n){allocs++;return __real_malloc(n);}
void *__wrap_calloc(size_t a,size_t b){allocs++;return __real_calloc(a,b);}
void __wrap_free(void *p){frees++;__real_free(p);}
r1d_stream *r1d_create(const unsigned char*w,size_t wn,const unsigned char*t,size_t tn,r1d_sink sink,void*u){
    if(!w||!t||wn!=1||tn!=1||w[0]!=7||t[0]!=7)return NULL;
    r1d_stream*s=calloc(1,sizeof(*s));if(s){s->sink=sink;s->user=u;}return s;
}
int r1d_feed(r1d_stream*s,const unsigned char*p,size_t n,size_t*used){
    *used=0;core_calls++;if(fail_feed)return -9;
    for(size_t i=0;i<n;i++){
        (*used)++;if(!s->have){s->low=p[i];s->have=1;continue;}
        unsigned token=s->low+256u*p[i];s->have=0;
        double f[400]={0},z[R1_SDK_VOCAB];for(unsigned j=0;j<R1_SDK_VOCAB;j++)z[j]=-1.;
        if(token==65535u)z[0]=NAN;else z[token%R1_SDK_VOCAB]=1.;
        uint64_t row=s->rows++;
        int rc=s->sink(s->user,row,720+480*row,720+480*row,f,z);if(rc)return rc;
    }return 0;
}
int r1d_finish(r1d_stream*s){return s->have?-8:0;}
int r1d_reset(r1d_stream*s){if(fail_reset)return -7;s->rows=0;s->have=0;return 0;}
void r1d_destroy(r1d_stream*s){free(s);}
uint64_t r1d_frames(const r1d_stream*s){return s->rows;}
size_t r1d_state_bytes(void){return sizeof(r1d_stream);}
size_t r1d_weight_bytes(void){return 1;}
static r1_sdk *active;static int reenter,callback_fail;static unsigned count;static r1_proposal last;
static int sink(void*u,const r1_proposal*p){(void)u;count++;last=*p;
    if(reenter){size_t n=99;CHECK(r1_sdk_feed(active,0,NULL,0,&n)==R1_SDK_STATE&&n==0);
        CHECK(r1_sdk_gap(active,0)==R1_SDK_STATE);CHECK(r1_sdk_finish(active)==R1_SDK_STATE);
        CHECK(r1_sdk_destroy(active)==R1_SDK_STATE);}
    return callback_fail;
}
static unsigned char seven=7;
static void start(void){count=0;callback_fail=reenter=fail_feed=fail_reset=0;active=r1_sdk_create(&seven,1,&seven,1,sink,NULL);CHECK(active);}
static int push(uint64_t at,const unsigned*t,size_t n){unsigned char b[128];CHECK(n<=64);
    for(size_t i=0;i<n;i++){b[2*i]=(unsigned char)t[i];b[2*i+1]=(unsigned char)(t[i]>>8);}
    size_t used;return r1_sdk_feed(active,at,b,n*2,&used);
}
static unsigned groups;
static const unsigned k1[]={1027,1401,1462,2093},k2[]={1462,2093,1462,2093};
int main(void){
    r1_proposal_decoder d,before;unsigned k=0;
    r1_proposal_reset(&d);for(unsigned i=0;i<4;i++)CHECK(!r1_proposal_step(&d,k1[i],i,&k));CHECK(k==1);groups++;
    r1_proposal_reset(&d);for(unsigned i=0;i<4;i++)CHECK(!r1_proposal_step(&d,k2[i],i,&k));CHECK(k==2);groups++;
    r1_proposal_reset(&d);CHECK(!r1_proposal_step(&d,1462,0,&k)&&!k);CHECK(!r1_proposal_step(&d,2093,1,&k)&&!k);groups++;
    r1_proposal_reset(&d);unsigned repeat[]={1027,1027,1401,1401,1462,2093};for(unsigned i=0;i<6;i++)CHECK(!r1_proposal_step(&d,repeat[i],i,&k));CHECK(k==1);groups++;
    r1_proposal_reset(&d);unsigned blankrepeat[]={1027,0,1027,1401,1462,2093};for(unsigned i=0;i<6;i++)CHECK(!r1_proposal_step(&d,blankrepeat[i],i,&k));CHECK(k==1&&d.events==1);groups++;
    r1_proposal_reset(&d);unsigned broken[]={1027,1401,12,1462,2093};for(unsigned i=0;i<5;i++)CHECK(!r1_proposal_step(&d,broken[i],i,&k));CHECK(!k);groups++;
    r1_proposal_reset(&d);CHECK(!r1_proposal_step(&d,1027,0,&k));for(unsigned i=1;i<=151;i++)CHECK(!r1_proposal_step(&d,0,i,&k));for(unsigned i=1;i<4;i++)CHECK(!r1_proposal_step(&d,k1[i],151+i,&k));CHECK(!k);groups++;
    r1_proposal_reset(&d);for(unsigned i=0;i<8;i++)CHECK(!r1_proposal_step(&d,k1[i%4],i,&k));CHECK(d.events==1);for(unsigned i=8;i<53;i++)CHECK(!r1_proposal_step(&d,0,i,&k));for(unsigned i=0;i<4;i++)CHECK(!r1_proposal_step(&d,k1[i],53+i,&k));CHECK(d.events==2);groups++;
    before=d;CHECK(r1_proposal_step(&d,2601,d.rows,&k)==R1_SDK_ARGUMENT);CHECK(!memcmp(&before,&d,sizeof(d)));CHECK(r1_proposal_step(&d,1,0,&k)==R1_SDK_ARGUMENT);CHECK(!memcmp(&before,&d,sizeof(d)));groups++;
    CHECK(!r1_sdk_create(NULL,0,NULL,0,sink,NULL));CHECK(!r1_sdk_create(&seven,1,&seven,1,NULL,NULL));groups++;
    start();unsigned a=allocs,c=core_calls;size_t used=99;
    CHECK(r1_sdk_feed(active,1,NULL,0,&used)==R1_SDK_OFFSET&&used==0);CHECK(core_calls==c);
    CHECK(r1_sdk_feed(active,0,NULL,1,&used)==R1_SDK_ARGUMENT);CHECK(core_calls==c);
    CHECK(r1_sdk_feed(active,0,&seven,6401,&used)==R1_SDK_ARGUMENT);CHECK(core_calls==c);groups++;
    CHECK(!push(0,k1,4));CHECK(count==1&&last.keyword_id==1&&last.epoch==0);CHECK(allocs==a);CHECK(r1_sdk_frames(active)==4);CHECK(r1_sdk_proposals(active)==1);groups++;
    CHECK(!r1_sdk_finish(active));CHECK(r1_sdk_finish(active)==R1_SDK_STATE);CHECK(push(8,k1,4)==R1_SDK_STATE);groups++;
    CHECK(r1_sdk_gap(active,3)==R1_SDK_OFFSET);CHECK(!r1_sdk_gap(active,100));CHECK(r1_sdk_frames(active)==0);
    CHECK(!push(200,k2,4));CHECK(count==2&&last.keyword_id==2&&last.epoch==1&&last.required_end_sample==2260);CHECK(allocs==a);CHECK(!r1_sdk_destroy(active));groups++;
    start();CHECK(!push(0,k1,2));CHECK(!r1_sdk_gap(active,200));CHECK(!push(400,k1+2,2));CHECK(count==0);CHECK(!r1_sdk_destroy(active));groups++;
    start();reenter=1;CHECK(!push(0,k1,4));CHECK(count==1);CHECK(!r1_sdk_destroy(active));groups++;
    start();callback_fail=1;CHECK(push(0,k1,4)!=0);CHECK(push(8,k1,4)==R1_SDK_STATE);callback_fail=0;CHECK(!r1_sdk_gap(active,100));CHECK(!push(200,k1,4));CHECK(count==2);CHECK(!r1_sdk_destroy(active));groups++;
    start();unsigned nan_token=65535;CHECK(push(0,&nan_token,1)!=0);CHECK(push(2,k1,4)==R1_SDK_STATE);CHECK(!r1_sdk_destroy(active));groups++;
    start();CHECK(!r1_sdk_feed(active,0,&seven,1,&used));CHECK(r1_sdk_finish(active)!=0);CHECK(!r1_sdk_gap(active,1));CHECK(!push(2,k1,4));CHECK(count==1);CHECK(!r1_sdk_destroy(active));groups++;
    start();fail_feed=1;CHECK(push(0,k1,4)==-9);CHECK(push(0,k1,4)==R1_SDK_STATE);fail_feed=0;CHECK(!r1_sdk_gap(active,0));fail_reset=1;CHECK(r1_sdk_gap(active,0)==-7);CHECK(push(0,k1,4)==R1_SDK_STATE);fail_reset=0;CHECK(!r1_sdk_gap(active,0));CHECK(!r1_sdk_destroy(active));groups++;
    start();CHECK(!r1_sdk_gap(active,UINT64_MAX/2));CHECK(r1_sdk_gap(active,UINT64_MAX)==R1_SDK_OFFSET);CHECK(push(UINT64_MAX-1,k1,4)==R1_SDK_ARGUMENT);CHECK(!r1_sdk_destroy(active));groups++;
    printf("{\"groups\":%u,\"model_forwards\":0,\"fixture_only\":true,\"status\":\"PASS\",\"allocations\":%u,\"frees\":%u}\n",groups,allocs,frees);return 0;
}
