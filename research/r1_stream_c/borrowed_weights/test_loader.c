/* SPDX-License-Identifier: Apache-2.0
 * Invented structural payloads only. Real hash/loader checks; zero forwards.
 * The fixture hash is deliberately NOT the public SDK's pinned model hash.
 */
#include "r1dfsmn.h"
#include "sha256.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fenv.h>

static unsigned checks;
#define CHECK(expr) do { checks++; if(!(expr)){fprintf(stderr,"loader check failed: %u: %s\n",__LINE__,#expr);exit(1);} } while(0)

static void fixture(unsigned char *p) {
    memset(p,0,R1DFSMN_Q8_PACKAGE_BYTES);
    memcpy(p,"R1HQ8V1",7);
    const uint32_t fields[4]={1,2601,140,5};
    memcpy(p+8,fields,sizeof(fields));
    memcpy(p+24,"e9a3c3360bab0485b309eb2a3ef50869a8e917f50f28dbfd30673f57a88c86bb",64);
    const float one=1.0f;
    for(unsigned i=400;i<800;i++)memcpy(p+R1DFSMN_Q8_HEADER_BYTES+4u*i,&one,4);
    const double scale=1.0;
    for(unsigned i=0;i<2601;i++)memcpy(p+R1DFSMN_Q8_SCALE_OFFSET+8u*i,&scale,8);
}

static void rejected(r1dfsmn_model *m,const unsigned char *p,size_t n,
                     unsigned order,const char *hash,int expected) {
    r1dfsmn_model before;
    char payload_before[65],payload_after[65];
    if(p)r1d_sha256_memory_hex(p,n,payload_before);
    memcpy(&before,m,sizeof(before));
    CHECK((int)r1dfsmn_load_view(m,p,n,order,hash)==expected);
    CHECK(!memcmp(&before,m,sizeof(before)));
    if(p){r1d_sha256_memory_hex(p,n,payload_after);CHECK(!strcmp(payload_before,payload_after));}
}

int main(void) {
    const size_t n=R1DFSMN_Q8_PACKAGE_BYTES;
    unsigned char *p=(unsigned char *)malloc(n+8);
    float *copy=(float *)malloc(n);
    CHECK(p && copy && (uintptr_t)p%R1DFSMN_VIEW_ALIGNMENT==0);
    memset(p,0,n+8);
    fixture(p);
    char hash[65],after_hash[65];
    r1d_sha256_memory_hex(p,n,hash);
    r1dfsmn_model view,copied;
    memset(&view,0xa5,sizeof(view));memset(&copied,0xa5,sizeof(copied));
    CHECK(r1dfsmn_load_view(&view,p,n,1,hash)==R1DFSMN_OK);
    CHECK(view.weights==(const float *)(p+R1DFSMN_Q8_HEADER_BYTES));
    CHECK(view.qhead==(const int8_t *)(p+R1DFSMN_Q8_QUANT_OFFSET));
    CHECK(view.scales==(const double *)(p+R1DFSMN_Q8_SCALE_OFFSET));
    CHECK(view.biases==(const float *)(p+R1DFSMN_Q8_BIAS_OFFSET));
    CHECK(view.precise_head==(const float *)(p+R1DFSMN_Q8_PRECISE_OFFSET));
    for(size_t i=0;i<R1DFSMN_CACHE_FLOATS;i++)CHECK(view.cache[i]==0.0);
    CHECK(view.fault==0);
    r1d_sha256_memory_hex(p,n,after_hash);CHECK(!strcmp(hash,after_hash));
    CHECK(r1dfsmn_load(&copied,copy,n/4,p,n,1,hash)==R1DFSMN_OK);
    CHECK(copied.weights==(const float *)((unsigned char *)copy+R1DFSMN_Q8_HEADER_BYTES));
    CHECK(copied.weights!=view.weights && !memcmp(copy,p,n));

    rejected(&view,NULL,n,1,hash,R1DFSMN_ARGUMENT);
    rejected(&view,p,n-1,1,hash,R1DFSMN_SIZE);
    for(unsigned offset=1;offset<8;offset++)
        rejected(&view,p+offset,n,1,hash,R1DFSMN_ARGUMENT);
    rejected(&view,p,n,2,hash,R1DFSMN_ENVIRONMENT);
    rejected(&view,p,n,1,NULL,R1DFSMN_ARGUMENT);
    char bad_hash[65];memcpy(bad_hash,hash,sizeof(hash));bad_hash[0]='G';
    rejected(&view,p,n,1,bad_hash,R1DFSMN_IDENTITY);
    rejected(&view,p,n,1,(const char *)&view,R1DFSMN_OVERLAP);
    int rounding=fegetround();CHECK(fesetround(FE_DOWNWARD)==0);
    rejected(&view,p,n,1,hash,R1DFSMN_ENVIRONMENT);
    CHECK(fesetround(rounding)==0);

    unsigned char *snapshot=(unsigned char *)malloc(sizeof(view));CHECK(snapshot);
    memcpy(snapshot,p,sizeof(view));
    CHECK(r1dfsmn_load_view((r1dfsmn_model *)p,p,n,1,hash)==R1DFSMN_OVERLAP);
    CHECK(!memcmp(snapshot,p,sizeof(view)));free(snapshot);
    p[n-1]^=1u;rejected(&view,p,n,1,hash,R1DFSMN_IDENTITY);p[n-1]^=1u;

    struct corruption {size_t offset;uint64_t bits;size_t bytes;int status;} cases[]={
        {8,2,4,R1DFSMN_IDENTITY},
        {R1DFSMN_Q8_PAD_OFFSET,1,1,R1DFSMN_IDENTITY},
        {R1DFSMN_Q8_HEADER_BYTES,UINT32_C(0x7fc00000),4,R1DFSMN_NONFINITE},
        {R1DFSMN_Q8_HEADER_BYTES+400u*4u,0,4,R1DFSMN_ARGUMENT},
        {R1DFSMN_Q8_QUANT_OFFSET,0x80,1,R1DFSMN_ARGUMENT},
        {R1DFSMN_Q8_SCALE_OFFSET,0,8,R1DFSMN_NONFINITE},
        {R1DFSMN_Q8_SCALE_OFFSET,UINT64_C(0x7ff8000000000000),8,R1DFSMN_NONFINITE},
        {R1DFSMN_Q8_BIAS_OFFSET,UINT32_C(0x7f800000),4,R1DFSMN_NONFINITE},
        {R1DFSMN_Q8_PRECISE_OFFSET,UINT32_C(0x7fc00000),4,R1DFSMN_NONFINITE}
    };
    for(size_t i=0;i<sizeof(cases)/sizeof(cases[0]);i++) {
        fixture(p);memcpy(p+cases[i].offset,&cases[i].bits,cases[i].bytes);
        r1d_sha256_memory_hex(p,n,hash);
        rejected(&view,p,n,1,hash,cases[i].status);
    }
    free(copy);free(p);
    printf("{\"fixture\":\"invented_structural_payload\",\"model_forwards\":0,\"checks\":%u}\n",checks);
    return 0;
}
