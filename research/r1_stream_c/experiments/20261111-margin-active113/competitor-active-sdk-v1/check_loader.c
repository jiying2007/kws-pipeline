/* SPDX-License-Identifier: Apache-2.0
 * Invented payload checks only. NEVER calls any network/PCM/feature step. */
#include "r1active113.h"
#include "active113_sdk.h"
#include "active113_layout.h"
#include "sha256.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned assertions;
static void check(int yes,const char *name){assertions++;if(!yes){fprintf(stderr,"loader-check:%s\n",name);exit(1);}}
static void put32(unsigned char *p,uint32_t x){p[0]=(unsigned char)x;p[1]=(unsigned char)(x>>8);p[2]=(unsigned char)(x>>16);p[3]=(unsigned char)(x>>24);}
static void fixture(unsigned char *p){
    memset(p,0,R1ACTIVE113_PAYLOAD_BYTES);memcpy(p,R1ACTIVE113_EXPECTED_HEADER,256);
    for(unsigned i=0;i<5;i++)put32(p+R1ACTIVE113_IDS_OFFSET+i*4u,R1ACTIVE113_PRECISE_IDS[i]);
    for(unsigned i=0;i<113;i++)put32(p+R1ACTIVE113_IDS_OFFSET+(i+5u)*4u,R1ACTIVE113_OTHER_IDS[i]);
    for(unsigned i=400;i<800;i++)put32(p+R1ACTIVE113_TRUNK_OFFSET+i*4u,UINT32_C(0x3f800000));
    for(unsigned i=0;i<113;i++){double one=1.0;memcpy(p+R1ACTIVE113_SCALE_OFFSET+i*8u,&one,8u);}
}
static void bad_both(unsigned char *p,void *storage,r1active113_status expected,const char *label){
    char hash[65];r1a113_sha256_memory_hex(p,R1ACTIVE113_PAYLOAD_BYTES,hash);
    r1active113_model model,before;memset(&model,0xA5,sizeof(model));memcpy(&before,&model,sizeof(model));
    memset(storage,0xCD,R1ACTIVE113_PAYLOAD_BYTES);
    check(r1active113_load(&model,storage,R1ACTIVE113_PAYLOAD_BYTES,p,R1ACTIVE113_PAYLOAD_BYTES,1,hash)==expected,label);
    check(!memcmp(&model,&before,sizeof(model)),"failed copy leaves model unchanged");
    const unsigned char *b=(const unsigned char *)storage;int untouched=1;
    for(size_t i=0;i<R1ACTIVE113_PAYLOAD_BYTES;i++)if(b[i]!=0xCD){untouched=0;break;}
    check(untouched,"failed copy leaves storage unchanged");
    check(r1active113_load_view(&model,p,R1ACTIVE113_PAYLOAD_BYTES,1,hash)==expected,label);
    check(!memcmp(&model,&before,sizeof(model)),"failed view leaves model unchanged");
}
int main(void){
    unsigned char *p=malloc(R1ACTIVE113_PAYLOAD_BYTES+8u),*storage=malloc(R1ACTIVE113_PAYLOAD_BYTES+8u);
    check(p && storage,"allocation");check((uintptr_t)p%8u==0 && (uintptr_t)storage%8u==0,"malloc alignment");
    fixture(p);char hash[65];r1a113_sha256_memory_hex(p,R1ACTIVE113_PAYLOAD_BYTES,hash);
    r1active113_model m;
    check(r1active113_load(&m,storage,R1ACTIVE113_PAYLOAD_BYTES,p,R1ACTIVE113_PAYLOAD_BYTES,1,hash)==R1ACTIVE113_OK,"valid copy");
    check(m.payload==storage && !memcmp(p,storage,R1ACTIVE113_PAYLOAD_BYTES),"copy ownership and bytes");
    check(r1active113_other_winner(&m)==UINT32_MAX,"no invented winner before step");
    check(r1active113_load_view(&m,p,R1ACTIVE113_PAYLOAD_BYTES,1,hash)==R1ACTIVE113_OK && m.payload==p,"valid borrowed view");
    r1active113_model before;memcpy(&before,&m,sizeof(m));
    check(r1active113_load_view(&m,p+1,R1ACTIVE113_PAYLOAD_BYTES,1,hash)==R1ACTIVE113_ARGUMENT,"misaligned view");
    check(r1active113_load(&m,storage+1,R1ACTIVE113_PAYLOAD_BYTES,p,R1ACTIVE113_PAYLOAD_BYTES,1,hash)==R1ACTIVE113_ARGUMENT,"misaligned copy destination");
    check(r1active113_load_view(&m,p,R1ACTIVE113_PAYLOAD_BYTES-1u,1,hash)==R1ACTIVE113_SIZE,"truncated payload");
    check(r1active113_load_view(&m,p,R1ACTIVE113_PAYLOAD_BYTES+1u,1,hash)==R1ACTIVE113_SIZE,"oversize payload");
    check(r1active113_load(&m,p,R1ACTIVE113_PAYLOAD_BYTES,p,R1ACTIVE113_PAYLOAD_BYTES,1,hash)==R1ACTIVE113_OVERLAP,"copy aliases source");
    check(r1active113_load_view(&m,p,R1ACTIVE113_PAYLOAD_BYTES,0,hash)==R1ACTIVE113_ENVIRONMENT,"wrong byte order");
    char wrong[65];memcpy(wrong,hash,65);wrong[0]=wrong[0]=='0'?'1':'0';
    check(r1active113_load_view(&m,p,R1ACTIVE113_PAYLOAD_BYTES,1,wrong)==R1ACTIVE113_IDENTITY,"wrong trusted hash");
    wrong[0]='A';check(r1active113_load_view(&m,p,R1ACTIVE113_PAYLOAD_BYTES,1,wrong)==R1ACTIVE113_IDENTITY,"nonlowercase hash");
    check(!memcmp(&m,&before,sizeof(m)),"argument failures leave loaded state unchanged");
    p[0]^=1;bad_both(p,storage,R1ACTIVE113_IDENTITY,"wrong magic");fixture(p);
    p[200]=1;bad_both(p,storage,R1ACTIVE113_IDENTITY,"nonzero reserved header");fixture(p);
    p[28]^=1;bad_both(p,storage,R1ACTIVE113_IDENTITY,"wrong other count");fixture(p);
    p[72]^=1;bad_both(p,storage,R1ACTIVE113_IDENTITY,"wrong original-model identity");fixture(p);
    put32(p+R1ACTIVE113_IDS_OFFSET+20u,0);bad_both(p,storage,R1ACTIVE113_IDENTITY,"wrong selected row ID");fixture(p);
    p[R1ACTIVE113_SCALE_OFFSET-1u]=1;bad_both(p,storage,R1ACTIVE113_IDENTITY,"nonzero alignment padding");fixture(p);
    put32(p+R1ACTIVE113_TRUNK_OFFSET,UINT32_C(0x7fc00000));bad_both(p,storage,R1ACTIVE113_NONFINITE,"nonfinite trunk");fixture(p);
    put32(p+R1ACTIVE113_TRUNK_OFFSET+400u*4u,0);bad_both(p,storage,R1ACTIVE113_ARGUMENT,"nonpositive CMVN scale");fixture(p);
    p[R1ACTIVE113_QUANT_OFFSET]=128;bad_both(p,storage,R1ACTIVE113_ARGUMENT,"forbidden quant -128");fixture(p);
    memset(p+R1ACTIVE113_SCALE_OFFSET,0,8);bad_both(p,storage,R1ACTIVE113_NONFINITE,"zero row scale");fixture(p);
    put32(p+R1ACTIVE113_OTHER_BIAS_OFFSET,UINT32_C(0x7f800000));bad_both(p,storage,R1ACTIVE113_NONFINITE,"nonfinite other bias");fixture(p);
    put32(p+R1ACTIVE113_PRECISE_OFFSET,UINT32_C(0x7fc00000));bad_both(p,storage,R1ACTIVE113_NONFINITE,"nonfinite precise weight");fixture(p);
    put32(p+R1ACTIVE113_PRECISE_BIAS_OFFSET,UINT32_C(0x7f800000));bad_both(p,storage,R1ACTIVE113_NONFINITE,"nonfinite precise bias");fixture(p);
    double scores[6]={0,2,0,0,0,2};uint32_t winner=UINT32_MAX;
    check(R1ACTIVE113_OTHER_IDS[0]<1027u && R1ACTIVE113_OTHER_IDS[112]>1027u,"tie fixture IDs span precise ID");
    check(active113_sdk_pick_original_token(scores,6,R1ACTIVE113_OTHER_IDS[0],&winner)==0 && winner==R1ACTIVE113_OTHER_IDS[0],"lower original OTHER wins tie");
    check(active113_sdk_pick_original_token(scores,6,R1ACTIVE113_OTHER_IDS[112],&winner)==0 && winner==1027u,"lower original precise wins tie");
    scores[5]=3;check(active113_sdk_pick_original_token(scores,6,R1ACTIVE113_OTHER_IDS[112],&winner)==0 && winner==R1ACTIVE113_OTHER_IDS[112],"actual OTHER ID retained");
    uint32_t saved=winner;scores[0]=NAN;
    check(active113_sdk_pick_original_token(scores,6,R1ACTIVE113_OTHER_IDS[0],&winner)==ACTIVE113_BACKEND_ERROR && winner==saved,"nonfinite scores rejected atomically");
    scores[0]=0;check(active113_sdk_pick_original_token(scores,2601,R1ACTIVE113_OTHER_IDS[0],&winner)==ACTIVE113_BAD_ARGUMENT && winner==saved,"full-vocabulary count rejected");
    check(active113_sdk_pick_original_token(scores,6,1,&winner)==ACTIVE113_BAD_ARGUMENT && winner==saved,"sentinel OTHER ID rejected");
    free(storage);free(p);
    printf("{\"status\":\"PASS_INVENTED_LOADER_AND_ID_CHECKS\",\"assertions\":%u,\"neural_model_calls\":0,\"pcm_calls\":0}\n",assertions);
    return 0;
}
