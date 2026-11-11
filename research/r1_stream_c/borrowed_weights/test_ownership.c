/* SPDX-License-Identifier: Apache-2.0
 * Explicit link-time mocks isolate allocation/ownership and failure handling.
 * No genuine model is loaded. A network step aborts this test immediately.
 */
#include "fsmn_sdk.h"
#include "r1dfsmn.h"
#include "r1d_pcm.h"
#include "reference64_fbank.h"
#include "model_identity.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned checks,copy_allocations,copy_frees,caller_free_attempts;
static unsigned copied_calls,view_calls,calloc_calls;
static unsigned state_allocations,state_frees,caller_table_free_attempts;
static void *live_states[8];
static unsigned fail_calloc_at;
static int fail_model_alloc,fail_load,fail_bank,fail_pcm,bad_table_hash;
static unsigned char *caller_model,*caller_tables;
static void *owned_model;
#define CHECK(expr) do { checks++; if(!(expr)){fprintf(stderr,"ownership check failed: %u: %s\n",__LINE__,#expr);exit(1);} } while(0)

void *__real_malloc(size_t);
void *__real_calloc(size_t,size_t);
void __real_free(void *);
void *__wrap_malloc(size_t n) {
    if(n==FSMN_MODEL_BYTES && fail_model_alloc)return NULL;
    void *p=__real_malloc(n);
    if(n==FSMN_MODEL_BYTES && p){CHECK(!owned_model);owned_model=p;copy_allocations++;}
    return p;
}
void *__wrap_calloc(size_t n,size_t bytes) {
    calloc_calls++;
    if(fail_calloc_at && calloc_calls==fail_calloc_at)return NULL;
    void *p=__real_calloc(n,bytes);
    if(p) {
        unsigned slot=0;
        while(slot<8 && live_states[slot])slot++;
        CHECK(slot<8);live_states[slot]=p;state_allocations++;
    }
    return p;
}
void __wrap_free(void *p) {
    if(p==caller_model && p){caller_free_attempts++;return;}
    if(p==caller_tables && p){caller_table_free_attempts++;return;}
    if(p==owned_model && p){copy_frees++;owned_model=NULL;}
    for(unsigned i=0;i<8;i++)if(p && p==live_states[i]) {
        state_frees++;live_states[i]=NULL;break;
    }
    __real_free(p);
}
void __wrap_r1d_sha256_memory_hex(const unsigned char *p,size_t n,char out[65]) {
    CHECK(p==caller_tables && n==FSMN_TABLE_BYTES);
    memcpy(out,R1D_FRONTEND_SHA,65);
    if(bad_table_hash)out[0]=out[0]=='a'?'b':'a';
}
r1dfsmn_status __wrap_r1dfsmn_load(r1dfsmn_model *m,float *w,size_t nw,
        const unsigned char *p,size_t n,unsigned order,const char hash[65]) {
    copied_calls++;
    CHECK(p==caller_model && w==owned_model && nw==FSMN_MODEL_BYTES/4u);
    CHECK(n==FSMN_MODEL_BYTES && order==1 && !strcmp(hash,R1D_MODEL_SHA));
    m->weights=w;
    return fail_load?R1DFSMN_IDENTITY:R1DFSMN_OK;
}
r1dfsmn_status __wrap_r1dfsmn_load_view(r1dfsmn_model *m,const unsigned char *p,
        size_t n,unsigned order,const char hash[65]) {
    view_calls++;
    CHECK(p==caller_model && n==FSMN_MODEL_BYTES && order==1 && !strcmp(hash,R1D_MODEL_SHA));
    CHECK(!owned_model);
    m->weights=(const float *)p;
    return fail_load?R1DFSMN_IDENTITY:R1DFSMN_OK;
}
int __wrap_r1d_fbank_fbank_init(r1d_fbank_fbank *s,const r1d_fbank_tables *tables) {
    (void)s;(void)tables;return fail_bank?R1DCF_FBANK_TABLE:0;
}
int __wrap_r1d_pcm_init(r1d_pcm *s,r1d_fbank_bank_fn bank,void *bank_user,
                       r1d_fbank_emit_fn emit,void *emit_user) {
    (void)s;(void)bank;(void)bank_user;(void)emit;(void)emit_user;
    return fail_pcm?R1DCF_ARGUMENT:0;
}
r1dfsmn_status __wrap_r1dfsmn_step(r1dfsmn_model *m,const double *x,size_t nx,
        double *y,size_t ny,double *t,size_t nt) {
    (void)m;(void)x;(void)nx;(void)y;(void)ny;(void)t;(void)nt;
    fputs("ERROR: a model forward was attempted in a source-only test\n",stderr);abort();
}

static void run_case(int view,int success) {
    unsigned allocations_before=copy_allocations,frees_before=copy_frees;
    unsigned copy_before=copied_calls,view_before=view_calls;
    calloc_calls=0;
    fsmn_sdk *sdk=view?
        fsmn_sdk_create_view(caller_model,FSMN_MODEL_BYTES,caller_tables,FSMN_TABLE_BYTES,0,NULL,NULL,NULL):
        fsmn_sdk_create(caller_model,FSMN_MODEL_BYTES,caller_tables,FSMN_TABLE_BYTES,0,NULL,NULL,NULL);
    CHECK((sdk!=NULL)==success);
    if(sdk) {
        CHECK(!fsmn_sdk_frame_count(sdk) && !fsmn_sdk_consumed_bytes(sdk));
        CHECK(!fsmn_sdk_event_count(sdk));
        CHECK(fsmn_sdk_destroy(sdk)==FSMN_OK);
    }
    CHECK(caller_free_attempts==0 && caller_table_free_attempts==0 && !owned_model);
    CHECK(state_allocations==state_frees);
    CHECK(caller_model[0]==0x37 && caller_model[FSMN_MODEL_BYTES-1]==0x37);
    CHECK(copy_allocations-allocations_before==copy_frees-frees_before);
    if(view) {
        CHECK(copy_allocations==allocations_before && copied_calls==copy_before);
        if(success)CHECK(view_calls==view_before+1);
    } else {
        CHECK(view_calls==view_before);
        if(success)CHECK(copy_allocations==allocations_before+1 && copied_calls==copy_before+1);
    }
}

int main(void) {
    caller_model=(unsigned char *)__real_malloc(FSMN_MODEL_BYTES);
    caller_tables=(unsigned char *)__real_calloc(1,FSMN_TABLE_BYTES);
    CHECK(caller_model && caller_tables);
    memset(caller_model,0x37,FSMN_MODEL_BYTES);
    for(int view=0;view<=1;view++) {
        run_case(view,1);
        fail_load=1;run_case(view,0);fail_load=0;
        fail_bank=1;run_case(view,0);fail_bank=0;
        fail_pcm=1;run_case(view,0);fail_pcm=0;
        bad_table_hash=1;run_case(view,0);bad_table_hash=0;
        fail_calloc_at=1;run_case(view,0);
        fail_calloc_at=2;run_case(view,0);fail_calloc_at=0;
    }
    fail_model_alloc=1;run_case(0,0);run_case(1,1);fail_model_alloc=0;
    CHECK(copy_allocations==copy_frees && caller_free_attempts==0 && caller_table_free_attempts==0);
    CHECK(state_allocations==state_frees);
    __real_free(caller_tables);__real_free(caller_model);
    printf("{\"fixture\":\"explicit_constructor_link_stubs\",\"model_forwards\":0,\"checks\":%u,\"copy_allocations\":%u,\"copy_frees\":%u,\"caller_frees_by_sdk\":%u,\"caller_table_frees_by_sdk\":%u,\"state_allocations\":%u,\"state_frees\":%u}\n",
           checks,copy_allocations,copy_frees,caller_free_attempts,caller_table_free_attempts,state_allocations,state_frees);
    return 0;
}
