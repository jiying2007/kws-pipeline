#include "../baseline/decoder/a20_decoder.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
/* Static storage: no heap and no large thread-stack allocation required. */
static a20d_decoder state, saved;
static a20d_workspace scratch;
static float rows[512][6];
static void hot(float *row,int token) {
    for(int j=0;j<6;++j)row[j]=.014f;
    row[token]=.93f;
}
static void check_invariants(void) {
    assert(state.hyp_count<=A20D_PATH_BEAM && state.node_count<=A20D_NODE_CAP);
    for(size_t h=0;h<state.hyp_count;++h) {
        assert(state.hyps[h].len<=A20D_PREFIX_CAP);
        assert(isfinite(state.hyps[h].pb) && isfinite(state.hyps[h].pnb));
        for(size_t j=0;j<state.hyps[h].len;++j) {
            assert(state.hyps[h].node[j]<state.node_count);
            const a20d_node *n=&state.nodes[state.hyps[h].node[j]];
            assert(n->token>=1 && n->token<=5 && isfinite(n->prob));
        }
    }
}
int main(void) {
    a20d_result r, before; float out[6];int32_t top[3];
    assert(a20d_init(NULL)==A20D_INVALID);
    assert(a20d_init(&state)==A20D_OK);
    assert(a20d_process_probs(&state,&scratch,NULL,0,&r)==A20D_OK && !r.valid);
    for(int i=0;i<10;++i)hot(rows[i],i<4?i+1:0);
    assert(a20d_process_probs(&state,&scratch,&rows[0][0],10,&r)==A20D_OK);
    assert(r.state==1 && r.keyword==1 && r.start_frame==0 && r.end_frame==9);
    assert(r.rows_decoded==4 && state.total_frames==30 && state.last_active_pos==9);
    check_invariants();
    assert(a20d_reset(&state)==A20D_OK && state.total_frames==30 && state.last_active_pos==9);
    assert(a20d_reset_all(&state)==A20D_OK && state.total_frames==0 && state.last_active_pos==-1);
    float same[6]={0,0,0,0,0,0};
    assert(a20d_softmax6(same,out)==A20D_OK);
    assert(a20d_top3(out,top)==A20D_OK && top[0]==3 && top[1]==5 && top[2]==4);
    float extreme[6]={FLT_MAX,-FLT_MAX,FLT_MAX,-FLT_MAX,0,1};
    assert(a20d_softmax6(extreme,out)==A20D_OK && out[0]==.5f && out[2]==.5f);
    saved=state;before=r;
    rows[0][0]=NAN;
    assert(a20d_process_probs(&state,&scratch,&rows[0][0],1,&r)==A20D_INVALID);
    assert(memcmp(&state,&saved,sizeof(state))==0 && memcmp(&r,&before,sizeof(r))==0);
    assert(a20d_process_logits(&state,&scratch,&rows[0][0],1,&r)==A20D_INVALID);
    assert(a20d_process_probs(&state,&scratch,NULL,1,&r)==A20D_INVALID);
    assert(a20d_process_probs(&state,&scratch,&rows[0][0],SIZE_MAX,&r)==A20D_OVERFLOW);
    for(int i=0;i<130;++i)hot(rows[i],1+i%2);
    assert(a20d_process_probs(&state,&scratch,&rows[0][0],130,&r)==A20D_CAPACITY);
    assert(memcmp(&state,&saved,sizeof(state))==0);
    state.total_frames=INT64_MAX-1;saved=state;
    assert(a20d_process_probs(&state,&scratch,&rows[0][0],1,&r)==A20D_OVERFLOW);
    assert(memcmp(&state,&saved,sizeof(state))==0);
    /* Deterministic finite synthetic stress, all six tokens and branch states. */
    uint32_t rng=UINT32_C(0x425a20);
    size_t processed=0;
    for(int stream=0;stream<200;++stream) {
        assert(a20d_reset_all(&state)==A20D_OK);
        for(int chunk=0;chunk<30;++chunk) {
            for(int i=0;i<12;++i) {
                double sum=0;
                for(int j=0;j<6;++j) {rng=rng*UINT32_C(1664525)+UINT32_C(1013904223);
                    rows[i][j]=(float)(1+(rng>>16));sum+=rows[i][j];}
                for(int j=0;j<6;++j)rows[i][j]=(float)(rows[i][j]/sum);
            }
            assert(a20d_process_probs(&state,&scratch,&rows[0][0],12,&r)==A20D_OK);
            processed+=12;check_invariants();
        }
    }
    printf("{\"native_c_tests\":\"passed\",\"synthetic_stress_rows\":%zu,"
           "\"decoder_bytes\":%zu,\"workspace_bytes\":%zu,\"result_bytes\":%zu}\n",
           processed,sizeof(state),sizeof(scratch),sizeof(r));
    return 0;
}
