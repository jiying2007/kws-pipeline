/* Seed the existing public caller-owned state; no production test API or long run. */
#include "donor_fbank.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
_Static_assert(sizeof(((donor_fbank_state*)0)->total_samples)==8,"sample timeline must be64bit");
_Static_assert(sizeof(((donor_fbank_state*)0)->frame_index)==8,"frame timeline must be64bit");
static uint64_t want_index,want_end;
static unsigned callbacks;
static void received(void*u,const float row[80],uint64_t index,uint64_t end){
 (void)u;(void)row;assert(index==want_index);assert(end==want_end);
 want_index++;want_end+=UINT64_C(160);callbacks++;
}
static void two_frames(uint64_t first_index){
 donor_fbank_state state;int16_t zero[320]={0};assert(donor_fbank_init(&state)==0);
 /* Reachable all-zero steady state immediately after the previous frame. */
 state.frame_index=first_index;state.total_samples=UINT64_C(240)+UINT64_C(160)*first_index;state.used=240;
 uint64_t before=state.total_samples;want_index=first_index;want_end=UINT64_C(400)+UINT64_C(160)*first_index;callbacks=0;
 assert(donor_fbank_feed(&state,zero,320,received,0)==0);assert(callbacks==2);
 assert(state.frame_index==first_index+UINT64_C(2));assert(state.total_samples==before+UINT64_C(320));
 assert(donor_fbank_finish(&state)==0);assert(donor_fbank_reset(&state)==0);assert(state.total_samples==0&&state.frame_index==0);
}
int main(void){
 uint64_t boundaries[2]={UINT64_C(1)<<31,UINT64_C(1)<<32};
 for(unsigned i=0;i<2;i++){
  uint64_t b=boundaries[i];uint64_t j=(b-UINT64_C(400))/UINT64_C(160);
  assert(UINT64_C(400)+UINT64_C(160)*j<b);assert(UINT64_C(400)+UINT64_C(160)*(j+1)>b);
  two_frames(j);two_frames(b-1);
 }
 donor_fbank_state state,before;int16_t zeros[160]={0};assert(donor_fbank_init(&state)==0);
 state.total_samples=UINT64_MAX-UINT64_C(100);state.frame_index=(state.total_samples-UINT64_C(240))/UINT64_C(160);state.used=(size_t)((state.total_samples-UINT64_C(240))%UINT64_C(160))+240;
 before=state;assert(donor_fbank_feed(&state,zeros,160,received,0)==DONOR_FBANK_ARGUMENT);assert(!memcmp(&before,&state,sizeof(state)));
 puts("long counters: sample and frame crossings at2^31/2^32 plusUINT64 overflow rejection PASS");return 0;
}
