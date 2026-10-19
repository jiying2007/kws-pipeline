/* SPDX-License-Identifier: Apache-2.0.
 * Real adapter/PCM scheduler; only mocked acoustic and decoder backends.
 * Reuse private fixtures without changing their historical eight-group budget.
 */
#define main legacy_fixture_main
#include "test_b_stream_adapter.c"
#undef main

#include "bsa_capture.h"
static void capture_gap(void) {
 clear_controls(); bsa_handle *h=make(.7);
 /* Live event delivered before EOF; then a dangling low byte at an XRUN. */
 feed_ok(h,0,1440); feed_ok(h,1440,1);
 bsa_status before=status(h);
 CHECK(before.real_samples==720 && before.zero_samples==0 && h->pcm.have_low);
 CHECK(callbacks_local==1 && !last_row.eof_derived);
 /* Hardware reports next complete PCM sample at absolute index 1120.
  * Drop incomplete transport sample and resynchronise PCM16LE at its low byte.
  * Do NOT finish: doing so would fabricate a tail and events at the gap. */
 bsa_capture_anchor anchor={1120,UINT64_C(1070000000),4,1,1};
 bsa_capture_clock clock; bsa_capture_endpoint endpoint;
 unsigned callbacks_before=budget.callbacks;
 clear_controls(); CHECK(++budget.generations<=MAX_GENERATIONS);
 CHECK(bsa_capture_reset(h,"post-gap",receive,NULL,&clock,&anchor,BSA_CAPTURE_XRUN)==BSA_OK);
 bsa_status after=status(h);
 CHECK(after.generation==before.generation+1 && after.aborted_generations==1);
 CHECK(after.zero_samples==0 && after.finish_calls==0 && budget.callbacks==callbacks_before);
 CHECK(h->pcm.have_low==0 && after.real_samples==0);
 feed_ok(h,0,1); feed_ok(h,1,1439);
 CHECK(last_row.row==0 && last_row.generation==after.generation && !last_row.eof_derived);
 CHECK(last_row.required_padded_sample==720 && last_row.available_real_sample==720);
 CHECK(bsa_capture_map_real(&clock,last_row.generation,last_row.required_padded_sample,&endpoint)==BSA_OK);
 CHECK(endpoint.sample_known && endpoint.time_known && endpoint.clock_epoch==4);
 CHECK(endpoint.absolute_sample==1840 && endpoint.monotonic_ns==UINT64_C(1115000000));
 CHECK(bsa_capture_map_real(&clock,before.generation,720,&endpoint)==BSA_STATE);
 finish_ok(h); bsa_status closed=status(h);
 CHECK(bsa_finish(h)==BSA_STATE && status(h).zero_samples==closed.zero_samples);
 size_t consumed=5; CHECK(bsa_feed(h,pcm_bytes,2,&consumed)==BSA_STATE && consumed==0);
 /* Clock reset starts a new epoch; unknown absolute sample loss is retained. */
 anchor=(bsa_capture_anchor){0,UINT64_C(9000000000),5,0,1};
 clear_controls(); CHECK(++budget.generations<=MAX_GENERATIONS);
 CHECK(bsa_capture_reset(h,"new-clock",receive,NULL,&clock,&anchor,BSA_CAPTURE_CLOCK_RESET)==BSA_OK);
 CHECK(clock.generation==after.generation+1 && clock.reason==BSA_CAPTURE_CLOCK_RESET);
 feed_ok(h,0,1440);
 CHECK(bsa_capture_map_real(&clock,last_row.generation,720,&endpoint)==BSA_OK);
 CHECK(!endpoint.sample_known && endpoint.time_known && endpoint.clock_epoch==5);
 CHECK(endpoint.monotonic_ns==UINT64_C(9045000000));
 CHECK(status(h).finish_calls==0 && status(h).zero_samples==0);
 drop(h);
 /* Overflow rejection leaves outputs untouched. Unknown loss stays unknown. */
 endpoint.absolute_sample=11; endpoint.monotonic_ns=12;
 clock.anchor.sample_origin_known=1; clock.anchor.sample_origin=UINT64_MAX;
 CHECK(bsa_capture_map_real(&clock,clock.generation,1,&endpoint)==BSA_OVERFLOW);
 CHECK(endpoint.absolute_sample==11 && endpoint.monotonic_ns==12);
 clock.anchor.sample_origin=0; clock.anchor.anchor_ns=UINT64_MAX;
 CHECK(bsa_capture_map_real(&clock,clock.generation,1,&endpoint)==BSA_OVERFLOW);
 CHECK(endpoint.absolute_sample==11 && endpoint.monotonic_ns==12);
 clock.anchor.sample_origin_known=clock.anchor.timestamp_known=0;
 CHECK(bsa_capture_map_real(&clock,clock.generation,UINT64_MAX,&endpoint)==BSA_OK);
 CHECK(!endpoint.sample_known && !endpoint.time_known);
 /* Invalid reset arguments do not invoke the adapter; invalid handle cannot
  * leave a valid old mapping available for use. */
 CHECK(bsa_capture_reset(NULL,"bad",receive,NULL,&clock,&anchor,BSA_CAPTURE_XRUN)==BSA_STATE);
 CHECK(!clock.valid);

}
static void continuous_chunks(void) {
 const size_t chunks[]={1,159,321,7,640,1272};
 uint64_t expected=0;
 for(unsigned lane=0;lane<2;lane++) {
  clear_controls(); bsa_handle *h=make(.7);
  if(!lane) feed_ok(h,0,sizeof(pcm_bytes));
  else {size_t off=0;for(unsigned i=0;i<sizeof(chunks)/sizeof(chunks[0]);i++) {
   feed_ok(h,off,chunks[i]); off+=chunks[i]; feed_ok(h,off,0);
  } CHECK(off==sizeof(pcm_bytes));}
  CHECK(callbacks_local==2 && status(h).real_samples==1200 && !last_row.eof_derived);
  if(!lane) expected=digest; else CHECK(digest==expected);
  drop(h);
 }
}
int main(void) {
 for(unsigned i=0;i<sizeof(pcm_bytes)/2;i++) {
  uint16_t word=(uint16_t)edge_values[i%4]; pcm_bytes[2*i]=(uint8_t)word;
  pcm_bytes[2*i+1]=(uint8_t)(word>>8);
 }
 capture_gap(); continuous_chunks();
 CHECK(budget.handles==3 && budget.generations==5 && budget.zero_samples==4800);
 puts("capture contract groups=2; gap reset and chunk invariance; mock only");
 return 0;
}
