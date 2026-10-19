/* SPDX-License-Identifier: Apache-2.0. Optional capture-side example helpers.
 * No hardware detection, allocation, new exported symbol or acoustic arithmetic.
 */
#ifndef BSA_CAPTURE_H
#define BSA_CAPTURE_H
#include "b_stream_adapter.h"
#ifdef __cplusplus
extern "C" {
#endif
enum { BSA_CAPTURE_XRUN=1, BSA_CAPTURE_ROUTE_CHANGE=2,
       BSA_CAPTURE_CLOCK_RESET=3, BSA_CAPTURE_SUSPEND=4 };
typedef struct {
 uint64_t sample_origin, anchor_ns, clock_epoch;
 int sample_origin_known, timestamp_known;
} bsa_capture_anchor;
typedef struct {
 bsa_capture_anchor anchor;
 uint64_t generation;
 int reason, valid;
} bsa_capture_clock;
typedef struct {
 uint64_t absolute_sample, monotonic_ns, clock_epoch;
 int sample_known, time_known;
} bsa_capture_endpoint;
/* Caller stops old feed, drops partial sample, and resynchronises byte alignment.
 * Success starts an empty generation without an EOF tail. Failure invalidates
 * the supplied mapping; inspect adapter status and do not resume blindly.
 * Existing clock must not be accessed concurrently or from another handle.
 */
static inline int bsa_capture_reset(bsa_handle *h,const char *id,bsa_callback cb,
 void *user,bsa_capture_clock *clock,const bsa_capture_anchor *anchor,int reason) {
 bsa_status status; kws3_summary summary; uint64_t rejected; int rc;
 if(!clock) return BSA_ARGUMENT;
 clock->valid=0;
 if(!anchor || reason<BSA_CAPTURE_XRUN || reason>BSA_CAPTURE_SUSPEND ||
    (anchor->sample_origin_known!=0 && anchor->sample_origin_known!=1) ||
    (anchor->timestamp_known!=0 && anchor->timestamp_known!=1)) return BSA_ARGUMENT;
 /* Preserve aliased anchor when clock is replaced below. */
 bsa_capture_anchor saved=*anchor;
 rc=bsa_reset(h,id,cb,user); if(rc) return rc;
 rc=bsa_get(h,&status,&summary,&rejected); if(rc) return rc;
 clock->anchor=saved; clock->generation=status.generation;
 clock->reason=reason; clock->valid=1; return BSA_OK;
}
/* Real endpoint only: never map required_padded_sample for EOF-derived rows.
 * Clock/row generation matching prevents accidental reuse across resets.
 * Unknown coordinates stay explicitly unknown. Output is unchanged on failure.
 */
static inline int bsa_capture_map_real(const bsa_capture_clock *clock,
 uint64_t generation,uint64_t relative,bsa_capture_endpoint *out) {
 bsa_capture_endpoint result={0,0,0,0,0};
 if(!clock || !out || !clock->valid || generation!=clock->generation) return BSA_STATE;
 if((clock->anchor.sample_origin_known && relative>UINT64_MAX-clock->anchor.sample_origin) ||
    (clock->anchor.timestamp_known && relative>(UINT64_MAX-clock->anchor.anchor_ns)/UINT64_C(62500)))
  return BSA_OVERFLOW;
 result.clock_epoch=clock->anchor.clock_epoch;
 result.sample_known=clock->anchor.sample_origin_known;
 result.time_known=clock->anchor.timestamp_known;
 if(result.sample_known) result.absolute_sample=clock->anchor.sample_origin+relative;
 if(result.time_known) result.monotonic_ns=clock->anchor.anchor_ns+relative*UINT64_C(62500);
 *out=result; return BSA_OK;
}
#ifdef __cplusplus
}
#endif
#endif
