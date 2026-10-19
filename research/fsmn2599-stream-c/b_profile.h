/* SPDX-License-Identifier: Apache-2.0. Opt-in diagnostics; not model qualification. */
#ifndef B_PROFILE_H
#define B_PROFILE_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define B_PROFILE_ABI 1u
#define B_PROFILE_STAGES 5u
#define B_PROFILE_BUCKETS 16u
enum { B_PROFILE_FRONTEND=0, B_PROFILE_BODY=1, B_PROFILE_HEAD=2,
       B_PROFILE_SOFTMAX=3, B_PROFILE_DECODER=4 };
typedef struct {
 uint64_t calls, failed_calls, timed_calls, total_ns, min_ns, max_ns;
 uint64_t histogram[B_PROFILE_BUCKETS];
} b_profile_stage;
typedef struct {
 uint32_t abi, stages, buckets, saturated;
 uint64_t timer_errors;
 b_profile_stage stage[B_PROFILE_STAGES];
} b_profile_snapshot;
typedef struct {uint64_t ns; int valid;} b_profile_stamp;
/* All stats are per calling thread, not per handle. Reset/get must occur on the
 * same worker thread as inference. Snapshot requires exact sizeof handshake.
 * No allocation/I/O. Errors affect diagnostics only, never component rc.
 */
void b_profile_reset(void);
size_t b_profile_size(void);
int b_profile_get(b_profile_snapshot *out,size_t bytes);
/* Internal cross-TU instrumentation; excluded from exported DSO ABI. */
#if defined(__GNUC__)
#define B_PROFILE_INTERNAL __attribute__((visibility("hidden")))
#else
#define B_PROFILE_INTERNAL
#endif
B_PROFILE_INTERNAL b_profile_stamp b_profile_begin(void);
B_PROFILE_INTERNAL void b_profile_end(unsigned stage,b_profile_stamp start,int success);
#ifdef __cplusplus
}
#endif
#endif
