/* SPDX-License-Identifier: Apache-2.0
 * Independent ACTIVE113 research ABI. Six scores plus OTHER original ID.
 * Fixed original backbone/frontend; no full2601 probability/CTC claim.
 * Synchronous and single-threaded; callbacks must not re-enter this handle.
 * Input signed16 LE,16000Hz mono PCM after caller-owned AFE. */
#ifndef ACTIVE113_SDK_H
#define ACTIVE113_SDK_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define ACTIVE113_SDK_ABI UINT32_C(0xA1130001)
#define ACTIVE113_MAX_INPUT_BYTES 8192u
#define ACTIVE113_MODEL_BYTES 1582624u
#define ACTIVE113_TABLE_BYTES 83840u
#define ACTIVE113_SCORE_COUNT 6u
#define ACTIVE113_FEATURE_COUNT 400u
#define ACTIVE113_MODEL_VIEW_ALIGNMENT 8u
typedef struct active113_sdk active113_sdk;
typedef enum {
    ACTIVE113_OK=0,ACTIVE113_BAD_ARGUMENT=-1,ACTIVE113_BAD_STATE=-2,
    ACTIVE113_BACKEND_ERROR=-3,ACTIVE113_CALLBACK_ERROR=-4,ACTIVE113_RANGE_ERROR=-5
} active113_status;
typedef struct {
    uint32_t abi,keyword_id;
    uint64_t segment,model_frame,first_token_frame;
    uint64_t dependency_end_sample,delivery_end_sample;
} active113_event;
typedef int (*active113_event_fn)(void *,const active113_event *);
/* scores[0..4] map to original IDs0,1027,1401,1462,2093. scores[5] is
 * selected-OTHER maximum and maps to the supplied other_original_id.
 * Six values MUST NOT be consumed as a2601-logit array. Feature count400.
 * Borrowed arrays are valid only during callback. Times remain the original
 * dependency/delivery sample clocks, not acoustic keyword annotations. */
typedef int (*active113_frame_fn)(void *,uint64_t frame,uint64_t dependency_end,
    uint64_t delivery_end,const double *features,const double *scores,uint32_t other_original_id);
/* Original-ID tie ordering, including between precise and OTHER scores.
 * On invalid count/ID/nonfinite input, winner remains unchanged. */
int active113_sdk_pick_original_token(const double *scores,size_t count,
    uint32_t other_original_id,uint32_t *winner);
active113_sdk *active113_sdk_create(const unsigned char *model,size_t model_bytes,
    const unsigned char *tables,size_t table_bytes,uint64_t origin,
    active113_event_fn,active113_frame_fn,void *user);
/* Immutable8-byte-aligned malloc/read or read-only mmap storage must outlive
 * the handle; never mutate it or originate it from a declared byte array.
 * Model ownership remains with caller, including on failure. Tables are
 * copied. Normal create instead makes one private model copy. */
active113_sdk *active113_sdk_create_view(const unsigned char *model,size_t model_bytes,
    const unsigned char *tables,size_t table_bytes,uint64_t origin,
    active113_event_fn,active113_frame_fn,void *user);
int active113_sdk_feed(active113_sdk *,const unsigned char *pcm,size_t bytes,size_t *consumed);
int active113_sdk_finish(active113_sdk *);
int active113_sdk_reset(active113_sdk *,uint64_t origin);
int active113_sdk_notify_gap(active113_sdk *,uint64_t next_origin);
int active113_sdk_destroy(active113_sdk *);
uint64_t active113_sdk_frame_count(const active113_sdk *);
uint64_t active113_sdk_event_count(const active113_sdk *);
uint64_t active113_sdk_consumed_bytes(const active113_sdk *);
size_t active113_sdk_state_bytes(void);
size_t active113_sdk_weight_bytes(void);
const char *active113_sdk_model_sha256(void);
const char *active113_sdk_table_sha256(void);
const char *active113_sdk_selection_sha256(void);
const char *active113_sdk_policy_id(void);
#ifdef __cplusplus
}
#endif
#endif
