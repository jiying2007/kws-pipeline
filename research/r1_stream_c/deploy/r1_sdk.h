/* SPDX-License-Identifier: Apache-2.0
 * R1 reference64 research SDK. Proposals are NOT calibrated wake decisions.
 * One owner thread. No callback re-entry; pointers are valid during callback.
 * Init/destroy may allocate. Feed/gap/finish have no allocation or file I/O.
 * Exact model/table SHA256 are checked by the underlying R1 loader.
 */
#ifndef R1_REFERENCE64_SDK_H
#define R1_REFERENCE64_SDK_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define R1_SDK_MAX_BLOCK_BYTES 6400u
#define R1_SDK_VOCAB 2601u
#define R1_SDK_POLICY "r1-original-greedy-span150-cooldown50-v1"
enum { R1_SDK_OK=0, R1_SDK_ARGUMENT=-100, R1_SDK_STATE=-101,
       R1_SDK_OFFSET=-102, R1_SDK_CALLBACK=-103, R1_SDK_NUMERIC=-104 };
typedef struct {
    uint64_t epoch, keyword_id, model_frame, required_end_sample;
    uint64_t delivery_end_sample, proposal_index;
} r1_proposal;
typedef int (*r1_proposal_sink)(void *, const r1_proposal *);
typedef struct r1_sdk r1_sdk;
r1_sdk *r1_sdk_create(const unsigned char *weights,size_t weights_bytes,
                     const unsigned char *tables,size_t tables_bytes,
                     r1_proposal_sink sink,void *user);
/* absolute_byte must equal the next byte offset. Odd-size partitions are legal;
 * the final stream must contain complete PCM16 samples. No automatic gap fill. */
int r1_sdk_feed(r1_sdk *,uint64_t absolute_byte,const unsigned char *,size_t,size_t *consumed);
/* Discard all acoustic/prefix/cooldown state; next segment starts at sample.
 * Backward/overlapping epochs are rejected. Same position is an explicit reset. */
int r1_sdk_gap(r1_sdk *,uint64_t next_sample);
int r1_sdk_finish(r1_sdk *);
int r1_sdk_destroy(r1_sdk *);
uint64_t r1_sdk_frames(const r1_sdk *);
uint64_t r1_sdk_proposals(const r1_sdk *);
size_t r1_sdk_state_bytes(void);
size_t r1_sdk_weight_bytes(void);
/* Small independent greedy-CTC proposal primitive for contract tests.
 * Inputs are actual argmax IDs, NOT presumed labels. Model SDK never accepts
 * injected IDs. Reset this state at every discontinuity. */
typedef struct {
    uint64_t rows,events,frame[4],cooldown[2];
    unsigned last,used,ids[4];
} r1_proposal_decoder;
void r1_proposal_reset(r1_proposal_decoder *);
int r1_proposal_step(r1_proposal_decoder *,unsigned token,uint64_t frame,unsigned *keyword);
#ifdef __cplusplus
}
#endif
#endif
