/* SPDX-License-Identifier: Apache-2.0. Research adapter, no production claim. */
#ifndef B_STREAM_ADAPTER_H
#define B_STREAM_ADAPTER_H
#include <stddef.h>
#include <stdint.h>
#include "kws3.h"
#ifdef __cplusplus
extern "C" {
#endif
#define BSA_ABI 1u
#define BSA_MODEL_ID "fsmn-B-a366d9f5"
typedef struct bsa_handle bsa_handle;
typedef struct {
 uint32_t abi, inputs, outputs, sample_rate, channels, sample_bits, byte_order;
 uint64_t payload_bytes;
 const char *id, *checkpoint_sha256, *payload_sha256, *contract_sha256;
} bsa_descriptor;
enum { BSA_OK=0, BSA_ARGUMENT=1, BSA_STATE=2, BSA_ALLOCATION=3,
 BSA_ENVIRONMENT=4, BSA_RESTORE=5, BSA_OVERFLOW=6, BSA_CALLBACK=7,
 BSA_INVARIANT=8, BSA_COMPONENT=9 };
enum { BSA_NEW=0, BSA_OPEN=1, BSA_CLOSED=2, BSA_FAILED=3 };
typedef struct {
 uint64_t generation, row, required_padded_sample, available_real_sample;
 uint64_t processed_real_sample;
 int valid_so_far, eof_derived;
} bsa_row;
typedef int (*bsa_callback)(void *,const bsa_row *,const float *,size_t,
                            const kws3_event *,size_t);
typedef struct {
 uint64_t generation, supplied_bytes, processed_bytes, real_samples, padded_samples;
 uint64_t fbank_reserved, fbank_completed, network_reserved, network_completed;
 uint64_t softmax_reserved, softmax_completed, decoder_reserved, decoder_completed;
 uint64_t callbacks_completed, events_delivered, finish_calls, zero_samples;
 uint64_t aborted_generations;
 double threshold;
 int phase, error, component, component_error, valid_so_far;
} bsa_status;
/* Only the returned registered descriptor pointer is accepted. Metadata/hash
 * arguments from an arbitrary caller are not a trust root. Immutable for life. */
const bsa_descriptor *bsa_model(void);
size_t bsa_abi_size(unsigned); /* 1=row,2=status,3=event,4=summary,5=descriptor */
size_t bsa_storage_bytes(void); /* excludes caller payload, constants and stack */
int bsa_create(const bsa_descriptor *,const uint8_t *,size_t,double,bsa_handle **);
int bsa_begin(bsa_handle *,const char *,bsa_callback,void *);
/* NULL data allowed iff bytes==0. consumed required. Caller buffers must not
 * alias handle/state; callback must not reenter or mutate data or FP mode.
 * Entire supplied block is available when called. A dangling low byte is
 * retained; only complete PCM16LE samples count toward availability. */
int bsa_feed(bsa_handle *,const uint8_t *,size_t,size_t *consumed);
/* Exactly one explicit canonical4800 tail; no transport-boundary flush. */
int bsa_finish(bsa_handle *);
/* May explicitly abandon OPEN/FAILED stream. Clears all per-stream state,
 * preserves registered model and threshold (both signed zeros canonicalize to
 * +0, all nonzero values exact), increments generation. */
int bsa_reset(bsa_handle *,const char *,bsa_callback,void *);
int bsa_get(bsa_handle *,bsa_status *,kws3_summary *,uint64_t *threshold_rejections);
int bsa_destroy(bsa_handle *);
/* Events are delivered synchronously during feed; no EOF is needed for a live
 * wakeup. valid_so_far means no failure yet, not numerical qualification.
 * Later fatal failure invalidates the generation, cannot undo delivered events,
 * and never causes automatic replay. EOF-derived events are marked separately.
 * Offline scoring may require a completed valid recording. All callback views
 * expire on return. Single-threaded, non-reentrant; no feed allocations.
 * Each numerical API saves/masks/restores fenv. RNE/gradual-underflow required;
 * no rounding or FTZ/DAZ changes. Restore failure reports possibly changed FP
 * environment. Existing frontend/network strict FAIL remains unchanged. */
#ifdef __cplusplus
}
#endif
#endif
