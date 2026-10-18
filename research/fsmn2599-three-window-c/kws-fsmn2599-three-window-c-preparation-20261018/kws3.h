/* SPDX-License-Identifier: Apache-2.0
 * Derived beam/detection semantics: Copyright (c) 2023 Jing Du
 * (thuduj12@163.com). See NOTICE for the fixed source identities.
 * Research candidate decoder only. Not an acoustic-model/FAR admission.
 */
#ifndef KWS3_H
#define KWS3_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define KWS3_CLASSES 2599
#define KWS3_WINDOW_ROWS 126
#define KWS3_HOP_ROWS 42
#define KWS3_WINDOWS 3
#define KWS3_PATHS 20
#define KWS3_NEXT_PATHS 80
#define KWS3_DEDUP_KEYS 378
#define KWS3_ID_BYTES 256
#define KWS3_ROW_EVENTS 3
/* Clock/counter contract, independent of sizeof(size_t). */
#define KWS3_MAX_ROWS ((INT64_MAX - INT64_C(378)) / INT64_C(3))

typedef enum {
    KWS3_OK = 0, KWS3_ARGUMENT = 1, KWS3_POSTERIOR = 2,
    KWS3_CAPACITY = 3, KWS3_CLOCK = 4, KWS3_FINISHED = 5,
    KWS3_INVALID = 6, KWS3_ENVIRONMENT = 7, KWS3_ALLOCATION = 8,
    KWS3_INVARIANT = 9
} kws3_status;
typedef struct kws3_decoder kws3_decoder;
typedef struct {
    uint64_t event_index;
    int64_t node_frames[4], start_frame, end_frame;
    int64_t window_start, decision_row, decision_frame;
    double score, grid_seconds;
    uint32_t keyword_id, diagnostic_candidate;
    char recording_id[KWS3_ID_BYTES];
} kws3_event;
typedef struct {
    uint64_t rows_decoded, beam_updates, topk_computations;
    uint64_t windows_started, windows_retired, eof_windows_closed;
    uint64_t candidates_seen, duplicate_candidates, refractory_rejections;
    uint64_t events_emitted, zero_mass_window_rows, empty_beam_window_rows;
    uint64_t max_live_windows, max_prefix_seen, max_next_hypotheses;
    uint64_t max_dedup_keys, max_live_node_references;
    uint64_t live_windows, dedup_keys;
    int finished, valid, numerical_support_ok;
    kws3_status invalid_reason;
    char recording_id[KWS3_ID_BYTES];
} kws3_summary;
/* Sink is synchronous and cannot cancel a chunk. It must not re-enter or
 * mutate the decoder. Copy the event if it must outlive the callback.
 * All candidate events remain provisional until the recording finishes valid. */
typedef void (*kws3_event_sink)(void *user, const kws3_event *event);

/* Floating-point contract: IEEE binary64/binary32, round-to-nearest-even,
 * gradual underflow, no FTZ/DAZ, no fast-math or expression contraction.
 * Build with -fno-fast-math -ffp-contract=off -fexcess-precision=standard.
 * Caller must keep FP exception traps masked; portable probes cannot verify
 * trap masks. Environment probes may set status flags, never change modes.
 * Do not change the FP environment during a decoder call. */

/* Only create/destroy allocate/free. One handle per stream, no global mutable
 * state. Nonempty NUL-terminated identity <=255 bytes; longer IDs are errors. */
kws3_status kws3_create(const char *recording_id, kws3_decoder **out);
/* Explicit immutable finite threshold in [0,1]. Old create is exactly theta0.
 * reset preserves threshold and clears rejection count. Reject before refractory
 * but after seen registration; score==threshold is accepted.
 * Existing event/summary layouts and original 11 symbols are unchanged. */
kws3_status kws3_create_with_threshold(const char *recording_id, double threshold,
                                       kws3_decoder **out);
kws3_status kws3_get_threshold_status(const kws3_decoder *decoder, double *threshold,
                                      uint64_t *rejections);
void kws3_destroy(kws3_decoder *decoder);
size_t kws3_storage_bytes(void);
kws3_status kws3_reset(kws3_decoder *decoder, const char *recording_id);
/* cap must be >=3; never silently drop an event. On failure *count=0 and
 * committed row state is unchanged except the failure latch. */
kws3_status kws3_step(kws3_decoder *decoder, const double *posterior,
                     size_t classes, kws3_event *events, size_t cap, size_t *count);
/* Exact binary32-to-binary64 promotion, no re-softmax or renormalization. */
kws3_status kws3_step_f32(kws3_decoder *decoder, const float *posterior,
                         size_t classes, kws3_event *events, size_t cap, size_t *count);
/* Contiguous/strided rows, stride in double elements. A valid chunk is fully
 * consumed, even after events. On failure consumed is the committed prefix;
 * the failed row and all later rows are not consumed, and recording is invalid.
 * sink=NULL explicitly counts/discards events; callers needing them supply sink.
 * num_rows=0 is a no-op. Input must not overlap decoder-owned storage. */
kws3_status kws3_process(kws3_decoder *decoder, const double *rows,
                        size_t num_rows, size_t stride, kws3_event_sink sink,
                        void *user, size_t *consumed);
kws3_status kws3_finish(kws3_decoder *decoder);
kws3_status kws3_get_summary(const kws3_decoder *decoder, kws3_summary *out);
const char *kws3_status_string(kws3_status status);
const char *kws3_keyword(uint32_t keyword_id);
#ifdef __cplusplus
}
#endif
#endif
