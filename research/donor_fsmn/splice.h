#ifndef DONOR_FSMN_SPLICE_H
#define DONOR_FSMN_SPLICE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Isolated feature-level adapter for the pinned WeKws accept_wave contract.
 * This does not buffer PCM or calculate fbank. Accumulate application ingress
 * into 4800-sample reference calls (plus ONE actual short final call) outside
 * this API. For EACH such call:
 *   1. plan_call reports the concatenated pending-wave + new-call length.
 *   2. If fbank_rows != 0, compute exactly that many 400/160 snip-edge fbank80
 *      rows on that waveform. Retain PCM starting at consumed_samples.
 *      Otherwise retain all PCM; the source has an 800-sample gate.
 *   3. accept_call consumes those NEW fbank rows together, preserving that
 *      call's splice/skip grouping. Pass NULL,0 below the gate.
 *   4. At EOF call finish; it emits nothing. Reset BOTH this state and the
 *      external PCM/fbank state before another file.
 *
 * In particular, an unrestricted donor_fbank_feed callback is not itself an
 * accept_wave call: it emits from 400 samples, before this adapter's gate.
 * Generic chunk repartitioning is NOT claimed equivalent. Tiny-call support
 * permits source edge-case tests; after a three-row call feature_remained has
 * THREE rows, not a reconstructed four-row continuous-context history.
 * Normal decode-visible calls must remain 4800 samples plus the final tail.
 */
#define DONOR_SPLICE_FBANK_BINS 80
#define DONOR_SPLICE_FEATURE_DIM 400
#define DONOR_SPLICE_REFERENCE_SAMPLES 4800
#define DONOR_SPLICE_MAX_CALL_SAMPLES 16000

typedef struct {
    uint32_t initialized;
    uint32_t finished;
    uint32_t wave_samples;
    uint32_t feature_count;
    uint32_t offset;
    uint64_t total_samples;
    uint64_t total_fbank;
    uint64_t total_selected;
    float feature_remained[4][DONOR_SPLICE_FBANK_BINS];
} donor_splice_state;

typedef struct {
    size_t waveform_samples;
    size_t fbank_rows;
    size_t consumed_samples;
    size_t retained_samples;
    size_t splice_rows;
    size_t selected_rows;
    uint32_t next_offset;
} donor_splice_plan;

/* Borrowed row valid only during callback; no API reentry or state/input
 * mutation from callbacks. center_index identifies the actual fbank center,
 * NOT the upstream decoder timestamp. available_samples is the total PCM
 * received by this reference call. CMVN400 follows this callback's splice400.
 * Use separate caller-owned states for independent streams. */
typedef void (*donor_splice_callback)(void *user,
    const float row[DONOR_SPLICE_FEATURE_DIM], uint64_t center_index,
    uint64_t available_samples);

enum {
    DONOR_SPLICE_OK = 0,
    DONOR_SPLICE_ARGUMENT = 1,
    DONOR_SPLICE_STATE = 2
};

size_t donor_splice_state_bytes(void);
int donor_splice_init(donor_splice_state *state);
int donor_splice_reset(donor_splice_state *state);
/* Pure query: successful plan writes only *plan. plan must not overlap state.
 * The per-call safety cap is 16000 samples; production grouping remains4800.
 * Rejected calls never mutate state or emit callbacks. */
int donor_splice_plan_call(const donor_splice_state *state, size_t call_samples,
                          donor_splice_plan *plan);
/* new_fbank is contiguous [fbank_rows,80] and must not overlap state.
 * It must have exactly the row count returned by plan_call. Nonfinite input,
 * wrong count, overflow, invalid state and missing required callback fail
 * atomically. No allocation or PCM access is performed by this API. */
int donor_splice_accept_call(donor_splice_state *state, size_t call_samples,
    const float *new_fbank, size_t fbank_rows, donor_splice_callback callback,
    void *user);
/* Mark EOF only. No padding, right replication, delayed draining, or callback.
 * Repeated finish or accept/plan after finish returns DONOR_SPLICE_STATE. */
int donor_splice_finish(donor_splice_state *state);

#ifdef __cplusplus
}
#endif
#endif
