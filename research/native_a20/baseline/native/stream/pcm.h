#ifndef DONOR_FSMN_PCM_H
#define DONOR_FSMN_PCM_H

#include "splice.h"
#include "../donor_fbank/donor_fbank.h"

#ifdef __cplusplus
extern "C" {
#endif

/* PCM composition only: no model, CMVN, decoder, allocation or file I/O.
 * Input is native int16_t samples already decoded from mono16k PCM16LE.
 * Application ingress may have arbitrary lengths. The adapter presents ONLY
 * 4800-sample calls and ONE actual short final call to the pinned accept_wave
 * semantics. Its separate waveform buffer preserves the 800-sample gate;
 * donor_fbank_feed's 400-sample gate is deliberately not used.
 */
#define DONOR_PCM_REFERENCE_SAMPLES DONOR_SPLICE_REFERENCE_SAMPLES
#define DONOR_PCM_MAX_WAVE_SAMPLES (DONOR_PCM_REFERENCE_SAMPLES + 800)
#define DONOR_PCM_MAX_FBANK_ROWS 33
#define DONOR_PCM_MAX_SELECTED_ROWS 11

typedef struct {
    uint32_t initialized, finished, faulted, busy;
    size_t ingress_samples;
    uint64_t total_ingress, total_calls;
    int16_t ingress[DONOR_PCM_REFERENCE_SAMPLES];
    int16_t waveform[DONOR_PCM_MAX_WAVE_SAMPLES];
    donor_splice_state splice;
    donor_fbank_state frontend;
    donor_fbank_trace trace;
    float fbank[DONOR_PCM_MAX_FBANK_ROWS][DONOR_SPLICE_FBANK_BINS];
    float rows[DONOR_PCM_MAX_SELECTED_ROWS][DONOR_SPLICE_FEATURE_DIM];
    uint64_t centers[DONOR_PCM_MAX_SELECTED_ROWS];
} donor_pcm_state;

typedef struct {
    uint64_t call_index;             /* zero based, canonical calls only */
    uint64_t available_samples;      /* end of this call, not ingress size */
    size_t call_samples;
    size_t waveform_samples;
    size_t fbank_rows;
    size_t splice_rows;
    size_t selected_rows;
    uint32_t is_final_short;         /* actual nonempty short call at finish */
    const float *fbank;              /* contiguous [fbank_rows,80] */
    const float *rows;               /* contiguous [selected_rows,400] */
    const uint64_t *centers;         /* actual global fbank center indices */
    uint32_t wave_samples;           /* retained waveform after this call */
    uint32_t feature_count;          /* retained NEW fbank rows:0,3,or4 */
    uint32_t offset;                 /* next splice/skip offset */
} donor_pcm_batch;

/* One callback per nonempty canonical call, INCLUDING calls with zero rows.
 * All pointers are borrowed and valid only during callback. The callback must
 * not mutate input/state or reenter this API. Independent states may be used
 * on separate threads. Full calls emit as soon as available; is_final_short
 * therefore does NOT mark EOF for a file whose length is a multiple of4800.
 * finish emits no empty call, padding, right replication or delayed output.
 */
typedef void (*donor_pcm_callback)(void *user, const donor_pcm_batch *batch);

enum {
    DONOR_PCM_OK = 0,
    DONOR_PCM_ARGUMENT = 1,
    DONOR_PCM_STATE = 2,
    DONOR_PCM_NUMERIC = 3
};

size_t donor_pcm_state_bytes(void);
int donor_pcm_init(donor_pcm_state *state);
int donor_pcm_reset(donor_pcm_state *state);
/* callback is required even when this ingress does not yet complete a call.
 * pcm must not overlap state. Zero count permits NULL pcm. Invalid arguments
 * are rejected before mutation/output. A numeric fault emits no batch for the
 * affected canonical call and latches faulted; earlier completed batches in
 * this feed remain valid. Reset is mandatory after any latched fault.
 */
int donor_pcm_feed(donor_pcm_state *state, const int16_t *pcm, size_t count,
                   donor_pcm_callback callback, void *user);
/* Processes pending application ingress once, then marks EOF without changing
 * retained waveform/features. Repeated finish or feed after finish fails.
 */
int donor_pcm_finish(donor_pcm_state *state, donor_pcm_callback callback,
                     void *user);

#ifdef __cplusplus
}
#endif
#endif
