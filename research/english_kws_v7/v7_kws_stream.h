#ifndef ENGLISH_V7_KWS_STREAM_H
#define ENGLISH_V7_KWS_STREAM_H
#include <stddef.h>
#include <stdint.h>
#include "v7_ctc.h"

/* 43,876-parameter candidate-relative causal FSMN-CTC research runtime.
 * Complete 16 kHz PCM16 -> periodic Hann512/400 -> 40 HTK log1p Mel ->
 * train-only normalization -> 4 x 96 causal FSMN -> candidate-specific CTC.
 * Candidate-specific weights and frontend are supplied by the immutable KWSV7C01 pack.
 * Experimental, not SSC305-qualified, not calibrated FAR/FRR.
 * One caller owns the state/model memory. No allocation/I/O in this library.
 */
#define EN_V7_KWS_FFT 512u
#define EN_V7_KWS_WINDOW 400u
#define EN_V7_KWS_HOP 160u
#define EN_V7_KWS_MELS 40u
#define EN_V7_KWS_HIDDEN 96u
#define EN_V7_KWS_BLOCKS 4u
#define EN_V7_KWS_LEFT 4u
#define EN_V7_KWS_PARAMS 43876u
#define EN_V7_KWS_MEL_WEIGHTS (257u*40u)
#define EN_V7_KWS_FLOATS (EN_V7_KWS_PARAMS+EN_V7_KWS_MEL_WEIGHTS+80u)
#define EN_V7_KWS_PACK_BYTES (16u+4u*EN_V7_KWS_FLOATS)
#define EN_V7_KWS_MAX_FEED 16000u

enum en_v7_kws_status {
    EN_V7_KWS_OK=0, EN_V7_KWS_BAD_ARGUMENT=-100,
    EN_V7_KWS_BAD_MODEL=-101, EN_V7_KWS_BAD_STATE=-102,
    EN_V7_KWS_BAD_CLOCK=-103, EN_V7_KWS_BAD_NUMERIC=-104,
    EN_V7_KWS_BAD_DECODER=-105
};
typedef struct {
    uint32_t magic;
    uint32_t candidate_id;
    float params[EN_V7_KWS_PARAMS];
    /* Unchanged original [fft_frequency=257][mel=40] row-major. */
    float mel[EN_V7_KWS_MEL_WEIGHTS];
    float mean[EN_V7_KWS_MELS], scale[EN_V7_KWS_MELS];
} en_v7_kws_model;

typedef struct {
    uint32_t magic;
    const en_v7_kws_model *model;
    uint64_t next_sample, frame_index;
    size_t buffered;
    int16_t pending[EN_V7_KWS_FFT];
    /* Depthwise left memory: [block][oldest..newest][channel]. */
    float memory[EN_V7_KWS_BLOCKS][EN_V7_KWS_LEFT+1u][EN_V7_KWS_HIDDEN];
    float re[EN_V7_KWS_FFT], im[EN_V7_KWS_FFT];
    float tw_cos[EN_V7_KWS_FFT/2u], tw_sin[EN_V7_KWS_FFT/2u];
    float hann[EN_V7_KWS_WINDOW];
    en_v7_ctc_state ctc;
} en_v7_kws_stream;

/* Exactly one little-endian payload with 8-byte magic "KWSV7C01",
 * uint32 version 1, uint32 float count EN_V7_KWS_FLOATS, then model
 * state_dict weights (source order), Mel257x40, train mean40, train scale40.
 * SHA-256 provenance is verified by the packer / orchestration.
 * This loader also rejects malformed sizes, nonfinite weights and scale<=0.
 * Caller must allocate model storage (e.g. static, not on small stack).
 */
int en_v7_kws_model_load(en_v7_kws_model *m, const void *bytes, size_t length, uint32_t candidate_id);
size_t en_v7_kws_model_bytes(void);
size_t en_v7_kws_stream_bytes(void);

/* max_gap/cooldown are research decoder policy, not calibrated thresholds.
 * For frozen full-sequence parity, set max_gap larger than the clip duration
 * and cooldown=0. Only 0(blank),1(token-1),2(token-2),3(token-3) can emit a candidate event.
 */
int en_v7_kws_stream_init(en_v7_kws_stream *s,
                          const en_v7_kws_model *m,
                          uint64_t max_gap_samples,
                          uint64_t cooldown_samples);
int en_v7_kws_stream_reset(en_v7_kws_stream *s);
/* A genuine missing-sample gap must explicitly reset PCM/FSMN/CTC state.
 * The next feed begins at the supplied absolute monotonically increasing
 * PCM sample index. The decoder preserves aggregate event_count.
 */
int en_v7_kws_discontinuity(en_v7_kws_stream *s, uint64_t next_sample);

/* Called once per 512-input/160-hop frame; all pointers valid only for
 * the callback duration. available_after_sample is an evidence clock, NOT
 * word-end timing. Do not re-enter the same stream from the callback.
 */
typedef void (*en_v7_kws_frame_callback)(
    void *user, const float logits[4], uint64_t frame_index,
    uint64_t available_after_sample, const en_v7_ctc_event *event);

/* Supply contiguous PCM chunks in [1,16000], starting at sample 0 after init.
 * No trailing padding. Returns no false success if clock or state invalid.
 * Each complete frame delivers four raw CTC logits to callback. */
int en_v7_kws_feed(en_v7_kws_stream *s, const int16_t *pcm,
                   size_t count, uint64_t first_sample,
                   en_v7_kws_frame_callback callback, void *user);
#endif
