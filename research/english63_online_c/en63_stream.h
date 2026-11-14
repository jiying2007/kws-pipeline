#ifndef EN63_STREAM_H
#define EN63_STREAM_H

#include <stddef.h>
#include <stdint.h>

/* Standalone English research ABI. It does not change the KWSP product ABI. */
#define EN63_FRAME 400u
#define EN63_HOP 160u
#define EN63_BANDS 32u
#define EN63_HIDDEN 32u
#define EN63_WINDOW 200u
#define EN63_MAX_CHUNK 3199u
#define EN63_GRU 1u
#define EN63_DSC_TCN 2u
#define EN63_GRU_PARAMETERS 6531u
#define EN63_DSC_PARAMETERS 4803u

enum { EN63_OK = 0, EN63_ARGUMENT = -1, EN63_STATE = -2, EN63_NUMERIC = -3 };

typedef struct {
    uint32_t kind;
    size_t parameter_count;
    const float *parameters;
    const float *window;             /* 400 exact exported FP32 Hann values */
    const float *mean, *std;         /* original train-only FP32 normalizer */
    const uint16_t *mel_start, *mel_count, *mel_offset; /* 32, 32, 33 */
    const float *mel_weights;        /* exact nonzero exported FP32 values */
    size_t mel_weights_count;
    const char *checkpoint_sha256;
} en63_model;

typedef struct {
    const en63_model *model;
    uint32_t magic, finished, faulted, busy;
    uint32_t pcm_fill, pool_count, pool_write, candidate, confirmations;
    uint64_t samples, frames, blocked_until, events;
    int16_t pcm[EN63_FRAME];
    float hidden[EN63_HIDDEN];
    float history[3][8][EN63_HIDDEN];
    float pool[EN63_WINDOW][EN63_HIDDEN];
    double re[512], im[512], stage_re[9], stage_im[9];
    float features[EN63_BANDS], logits[3], probabilities[3];
} en63_state;

/* Callback data is valid only during the call. No callbacks occur at EOF.
 * callback must not mutate/reenter state. The core does no I/O or allocation. */
typedef void (*en63_callback)(void *user, uint64_t frame, uint64_t available_sample,
                             const float features[32], const float logits[3],
                             const float probabilities[3], uint32_t proposal_class);

int en63_init(en63_state *state, const en63_model *model);
int en63_reset(en63_state *state);
int en63_feed(en63_state *state, const int16_t *pcm, size_t count,
              en63_callback callback, void *user);
int en63_finish(en63_state *state);
size_t en63_state_bytes(void);

#endif
