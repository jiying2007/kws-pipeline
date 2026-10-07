#ifndef A20_TRAINING_FEATURE_COLLECTOR_H
#define A20_TRAINING_FEATURE_COLLECTOR_H
#include "pcm_fft64.h"
#define A20_RECIPE_MAX_SAMPLES 46080u
#define A20_RECIPE_MAX_ROWS 95u
#define A20_RECIPE_MAX_CALLS 10u

typedef struct {
    uint64_t call_index, available_samples;
    size_t call_samples, waveform_samples, fbank_rows, splice_rows, selected_rows;
    uint32_t is_final_short, wave_samples, feature_count, offset;
    uint64_t centers[DONOR_FFT64_PCM_MAX_SELECTED_ROWS];
} a20_recipe_call;

typedef struct {
    donor_fft64_pcm_state frontend;
    float rows[A20_RECIPE_MAX_ROWS][400]; /* PRE-CMVN, valid only on success */
    a20_recipe_call calls[A20_RECIPE_MAX_CALLS];
    size_t row_count, call_count;
    int failed;
} a20_recipe_features;

/* Preparation-only library function; no main, model, decoder, CMVN or file I/O.
 * Caller provides nonoverlapping workspace and native int16 PCM already checked
 * against the frozen WAV/PCM manifest. Every call resets the entire workspace.
 * Exactly 4800 samples per full feed, actual short tail once, finish once.
 * Partial outputs are INVALID on any nonzero return. No extraction ran in prep.
 */
int a20_recipe_collect(a20_recipe_features *out, const int16_t *pcm, size_t count);
#endif
