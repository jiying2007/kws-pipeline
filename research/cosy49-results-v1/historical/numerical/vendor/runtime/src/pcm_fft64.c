/* PCM composition of the separately reviewed donor fbank and WeKws splice
 * adapters. See splice.c and LICENSE.wekws for original source attribution.
 * This file changes neither frontend arithmetic nor feature/phase semantics.
 */
#include "pcm_fft64.h"

#include <math.h>
#include <string.h>

#define PCM_MAGIC UINT32_C(0x50433634)

static int overlaps(const void *a, size_t na, const void *b, size_t nb)
{
    uintptr_t x = (uintptr_t)a;
    uintptr_t y = (uintptr_t)b;
    if (na == 0 || nb == 0) return 0;
    if (x > UINTPTR_MAX - na || y > UINTPTR_MAX - nb) return 1;
    return x < y + nb && y < x + na;
}

static int valid_state(const donor_fft64_pcm_state *state)
{
    donor_splice_plan plan;
    if (state == NULL || !donor_fft64_environment_ok() || state->initialized != PCM_MAGIC ||
        state->finished != 0 || state->faulted != 0 || state->busy != 0 ||
        state->ingress_samples >= DONOR_FFT64_PCM_REFERENCE_SAMPLES ||
        state->total_ingress < state->splice.total_samples ||
        state->total_ingress - state->splice.total_samples != state->ingress_samples ||
        state->splice.total_samples % DONOR_FFT64_PCM_REFERENCE_SAMPLES != 0 ||
        state->total_calls != state->splice.total_samples / DONOR_FFT64_PCM_REFERENCE_SAMPLES ||
        state->frontend.finished != 0 || state->frontend.used != 0 ||
        state->frontend.total_samples != 0 || state->frontend.frame_index != 0)
        return 0;
    return donor_splice_plan_call(&state->splice, 0, &plan) == DONOR_SPLICE_OK;
}

static int finite_history(donor_fft64_pcm_state *state)
{
    size_t i, j;
    for (i = 0; i < state->splice.feature_count; ++i)
        for (j = 0; j < DONOR_SPLICE_FBANK_BINS; ++j)
            if (!isfinite(state->splice.feature_remained[i][j])) {
                state->faulted = 1;
                return 0;
            }
    return 1;
}

size_t donor_fft64_pcm_state_bytes(void)
{
    return sizeof(donor_fft64_pcm_state);
}

int donor_fft64_pcm_init(donor_fft64_pcm_state *state)
{
    if (state == NULL) return DONOR_FFT64_PCM_ARGUMENT;
    if (!donor_fft64_environment_ok()) return DONOR_FFT64_PCM_STATE;
    memset(state, 0, sizeof(*state));
    state->initialized = PCM_MAGIC;
    (void)donor_splice_init(&state->splice);
    (void)donor_fft64_init(&state->frontend);
    return DONOR_FFT64_PCM_OK;
}

int donor_fft64_pcm_reset(donor_fft64_pcm_state *state)
{
    return donor_fft64_pcm_init(state);
}

typedef struct {
    donor_fft64_pcm_state *state;
    size_t count;
} row_collector;

static void collect_row(void *user, const float row[DONOR_SPLICE_FEATURE_DIM],
                        uint64_t center, uint64_t available_samples)
{
    row_collector *collector = (row_collector *)user;
    (void)available_samples;
    /* process_call has already bounded the exact planned number of rows. */
    memcpy(collector->state->rows[collector->count], row,
           DONOR_SPLICE_FEATURE_DIM * sizeof(float));
    collector->state->centers[collector->count++] = center;
}

static int process_call(donor_fft64_pcm_state *state, size_t count, uint32_t final_short,
                        donor_fft64_pcm_callback callback, void *user)
{
    donor_splice_plan plan;
    donor_fft64_pcm_batch batch;
    row_collector collector = {state, 0};
    size_t i, j;
    int result = donor_splice_plan_call(&state->splice, count, &plan);
    if (result != DONOR_SPLICE_OK ||
        plan.waveform_samples > DONOR_FFT64_PCM_MAX_WAVE_SAMPLES ||
        plan.fbank_rows > DONOR_FFT64_PCM_MAX_FBANK_ROWS ||
        plan.selected_rows > DONOR_FFT64_PCM_MAX_SELECTED_ROWS) {
        state->faulted = 1;
        return DONOR_FFT64_PCM_STATE;
    }
    memcpy(state->waveform + state->splice.wave_samples, state->ingress,
           count * sizeof(int16_t));
    for (i = 0; i < plan.fbank_rows; ++i) {
        result = donor_fft64_analyze_frame(&state->frontend,
                     state->waveform + i * DONOR_FFT64_HOP, &state->trace);
        if (result != DONOR_FFT64_OK) {
            state->faulted = 1;
            return DONOR_FFT64_PCM_STATE;
        }
        for (j = 0; j < DONOR_SPLICE_FBANK_BINS; ++j)
            if (!isfinite(state->trace.logfbank[j])) {
                state->faulted = 1;
                return DONOR_FFT64_PCM_NUMERIC;
            }
        memcpy(state->fbank[i], state->trace.logfbank,
               DONOR_SPLICE_FBANK_BINS * sizeof(float));
    }
    result = donor_splice_accept_call(&state->splice, count,
        plan.fbank_rows == 0 ? NULL : &state->fbank[0][0], plan.fbank_rows,
        collect_row, &collector);
    if (result != DONOR_SPLICE_OK || collector.count != plan.selected_rows) {
        state->faulted = 1;
        return DONOR_FFT64_PCM_STATE;
    }
    memmove(state->waveform, state->waveform + plan.consumed_samples,
            plan.retained_samples * sizeof(int16_t));
    memset(state->waveform + plan.retained_samples, 0,
           (DONOR_FFT64_PCM_MAX_WAVE_SAMPLES - plan.retained_samples) * sizeof(int16_t));
    memset(&batch, 0, sizeof(batch));
    batch.call_index = state->total_calls++;
    batch.available_samples = state->splice.total_samples;
    batch.call_samples = count;
    batch.waveform_samples = plan.waveform_samples;
    batch.fbank_rows = plan.fbank_rows;
    batch.splice_rows = plan.splice_rows;
    batch.selected_rows = collector.count;
    batch.is_final_short = final_short;
    batch.fbank = &state->fbank[0][0];
    batch.rows = &state->rows[0][0];
    batch.centers = state->centers;
    batch.wave_samples = state->splice.wave_samples;
    batch.feature_count = state->splice.feature_count;
    batch.offset = state->splice.offset;
    state->ingress_samples = 0;
    memset(state->ingress, 0, sizeof(state->ingress));
    callback(user, &batch);
    return DONOR_FFT64_PCM_OK;
}

int donor_fft64_pcm_feed(donor_fft64_pcm_state *state, const int16_t *pcm, size_t count,
                   donor_fft64_pcm_callback callback, void *user)
{
    size_t consumed = 0;
    int result = DONOR_FFT64_PCM_OK;
    if (state == NULL || callback == NULL || (pcm == NULL && count != 0) ||
        count > SIZE_MAX / sizeof(int16_t)) return DONOR_FFT64_PCM_ARGUMENT;
    if (!valid_state(state)) return DONOR_FFT64_PCM_STATE;
    if ((uint64_t)count > UINT64_MAX - state->total_ingress ||
        overlaps(state, sizeof(*state), pcm, count * sizeof(int16_t)))
        return DONOR_FFT64_PCM_ARGUMENT;
    if (!finite_history(state)) return DONOR_FFT64_PCM_NUMERIC;
    state->busy = 1;
    while (consumed < count) {
        size_t take = DONOR_FFT64_PCM_REFERENCE_SAMPLES - state->ingress_samples;
        if (take > count - consumed) take = count - consumed;
        memcpy(state->ingress + state->ingress_samples, pcm + consumed,
               take * sizeof(int16_t));
        state->ingress_samples += take;
        state->total_ingress += (uint64_t)take;
        consumed += take;
        if (state->ingress_samples == DONOR_FFT64_PCM_REFERENCE_SAMPLES) {
            result = process_call(state, DONOR_FFT64_PCM_REFERENCE_SAMPLES, 0,
                                  callback, user);
            if (result != DONOR_FFT64_PCM_OK) break;
        }
    }
    state->busy = 0;
    return result;
}

int donor_fft64_pcm_finish(donor_fft64_pcm_state *state, donor_fft64_pcm_callback callback,
                     void *user)
{
    int result = DONOR_FFT64_PCM_OK;
    if (state == NULL || callback == NULL) return DONOR_FFT64_PCM_ARGUMENT;
    if (!valid_state(state)) return DONOR_FFT64_PCM_STATE;
    if (!finite_history(state)) return DONOR_FFT64_PCM_NUMERIC;
    state->busy = 1;
    if (state->ingress_samples != 0)
        result = process_call(state, state->ingress_samples, 1, callback, user);
    if (result == DONOR_FFT64_PCM_OK) {
        if (donor_splice_finish(&state->splice) != DONOR_SPLICE_OK) {
            state->faulted = 1;
            result = DONOR_FFT64_PCM_STATE;
        } else {
            state->finished = 1;
        }
    }
    state->busy = 0;
    return result;
}
