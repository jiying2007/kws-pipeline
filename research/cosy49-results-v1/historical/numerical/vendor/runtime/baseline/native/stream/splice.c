/* Splice/skip semantics adapted from WeKws stream_kws_ctc.py:
 * Copyright (c) 2023 Jing Du (thuduj12@163.com), Apache-2.0.
 * Source SHA256: 2a5d462f1c0830beee844427cbf0063e38f7b981dcd6e7990ec7a633acf46e53.
 * See repository LICENSE; this implementation has no model or decoder. */
#include "splice.h"

#include <math.h>
#include <string.h>

#define SPLICE_MAGIC UINT32_C(0x53504c31)

static int overlaps(const void *a, size_t na, const void *b, size_t nb)
{
    uintptr_t x = (uintptr_t)a;
    uintptr_t y = (uintptr_t)b;
    if (na == 0 || nb == 0) return 0;
    if (x > UINTPTR_MAX - na || y > UINTPTR_MAX - nb) return 1;
    return x < y + nb && y < x + na;
}

static int valid_state(const donor_splice_state *state)
{
    if (state == NULL || state->initialized != SPLICE_MAGIC ||
        state->finished != 0 || state->wave_samples >= 800 ||
        state->offset > 2 ||
        (state->feature_count != 0 && state->feature_count != 3 &&
         state->feature_count != 4)) return 0;
    if (state->total_fbank > (UINT64_MAX - state->wave_samples) / 160 ||
        state->total_samples != state->total_fbank * 160 + state->wave_samples)
        return 0;
    if (state->feature_count == 0)
        return state->total_fbank == 0 && state->offset == 0 &&
               state->total_selected == 0;
    return state->total_fbank >= state->feature_count &&
           state->total_selected <= state->total_fbank;
}

size_t donor_splice_state_bytes(void)
{
    return sizeof(donor_splice_state);
}

int donor_splice_init(donor_splice_state *state)
{
    if (state == NULL) return DONOR_SPLICE_ARGUMENT;
    memset(state, 0, sizeof(*state));
    state->initialized = SPLICE_MAGIC;
    return DONOR_SPLICE_OK;
}

int donor_splice_reset(donor_splice_state *state)
{
    return donor_splice_init(state);
}

int donor_splice_plan_call(const donor_splice_state *state, size_t call_samples,
                          donor_splice_plan *plan)
{
    donor_splice_plan next;
    size_t prefix;
    size_t remainder;
    if (state == NULL || plan == NULL) return DONOR_SPLICE_ARGUMENT;
    if (!valid_state(state)) return DONOR_SPLICE_STATE;
    if (call_samples > DONOR_SPLICE_MAX_CALL_SAMPLES ||
        (uint64_t)call_samples > UINT64_MAX - state->total_samples ||
        overlaps(state, sizeof(*state), plan, sizeof(*plan)))
        return DONOR_SPLICE_ARGUMENT;
    memset(&next, 0, sizeof(next));
    next.waveform_samples = state->wave_samples + call_samples;
    next.retained_samples = next.waveform_samples;
    next.next_offset = state->offset;
    if (next.waveform_samples >= 800) {
        next.fbank_rows = 1 + (next.waveform_samples - 400) / 160;
        next.consumed_samples = next.fbank_rows * 160;
        next.retained_samples -= next.consumed_samples;
        prefix = state->feature_count == 0 ? 2 : state->feature_count;
        next.splice_rows = prefix + next.fbank_rows - 4;
        if (next.splice_rows > state->offset)
            next.selected_rows = 1 + (next.splice_rows - 1 - state->offset) / 3;
        remainder = (next.splice_rows +
                     (state->offset == 0 ? 0 : 3 - state->offset)) % 3;
        next.next_offset = (uint32_t)(remainder == 0 ? 0 : 3 - remainder);
        if ((uint64_t)next.fbank_rows > UINT64_MAX - state->total_fbank ||
            (uint64_t)next.selected_rows > UINT64_MAX - state->total_selected)
            return DONOR_SPLICE_ARGUMENT;
    }
    *plan = next;
    return DONOR_SPLICE_OK;
}

int donor_splice_accept_call(donor_splice_state *state, size_t call_samples,
    const float *new_fbank, size_t fbank_rows, donor_splice_callback callback,
    void *user)
{
    donor_splice_plan plan;
    donor_splice_state old;
    float row[DONOR_SPLICE_FEATURE_DIM];
    size_t i;
    size_t j;
    size_t prefix;
    size_t keep;
    int result = donor_splice_plan_call(state, call_samples, &plan);
    if (result != DONOR_SPLICE_OK) return result;
    if (fbank_rows != plan.fbank_rows ||
        (fbank_rows != 0 && new_fbank == NULL) ||
        (plan.selected_rows != 0 && callback == NULL))
        return DONOR_SPLICE_ARGUMENT;
    if (fbank_rows != 0 &&
        overlaps(state, sizeof(*state), new_fbank,
                 fbank_rows * DONOR_SPLICE_FBANK_BINS * sizeof(float)))
        return DONOR_SPLICE_ARGUMENT;
    for (i = 0; i < fbank_rows * DONOR_SPLICE_FBANK_BINS; ++i)
        if (!isfinite(new_fbank[i])) return DONOR_SPLICE_ARGUMENT;

    old = *state;
    state->wave_samples = (uint32_t)plan.retained_samples;
    state->total_samples += (uint64_t)call_samples;
    if (fbank_rows == 0) return DONOR_SPLICE_OK;
    state->total_fbank += (uint64_t)fbank_rows;
    state->total_selected += (uint64_t)plan.selected_rows;
    state->offset = plan.next_offset;
    keep = fbank_rows < 4 ? fbank_rows : 4;
    state->feature_count = (uint32_t)keep;
    memset(state->feature_remained, 0, sizeof(state->feature_remained));
    memcpy(state->feature_remained,
           new_fbank + (fbank_rows - keep) * DONOR_SPLICE_FBANK_BINS,
           keep * DONOR_SPLICE_FBANK_BINS * sizeof(float));

    prefix = old.feature_count == 0 ? 2 : old.feature_count;
    for (i = old.offset; i < plan.splice_rows; i += 3) {
        uint64_t center;
        for (j = 0; j < 5; ++j) {
            size_t index = i + j;
            const float *source;
            if (index < prefix) {
                source = old.feature_count == 0 ? new_fbank :
                         old.feature_remained[index];
            } else {
                source = new_fbank + (index - prefix) * DONOR_SPLICE_FBANK_BINS;
            }
            memcpy(row + j * DONOR_SPLICE_FBANK_BINS, source,
                   DONOR_SPLICE_FBANK_BINS * sizeof(float));
        }
        center = old.feature_count == 0 ? (uint64_t)i :
                 old.total_fbank - old.feature_count + (uint64_t)i + 2;
        callback(user, row, center, state->total_samples);
    }
    return DONOR_SPLICE_OK;
}

int donor_splice_finish(donor_splice_state *state)
{
    if (state == NULL) return DONOR_SPLICE_ARGUMENT;
    if (!valid_state(state)) return DONOR_SPLICE_STATE;
    state->finished = 1;
    return DONOR_SPLICE_OK;
}
