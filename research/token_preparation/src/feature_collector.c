#include "feature_collector.h"
#include <math.h>
#include <string.h>

static void collect(void *user, const donor_fft64_pcm_batch *batch)
{
    a20_recipe_features *out = user;
    a20_recipe_call *call;
    size_t i;
    if (out->failed) return;
    if (batch->call_index != out->call_count || out->call_count >= A20_RECIPE_MAX_CALLS ||
        batch->selected_rows > DONOR_FFT64_PCM_MAX_SELECTED_ROWS ||
        batch->selected_rows > A20_RECIPE_MAX_ROWS - out->row_count) {
        out->failed = 1;
        return;
    }
    for (i = 0; i < batch->selected_rows * 400; ++i) {
        if (!isfinite(batch->rows[i])) { out->failed = 1; return; }
    }
    call = &out->calls[out->call_count++];
    call->call_index = batch->call_index;
    call->available_samples = batch->available_samples;
    call->call_samples = batch->call_samples;
    call->waveform_samples = batch->waveform_samples;
    call->fbank_rows = batch->fbank_rows;
    call->splice_rows = batch->splice_rows;
    call->selected_rows = batch->selected_rows;
    call->is_final_short = batch->is_final_short;
    call->wave_samples = batch->wave_samples;
    call->feature_count = batch->feature_count;
    call->offset = batch->offset;
    if (batch->selected_rows) {
        memcpy(call->centers, batch->centers, batch->selected_rows * sizeof(uint64_t));
        memcpy(out->rows[out->row_count], batch->rows, batch->selected_rows * 400 * sizeof(float));
    }
    out->row_count += batch->selected_rows;
}

int a20_recipe_collect(a20_recipe_features *out, const int16_t *pcm, size_t count)
{
    size_t start;
    uintptr_t a = (uintptr_t)out, b = (uintptr_t)pcm;
    int rc;
    if (!out || !pcm || count == 0 || count > A20_RECIPE_MAX_SAMPLES) return 1;
    if (a > UINTPTR_MAX - sizeof(*out) || b > UINTPTR_MAX - count * sizeof(*pcm) ||
        (a < b + count * sizeof(*pcm) && b < a + sizeof(*out))) return 1;
    memset(out, 0, sizeof(*out));
    rc = donor_fft64_pcm_reset(&out->frontend);
    if (rc) return rc;
    for (start = 0; start < count; start += 4800) {
        size_t n = count - start;
        if (n > 4800) n = 4800;
        rc = donor_fft64_pcm_feed(&out->frontend, pcm + start, n, collect, out);
        if (rc || out->failed) return rc ? rc : 4;
    }
    rc = donor_fft64_pcm_finish(&out->frontend, collect, out);
    if (rc || out->failed) return rc ? rc : 4;
    return 0;
}
