/* Read-only ABI queries. No state allocation, model loading, or arithmetic. */
#include "a20_stream_fft64.h"
#include <stddef.h>
size_t a20fft64s_state_alignment(void) { return _Alignof(a20fft64s_state); }
size_t donor_fft64_pcm_state_alignment(void) { return _Alignof(donor_fft64_pcm_state); }
size_t donor_fft64_state_alignment(void) { return _Alignof(donor_fft64_state); }
const float *a20fft64s_cache(const a20fft64s_state *s) { return s ? s->model.cache : NULL; }
const donor_fft64_pcm_state *a20fft64s_pcm(const a20fft64s_state *s) { return s ? &s->pcm : NULL; }
