#ifndef A20_STREAM_FFT64_H
#define A20_STREAM_FFT64_H
#include "../baseline/precision64/model/a20_fsmn.h"
#include "pcm_fft64.h"
#include "../baseline/decoder/a20_decoder.h"
/* Research-only explicit caller arena. No product RNN ABI compatibility.
 * Caller owns immutable model bytes and all state; single-thread/non-reentrant. */
typedef void (*a20fft64s_trace_callback)(void*,uint64_t,size_t,const float*,const float*,const float*);
typedef struct {
 donor_fft64_pcm_state pcm;
 a20_model model;
 a20d_decoder decoder;
 a20d_workspace decoder_work;
 float logits[DONOR_FFT64_PCM_MAX_SELECTED_ROWS][A20D_CLASSES];
 uint32_t initialized, fault, busy;
 /* Optional borrowed diagnostic trace, absent in normal operation. */
 float *trace_buffer;
 a20fft64s_trace_callback trace_callback;
 void *trace_user;
} a20fft64s_state;
typedef struct {
 const donor_fft64_pcm_batch *frontend;
 const float *logits;        /* borrowed [selected_rows][6] */
 const float *cache;         /* borrowed [128][11][4] after whole batch */
 const a20d_result *result;  /* borrowed chunk return; at most one event */
} a20fft64s_batch;
typedef void (*a20fft64s_callback)(void *, const a20fft64s_batch *);
size_t a20fft64s_state_bytes(void);
int a20fft64s_init(a20fft64s_state*,const float *weights,size_t floats);
int a20fft64s_reset(a20fft64s_state*); /* resets all timelines, frontend, cache and beams */
int a20fft64s_set_trace(a20fft64s_state*,float *trace21x250,a20fft64s_trace_callback,void*);
int a20fft64s_feed(a20fft64s_state*,const int16_t*,size_t,a20fft64s_callback,void*);
int a20fft64s_finish(a20fft64s_state*,a20fft64s_callback,void*); /* actual retained tail only */
#endif
