#ifndef A20_STREAM_H
#define A20_STREAM_H
#include "../model/a20_fsmn.h"
#include "../stream/pcm.h"
#include "../../decoder/a20_decoder.h"
/* Research-only explicit caller arena. No product RNN ABI compatibility.
 * Caller owns immutable model bytes and all state; single-thread/non-reentrant. */
typedef void (*a20s_trace_callback)(void*,uint64_t,size_t,const float*,const float*,const float*);
typedef struct {
 donor_pcm_state pcm;
 a20_model model;
 a20d_decoder decoder;
 a20d_workspace decoder_work;
 float logits[DONOR_PCM_MAX_SELECTED_ROWS][A20D_CLASSES];
 uint32_t initialized, fault, busy;
 /* Optional borrowed diagnostic trace, absent in normal operation. */
 float *trace_buffer;
 a20s_trace_callback trace_callback;
 void *trace_user;
} a20s_state;
typedef struct {
 const donor_pcm_batch *frontend;
 const float *logits;        /* borrowed [selected_rows][6] */
 const float *cache;         /* borrowed [128][11][4] after whole batch */
 const a20d_result *result;  /* borrowed chunk return; at most one event */
} a20s_batch;
typedef void (*a20s_callback)(void *, const a20s_batch *);
size_t a20s_state_bytes(void);
int a20s_init(a20s_state*,const float *weights,size_t floats);
int a20s_reset(a20s_state*); /* resets all timelines, frontend, cache and beams */
int a20s_set_trace(a20s_state*,float *trace21x250,a20s_trace_callback,void*);
int a20s_feed(a20s_state*,const int16_t*,size_t,a20s_callback,void*);
int a20s_finish(a20s_state*,a20s_callback,void*); /* actual retained tail only */
#endif
