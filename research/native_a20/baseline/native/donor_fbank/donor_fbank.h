#ifndef DONOR_FBANK_H
#define DONOR_FBANK_H
#include <stddef.h>
#include <stdint.h>
/* Isolated donor-specific research contract. Mono16k PCM16, unscaled float32;
 *400/160 framegrid phase0, no EOF padding. Not the product frontend ABI. */
#define DONOR_FBANK_BINS 80
#define DONOR_FBANK_FRAME 400
#define DONOR_FBANK_HOP 160
#define DONOR_FBANK_FFT 512
#define DONOR_FBANK_MAX_FEED 16000
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
 uint32_t initialized;
 int finished;
 size_t used;
 uint64_t total_samples,frame_index;
 int16_t pcm[400];
 float re[512],im[512];
} donor_fbank_state;
typedef struct {
 float dc[400],preemphasis[400],windowed[512];
 float power[257],mel[80],logfbank[80];
} donor_fbank_trace;
/* Callback row lives for call duration only; index starts0, end_sample is
 *exclusive400+160*index. Single-threaded/non-reentrant; no API calls in callback. */
typedef void (*donor_fbank_callback)(void*,const float row[80],uint64_t index,uint64_t end_sample);
enum {DONOR_FBANK_OK=0,DONOR_FBANK_ARGUMENT=1,DONOR_FBANK_STATE=2};
size_t donor_fbank_state_bytes(void);
int donor_fbank_init(donor_fbank_state*);
int donor_fbank_reset(donor_fbank_state*);
int donor_fbank_feed(donor_fbank_state*,const int16_t*,size_t,donor_fbank_callback,void*);
int donor_fbank_finish(donor_fbank_state*);
/* Diagnostic single-frame operation; uses caller-owned state scratch, does not
 *advance stream counters. Not for concurrent use with feed. */
int donor_fbank_analyze_frame(donor_fbank_state*,const int16_t pcm[400],donor_fbank_trace*);
/* Diagnostic FFT512 power-only hook for independent scaling/Parseval/bin tests.
 *No frame preprocessing; does not advance stream counters. */
int donor_fbank_fft_power(donor_fbank_state*,const float input[512],float output[257]);
/* Donor-normalization component only: input is an already correctly spliced
 *400-vector. No80dim optimization, splice, subsampling or statistic fitting.
 *Exact input==output in-place is allowed; partial input/output overlap is
 *rejected. Statistic buffers must not overlap output. Computed nonfinite
 *results are rejected before any output write. */
int donor_cmvn400(const float input[400],const float mean[400],const float inverse_std[400],float output[400]);
#ifdef __cplusplus
}
#endif
#endif
