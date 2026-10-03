#ifndef DONOR_FFT64_H
#define DONOR_FFT64_H
#include <stddef.h>
#include <stdint.h>
/* Isolated donor-specific research contract. Mono16k PCM16, unscaled float32;
 *400/160 framegrid phase0, no EOF padding. Not the product frontend ABI. */
#define DONOR_FFT64_BINS 80
#define DONOR_FFT64_FRAME 400
#define DONOR_FFT64_HOP 160
#define DONOR_FFT64_FFT 512
#define DONOR_FFT64_MAX_FEED 16000
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
 /* Research-only binary64 complex workspace; original float ABI is untouched. */
 double fft_re[512],fft_im[512];
} donor_fft64_state;
typedef struct {
 float dc[400],preemphasis[400],windowed[512];
 float power[257],mel[80],logfbank[80];
} donor_fft64_trace;
/* Callback row lives for call duration only; index starts0, end_sample is
 *exclusive400+160*index. Single-threaded/non-reentrant; no API calls in callback. */
typedef void (*donor_fft64_callback)(void*,const float row[80],uint64_t index,uint64_t end_sample);
enum {DONOR_FFT64_OK=0,DONOR_FFT64_ARGUMENT=1,DONOR_FFT64_STATE=2};
/* Required for all arithmetic calls: RNE, non-FTZ/DAZ IEEE32/64. */
int donor_fft64_environment_ok(void);
size_t donor_fft64_state_bytes(void);
int donor_fft64_init(donor_fft64_state*);
int donor_fft64_reset(donor_fft64_state*);
int donor_fft64_feed(donor_fft64_state*,const int16_t*,size_t,donor_fft64_callback,void*);
int donor_fft64_finish(donor_fft64_state*);
/* Diagnostic single-frame operation; uses caller-owned state scratch, does not
 *advance stream counters. Not for concurrent use with feed. */
int donor_fft64_analyze_frame(donor_fft64_state*,const int16_t pcm[400],donor_fft64_trace*);
/* Diagnostic FFT512 power-only hook for independent scaling/Parseval/bin tests.
 *No frame preprocessing; does not advance stream counters. */
int donor_fft64_fft_power(donor_fft64_state*,const float input[512],float output[257]);
/* Donor-normalization component only: input is an already correctly spliced
 *400-vector. No80dim optimization, splice, subsampling or statistic fitting.
 *Exact input==output in-place is allowed; partial input/output overlap is
 *rejected. Statistic buffers must not overlap output. Computed nonfinite
 *results are rejected before any output write. */
int donor_fft64_cmvn400(const float input[400],const float mean[400],const float inverse_std[400],float output[400]);
#ifdef __cplusplus
}
#endif
#endif
