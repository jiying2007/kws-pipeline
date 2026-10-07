#ifndef A20_FSMN_H
#define A20_FSMN_H
#include <stddef.h>
#define A20_FLOATS 391320u
#define A20_OUTPUTS 6u
#define A20_TRACE_STRIDE 250u
#define A20_CACHE_FLOATS 5632u
/* Cache layout matches Python [128][11][4]. Weights remain caller-owned. */
typedef struct { const float *weights; float cache[A20_CACHE_FLOATS]; int fault; } a20_model;
size_t a20_model_bytes(void);
/* Research variant: affine/FSMN accumulation only; storage and CMVN FP32. */
int a20_accumulation_bits(void);
int a20_init_verified(a20_model*, const float*, size_t);
/* Unverified initializer is for synthetic unit tests only. Runtime uses verified. */
int a20_init(a20_model*, const float*, size_t);
void a20_reset(a20_model*);
/* CMVN input/output must not overlap; on error all output is invalid. */
int a20_cmvn(const a20_model*, const float*, float*);
/* One 400D spliced row -> 6 raw logits. Optional trace [21][250],
   On nonfinite arithmetic the stream faults until reset; outputs must be discarded.
   stage dimensions 140,250,250,(128,128,250,250)x4,140,6. */
int a20_step(a20_model*, const float*, float*, float*);
#endif
