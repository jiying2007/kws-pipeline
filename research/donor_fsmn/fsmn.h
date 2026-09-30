#ifndef DONOR_FSMN_H
#define DONOR_FSMN_H
#include <stddef.h>
#define DF_FLOATS 756933u
#define DF_CACHE_FLOATS 5632u
/* Cache layout matches Python [128][11][4]. Weights remain caller-owned. */
typedef struct { const float *weights; float cache[DF_CACHE_FLOATS]; int fault; } df_model;
int df_init(df_model*, const float*, size_t);
void df_reset(df_model*);
/* CMVN input/output must not overlap; on error all output is invalid. */
int df_cmvn(const df_model*, const float*, float*);
/* One 400D spliced row -> 2599 raw logits. Optional trace [21][2599],
   On nonfinite arithmetic the stream faults until reset; outputs must be discarded.
   stage dimensions 140,250,250,(128,128,250,250)x4,140,2599. */
int df_step(df_model*, const float*, float*, float*);
#endif
