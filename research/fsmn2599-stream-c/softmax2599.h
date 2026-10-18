/* SPDX-License-Identifier: Apache-2.0
 * Independent research candidate; not Torch softmax parity or a release gate. */
#ifndef KWS_SOFTMAX2599_H
#define KWS_SOFTMAX2599_H
#include <stddef.h>
#define KSX_CLASSES 2599u
typedef enum { KSX_OK=0, KSX_ARGUMENT=1, KSX_OVERLAP=2,
    KSX_ENVIRONMENT=3, KSX_NONFINITE=4, KSX_ARITHMETIC=5 } ksx_status;
typedef struct { float exponentials[KSX_CLASSES], probabilities[KSX_CLASSES]; } ksx_workspace;
/* Full2599 finite float32 logits -> float32 posterior. All buffers disjoint;
 * workspace and output must remain unmodified by the caller during this call.
 * Ascending-token max -> float32 shifted argument -> expf -> ascending-token
 * binary64 sum -> binary64 division -> one float32 rounding. No crop, floor,
 * clipping of probabilities, residual correction, or secondary renormalization.
 * On any error output is unchanged; workspace may be overwritten. No allocation.
 * RNE / gradual underflow / binary32+binary64 / masked FP traps are required.
 * Environment probes cannot verify trap masks. No global FP mode is changed.
 * Build without fast-math, reassociation, FMA contraction or excess precision.
 * The result is a NEW mixed-precision softmax contract, not Torch bit parity.
 * Feed the complete output to kws3_step_f32 for exact binary64 promotion. */
ksx_status ksx_softmax2599(const float *logits,size_t logit_count,
                         float *posterior,size_t posterior_count,
                         ksx_workspace *workspace);
#endif
