/* SPDX-License-Identifier: Apache-2.0
 * Research-only, opt-in exact OTHER maximum with conservative time bound.
 * Full original 2,601-row quantized head remains the backing store.
 * No probability, all-logit, acoustic, or device qualification claim.
 */
#ifndef R1_CERTIFIED_OTHER_FALLBACK_H
#define R1_CERTIFIED_OTHER_FALLBACK_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define R1_CERT_CLASSES 2601u
#define R1_CERT_HIDDEN 140u
#define R1_CERT_PRECISE_COUNT 5u
#define R1_CERT_SELECTED_COUNT 113u

enum {
    R1_CERT_OK = 0,
    R1_CERT_BAD_ARGUMENT = 1,
    R1_CERT_BAD_MODEL = 2,
    R1_CERT_BAD_ENVIRONMENT = 3,
    R1_CERT_NONFINITE = 4
};

typedef struct {
    double other_max;
    uint32_t other_original_id;
    uint32_t evaluated_other_rows;
    uint32_t certified_skips;
    uint32_t exact_fallbacks;
    int certified_this_frame; /* 1: skipped all residual rows; 0: exact scan */
} r1_cert_result;

typedef struct {
    /* Caller-owned immutable typed arrays; valid for the full handle lifetime.
     * quant_rows is 2601 consecutive 140-byte int8 rows, including the five
     * unused precise IDs. scales/biases are in original class-ID order. */
    const int8_t *quant_rows;
    const double *scales;
    const float *biases;
    /* 0=residual OTHER, 1=selected OTHER, 2=precise */
    unsigned char role[R1_CERT_CLASSES];
    uint32_t selected_ids[R1_CERT_SELECTED_COUNT];
    double residual_abs_coeff[R1_CERT_HIDDEN];
    double residual_bias_cap;
    double prior_hidden[R1_CERT_HIDDEN];
    double prior_residual_upper;
    double prior_magnitude;
    uint32_t certified_skips;
    uint32_t exact_fallbacks;
    int have_prior;
} r1_cert_head;

/* Expected selected IDs must be the already-frozen ACTIVE113 list, in ascending
 * original-ID order. The caller checks its frozen SHA provenance separately.
 * No access to owner audio, model files, or new model evaluation occurs in init.
 * Requires IEEE binary64, RNE, gradual underflow, no fast-math or contraction.
 * Caller must use the same arithmetic flags as the original par4 kernel. */
int r1_cert_init(r1_cert_head *state, const int8_t *quant_rows,
                 const double *scales, const float *biases,
                 const uint32_t *selected_ids, size_t count);
int r1_cert_reset(r1_cert_head *state);

/* Computes exact par4 selected OTHER scores first. When a conservative
 * rounded upper bound on all residual OTHER scores is STRICTLY BELOW the
 * selected maximum, no residual row is evaluated; otherwise every residual
 * original row is evaluated in ascending original-ID order. Both paths return
 * the original full-head max OTHER score and original tie-breaking ID, not
 * merely the overall token winner. No changes to the five precise logits,
 * decoder, clock, or default SDK. On failure, state/result remain unchanged.
 * Calling with missing or changed full model arrays violates the contract.
 * Not reentrant or thread safe. A gap/reset clears the temporal certificate.
 */
int r1_cert_step(r1_cert_head *state, const double hidden[R1_CERT_HIDDEN],
                 r1_cert_result *result);

#ifdef __cplusplus
}
#endif
#endif
