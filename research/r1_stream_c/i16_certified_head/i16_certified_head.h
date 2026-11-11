/* SPDX-License-Identifier: Apache-2.0
 * Opt-in original-par4 exact OTHER maximum using int16 quantized-hidden
 * candidate screening. This is a full-weight research head, not ACTIVE113's
 * six-score CTC/probability replacement or a target-board qualification. */
#ifndef R1_I16_CERTIFIED_HEAD_H
#define R1_I16_CERTIFIED_HEAD_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define R1_I16_CLASSES 2601u
#define R1_I16_HIDDEN 140u
#define R1_I16_PRECISE 5u
#define R1_I16_OTHER 2596u

enum {
    R1_I16_OK = 0,
    R1_I16_BAD_ARGUMENT = 1,
    R1_I16_BAD_MODEL = 2,
    R1_I16_BAD_ENVIRONMENT = 3,
    R1_I16_NONFINITE = 4
};

typedef struct {
    /* Caller-validated complete qhead8 payload, immutable while in use.
     * Original class ID layout, including five precise class slots.
     * No copy or ownership transfer. All typed arrays must be aligned. */
    const int8_t *quant_rows;     /* [2601][140], signed byte */
    const double *scales;        /* [2601] */
    const float *biases;         /* [2601] */
    uint16_t abs_weight_sum[R1_I16_CLASSES];
    double candidate_upper[R1_I16_CLASSES]; /* reusable scratch, not output */
} r1_i16_head;

typedef struct {
    double exact_other_max;
    uint32_t other_original_id;
    uint32_t approx_rows;    /* normally 2596, 0 on numeric fallback */
    uint32_t exact_rows;     /* candidates or all 2596 on numeric fallback */
    uint32_t numeric_fallback;
} r1_i16_result;

/* Validates supplied rows/scale/bias and constructs L1 row sums. The caller
 * must authenticate the original complete model SHA256 independently.
 * Requires IEEE binary32/64, little endian, FE_TONEAREST and subnormals.
 * No fast-math, FMA contraction or excess precision. No PCM or inference. */
int r1_i16_init(r1_i16_head *head, const int8_t *quant_rows,
                const double *scales, const float *biases);

/* Every OTHER row gets an int16(hidden) x int8(weight) integer dot product
 * and an outward-rounded affine interval; all possible winning rows are
 * re-evaluated with original par4 binary64 ordering in ascending ORIGINAL ID.
 * On unsupported intermediate values, evaluate all OTHER rows exactly.
 * Five precise output scores and decoder remain caller-owned and unchanged.
 * Missing/nonfinite input fails without modifying the caller's result.
 * Single-threaded, non-reentrant. No dynamic memory, no global mutable data. */
int r1_i16_step(r1_i16_head *head, const double hidden[R1_I16_HIDDEN],
                r1_i16_result *result);

#ifdef __cplusplus
}
#endif
#endif
