/* SPDX-License-Identifier: Apache-2.0
 * Exact full-OTHER original par4 maximum, with an optional conservative
 * temporal certificate. No model/PCM IO, no allocations, no runtime switch.
 */
#include "certified_other.h"
#include <float.h>
#include <fenv.h>
#include <math.h>
#include <string.h>

static const uint32_t precise_ids[R1_CERT_PRECISE_COUNT] = {
    0u, 1027u, 1401u, 1462u, 2093u
};

/* The original par4 kernel performs 35 sequential terms in each lane,
 * combines (s0+s1)+(s2+s3), scales in binary64, then adds an f32 bias.
 * It must be built with no fast-math, contraction or excess precision. */
static double exact_par4(const r1_cert_head *s, uint32_t id,
                         const double *x)
{
    const int8_t *row = s->quant_rows + (size_t)id * R1_CERT_HIDDEN;
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    for (size_t i = 0; i < R1_CERT_HIDDEN; i += 4u) {
        s0 += x[i]     * (double)row[i];
        s1 += x[i + 1u] * (double)row[i + 1u];
        s2 += x[i + 2u] * (double)row[i + 2u];
        s3 += x[i + 3u] * (double)row[i + 3u];
    }
    double sum = (s0 + s1) + (s2 + s3);
    sum *= s->scales[id];
    return sum + (double)s->biases[id];
}

static int valid_fp_environment(void)
{
    const uint16_t one = 1u;
    volatile double lo = DBL_MIN, half = 0.5;
    volatile double sub = lo * half;
    return sizeof(double) == 8u && DBL_MANT_DIG == 53 &&
           sizeof(float) == 4u && FLT_MANT_DIG == 24 &&
           FLT_RADIX == 2 && FLT_EVAL_METHOD == 0 &&
           *(const unsigned char *)&one == 1u &&
           fegetround() == FE_TONEAREST && sub > 0.0;
}

int r1_cert_init(r1_cert_head *state, const int8_t *rows,
                 const double *scales, const float *biases,
                 const uint32_t *selected_ids, size_t count)
{
    if (!state || !rows || !scales || !biases || !selected_ids ||
        count != R1_CERT_SELECTED_COUNT)
        return R1_CERT_BAD_ARGUMENT;
    if (!valid_fp_environment())
        return R1_CERT_BAD_ENVIRONMENT;
    /* Do not publish partially validated state. */
    r1_cert_head trial;
    memset(&trial, 0, sizeof(trial));
    trial.quant_rows = rows;
    trial.scales = scales;
    trial.biases = biases;
    for (size_t i = 0; i < R1_CERT_PRECISE_COUNT; ++i)
        trial.role[precise_ids[i]] = 2u;
    for (size_t i = 0; i < count; ++i) {
        uint32_t id = selected_ids[i];
        if (id >= R1_CERT_CLASSES ||
            trial.role[id] != 0u ||
            (i && id <= selected_ids[i - 1u]))
            return R1_CERT_BAD_MODEL;
        trial.role[id] = 1u;
        trial.selected_ids[i] = id;
    }
    size_t residual = 0u;
    for (uint32_t id = 0u; id < R1_CERT_CLASSES; ++id) {
        if (trial.role[id] == 2u)
            continue;
        if (!isfinite(scales[id]) || scales[id] <= 0.0 ||
            !isfinite((double)biases[id]))
            return R1_CERT_BAD_MODEL;
        const int8_t *row = rows + (size_t)id * R1_CERT_HIDDEN;
        for (size_t i = 0u; i < R1_CERT_HIDDEN; ++i)
            if (row[i] == INT8_MIN)
                return R1_CERT_BAD_MODEL;
        if (trial.role[id] != 0u)
            continue;
        ++residual;
        double b = fabs((double)biases[id]);
        if (b > trial.residual_bias_cap)
            trial.residual_bias_cap = b;
        for (size_t i = 0u; i < R1_CERT_HIDDEN; ++i) {
            int value = (int)row[i];
            if (value < 0)
                value = -value;
            double c = scales[id] * (double)value;
            if (!isfinite(c))
                return R1_CERT_BAD_MODEL;
            if (c != 0.0)
                c = nextafter(c, INFINITY);
            if (c > trial.residual_abs_coeff[i])
                trial.residual_abs_coeff[i] = c;
        }
    }
    if (residual != R1_CERT_CLASSES - R1_CERT_PRECISE_COUNT -
                    R1_CERT_SELECTED_COUNT)
        return R1_CERT_BAD_MODEL;
    *state = trial;
    return R1_CERT_OK;
}

int r1_cert_reset(r1_cert_head *s)
{
    if (!s || !s->quant_rows || !s->scales || !s->biases)
        return R1_CERT_BAD_ARGUMENT;
    s->have_prior = 0;
    s->prior_residual_upper = 0.0;
    s->prior_magnitude = 0.0;
    s->certified_skips = 0u;
    s->exact_fallbacks = 0u;
    memset(s->prior_hidden, 0, sizeof(s->prior_hidden));
    return R1_CERT_OK;
}

/* A finite upper envelope of the exact real dot magnitude for any omitted
 * row, plus its bias. The outward-rounded coefficient maxima are fixed at
 * initialization and only cover residual rows. */
static int magnitude(const r1_cert_head *s, const double *x, double *out)
{
    double v = 1.0 + s->residual_bias_cap;
    for (size_t i = 0u; i < R1_CERT_HIDDEN; ++i) {
        if (!isfinite(x[i]))
            return 0;
        double contribution = fabs(x[i]) * s->residual_abs_coeff[i];
        if (!isfinite(contribution))
            return 0;
        v += contribution;
        if (!isfinite(v))
            return 0;
    }
    *out = v;
    return 1;
}

/* For any omitted original row j, the *exact* real affine change from x0
 * to x1 is bounded by sum_i max_j |scale_j*q_ji| * |x1_i-x0_i|.
 * The original 4x35 binary64 accumulation plus scale and bias introduces
 * bounded rounding. 8192*DBL_EPSILON is deliberately much larger than the
 * sub-50-operation gamma bound per output; it also covers the 140-term
 * envelope, subtractions and propagation arithmetic, with +1 for subnormals.
 * If any intermediate is nonfinite, no certificate is issued.
 * We NEVER certify on equality: full fallback retains original-ID tie order.
 * Target build must not use contraction, fast-math, excess precision, FTZ,
 * or non-nearest rounding, and both steps must use the same model bytes.
 */
static int next_upper(const r1_cert_head *s, const double *x,
                      double now_magnitude, double *next)
{
    if (!s->have_prior)
        return 0;
    double delta = 0.0;
    for (size_t i = 0u; i < R1_CERT_HIDDEN; ++i) {
        double change = fabs(x[i] - s->prior_hidden[i]);
        double term = change * s->residual_abs_coeff[i];
        if (!isfinite(term))
            return 0;
        delta += term;
        if (!isfinite(delta))
            return 0;
    }
    double envelope = 1.0 + s->prior_magnitude;
    envelope += now_magnitude;
    envelope += delta;
    envelope += fabs(s->prior_residual_upper);
    if (!isfinite(envelope))
        return 0;
    double guard = (8192.0 * DBL_EPSILON) * envelope;
    double upper = s->prior_residual_upper + delta;
    upper += guard;
    if (!isfinite(upper))
        return 0;
    upper = nextafter(upper, INFINITY);
    if (!isfinite(upper))
        return 0;
    *next = upper;
    return 1;
}

int r1_cert_step(r1_cert_head *s, const double hidden[R1_CERT_HIDDEN],
                 r1_cert_result *result)
{
    if (!s || !s->quant_rows || !s->scales || !s->biases ||
        !hidden || !result)
        return R1_CERT_BAD_ARGUMENT;
    if (!valid_fp_environment())
        return R1_CERT_BAD_ENVIRONMENT;
    for (size_t i = 0u; i < R1_CERT_HIDDEN; ++i)
        if (!isfinite(hidden[i]))
            return R1_CERT_NONFINITE;

    /* Every selected row uses its unchanged original par4 arithmetic. */
    double selected_max = -INFINITY;
    uint32_t selected_id = UINT32_MAX;
    for (size_t i = 0u; i < R1_CERT_SELECTED_COUNT; ++i) {
        uint32_t id = s->selected_ids[i];
        double value = exact_par4(s, id, hidden);
        if (!isfinite(value))
            return R1_CERT_NONFINITE;
        if (value > selected_max ||
            (value == selected_max && id < selected_id)) {
            selected_max = value;
            selected_id = id;
        }
    }

    double mag = 0.0, upper = 0.0;
    int can_bound = magnitude(s, hidden, &mag);
    int certified = can_bound && next_upper(s, hidden, mag, &upper) &&
                    upper < selected_max;
    double best = selected_max;
    uint32_t best_id = selected_id;
    double residual_max = -INFINITY;
    uint32_t residual_id = UINT32_MAX;
    if (!certified) {
        for (uint32_t id = 0u; id < R1_CERT_CLASSES; ++id) {
            if (s->role[id] != 0u)
                continue;
            double value = exact_par4(s, id, hidden);
            if (!isfinite(value))
                return R1_CERT_NONFINITE;
            if (value > residual_max ||
                (value == residual_max && id < residual_id)) {
                residual_max = value;
                residual_id = id;
            }
        }
        if (residual_max > best ||
            (residual_max == best && residual_id < best_id)) {
            best = residual_max;
            best_id = residual_id;
        }
    }

    r1_cert_result fresh;
    fresh.other_max = best;
    fresh.other_original_id = best_id;
    fresh.certified_this_frame = certified;
    fresh.evaluated_other_rows = certified ? R1_CERT_SELECTED_COUNT :
                                       R1_CERT_CLASSES - R1_CERT_PRECISE_COUNT;
    fresh.certified_skips = s->certified_skips + (unsigned)certified;
    fresh.exact_fallbacks = s->exact_fallbacks + (unsigned)!certified;

    /* Commit atomically only after all exact evaluations succeed. */
    memcpy(s->prior_hidden, hidden, sizeof(s->prior_hidden));
    s->have_prior = can_bound;
    s->prior_magnitude = can_bound ? mag : 0.0;
    s->prior_residual_upper = certified ? upper : residual_max;
    s->certified_skips = fresh.certified_skips;
    s->exact_fallbacks = fresh.exact_fallbacks;
    *result = fresh;
    return R1_CERT_OK;
}
