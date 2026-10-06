#include "k1_fixed_slot_ctc.h"

#include <float.h>
#include <limits.h>
#include <math.h>

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)
#error "K1 requires IEEE infinity/NaN semantics; do not enable fast-math"
#endif

_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128,
               "K1 requires binary32 float");
_Static_assert(DBL_MANT_DIG == 53 && DBL_MAX_EXP == 1024,
               "K1 requires binary64 double");
_Static_assert(CHAR_BIT == 8 && sizeof(float) == 4 && sizeof(double) == 8,
               "K1 requires 32-bit float and 64-bit double storage");
_Static_assert(FLT_EVAL_METHOD == 0 || FLT_EVAL_METHOD == 1,
               "K1 does not permit extended-precision double evaluation");
_Static_assert(SIZE_MAX <= UINT64_MAX, "K1 supports at most 64-bit size_t");

static int same_binding(const k1_binding *x, const k1_binding *y)
{
    return x->source_id == y->source_id && x->epoch == y->epoch &&
           x->generation == y->generation && x->range_id == y->range_id &&
           x->prefix_begin == y->prefix_begin && x->a == y->a && x->b == y->b &&
           x->evidence_horizon == y->evidence_horizon;
}

static double add_logs(double x, double y)
{
    double t;
    if (x == -INFINITY) return y;
    if (y == -INFINITY) return x;
    if (y > x) { t = x; x = y; y = t; }
    return x + log1p(exp(y - x));
}

static double log_probability(float x)
{
    return x == 0.0f ? -INFINITY : log((double)x);
}

/* uintptr_t makes the supported platform's object-address representation
 * explicit. No pointer subtraction/order is used across unrelated objects.
 */
static enum k1_error disjoint_objects(const k1_view *v, const k1_prefix *p,
                                      const k1_slot *s, k1_result *out)
{
    const void *ptr[6] = {v, p, s, v->rows, p->raw_path, out};
    size_t bytes[6] = {sizeof *v, sizeof *p, sizeof *s,
                      v->capacity_rows * sizeof *v->rows,
                      p->path_capacity * sizeof *p->raw_path, sizeof *out};
    uintptr_t begin[6], end[6];
    size_t i, j;
    for (i = 0; i < 6; ++i) {
        if (ptr[i] == NULL) return K1_ERR_ARGUMENT;
        begin[i] = (uintptr_t)ptr[i];
        if ((uintmax_t)bytes[i] > (uintmax_t)UINTPTR_MAX ||
            begin[i] > UINTPTR_MAX - (uintptr_t)bytes[i])
            return K1_ERR_OVERFLOW;
        end[i] = begin[i] + (uintptr_t)bytes[i];
    }
    for (i = 0; i < 6; ++i)
        for (j = i + 1; j < 6; ++j)
            if (bytes[i] != 0 && bytes[j] != 0 &&
                begin[i] < end[j] && begin[j] < end[i])
                return K1_ERR_OVERLAP;
    return K1_OK;
}

enum k1_error k1_score(const k1_view *v, const k1_prefix *p,
                       const k1_slot *s, k1_result *out)
{
    static const uint8_t target[3] = {K1_NI, K1_HAO, K1_XIAO};
    const k1_binding *h;
    uint64_t available_end, used64, prefix64;
    size_t offset, used, prefix_count, i, j, emitted = 0;
    uint8_t previous;
    double state[2][3] = {{0.0, -INFINITY, -INFINITY},
                          {0.0, -INFINITY, -INFINITY}};
    k1_result result = {0};
    enum k1_error error;

    if (v == NULL || p == NULL || s == NULL || out == NULL)
        return K1_ERR_ARGUMENT;
    h = &v->binding;
    if (!same_binding(h, &p->binding) || !same_binding(h, &s->binding) ||
        h->source_id == 0 || h->epoch == 0 || h->generation == 0 || h->range_id == 0)
        return K1_ERR_BINDING;
    if (h->prefix_begin > h->a || h->a > h->b)
        return K1_ERR_RANGE;
    if (v->class_count != K1_CLASS_COUNT || v->append_only != 1)
        return K1_ERR_ARGUMENT;
    if (v->contiguous != 1) return K1_ERR_GAPPED;
    if (v->capacity_rows > SIZE_MAX / sizeof *v->rows ||
        v->available_rows > UINT64_MAX - v->first_row)
        return K1_ERR_OVERFLOW;
    if (v->available_rows > v->capacity_rows || p->path_count > p->path_capacity)
        return K1_ERR_CAPACITY;
    available_end = v->first_row + (uint64_t)v->available_rows;
    if (v->immutable_through < v->first_row || v->immutable_through > available_end)
        return K1_ERR_RANGE;
    if (h->prefix_begin < v->first_row || h->b > available_end)
        return K1_ERR_UNAVAILABLE;
    if (s->sealed != 1 || p->immutable != 1 || p->run_closed != 1 ||
        v->immutable_through < h->b)
        return K1_ERR_UNSEALED;
    used64 = h->b - h->prefix_begin;
    prefix64 = h->a - h->prefix_begin;
    if (used64 > K1_RESEARCH_MAX_USED_ROWS)
        return K1_ERR_CAPACITY;
    if (prefix64 != (uint64_t)p->path_count || prefix64 < 3 ||
        p->entry_raw_label >= K1_CLASS_COUNT)
        return K1_ERR_PREFIX_PATH;
    error = disjoint_objects(v, p, s, out);
    if (error != K1_OK) return error;
    offset = (size_t)(h->prefix_begin - v->first_row);
    used = (size_t)used64;
    prefix_count = (size_t)prefix64;

    /* Validate all six probabilities on every consumed row. Never normalize. */
    for (i = 0; i < used; ++i) {
        double sum = 0.0;
        if (v->rows[offset + i].available_at > h->evidence_horizon)
            return K1_ERR_LATE;
        for (j = 0; j < K1_CLASS_COUNT; ++j) {
            double x = (double)v->rows[offset + i].p[j];
            if (!isfinite(x) || x < 0.0 || x > 1.0)
                return K1_ERR_PROBABILITY;
            sum += x;
        }
        if (fabs(sum - 1.0) > K1_FP32_SUM_TOLERANCE)
            return K1_ERR_NORMALIZATION;
    }

    previous = p->entry_raw_label;
    for (i = 0; i < prefix_count; ++i) {
        uint8_t raw = p->raw_path[i];
        float probability;
        if (raw >= K1_CLASS_COUNT) return K1_ERR_PREFIX_PATH;
        if (raw != K1_BLANK && raw != previous) {
            if (emitted >= 3 || raw != target[emitted])
                return K1_ERR_PREFIX_PATH;
            ++emitted;
        }
        previous = raw;
        probability = v->rows[offset + i].p[raw];
        if (probability == 0.0f) return K1_ERR_PREFIX_UNSUPPORTED;
        result.log_prefix_mass += log((double)probability);
    }
    if (emitted != 3 || (previous != K1_XIAO && previous != K1_BLANK))
        return K1_ERR_PREFIX_PATH;

    for (i = prefix_count; i < used; ++i) {
        const k1_row *row = &v->rows[offset + i];
        double lb = log_probability(row->p[K1_BLANK]);
        for (j = 0; j < 2; ++j) {
            double lq = log_probability(row->p[K1_WO + j]);
            double pre = lb + state[j][0];
            double run = lq + add_logs(state[j][0], state[j][1]);
            double post = lb + add_logs(state[j][1], state[j][2]);
            state[j][0] = pre;
            state[j][1] = run;
            state[j][2] = post;
        }
    }
    for (j = 0; j < 2; ++j) {
        result.log_conditional_mass[j] = add_logs(state[j][1], state[j][2]);
        result.log_joint_mass[j] = result.log_prefix_mass + result.log_conditional_mass[j];
    }
    if (prefix_count == used) result.support = K1_SUPPORT_EMPTY_INTERVAL;
    else if (result.log_conditional_mass[0] == -INFINITY)
        result.support = result.log_conditional_mass[1] == -INFINITY ?
                         K1_SUPPORT_NEITHER : K1_SUPPORT_WU_ONLY;
    else result.support = result.log_conditional_mass[1] == -INFINITY ?
                          K1_SUPPORT_WO_ONLY : K1_SUPPORT_BOTH;
    result.binding = *h;
    result.provenance = K1_CALLER_BOUND_UNVERIFIED;
    result.boundaries = K1_FIXED_COORDINATES_ACOUSTIC_BOUNDARIES_UNPROVEN;
    result.entry_raw_label = p->entry_raw_label;
    result.final_prefix_raw_label = previous;
    *out = result;
    return K1_OK;
}
