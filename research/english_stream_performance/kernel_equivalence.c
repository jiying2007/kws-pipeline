/* Run only in the authorized public Actions experiment, never locally.
 * The original pinned C is the independent oracle, not a rewritten convolution.
 * Define EKWS_BASELINE_FRONTEND to its quoted absolute source path at compile.
 * Synthetic values below verify arithmetic; they are not acoustic evidence. */
#ifndef EKWS_BASELINE_FRONTEND
#error "Define EKWS_BASELINE_FRONTEND to the pinned original frontend C path"
#endif
#include EKWS_BASELINE_FRONTEND
#include "ordered_3x1.h"
#include <fenv.h>
#include <stdio.h>
#include <stdlib.h>

static unsigned failures;
static unsigned checks;
static unsigned guard_checks;
static unsigned fallback_matcher_checks;

static void require_check(int condition, const char *label) {
    ++checks;
    if (!condition) { ++failures; fprintf(stderr, "failed: %s\n", label); }
}

static void require_rejected(int matched, const char *label) {
    ++fallback_matcher_checks;
    require_check(!matched, label);
}

static float from_bits(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static float test_value(size_t i, unsigned pattern, int weight) {
    uint32_t u = (uint32_t)i * UINT32_C(1664525) + UINT32_C(1013904223);
    switch (pattern) {
        case 0: return (float)((int32_t)((u >> 12) % 2049) - 1024) * 0x1p-10f;
        /* All products are -0. Starting from +0 must still produce +0 under
         * round-to-nearest, even with a final -0 bias. */
        case 1: return weight ? 1.0f : from_bits(UINT32_C(0x80000000));
        case 2:
            if (weight) return (i & 1u) ? -0.5f : 1.0f;
            return from_bits((uint32_t)(1 + i % 64) | ((i & 1u) ? UINT32_C(0x80000000) : 0));
        case 3:
            if (weight) return (i % 3 == 0) ? -1.0f : 1.0f;
            switch (i % 4) { case 0: return 0x1p24f; case 1: return 1.0f;
                            case 2: return -0x1p24f; default: return -1.0f; }
        case 4: return weight ? ((i & 1u) ? -4.0f : 4.0f) : FLT_MAX;
        default:
            if (i % 29 == 0) return from_bits(UINT32_C(0x7fc00001));
            if (i % 31 == 0) return from_bits(UINT32_C(0x7f800000));
            return weight ? 0.5f : -1.0f;
    }
}

static void descriptor(ef_tensor *t, uint32_t c, uint32_t h, uint32_t w, uint32_t storage) {
    memset(t, 0, sizeof(*t));
    t->rank = 4; t->dims[0] = 1; t->dims[1] = c; t->dims[2] = h; t->dims[3] = w;
    t->count = c * h * w; t->storage = storage;
}

static int equal_result(float a, float b) {
    if (isnan(a) || isnan(b)) return isnan(a) && isnan(b);
    /* Finite values, signed zeros, and signed infinities must be bit-identical. */
    return memcmp(&a, &b, sizeof(a)) == 0;
}

int main(void) {
    static const uint32_t dims[9][3] = {
        {24,74,32}, {48,36,16}, {48,34,16}, {72,32,8}, {72,30,8},
        {96,14,4}, {96,12,4}, {96,10,2}, {96,8,2}
    };
    static const int32_t params[16] = {2,1,3,1,1,1,1,1,0,0,0,0,0,0,0,0};
    unsigned shape, pattern, pairs = 0;
    size_t compared = 0, nonfinite_values = 0;
    if (FLT_EVAL_METHOD != 0 || fegetround() != FE_TONEAREST) {
        fprintf(stderr, "numeric test requires FLT_EVAL_METHOD=0 and FE_TONEAREST\n");
        return 2;
    }
    /* Refuse silent denormal flushing, rather than relaxing the candidate's
     * contract. Volatile prevents this environment probe being constant-folded. */
    {
        volatile float tiny = from_bits(1), one = 1.0f;
        float observed = tiny * one;
        require_check(memcmp(&observed, &(float){from_bits(1)}, sizeof(float)) == 0,
                      "subnormal multiplication preserved");
    }
    for (shape = 0; shape < 9; ++shape) {
        const uint32_t c = dims[shape][0], h = dims[shape][1], width = dims[shape][2];
        ef_tensor x, w, y;
        ef_node n;
        float *a, *weights, *bias, *reference, *candidate, *adjacent;
        size_t i;
        unsigned before;
        descriptor(&x, c, h, width, EF_WORK);
        descriptor(&y, c, h - 2, width, EF_WORK);
        descriptor(&w, c, 3, 1, EF_CONSTANT);
        w.dims[0] = c; w.count *= c;
        memset(&n, 0, sizeof(n)); n.op = EF_CONV; n.input_count = 3;
        n.inputs[0] = 0; n.inputs[1] = 1; n.inputs[2] = 2;
        memcpy(n.p, params, sizeof(params));
        a = malloc((size_t)x.count * sizeof(float));
        weights = malloc((size_t)w.count * sizeof(float));
        bias = malloc((size_t)c * sizeof(float));
        reference = malloc((size_t)y.count * sizeof(float));
        candidate = malloc((size_t)y.count * sizeof(float));
        adjacent = malloc(((size_t)x.count + y.count) * sizeof(float));
        if (!a || !weights || !bias || !reference || !candidate || !adjacent) {
            fprintf(stderr, "allocation failed\n"); return 2;
        }
        before = checks;
        require_check(ekws_3x1_matches(&x,&w,&y,&n,a,weights,bias,candidate), "fixed shape accepted");
        require_rejected(ekws_3x1_matches(&x,&w,&y,&n,a,weights,bias,a), "exact input alias rejected");
        require_rejected(ekws_3x1_matches(&x,&w,&y,&n,a,weights,bias,a+1), "partial input alias rejected");
        require_rejected(ekws_3x1_matches(&x,&w,&y,&n,a,weights,bias,a+x.count-1), "one-float overlap rejected");
        require_rejected(ekws_3x1_matches(&x,&w,&y,&n,a,weights,bias,weights), "weight alias rejected");
        require_rejected(ekws_3x1_matches(&x,&w,&y,&n,a,weights,bias,bias), "bias alias rejected");
        require_check(ekws_3x1_matches(&x,&w,&y,&n,adjacent,weights,bias,adjacent+x.count), "adjacent ranges accepted");
        n.inputs[2] = EF_NONE;
        require_check(ekws_3x1_matches(&x,&w,&y,&n,a,weights,NULL,candidate), "explicit absent bias accepted");
        require_rejected(ekws_3x1_matches(&x,&w,&y,&n,a,weights,bias,candidate), "EF_NONE with bias pointer rejected");
        n.inputs[2] = 2;
        require_rejected(ekws_3x1_matches(&x,&w,&y,&n,a,weights,NULL,candidate), "missing required bias pointer rejected");
        n.input_count = 2;
        require_rejected(ekws_3x1_matches(&x,&w,&y,&n,a,weights,bias,candidate), "extra bias pointer rejected");
        n.input_count = 4;
        require_rejected(ekws_3x1_matches(&x,&w,&y,&n,a,weights,bias,candidate), "extra input rejected");
        n.input_count = 3;
        n.p[8] = 1;
        require_rejected(ekws_3x1_matches(&x,&w,&y,&n,a,weights,bias,candidate), "padding rejected");
        n.p[8] = 0; n.p[4] = 2;
        require_rejected(ekws_3x1_matches(&x,&w,&y,&n,a,weights,bias,candidate), "stride rejected");
        n.p[4] = 1; n.p[1] = 2;
        require_rejected(ekws_3x1_matches(&x,&w,&y,&n,a,weights,bias,candidate), "groups rejected");
        n.p[1] = 1; n.p[6] = 2;
        require_rejected(ekws_3x1_matches(&x,&w,&y,&n,a,weights,bias,candidate), "dilation rejected");
        n.p[6] = 1; x.rank = 3;
        require_rejected(ekws_3x1_matches(&x,&w,&y,&n,a,weights,bias,candidate), "rank rejected");
        x.rank = 4; --x.count;
        require_rejected(ekws_3x1_matches(&x,&w,&y,&n,a,weights,bias,candidate), "wrong count rejected");
        ++x.count;
        guard_checks += checks - before;
        for (pattern = 0; pattern < 6; ++pattern) {
            const float *use_bias = pattern == 2 ? NULL : bias;
            n.input_count = use_bias ? 3 : 2;
            for (i = 0; i < x.count; ++i) a[i] = test_value(i, pattern, 0);
            for (i = 0; i < w.count; ++i) weights[i] = test_value(i + 7, pattern, 1);
            for (i = 0; i < c; ++i)
                bias[i] = pattern == 1 ? from_bits(UINT32_C(0x80000000)) : (float)((int)(i % 5) - 2) * 0.25f;
            require_check(ekws_3x1_matches(&x,&w,&y,&n,a,weights,use_bias,candidate), "numeric case matches");
            require_check(spatial_run(&x,&w,&y,&n,a,weights,use_bias,reference) == EF_OK, "pinned original oracle completed");
            ekws_3x1_ordered(&x,&y,a,weights,use_bias,candidate);
            require_check(finite_values(reference,y.count) == finite_values(candidate,y.count), "nonfinite classification equal");
            for (i = 0; i < y.count; ++i) {
                if (!equal_result(reference[i], candidate[i])) {
                    fprintf(stderr, "shape=%u pattern=%u output=%zu unequal\n", shape, pattern, i);
                    ++failures; break;
                }
                ++compared;
                if (!isfinite(reference[i])) ++nonfinite_values;
            }
            ++pairs;
        }
        /* Node 62 geometry must stay on the generic path. */
        descriptor(&x,96,3,1,EF_WORK); descriptor(&y,96,1,1,EF_WORK);
        descriptor(&w,96,3,1,EF_CONSTANT); w.dims[0]=96; w.count *= 96;
        n.input_count=2;
        before = checks;
        require_rejected(ekws_3x1_matches(&x,&w,&y,&n,a,weights,NULL,candidate), "node62 fallback");
        guard_checks += checks - before;
        free(a); free(weights); free(bias); free(reference); free(candidate); free(adjacent);
    }
    printf("{\"schema\":\"english-ordered-3x1-numeric-v1\",\"passed\":%s,"
           "\"synthetic_kernel_checks\":true,\"acoustic_evaluation\":false,"
           "\"full_model_executed\":false,\"numeric_pairs\":%u,\"cases\":%u,"
           "\"optimized_cases\":%u,\"fallback_cases\":0,"
           "\"fallback_matcher_checks\":%u,\"guard_checks\":%u,\"checks\":%u,"
           "\"compared_values\":%zu,\"nonfinite_values\":%zu,\"failures\":%u,"
           "\"finite_signedzero_and_infinity_comparison\":\"bitwise\","
           "\"nan_comparison\":\"classification_only\",\"FLT_EVAL_METHOD\":%d,"
           "\"fallback_scope\":\"matcher rejection checks; unchanged fallback kernel not rerun by this checker\"}\n",
           failures ? "false" : "true", pairs, pairs, pairs, fallback_matcher_checks,
           guard_checks, checks, compared, nonfinite_values, failures, FLT_EVAL_METHOD);
    return failures ? 1 : 0;
}
