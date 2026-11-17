#ifndef EKWS_ORDERED_3X1_CANDIDATE_H
#define EKWS_ORDERED_3X1_CANDIDATE_H

/* Experimental fixed-shape dispatch only. The unmodified generic spatial_run
 * remains the fallback. Compile both variants with -fno-fast-math and
 * -ffp-contract=off; this file deliberately uses no SIMD intrinsics or FMA.
 * ef_tensor, ef_node and EF_* are defined by the pinned fixed_frontend.h. */
#include <float.h>
#include <stddef.h>
#include <stdint.h>

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "The ordered 3x1 candidate requires strict floating-point compilation"
#endif
#if defined(__ASSOCIATIVE_MATH__) || defined(__NO_SIGNED_ZEROS__)
#error "Reassociation and unsigned-zero assumptions are not allowed"
#endif

/* Byte interval comparison uses the flat-address uintptr_t ABI of the reviewed
 * GNU/Clang hosts. Integer overflow or an empty/NULL span means no match. */
static int ekws_3x1_disjoint(const void *a, size_t a_count,
                             const void *b, size_t b_count) {
    uintptr_t a_begin, b_begin, a_bytes, b_bytes;
    if (!a || !b || !a_count || !b_count ||
        a_count > UINTPTR_MAX / sizeof(float) ||
        b_count > UINTPTR_MAX / sizeof(float)) return 0;
    a_begin = (uintptr_t)a;
    b_begin = (uintptr_t)b;
    a_bytes = (uintptr_t)a_count * sizeof(float);
    b_bytes = (uintptr_t)b_count * sizeof(float);
    if (a_begin > UINTPTR_MAX - a_bytes || b_begin > UINTPTR_MAX - b_bytes) return 0;
    return a_begin + a_bytes <= b_begin || b_begin + b_bytes <= a_begin;
}

static int ekws_3x1_matches(const ef_tensor *x, const ef_tensor *w,
                            const ef_tensor *y, const ef_node *n,
                            const float *a, const float *weights,
                            const float *bias, float *out) {
    /* IC, IH, IW, OC: the nine fixed embedding nodes with output plane >= 4.
     * Node 62 ([1,96,3,1] -> [1,96,1,1]) intentionally has no entry. */
    static const uint32_t shapes[9][4] = {
        {24, 74, 32, 24}, {48, 36, 16, 48}, {48, 34, 16, 48},
        {72, 32, 8, 72}, {72, 30, 8, 72}, {96, 14, 4, 96},
        {96, 12, 4, 96}, {96, 10, 2, 96}, {96, 8, 2, 96}
    };
    static const int32_t params[16] = {2, 1, 3, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
    size_t i;
    uint32_t ic, ih, iw, oc;
    int found = 0;
    if (FLT_EVAL_METHOD != 0) return 0;
    if (!x || !w || !y || !n || !a || !weights || !out) return 0;
    if (n->op != EF_CONV || (n->input_count != 2 && n->input_count != 3) ||
        (n->input_count == 2 && bias) ||
        (n->input_count == 3 && ((n->inputs[2] == EF_NONE) != (bias == NULL)))) return 0;
    if (x->rank != 4 || w->rank != 4 || y->rank != 4 ||
        x->storage != EF_WORK || w->storage != EF_CONSTANT || y->storage != EF_WORK ||
        x->dims[0] != 1 || y->dims[0] != 1) return 0;
    for (i = 0; i < 16; ++i) if (n->p[i] != params[i]) return 0;
    ic = x->dims[1]; ih = x->dims[2]; iw = x->dims[3]; oc = y->dims[1];
    for (i = 0; i < 9; ++i)
        if (ic == shapes[i][0] && ih == shapes[i][1] &&
            iw == shapes[i][2] && oc == shapes[i][3]) found = 1;
    if (!found || y->dims[2] != ih - 2 || y->dims[3] != iw ||
        w->dims[0] != oc || w->dims[1] != ic || w->dims[2] != 3 || w->dims[3] != 1) return 0;
    /* Whitelist dimensions make these uint64_t products bounded on 32-bit too. */
    if ((uint64_t)x->count != (uint64_t)ic * ih * iw ||
        (uint64_t)w->count != (uint64_t)oc * ic * 3 ||
        (uint64_t)y->count != (uint64_t)oc * (ih - 2) * iw) return 0;
    if (((uint64_t)(ih - 2) * iw) % 4 != 0) return 0;
    if (!ekws_3x1_disjoint(out, y->count, a, x->count) ||
        !ekws_3x1_disjoint(out, y->count, weights, w->count) ||
        (bias && !ekws_3x1_disjoint(out, y->count, bias, oc))) return 0;
    return 1;
}

/* Call only after ekws_3x1_matches. The restricted output is disjoint from all
 * reads. Read-only input/weights/bias are not required to be mutually disjoint.
 * Keep +0, ci order, kh=0,1,2 order and the optional final bias addition. */
static void ekws_3x1_ordered(const ef_tensor *x, const ef_tensor *y,
                             const float *a, const float *weights,
                             const float *bias, float *restrict out) {
    const size_t ic = x->dims[1], ih = x->dims[2], width = x->dims[3];
    const size_t oc = y->dims[1], in_plane = ih * width;
    const size_t out_plane = (ih - 2) * width;
    size_t co, ci, pos;
    for (co = 0; co < oc; ++co) {
        float *dst = out + co * out_plane;
        for (pos = 0; pos < out_plane; ++pos) dst[pos] = 0.0f;
        for (ci = 0; ci < ic; ++ci) {
            const float *src = a + ci * in_plane;
            const float *k = weights + (co * ic + ci) * 3;
            /* Four independent output positions expose legal SLP/loop SIMD.
             * The matcher proves a multiple of four; each accumulator keeps
             * its original ci/kh order, without a horizontal reduction. */
            for (pos = 0; pos < out_plane; pos += 4) {
                float s0 = dst[pos], s1 = dst[pos + 1];
                float s2 = dst[pos + 2], s3 = dst[pos + 3];
                s0 += src[pos] * k[0];
                s1 += src[pos + 1] * k[0];
                s2 += src[pos + 2] * k[0];
                s3 += src[pos + 3] * k[0];
                s0 += src[pos + width] * k[1];
                s1 += src[pos + width + 1] * k[1];
                s2 += src[pos + width + 2] * k[1];
                s3 += src[pos + width + 3] * k[1];
                s0 += src[pos + 2 * width] * k[2];
                s1 += src[pos + 2 * width + 1] * k[2];
                s2 += src[pos + 2 * width + 2] * k[2];
                s3 += src[pos + 2 * width + 3] * k[2];
                dst[pos] = s0; dst[pos + 1] = s1;
                dst[pos + 2] = s2; dst[pos + 3] = s3;
            }
        }
        if (bias)
            for (pos = 0; pos < out_plane; ++pos) dst[pos] += bias[co];
    }
}

#endif
