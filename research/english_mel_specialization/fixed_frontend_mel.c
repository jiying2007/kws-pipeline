#include "fixed_frontend.h"
#include <float.h>
#include <math.h>
#include <string.h>

#ifdef __FAST_MATH__
#error "fixed_frontend requires -fno-fast-math -ffp-contract=off"
#endif
_Static_assert(sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24,
               "IEEE754 binary32 required");

static int finite_values(const float *x, size_t n) {
    size_t i;
    if (!x) return 0;
    for (i = 0; i < n; ++i) if (!isfinite(x[i])) return 0;
    return 1;
}

static int same_shape(const ef_tensor *a, const ef_tensor *b) {
    uint32_t j;
    if (a->rank != b->rank) return 0;
    for (j = 0; j < a->rank; ++j) if (a->dims[j] != b->dims[j]) return 0;
    return 1;
}

static int broadcastable(const ef_tensor *a, const ef_tensor *out) {
    uint32_t j, shift;
    if (a->rank > out->rank) return 0;
    shift = out->rank - a->rank;
    for (j = 0; j < a->rank; ++j)
        if (a->dims[j] != 1 && a->dims[j] != out->dims[j + shift]) return 0;
    return 1;
}

static int spatial_valid(const ef_tensor *x, const ef_tensor *w,
                         const ef_tensor *bias, const ef_tensor *y,
                         const ef_node *n) {
    uint32_t sr, j;
    const int32_t *p = n->p;
    int pool = n->op == EF_MAX_POOL;
    if (x->rank != 3 && x->rank != 4) return 0;
    sr = x->rank - 2;
    if (y->rank != x->rank || y->dims[0] != x->dims[0] || p[0] != (int32_t)sr ||
        p[1] <= 0 || p[12] < 0 || p[12] > 1) return 0;
    if (pool) {
        if (p[1] != 1 || y->dims[1] != x->dims[1]) return 0;
    } else {
        if (!w || w->rank != x->rank || p[12] ||
            (uint64_t)w->dims[1] * (uint32_t)p[1] != x->dims[1] ||
            w->dims[0] % (uint32_t)p[1] || y->dims[1] != w->dims[0] ||
            (bias && (bias->rank != 1 || bias->dims[0] != w->dims[0]))) return 0;
    }
    for (j = 0; j < 2; ++j) {
        int64_t in = j < sr ? x->dims[j + 2] : 1;
        int64_t kernel = p[2 + j], stride = p[4 + j], dilation = p[6 + j];
        int64_t before = p[8 + j], after = p[10 + j], numerator, out;
        if (kernel <= 0 || stride <= 0 || dilation <= 0 || before < 0 || after < 0) return 0;
        if (j >= sr && (kernel != 1 || stride != 1 || dilation != 1 || before || after)) return 0;
        if (!pool && j < sr && kernel != w->dims[j + 2]) return 0;
        numerator = in + before + after - (dilation * (kernel - 1) + 1);
        numerator += p[12] ? stride - 1 : 0;
        /* C division truncates; floor division is required for negative values. */
        out = numerator >= 0 ? numerator / stride + 1 : -((-numerator + stride - 1) / stride) + 1;
        if (pool && p[12] && (out - 1) * stride >= in + before) --out;
        if (out <= 0 || (j < sr && out != y->dims[j + 2]) || (j >= sr && out != 1)) return 0;
    }
    return 1;
}

size_t ef_graph_workspace_bytes(const ef_graph *g) {
    if (!g || (uint64_t)g->workspace_floats * sizeof(float) > SIZE_MAX) return 0;
    return (size_t)g->workspace_floats * sizeof(float);
}

int ef_graph_validate(const ef_graph *g) {
    uint32_t i, j;
    if (!g || !g->tensors || !g->tensor_count || !g->nodes ||
        g->input >= g->tensor_count || g->output >= g->tensor_count ||
        (g->constant_count && !g->constants)) return EF_INVALID;
    if (g->tensors[g->input].storage != EF_INPUT) return EF_INVALID;
    for (i = 0; i < g->tensor_count; ++i) {
        const ef_tensor *t = &g->tensors[i];
        uint64_t count = 1, end = (uint64_t)t->offset + t->count;
        if (t->rank > EF_MAX_RANK || !t->count) return EF_INVALID;
        for (j = 0; j < t->rank; ++j) {
            if (!t->dims[j] || count > UINT32_MAX / t->dims[j]) return EF_INVALID;
            count *= t->dims[j];
        }
        if (count != t->count) return EF_INVALID;
        if (t->storage == EF_INPUT) {
            if (t->offset || t->count != g->tensors[g->input].count) return EF_INVALID;
        } else if (t->storage == EF_CONSTANT) {
            if (end > g->constant_count) return EF_INVALID;
        } else if (t->storage == EF_WORK) {
            if (end > g->workspace_floats) return EF_INVALID;
        } else return EF_INVALID;
    }
    for (i = 0; i < g->node_count; ++i) {
        const ef_node *n = &g->nodes[i];
        const ef_tensor *x, *y, *b = NULL;
        if (!n->input_count || n->input_count > EF_MAX_INPUTS ||
            n->inputs[0] >= g->tensor_count || n->output >= g->tensor_count) return EF_INVALID;
        x = &g->tensors[n->inputs[0]];
        y = &g->tensors[n->output];
        if (y->storage != EF_WORK) return EF_INVALID;
        for (j = 0; j < n->input_count; ++j) {
            if (n->inputs[j] == EF_NONE && ((n->op == EF_CLIP && j > 0) ||
                                          (n->op == EF_CONV && j == 2))) continue;
            if (n->inputs[j] >= g->tensor_count) return EF_INVALID;
        }
        if (n->input_count > 1 && n->inputs[1] != EF_NONE) b = &g->tensors[n->inputs[1]];
        if (n->op >= EF_ADD && n->op <= EF_MAX) {
            if (n->op != EF_MAX && n->input_count != 2) return EF_INVALID;
            for (j = 0; j < n->input_count; ++j)
                if (!broadcastable(&g->tensors[n->inputs[j]], y)) return EF_INVALID;
        } else if (n->op == EF_LOG || n->op == EF_LEAKY_RELU) {
            if (n->input_count != 1 || !same_shape(x, y) || !isfinite(n->alpha)) return EF_INVALID;
        } else if (n->op == EF_CLIP) {
            if (n->input_count != 3 || !same_shape(x, y)) return EF_INVALID;
            for (j = 1; j < 3; ++j)
                if (n->inputs[j] != EF_NONE && g->tensors[n->inputs[j]].rank != 0) return EF_INVALID;
        } else if (n->op == EF_TRANSPOSE) {
            uint32_t mask = 0;
            if (n->input_count != 1 || x->rank != y->rank) return EF_INVALID;
            for (j = 0; j < x->rank; ++j) {
                int32_t p = n->p[j];
                if (p < 0 || p >= (int32_t)x->rank || mask & (1u << p) || y->dims[j] != x->dims[p]) return EF_INVALID;
                mask |= 1u << p;
            }
        } else if (n->op == EF_REDUCE_MAX) {
            uint32_t k = 0, axes = (uint32_t)n->p[0];
            if (n->input_count != 1 || n->p[0] < 0 || axes >> x->rank ||
                (n->p[1] != 0 && n->p[1] != 1)) return EF_INVALID;
            for (j = 0; j < x->rank; ++j) {
                if (axes & (1u << j)) {
                    if (n->p[1] && (k >= y->rank || y->dims[k++] != 1)) return EF_INVALID;
                } else if (k >= y->rank || y->dims[k++] != x->dims[j]) return EF_INVALID;
            }
            if (k != y->rank) return EF_INVALID;
        } else if (n->op == EF_MATMUL) {
            uint32_t ak, bk, m, cols, ar, br, rank, dims[EF_MAX_RANK], yr = 0;
            if (n->input_count != 2 || !b || !x->rank || !b->rank) return EF_INVALID;
            ak = x->dims[x->rank - 1];
            bk = b->rank == 1 ? b->dims[0] : b->dims[b->rank - 2];
            if (ak != bk) return EF_INVALID;
            m = x->rank == 1 ? 1 : x->dims[x->rank - 2];
            cols = b->rank == 1 ? 1 : b->dims[b->rank - 1];
            ar = x->rank > 2 ? x->rank - 2 : 0;
            br = b->rank > 2 ? b->rank - 2 : 0;
            rank = ar > br ? ar : br;
            for (j = 0; j < rank; ++j) {
                uint32_t ad = j < rank - ar ? 1 : x->dims[j - (rank - ar)];
                uint32_t bd = j < rank - br ? 1 : b->dims[j - (rank - br)];
                if (ad != bd && ad != 1 && bd != 1) return EF_INVALID;
                dims[yr++] = ad > bd ? ad : bd;
            }
            if (x->rank != 1) dims[yr++] = m;
            if (b->rank != 1) dims[yr++] = cols;
            if (yr > EF_MAX_RANK || y->rank != yr) return EF_INVALID;
            for (j = 0; j < yr; ++j) if (y->dims[j] != dims[j]) return EF_INVALID;
        } else if (n->op == EF_CONV || n->op == EF_MAX_POOL) {
            const ef_tensor *bias = n->input_count > 2 && n->inputs[2] != EF_NONE ?
                                     &g->tensors[n->inputs[2]] : NULL;
            if ((n->op == EF_CONV && (n->input_count < 2 || n->input_count > 3)) ||
                (n->op == EF_MAX_POOL && n->input_count != 1) || !spatial_valid(x, b, bias, y, n)) return EF_INVALID;
        } else return EF_INVALID;
    }
    return EF_OK;
}

static const float *tensor_data(const ef_graph *g, uint32_t id,
                                const float *input, float *work) {
    const ef_tensor *t;
    if (id == EF_NONE) return NULL;
    t = &g->tensors[id];
    if (t->storage == EF_INPUT) return input;
    if (t->storage == EF_CONSTANT) return g->constants + t->offset;
    return work + t->offset;
}

static uint32_t broadcast_index(uint32_t flat, const ef_tensor *out, const ef_tensor *in) {
    uint32_t index = 0, stride = 1, j = out->rank;
    while (j) {
        uint32_t c;
        --j;
        c = flat % out->dims[j];
        flat /= out->dims[j];
        if (j + in->rank >= out->rank) {
            uint32_t d = in->dims[j + in->rank - out->rank];
            if (d != 1) index += c * stride;
            stride *= d;
        }
    }
    return index;
}

static void transpose(const ef_tensor *x, const ef_tensor *y, const ef_node *n,
                      const float *a, float *out) {
    uint32_t strides[EF_MAX_RANK], stride = 1, i, j;
    for (j = x->rank; j > 0;) { --j; strides[j] = stride; stride *= x->dims[j]; }
    for (i = 0; i < y->count; ++i) {
        uint32_t flat = i, src = 0;
        for (j = y->rank; j > 0;) {
            --j;
            src += (flat % y->dims[j]) * strides[n->p[j]];
            flat /= y->dims[j];
        }
        out[i] = a[src];
    }
}

static void reduce_max(const ef_tensor *x, const ef_tensor *y, const ef_node *n,
                       const float *a, float *out) {
    uint32_t i, j;
    for (i = 0; i < y->count; ++i) out[i] = -FLT_MAX;
    for (i = 0; i < x->count; ++i) {
        uint32_t flat = i, dst = 0, stride = 1;
        for (j = x->rank; j > 0;) {
            uint32_t c;
            --j;
            c = flat % x->dims[j]; flat /= x->dims[j];
            if (!(n->p[0] & (1u << j))) { dst += c * stride; stride *= x->dims[j]; }
        }
        if (a[i] > out[dst]) out[dst] = a[i];
    }
}

static void matmul(const ef_tensor *x, const ef_tensor *b, const ef_tensor *y,
                   const float *a, const float *v, float *out) {
    uint32_t m = x->rank == 1 ? 1 : x->dims[x->rank - 2], k = x->dims[x->rank - 1];
    uint32_t cols = b->rank == 1 ? 1 : b->dims[b->rank - 1];
    uint32_t ar = x->rank > 2 ? x->rank - 2 : 0, br = b->rank > 2 ? b->rank - 2 : 0;
    uint32_t rank = ar > br ? ar : br, dims[EF_MAX_RANK], batches = y->count / m / cols;
    uint32_t batch, row, col, z, j;
    for (j = 0; j < rank; ++j) {
        uint32_t ad = j < rank - ar ? 1 : x->dims[j - (rank - ar)];
        uint32_t bd = j < rank - br ? 1 : b->dims[j - (rank - br)];
        dims[j] = ad > bd ? ad : bd;
    }
    for (batch = 0; batch < batches; ++batch) {
        uint32_t flat = batch, ai = 0, bi = 0, ast = m * k, bst = k * cols;
        for (j = rank; j > 0;) {
            uint32_t c; --j;
            c = flat % dims[j]; flat /= dims[j];
            if (j >= rank - ar) {
                uint32_t d = x->dims[j - (rank - ar)];
                if (d != 1) ai += c * ast;
                ast *= d;
            }
            if (j >= rank - br) {
                uint32_t d = b->dims[j - (rank - br)];
                if (d != 1) bi += c * bst;
                bst *= d;
            }
        }
        for (row = 0; row < m; ++row) for (col = 0; col < cols; ++col) {
            float sum = 0.0f;
            for (z = 0; z < k; ++z) sum += a[ai + row * k + z] * v[bi + z * cols + col];
            out[(batch * m + row) * cols + col] = sum;
        }
    }
}

/* BEGIN FIXED MEL CONV1D SPECIALIZATION
 * Exact unpadded [1,1,16000] * [257,1,512] -> [1,257,97].
 * Only index/bounds work is removed: each output keeps the original 512
 * scalar multiply-then-add operations in tap order, starting from +0.0f.
 * No bias, reassociation, additional accumulator, or caller memory change.
 */
static int fixed_mel_conv1d_matches(const ef_tensor *x, const ef_tensor *w,
                                   const ef_tensor *y, const ef_node *n,
                                   const float *bias) {
    static const int32_t parameters[16] = {
        1, 1, 512, 1, 160, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0
    };
    return n->op == EF_CONV && n->input_count == 2 && !bias && w &&
           x->rank == 3 && w->rank == 3 && y->rank == 3 &&
           x->dims[0] == 1 && x->dims[1] == 1 && x->dims[2] == 16000 &&
           w->dims[0] == 257 && w->dims[1] == 1 && w->dims[2] == 512 &&
           y->dims[0] == 1 && y->dims[1] == 257 && y->dims[2] == 97 &&
           memcmp(n->p, parameters, sizeof(parameters)) == 0;
}

static void fixed_mel_conv1d_ordered(const float *a, const float *weights,
                                    float *out) {
    uint32_t channel, frame, tap;
    for (channel = 0; channel < 257; ++channel) {
        const float *kernel = weights + channel * 512;
        for (frame = 0; frame < 97; ++frame) {
            const float *window = a + frame * 160;
            float sum = 0.0f;
            for (tap = 0; tap < 512; ++tap) {
                float value = window[tap];
                sum += value * kernel[tap];
            }
            out[channel * 97 + frame] = sum;
        }
    }
}
/* END FIXED MEL CONV1D SPECIALIZATION */

static int spatial_run(const ef_tensor *x, const ef_tensor *w, const ef_tensor *y,
                       const ef_node *n, const float *a, const float *weights,
                       const float *bias, float *out) {
    const int32_t *p = n->p;
    /* BEGIN FIXED MEL CONV1D DISPATCH */
    if (fixed_mel_conv1d_matches(x, w, y, n, bias)) {
        fixed_mel_conv1d_ordered(a, weights, out);
        return EF_OK;
    }
    /* END FIXED MEL CONV1D DISPATCH */
    uint32_t ih = x->dims[2], iw = x->rank == 4 ? x->dims[3] : 1;
    uint32_t oh = y->dims[2], ow = y->rank == 4 ? y->dims[3] : 1;
    uint32_t ic = x->dims[1], oc = y->dims[1], batch, channel, h, col;
    int pool = n->op == EF_MAX_POOL;
    uint32_t per_group = pool ? 1 : w->dims[1], out_group = pool ? 1 : oc / (uint32_t)p[1];
    for (batch = 0; batch < x->dims[0]; ++batch) for (channel = 0; channel < oc; ++channel)
    for (h = 0; h < oh; ++h) for (col = 0; col < ow; ++col) {
        float sum = pool ? -FLT_MAX : 0.0f;
        uint32_t c;
        int32_t kh, kw;
        int found = 0;
        for (c = 0; c < per_group; ++c) for (kh = 0; kh < p[2]; ++kh) for (kw = 0; kw < p[3]; ++kw) {
            int64_t src_h = (int64_t)h * p[4] - p[8] + (int64_t)kh * p[6];
            int64_t src_w = (int64_t)col * p[5] - p[9] + (int64_t)kw * p[7];
            uint32_t src_channel = pool ? channel : channel / out_group * per_group + c;
            if (src_h >= 0 && src_h < ih && src_w >= 0 && src_w < iw) {
                float value = a[((batch * ic + src_channel) * ih + (uint32_t)src_h) * iw + (uint32_t)src_w];
                if (pool) { if (value > sum) sum = value; found = 1; }
                else sum += value * weights[((channel * per_group + c) * (uint32_t)p[2] + (uint32_t)kh) *
                                            (uint32_t)p[3] + (uint32_t)kw];
            }
        }
        if (pool && !found) return EF_INVALID;
        if (!pool && bias) sum += bias[channel];
        out[((batch * oc + channel) * oh + h) * ow + col] = sum;
    }
    return EF_OK;
}

int ef_graph_run(const ef_graph *g, const float *input, float *work, size_t work_count,
                 const float **output, ef_trace_fn trace, void *trace_context) {
    uint32_t ni;
    int status;
    if (output) *output = NULL;
    if (!output || ef_graph_validate(g) != EF_OK || !input) return EF_INVALID;
    if (work_count < g->workspace_floats || (g->workspace_floats && !work)) return EF_WORKSPACE;
    if (!finite_values(input, g->tensors[g->input].count)) return EF_NONFINITE;
    for (ni = 0; ni < g->node_count; ++ni) {
        const ef_node *n = &g->nodes[ni];
        const ef_tensor *x = &g->tensors[n->inputs[0]], *y = &g->tensors[n->output];
        const float *a = tensor_data(g, n->inputs[0], input, work);
        const float *b = n->input_count > 1 ? tensor_data(g, n->inputs[1], input, work) : NULL;
        float *out = work + y->offset;
        uint32_t i, j;
        status = EF_OK;
        if (n->op >= EF_ADD && n->op <= EF_MAX) {
            for (i = 0; i < y->count; ++i) {
                float v = a[broadcast_index(i, y, x)];
                for (j = 1; j < n->input_count; ++j) {
                    const ef_tensor *t = &g->tensors[n->inputs[j]];
                    float q = tensor_data(g, n->inputs[j], input, work)[broadcast_index(i, y, t)];
                    switch (n->op) {
                        case EF_ADD: v += q; break;
                        case EF_SUB: v -= q; break;
                        case EF_MUL: v *= q; break;
                        case EF_DIV: v /= q; break;
                        case EF_POW: v = powf(v, q); break;
                        case EF_MAX: if (q > v) v = q; break;
                        default: return EF_INVALID;
                    }
                }
                out[i] = v;
            }
        } else if (n->op == EF_LOG) {
            for (i = 0; i < y->count; ++i) out[i] = logf(a[i]);
        } else if (n->op == EF_LEAKY_RELU) {
            for (i = 0; i < y->count; ++i) out[i] = a[i] >= 0.0f ? a[i] : n->alpha * a[i];
        } else if (n->op == EF_CLIP) {
            const float *upper = tensor_data(g, n->inputs[2], input, work);
            float low = b ? b[0] : -FLT_MAX, high = upper ? upper[0] : FLT_MAX;
            if (!isfinite(low) || !isfinite(high)) return EF_NONFINITE;
            for (i = 0; i < y->count; ++i) {
                float v = a[i] < low ? low : a[i];
                out[i] = v > high ? high : v;
            }
        } else if (n->op == EF_TRANSPOSE) transpose(x, y, n, a, out);
        else if (n->op == EF_REDUCE_MAX) reduce_max(x, y, n, a, out);
        else if (n->op == EF_MATMUL) matmul(x, &g->tensors[n->inputs[1]], y, a, b, out);
        else if (n->op == EF_CONV || n->op == EF_MAX_POOL) {
            const float *bias = n->input_count > 2 ? tensor_data(g, n->inputs[2], input, work) : NULL;
            status = spatial_run(x, b ? &g->tensors[n->inputs[1]] : NULL, y, n, a, b, bias, out);
        } else return EF_INVALID;
        if (status != EF_OK) return status;
        if (!finite_values(out, y->count)) return EF_NONFINITE;
        if (trace && trace(trace_context, ni, y, out)) return EF_CALLBACK;
    }
    *output = tensor_data(g, g->output, input, work);
    if (!finite_values(*output, g->tensors[g->output].count)) { *output = NULL; return EF_NONFINITE; }
    return EF_OK;
}

static int expected_shape(const ef_tensor *t, uint32_t rank, const uint32_t *dims) {
    uint32_t i;
    if (t->rank != rank) return 0;
    for (i = 0; i < rank; ++i) if (t->dims[i] != dims[i]) return 0;
    return 1;
}

static int feature_graphs_valid(const ef_graph *mel, const ef_graph *embedding) {
    const uint32_t mi[] = {1, EF_PCM_WINDOW}, mo[] = {1, 1, EF_MEL_FRAMES, EF_MEL_BINS};
    const uint32_t ei[] = {1, EF_EMBED_FRAMES, EF_MEL_BINS, 1}, eo[] = {1, 1, 1, EF_EMBED_DIM};
    return ef_graph_validate(mel) == EF_OK && ef_graph_validate(embedding) == EF_OK &&
           expected_shape(&mel->tensors[mel->input], 2, mi) &&
           expected_shape(&mel->tensors[mel->output], 4, mo) &&
           expected_shape(&embedding->tensors[embedding->input], 4, ei) &&
           expected_shape(&embedding->tensors[embedding->output], 4, eo);
}

size_t ef_frontend_workspace_bytes(const ef_graph *mel, const ef_graph *embedding) {
    uint64_t count;
    if (!feature_graphs_valid(mel, embedding)) return 0;
    count = mel->workspace_floats > embedding->workspace_floats ?
            mel->workspace_floats : embedding->workspace_floats;
    count += EF_PCM_WINDOW + EF_EMBED_FRAMES * EF_MEL_BINS;
    return count * sizeof(float) <= SIZE_MAX ? (size_t)count * sizeof(float) : 0;
}

int ef_frontend_init(ef_frontend *s, const ef_graph *mel, const ef_graph *embedding,
                     float *workspace, size_t count) {
    size_t bytes;
    if (!s || !workspace || !(bytes = ef_frontend_workspace_bytes(mel, embedding))) return EF_INVALID;
    if (count < bytes / sizeof(float)) return EF_WORKSPACE;
    memset(s, 0, sizeof(*s));
    s->mel = mel; s->embedding = embedding; s->workspace = workspace; s->workspace_floats = count;
    s->initialized = 1; s->next_end = EF_PCM_WINDOW;
    return EF_OK;
}

void ef_frontend_reset(ef_frontend *s) {
    if (!s || !s->initialized) return;
    memset(s->ring, 0, sizeof(s->ring));
    s->write = s->filled = 0; s->observed = 0; s->next_end = EF_PCM_WINDOW; s->failed = 0;
}

int ef_frontend_discontinuity(ef_frontend *s, uint64_t next) {
    if (!s || !s->initialized) return EF_INVALID;
    if (next < s->observed || next > UINT64_MAX - EF_PCM_WINDOW) return EF_CLOCK;
    memset(s->ring, 0, sizeof(s->ring));
    s->write = s->filled = 0; s->observed = next; s->next_end = next + EF_PCM_WINDOW; s->failed = 0;
    return EF_OK;
}

static int emit_embedding(ef_frontend *s, ef_embedding_fn callback, void *context) {
    float *pcm = s->workspace, *normalized = pcm + EF_PCM_WINDOW;
    float *work = normalized + EF_EMBED_FRAMES * EF_MEL_BINS;
    size_t work_count = s->workspace_floats - EF_PCM_WINDOW - EF_EMBED_FRAMES * EF_MEL_BINS;
    const float *mel, *embedding;
    uint32_t i;
    int status;
    for (i = 0; i < EF_PCM_WINDOW; ++i) pcm[i] = (float)s->ring[(s->write + i) % EF_PCM_WINDOW];
    status = ef_graph_run(s->mel, pcm, work, work_count, &mel, NULL, NULL);
    if (status != EF_OK) return status;
    if (s->trace && s->trace(s->trace_context, 1, s->observed, mel, EF_MEL_FRAMES * EF_MEL_BINS)) return EF_CALLBACK;
    for (i = 0; i < EF_EMBED_FRAMES * EF_MEL_BINS; ++i)
        normalized[i] = mel[(EF_MEL_FRAMES - EF_EMBED_FRAMES) * EF_MEL_BINS + i] / 10.0f + 2.0f;
    if (s->trace && s->trace(s->trace_context, 2, s->observed, normalized, EF_EMBED_FRAMES * EF_MEL_BINS)) return EF_CALLBACK;
    status = ef_graph_run(s->embedding, normalized, work, work_count, &embedding, NULL, NULL);
    if (status != EF_OK) return status;
    if (s->trace && s->trace(s->trace_context, 3, s->observed, embedding, EF_EMBED_DIM)) return EF_CALLBACK;
    return callback(context, s->observed, embedding) ? EF_CALLBACK : EF_OK;
}

int ef_frontend_feed(ef_frontend *s, const int16_t *pcm, size_t count, uint64_t first,
                     ef_embedding_fn callback, void *context, size_t *consumed) {
    size_t i;
    int status;
    if (consumed) *consumed = 0;
    if (!s || !s->initialized || !consumed || s->failed || (count && (!pcm || !callback))) return EF_INVALID;
    if (first != s->observed || count > UINT64_MAX - s->observed ||
        (count && s->observed + count > UINT64_MAX - EF_HOP)) return EF_CLOCK;
    for (i = 0; i < count; ++i) {
        s->ring[s->write] = pcm[i]; s->write = (s->write + 1) % EF_PCM_WINDOW;
        if (s->filled < EF_PCM_WINDOW) ++s->filled;
        ++s->observed; *consumed = i + 1;
        if (s->observed == s->next_end) {
            if (s->filled != EF_PCM_WINDOW) { s->failed = 1; return EF_INVALID; }
            status = emit_embedding(s, callback, context);
            if (status != EF_OK) { s->failed = 1; return status; }
            s->next_end += EF_HOP;
        }
    }
    return EF_OK;
}
