/* SPDX-License-Identifier: Apache-2.0
 * Modified from the isolated A20 native FP32 model. Original WeKws FSMN:
 * Copyright 2022 Yueyue Nyy; 2023 Jing Du. See NOTICE.md for fixed source IDs.
 * Full2599 output, verified byte loader, bounded trace, explicit cache contract.
 * FP32 reductions are retained; no claim of equality with Torch is made. */
#include "fsmn2599.h"
#include "sha256.h"
#include <fenv.h>
#include <float.h>
#include <math.h>
#include <string.h>

#define FSMN2599_MAGIC UINT32_C(0x46323539)
_Static_assert(sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24,
               "requires IEEE binary32");
_Static_assert(FLT_EVAL_METHOD == 0, "requires binary32 evaluation without excess precision");
_Static_assert(FSMN2599_HEAD_BIAS_OFFSET + FSMN2599_OUTPUTS == FSMN2599_FLOATS,
               "last bias must end at payload boundary");
_Static_assert(FSMN2599_HEAD_WEIGHT_OFFSET + 140u * FSMN2599_OUTPUTS == FSMN2599_HEAD_BIAS_OFFSET,
               "full head weight/bias offsets disagree");

int fsmn2599_environment_ok(void) {
    const uint16_t one = 1;
    volatile float tiny = FLT_MIN;
    volatile float half = 0.5f;
    volatile float sub = tiny * half;
    volatile float recovered = sub + sub;
    return *(const unsigned char *)&one == 1 && fegetround() == FE_TONEAREST
           && sub > 0.0f && recovered == FLT_MIN;
}

static int overlaps(const void *a, size_t na, const void *b, size_t nb) {
    uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    if (na == 0 || nb == 0) return 0;
    if (x > UINTPTR_MAX - na || y > UINTPTR_MAX - nb) return 1;
    return x < y + nb && y < x + na;
}

static int valid(const fsmn2599_model *m) {
    return m && m->magic == FSMN2599_MAGIC && m->weights;
}

fsmn2599_status fsmn2599_load(fsmn2599_model *m, float *w, size_t nw,
        const unsigned char *payload, size_t bytes, unsigned order, const char hash[65]) {
    if (!m || !w || !payload || !hash) return FSMN2599_ARGUMENT;
    if (nw != FSMN2599_FLOATS || bytes != FSMN2599_PAYLOAD_BYTES) return FSMN2599_SIZE;
    if (order != FSMN2599_PAYLOAD_LITTLE_ENDIAN || !fsmn2599_environment_ok())
        return FSMN2599_ENVIRONMENT;
    if (overlaps(m, sizeof(*m), w, FSMN2599_PAYLOAD_BYTES)
        || overlaps(m, sizeof(*m), payload, bytes)
        || overlaps(w, FSMN2599_PAYLOAD_BYTES, payload, bytes)
        || overlaps(m, sizeof(*m), hash, 65)
        || overlaps(w, FSMN2599_PAYLOAD_BYTES, hash, 65)) return FSMN2599_OVERLAP;
    for (unsigned i = 0; i < 64; ++i)
        if (!((hash[i] >= '0' && hash[i] <= '9') || (hash[i] >= 'a' && hash[i] <= 'f')))
            return FSMN2599_IDENTITY;
    if (hash[64] != '\0') return FSMN2599_IDENTITY;
    char actual[65];
    kws_sha256_memory_hex(payload, bytes, actual);
    if (memcmp(actual, hash, 65) != 0) return FSMN2599_IDENTITY;
    for (size_t i = 0; i < FSMN2599_FLOATS; ++i) {
        float value;
        memcpy(&value, payload + i * 4u, 4u);
        if (!isfinite(value)) return FSMN2599_NONFINITE;
        if (i >= 400 && i < 800 && !(value > 0.0f)) return FSMN2599_ARGUMENT;
    }
    /* Byte reads and memcpy avoid unaligned payload access / float aliasing. */
    for (size_t i = 0; i < FSMN2599_FLOATS; ++i) memcpy(w + i, payload + i * 4u, 4u);
    m->weights = w;
    memset(m->cache, 0, sizeof(m->cache));
    m->magic = FSMN2599_MAGIC;
    m->fault = 0;
    return FSMN2599_OK;
}

fsmn2599_status fsmn2599_reset(fsmn2599_model *m) {
    if (!valid(m)) return FSMN2599_STATE;
    if (!fsmn2599_environment_ok()) return FSMN2599_ENVIRONMENT;
    memset(m->cache, 0, sizeof(m->cache));
    m->fault = 0;
    return FSMN2599_OK;
}

fsmn2599_status fsmn2599_copy_cache(const fsmn2599_model *m, float *out, size_t count) {
    if (!valid(m) || m->fault) return FSMN2599_STATE;
    if (!out) return FSMN2599_ARGUMENT;
    if (count != FSMN2599_CACHE_FLOATS) return FSMN2599_SIZE;
    if (overlaps(out, sizeof(m->cache), m, sizeof(*m))
        || overlaps(out, sizeof(m->cache), m->weights, FSMN2599_PAYLOAD_BYTES))
        return FSMN2599_OVERLAP;
    memcpy(out, m->cache, sizeof(m->cache));
    return FSMN2599_OK;
}

static void affine(const float *x, float *y, const float *w, const float *b, int ni, int no) {
    for (int o = 0; o < no; ++o) {
        float sum[4] = {0, 0, 0, 0};
        for (int i = 0; i < ni; ++i) sum[i % 4] += x[i] * w[o * ni + i];
        y[o] = ((sum[0] + sum[1]) + (sum[2] + sum[3])) + (b ? b[o] : 0.0f);
    }
}

static int finite_values(const float *x, size_t count) {
    for (size_t i = 0; i < count; ++i) if (!isfinite(x[i])) return 0;
    return 1;
}

static void relu(float *x, int count) {
    for (int i = 0; i < count; ++i) if (x[i] < 0) x[i] = 0;
}

static void trace_stage(float *t, unsigned stage, const float *x, size_t count) {
    if (t) memcpy(t + (stage == 20 ? FSMN2599_TRACE_FINAL_OFFSET : stage * 250u),
                  x, count * sizeof(float));
}

static void memory(fsmn2599_model *m, int layer, size_t offset, const float *p, float *out) {
    const float *w = m->weights;
    for (int c = 0; c < 128; ++c) {
        float left = 0.0f;
        for (int k = 0; k < 10; ++k)
            left += w[offset + 32000u + (size_t)c * 10u + (size_t)k]
                    * m->cache[((size_t)c * 11u + (size_t)k) * 4u + (size_t)layer];
        float right = w[offset + 33280u + (size_t)c * 2u]
                      * m->cache[((size_t)c * 11u + 10u) * 4u + (size_t)layer]
                      + w[offset + 33281u + (size_t)c * 2u] * p[c];
        out[c] = (m->cache[((size_t)c * 11u + 9u) * 4u + (size_t)layer] + left) + right;
        for (int k = 0; k < 10; ++k)
            m->cache[((size_t)c * 11u + (size_t)k) * 4u + (size_t)layer]
                = m->cache[((size_t)c * 11u + (size_t)k + 1u) * 4u + (size_t)layer];
        m->cache[((size_t)c * 11u + 10u) * 4u + (size_t)layer] = p[c];
    }
}

fsmn2599_status fsmn2599_step(fsmn2599_model *m, const float *x, size_t nx,
        float *y, size_t ny, float *t, size_t nt) {
    if (!valid(m) || m->fault) return FSMN2599_STATE;
    if (!x || !y || (!t && nt != 0)) return FSMN2599_ARGUMENT;
    if (nx != FSMN2599_INPUTS || ny != FSMN2599_OUTPUTS
        || (t && nt < FSMN2599_TRACE_FLOATS)) return FSMN2599_SIZE;
    if (!fsmn2599_environment_ok()) return FSMN2599_ENVIRONMENT;
    size_t xb = FSMN2599_INPUTS * sizeof(float), yb = FSMN2599_OUTPUTS * sizeof(float);
    size_t tb = t ? FSMN2599_TRACE_FLOATS * sizeof(float) : 0;
    if (overlaps(x, xb, m, sizeof(*m)) || overlaps(y, yb, m, sizeof(*m))
        || overlaps(t, tb, m, sizeof(*m)) || overlaps(x, xb, m->weights, FSMN2599_PAYLOAD_BYTES)
        || overlaps(y, yb, m->weights, FSMN2599_PAYLOAD_BYTES)
        || overlaps(t, tb, m->weights, FSMN2599_PAYLOAD_BYTES)
        || overlaps(x, xb, y, yb) || overlaps(x, xb, t, tb) || overlaps(y, yb, t, tb))
        return FSMN2599_OVERLAP;
    if (!finite_values(x, nx)) return FSMN2599_NONFINITE;
    if (t) memset(t, 0, tb);
    const float *w = m->weights;
    float z[400], a[250], b[250], p[128], mem[128];
    for (unsigned i = 0; i < 400; ++i) z[i] = (x[i] - w[i]) * w[400u + i];
    if (!finite_values(z, 400)) goto fault;
    affine(z, a, w + 800, w + 56800, 400, 140);
    if (!finite_values(a, 140)) goto fault;
    trace_stage(t, 0, a, 140);
    affine(a, b, w + 56940, w + 91940, 140, 250);
    if (!finite_values(b, 250)) goto fault;
    trace_stage(t, 1, b, 250); relu(b, 250); trace_stage(t, 2, b, 250);
    for (int layer = 0; layer < 4; ++layer) {
        size_t offset = 92190u + (size_t)layer * 65786u;
        affine(b, p, w + offset, NULL, 250, 128);
        if (!finite_values(p, 128)) goto fault;
        trace_stage(t, 3u + (unsigned)layer * 4u, p, 128);
        memory(m, layer, offset, p, mem);
        if (!finite_values(mem, 128)) goto fault;
        trace_stage(t, 4u + (unsigned)layer * 4u, mem, 128);
        affine(mem, b, w + offset + 33536u, w + offset + 65536u, 128, 250);
        if (!finite_values(b, 250)) goto fault;
        trace_stage(t, 5u + (unsigned)layer * 4u, b, 250);
        relu(b, 250); trace_stage(t, 6u + (unsigned)layer * 4u, b, 250);
    }
    affine(b, a, w + 355334u, w + 390334u, 250, 140);
    if (!finite_values(a, 140)) goto fault;
    trace_stage(t, 19, a, 140);
    affine(a, y, w + FSMN2599_HEAD_WEIGHT_OFFSET, w + FSMN2599_HEAD_BIAS_OFFSET,
           140, FSMN2599_OUTPUTS);
    if (!finite_values(y, FSMN2599_OUTPUTS)) goto fault;
    trace_stage(t, 20, y, FSMN2599_OUTPUTS);
    return FSMN2599_OK;
fault:
    m->fault = 1;
    return FSMN2599_NONFINITE;
}
