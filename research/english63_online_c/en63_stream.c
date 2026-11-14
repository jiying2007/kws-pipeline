/* The radix-2 FFT structure follows src/frontend.c. English's frozen 160-hop,
 * dense triangular normalized mel bank, log floor, and train-only norm differ
 * from that frontend, so its KWSP feature transform must not be substituted.
 * FP64 FFT scratch reduces frontend roundoff; exported model weights are FP32.
 */
#include "en63_stream.h"
#include <float.h>
#include <math.h>
#include <string.h>

#define EN63_MAGIC UINT32_C(0x454e3633)
#define EN63_PI 3.14159265358979323846264338327950288
_Static_assert(sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24,
               "IEEE binary32 is required");

static int model_valid(const en63_model *m)
{
    size_t i;
    if (!m || !m->parameters || !m->window || !m->mean || !m->std ||
        !m->mel_start || !m->mel_count || !m->mel_offset || !m->mel_weights ||
        (m->kind != EN63_GRU && m->kind != EN63_DSC_TCN) ||
        m->parameter_count != (m->kind == EN63_GRU ? EN63_GRU_PARAMETERS : EN63_DSC_PARAMETERS) ||
        m->mel_weights_count > 32u * 257u || m->mel_offset[0] != 0 ||
        m->mel_offset[32] != m->mel_weights_count) return 0;
    for (i = 0; i < m->parameter_count; ++i) if (!isfinite(m->parameters[i])) return 0;
    for (i = 0; i < 400; ++i)
        if (!isfinite(m->window[i]) || m->window[i] < 0 || m->window[i] > 1) return 0;
    for (i = 0; i < 32; ++i) {
        if (!isfinite(m->mean[i]) || !isfinite(m->std[i]) || m->std[i] < 0.5f ||
            m->mel_count[i] == 0 || (size_t)m->mel_start[i] + m->mel_count[i] > 257 ||
            (size_t)m->mel_offset[i] + m->mel_count[i] != m->mel_offset[i+1]) return 0;
    }
    for (i = 0; i < m->mel_weights_count; ++i)
        if (!isfinite(m->mel_weights[i]) || m->mel_weights[i] < 0) return 0;
    return 1;
}

size_t en63_state_bytes(void) { return sizeof(en63_state); }

int en63_init(en63_state *s, const en63_model *m)
{
    unsigned len, stage = 0;
    if (!s || !model_valid(m)) return EN63_ARGUMENT;
    memset(s, 0, sizeof(*s));
    s->model = m;
    s->magic = EN63_MAGIC;
    for (len = 2; len <= 512; len <<= 1) {
        double angle = -2.0 * EN63_PI / len;
        s->stage_re[stage] = cos(angle);
        s->stage_im[stage++] = sin(angle);
    }
    return EN63_OK;
}

int en63_reset(en63_state *s)
{
    if (!s || s->magic != EN63_MAGIC || s->busy) return EN63_STATE;
    return en63_init(s, s->model);
}

static void fft512(en63_state *s)
{
    unsigned i, j = 0, len, stage = 0;
    for (i = 1; i < 512; ++i) {
        unsigned bit = 256;
        while (j & bit) { j ^= bit; bit >>= 1; }
        j ^= bit;
        if (i < j) {
            double r = s->re[i], im = s->im[i];
            s->re[i] = s->re[j]; s->im[i] = s->im[j];
            s->re[j] = r; s->im[j] = im;
        }
    }
    for (len = 2; len <= 512; len <<= 1) {
        double cr = s->stage_re[stage], ci = s->stage_im[stage++];
        for (i = 0; i < 512; i += len) {
            double wr = 1, wi = 0;
            unsigned k;
            for (k = 0; k < len/2; ++k) {
                unsigned a = i+k, b = a+len/2;
                double vr = s->re[b]*wr-s->im[b]*wi, vi = s->re[b]*wi+s->im[b]*wr;
                double ur = s->re[a], ui = s->im[a], next_wr = wr*cr-wi*ci;
                s->re[a] = ur+vr; s->im[a] = ui+vi;
                s->re[b] = ur-vr; s->im[b] = ui-vi;
                wi = wr*ci+wi*cr; wr = next_wr;
            }
        }
    }
}

static void frontend(en63_state *s)
{
    float power[257];
    unsigned i, m;
    memset(s->re, 0, sizeof(s->re)); memset(s->im, 0, sizeof(s->im));
    for (i = 0; i < 400; ++i) {
        float x = (float)s->pcm[i] / 32768.0f;
        float windowed = x*s->model->window[i];
        s->re[i] = windowed;
    }
    fft512(s);
    for (i = 0; i < 257; ++i) {
        /* NumPy 2.3.5 rfft(float32) -> complex64; abs() and square are FP32. */
        float re = (float)s->re[i], im = (float)s->im[i];
        float magnitude = hypotf(re, im);
        power[i] = magnitude*magnitude;
    }
    for (m = 0; m < 32; ++m) {
        float energy = 0, logmel;
        unsigned start = s->model->mel_start[m], offset = s->model->mel_offset[m];
        for (i = 0; i < s->model->mel_count[m]; ++i)
            energy += power[start+i]*s->model->mel_weights[offset+i];
        logmel = logf(fmaxf(energy, 1e-8f));
        s->features[m] = (logmel-s->model->mean[m])/s->model->std[m];
    }
}

static float dot(const float *a, const float *b, unsigned n, float bias)
{
    unsigned i;
    float sum = 0;
    for (i = 0; i < n; ++i) sum += a[i]*b[i];
    return sum+bias;
}

static float sigmoid(float value)
{
    if (value >= 0) return 1.0f/(1.0f+expf(-value));
    { float x = expf(value); return x/(1.0f+x); }
}

static const float *encode(en63_state *s, float result[32])
{
    const float *p = s->model->parameters;
    unsigned i, j;
    if (s->model->kind == EN63_GRU) {
        float in[96], recurrent[96];
        for (i = 0; i < 96; ++i) {
            in[i] = dot(p+i*32, s->features, 32, p[6144+i]);
            recurrent[i] = dot(p+3072+i*32, s->hidden, 32, p[6240+i]);
        }
        for (i = 0; i < 32; ++i) {
            float reset = sigmoid(in[i]+recurrent[i]);
            float update = sigmoid(in[32+i]+recurrent[32+i]);
            float candidate = tanhf(in[64+i]+reset*recurrent[64+i]);
            result[i] = (s->hidden[i]-candidate)*update+candidate;
        }
        memcpy(s->hidden, result, sizeof(s->hidden));
        return p+6336;
    }
    for (i = 0; i < 32; ++i) result[i] = fmaxf(dot(p+i*32, s->features, 32, p[1024+i]), 0);
    p += 1056;
    for (j = 0; j < 3; ++j) {
        unsigned dilation = 1u << j, count = 2*dilation;
        float depthwise[32], next[32];
        for (i = 0; i < 32; ++i) {
            float x = p[3*i]*s->history[j][0][i];
            x += p[3*i+1]*s->history[j][dilation][i];
            x += p[3*i+2]*result[i];
            depthwise[i] = fmaxf(x+p[96+i], 0);
        }
        for (i = 0; i < 32; ++i)
            next[i] = result[i]+0.5f*dot(p+128+i*32, depthwise, 32, p[1152+i]);
        memmove(s->history[j][0], s->history[j][1], (count-1)*32*sizeof(float));
        memcpy(s->history[j][count-1], result, 32*sizeof(float));
        memcpy(result, next, 32*sizeof(float));
        p += 1184;
    }
    return p;
}

static uint32_t decode(en63_state *s)
{
    uint32_t chosen = 0;
    unsigned i;
    for (i = 1; i < 3; ++i) if (s->probabilities[i] > s->probabilities[chosen]) chosen = i;
    if (s->frames < 50 || s->frames < s->blocked_until || chosen == 0 || s->probabilities[chosen] < 0.8f) {
        s->candidate = 0; s->confirmations = 0; return 0;
    }
    s->confirmations = chosen == s->candidate ? s->confirmations+1 : 1;
    s->candidate = chosen;
    if (s->confirmations < 3) return 0;
    s->blocked_until = s->frames+100;
    s->candidate = 0; s->confirmations = 0; ++s->events;
    return chosen;
}

static int process_frame(en63_state *s, en63_callback callback, void *user)
{
    float encoded[32], pooled[64], largest, sum = 0;
    const float *classifier;
    unsigned i, j;
    uint32_t proposal;
    frontend(s);
    classifier = encode(s, encoded);
    memcpy(s->pool[s->pool_write], encoded, sizeof(encoded));
    s->pool_write = (s->pool_write+1)%200;
    if (s->pool_count < 200) ++s->pool_count;
    for (i = 0; i < 32; ++i) {
        float total = 0, maximum = -FLT_MAX;
        for (j = 0; j < s->pool_count; ++j) {
            unsigned index = (s->pool_write+200-s->pool_count+j)%200;
            float value = s->pool[index][i];
            total += value; if (value > maximum) maximum = value;
        }
        pooled[i] = total/(float)s->pool_count; pooled[32+i] = maximum;
    }
    for (i = 0; i < 3; ++i) s->logits[i] = dot(classifier+i*64, pooled, 64, classifier[192+i]);
    largest = fmaxf(s->logits[0], fmaxf(s->logits[1], s->logits[2]));
    for (i = 0; i < 3; ++i) { s->probabilities[i] = expf(s->logits[i]-largest); sum += s->probabilities[i]; }
    for (i = 0; i < 3; ++i) s->probabilities[i] /= sum;
    for (i = 0; i < 32; ++i) if (!isfinite(s->features[i]) || !isfinite(encoded[i])) return EN63_NUMERIC;
    for (i = 0; i < 3; ++i) if (!isfinite(s->logits[i]) || !isfinite(s->probabilities[i])) return EN63_NUMERIC;
    ++s->frames;
    proposal = decode(s);
    callback(user, s->frames, 400+(s->frames-1)*160, s->features, s->logits, s->probabilities, proposal);
    return EN63_OK;
}

int en63_feed(en63_state *s, const int16_t *pcm, size_t count, en63_callback callback, void *user)
{
    size_t i;
    uintptr_t source, target;
    if (!s || !pcm || !callback || count == 0 || count > EN63_MAX_CHUNK) return EN63_ARGUMENT;
    if (s->magic != EN63_MAGIC || s->finished || s->faulted || s->busy || s->pcm_fill >= 400 || s->pool_count > 200 || s->pool_write >= 200)
        return EN63_STATE;
    if ((uint64_t)count > UINT64_MAX-s->samples || s->samples > UINT64_MAX-16000) return EN63_ARGUMENT;
    source = (uintptr_t)pcm; target = (uintptr_t)s;
    if (source > UINTPTR_MAX-count*sizeof(*pcm) || target > UINTPTR_MAX-sizeof(*s) ||
        (source < target+sizeof(*s) && target < source+count*sizeof(*pcm))) return EN63_ARGUMENT;
    s->busy = 1;
    for (i = 0; i < count; ++i) {
        int status;
        s->pcm[s->pcm_fill++] = pcm[i]; ++s->samples;
        if (s->pcm_fill < 400) continue;
        status = process_frame(s, callback, user);
        if (status != EN63_OK) { s->faulted = 1; s->busy = 0; return status; }
        memmove(s->pcm, s->pcm+160, 240*sizeof(int16_t)); s->pcm_fill = 240;
    }
    s->busy = 0;
    return EN63_OK;
}

int en63_finish(en63_state *s)
{
    uint64_t expected;
    if (!s || s->magic != EN63_MAGIC || s->faulted || s->busy) return EN63_STATE;
    expected = s->samples < 400 ? 0 : 1+(s->samples-400)/160;
    if (expected != s->frames || s->pcm_fill >= 400) return EN63_STATE;
    s->finished = 1; /* No padding, frame, score, callback or event is produced. */
    return EN63_OK;
}
