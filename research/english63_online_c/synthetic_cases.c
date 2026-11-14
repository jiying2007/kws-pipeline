/* Invented zero/analytic weights only. Never include en63_generated.h here. */
#include "en63_stream.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static float parameters[EN63_GRU_PARAMETERS], window[400], mean[32], deviation[32], mel[32];
static uint16_t starts[32], counts[32], offsets[33];
static en63_state state_a, state_b, snapshot;
static int16_t pcm[3199];

typedef struct { uint64_t rows, events, hash, first_sample; } observed;

static void observe(void *opaque, uint64_t frame, uint64_t sample,
                    const float features[32], const float logits[3],
                    const float probabilities[3], uint32_t proposal)
{
    observed *o = opaque;
    unsigned i;
    assert(sample == 400+(frame-1)*160);
    ++o->rows;
    if (proposal) { if (!o->events) o->first_sample = sample; ++o->events; }
    for (i = 0; i < 38; ++i) {
        const float *value = i < 32 ? &features[i] : i < 35 ? &logits[i-32] : &probabilities[i-35];
        uint32_t bits;
        memcpy(&bits, value, 4);
        o->hash = (o->hash ^ bits)*UINT64_C(1099511628211);
    }
    o->hash = (o->hash ^ proposal)*UINT64_C(1099511628211);
}

static en63_model fixture(unsigned kind)
{
    unsigned i;
    memset(parameters, 0, sizeof(parameters));
    for (i = 0; i < 400; ++i) window[i] = 1;
    for (i = 0; i < 32; ++i) { starts[i] = 0; counts[i] = 1; offsets[i] = (uint16_t)i; deviation[i] = 1; mel[i] = 1; }
    offsets[32] = 32;
    { en63_model m = {kind, kind == EN63_GRU ? EN63_GRU_PARAMETERS : EN63_DSC_PARAMETERS,
                      parameters, window, mean, deviation, starts, counts, offsets, mel, 32, "synthetic-only"}; return m; }
}

static void feed(en63_state *s, size_t samples, int mixed, observed *o)
{
    static const size_t pattern[] = {1, 159, 161, 3199, 320, 7};
    size_t offset = 0, call = 0;
    while (offset < samples) {
        size_t i, take = mixed ? pattern[call++ % 6] : 160;
        if (take > samples-offset) take = samples-offset;
        for (i = 0; i < take; ++i) pcm[i] = (int16_t)((offset+i)%2048);
        assert(en63_feed(s, pcm, take, observe, o) == EN63_OK);
        offset += take;
    }
}

int main(void)
{
    unsigned kind, i;
    observed o = {0};
    en63_model m = fixture(EN63_GRU);
    assert(en63_init(&state_a, &m) == EN63_OK);
    snapshot = state_a;
    assert(en63_feed(&state_a, pcm, 0, observe, &o) == EN63_ARGUMENT);
    assert(en63_feed(&state_a, pcm, 3200, observe, &o) == EN63_ARGUMENT);
    assert(en63_feed(&state_a, state_a.pcm, 10, observe, &o) == EN63_ARGUMENT);
    assert(memcmp(&snapshot, &state_a, sizeof(state_a)) == 0);
    assert(en63_feed(&state_a, pcm, 399, observe, &o) == EN63_OK);
    assert(o.rows == 0 && en63_finish(&state_a) == EN63_OK && en63_finish(&state_a) == EN63_OK);
    assert(o.rows == 0 && o.events == 0 && state_a.pcm_fill == 399);
    assert(en63_feed(&state_a, pcm, 1, observe, &o) == EN63_STATE);
    assert(en63_reset(&state_a) == EN63_OK && state_a.samples == 0 && state_a.frames == 0);
    deviation[0] = 0;
    assert(en63_init(&state_b, &m) == EN63_ARGUMENT);

    /* PyTorch GRU reset applies to recurrent candidate bias: n=tanh(.4+.5*.2). */
    m = fixture(EN63_GRU);
    for (i = 0; i < 32; ++i) { parameters[6144+64+i] = .4f; parameters[6240+64+i] = .2f; }
    assert(en63_init(&state_a, &m) == EN63_OK);
    for (i = 0; i < 400; ++i) pcm[i] = 1000;
    o = (observed){0};
    assert(en63_feed(&state_a, pcm, 400, observe, &o) == EN63_OK);
    assert(fabsf(state_a.hidden[0]-.5f*tanhf(.5f)) < 1e-6f);
    assert(fabsf(state_a.features[0]-logf((400.0f*1000/32768)*(400.0f*1000/32768))) < 2e-5f);

    /* Analytic three causal dilated residual blocks, with no learned weights. */
    m = fixture(EN63_DSC_TCN);
    for (i = 0; i < 32; ++i) parameters[1024+i] = 1;
    for (kind = 0; kind < 3; ++kind) {
        float *p = parameters+1056+kind*1184;
        for (i = 0; i < 32; ++i) { p[3*i] = 1; p[3*i+1] = 2; p[3*i+2] = 3; p[128+i*32+i] = 1; }
    }
    assert(en63_init(&state_a, &m) == EN63_OK);
    assert(en63_feed(&state_a, pcm, 400, observe, &o) == EN63_OK);
    assert(state_a.pool[0][0] == 15.625f);
    assert(en63_feed(&state_a, pcm, 160, observe, &o) == EN63_OK);
    assert(state_a.pool[1][0] == 21.875f);

    for (kind = EN63_GRU; kind <= EN63_DSC_TCN; ++kind) {
        observed a = {0}, b = {0};
        m = fixture(kind);
        /* Constant class 1 logits exercise confirmation/cooldown, not acoustics. */
        parameters[m.parameter_count-2] = 4;
        assert(en63_init(&state_a, &m) == EN63_OK && en63_init(&state_b, &m) == EN63_OK);
        feed(&state_a, 480123, 0, &a); feed(&state_b, 480123, 1, &b);
        assert(a.rows == b.rows && a.rows == 2999 && a.hash == b.hash && a.events == b.events);
        assert(a.first_sample == 8560 && a.events > 20);
        assert(state_a.pool_count == 200 && state_a.pcm_fill < 400 && state_a.samples == 480123);
        assert(en63_finish(&state_a) == EN63_OK && en63_finish(&state_b) == EN63_OK);
        assert(en63_reset(&state_a) == EN63_OK && state_a.frames == 0 && state_a.events == 0);
        a = (observed){0}; feed(&state_a, 480123, 1, &a);
        assert(a.hash == b.hash && a.events == b.events);
    }
    puts("PASS synthetic GRU reset bias, causal TCN history, frontend DC, chunk identity, long stream, reset, EOF, invalid-input atomicity");
    return 0;
}
