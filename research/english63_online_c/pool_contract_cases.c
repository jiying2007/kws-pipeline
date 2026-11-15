/* Invented finite arrays, model parameters and PCM only. No saved inputs.
 * Compile this translation unit twice with renamed public entry points to
 * expose the actual static pooling helper, then link the fixture below.
 */
#ifdef EN63_POOL_FIXTURE_OBJECT
#include "en63_stream.c"
#ifndef EN63_POOL_FIXTURE_ENTRY
#error "A private fixture entry point is required"
#endif
void EN63_POOL_FIXTURE_ENTRY(const en63_state *s, float output[64])
{
    pool_window(s, output);
}
#else
#include "en63_stream.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DECLARE_VARIANT(prefix) \
    size_t prefix##_en63_state_bytes(void); \
    int prefix##_en63_init(en63_state *, const en63_model *); \
    int prefix##_en63_reset(en63_state *); \
    int prefix##_en63_discontinuity(en63_state *); \
    int prefix##_en63_feed(en63_state *, const int16_t *, size_t, en63_callback, void *); \
    int prefix##_en63_finish(en63_state *); \
    void prefix##_pool(const en63_state *, float[64])
DECLARE_VARIANT(pool0);
DECLARE_VARIANT(pool1);

typedef struct {
    size_t (*state_bytes)(void);
    int (*init)(en63_state *, const en63_model *);
    int (*reset)(en63_state *);
    int (*gap)(en63_state *);
    int (*feed)(en63_state *, const int16_t *, size_t, en63_callback, void *);
    int (*finish)(en63_state *);
    void (*pool)(const en63_state *, float[64]);
} implementation;

static const implementation variants[2] = {
    {pool0_en63_state_bytes, pool0_en63_init, pool0_en63_reset,
     pool0_en63_discontinuity, pool0_en63_feed, pool0_en63_finish, pool0_pool},
    {pool1_en63_state_bytes, pool1_en63_init, pool1_en63_reset,
     pool1_en63_discontinuity, pool1_en63_feed, pool1_en63_finish, pool1_pool}
};

static void require(int condition, const char *message)
{
    if (!condition) { fprintf(stderr, "POOL_SOURCE_CONTRACT_FAILED: %s\n", message); exit(1); }
}

static uint32_t bits(float value)
{
    uint32_t result;
    memcpy(&result, &value, sizeof(result));
    return result;
}

static float from_bits(uint32_t value)
{
    float result;
    memcpy(&result, &value, sizeof(result));
    return result;
}

static float fixture_value(unsigned pattern, unsigned time, unsigned channel)
{
    unsigned slot = time+channel;
    if (pattern == 0) return slot & 1u ? -0.0f : 0.0f;
    if (pattern == 1) return slot & 1u ? -FLT_MAX : FLT_MAX;
    if (pattern == 2) {
        static const float cancellation[4] = {0x1p100f, 1.0f, -0x1p100f, 0x1p-100f};
        return cancellation[slot % 4];
    }
    if (pattern == 3) {
        static const uint32_t tiny[8] = {
            UINT32_C(0x00000001), UINT32_C(0x80000001),
            UINT32_C(0x00800000), UINT32_C(0x80800000),
            UINT32_C(0x007fffff), UINT32_C(0x807fffff),
            UINT32_C(0x00000000), UINT32_C(0x80000000)
        };
        return from_bits(tiny[slot % 8]);
    }
    {
        uint32_t x = (uint32_t)(time+1)*UINT32_C(2654435761) ^
                     (uint32_t)(channel+17)*UINT32_C(2246822519);
        x ^= x >> 13; x *= UINT32_C(3266489917); x ^= x >> 16;
        /* Exponents <= 220 keep every 200-element sum finite. */
        return from_bits((x & UINT32_C(0x807fffff)) | ((1u+(x % 220u)) << 23));
    }
}

static void all_pool_geometries(void)
{
    static en63_state s;
    float reference[64], candidate[64];
    unsigned pattern, time, channel, count, write;
    unsigned long cases = 0;
    for (pattern = 0; pattern < 5; ++pattern) {
        for (time = 0; time < 200; ++time)
            for (channel = 0; channel < 32; ++channel) {
                s.pool[time][channel] = fixture_value(pattern, time, channel);
                require(isfinite(s.pool[time][channel]), "invented input must be finite");
            }
        for (count = 1; count <= 200; ++count) {
            s.pool_count = count;
            for (write = 0; write < 200; ++write) {
                unsigned oldest = (write+200-count)%200;
                s.pool_write = write;
                variants[0].pool(&s, reference); variants[1].pool(&s, candidate);
                for (channel = 0; channel < 64; ++channel) {
                    require(isfinite(reference[channel]) && isfinite(candidate[channel]),
                            "invented sums and maxima must remain finite");
                    if (bits(reference[channel]) != bits(candidate[channel])) {
                        fprintf(stderr, "pattern=%u count=%u write=%u component=%u ref=%08x candidate=%08x\n",
                                pattern, count, write, channel,
                                (unsigned)bits(reference[channel]), (unsigned)bits(candidate[channel]));
                        require(0, "FP32 pool byte identity");
                    }
                    if (pattern == 0 && channel >= 32)
                        require(bits(candidate[channel]) == bits(fixture_value(0, oldest, channel-32)),
                                "strict > retains the first signed-zero maximum");
                }
                ++cases;
            }
        }
    }
    require(cases == 200000ul, "all 5 x 200 x 200 pool fixtures executed");
    printf("PASS pool byte identity: %lu cases, all counts 1..200/write positions 0..199, 12800000 FP32 outputs\n", cases);
}

#define STREAM_SAMPLES 64403u
#define PREFIX_SAMPLES 33379u
#define RECORD_CAPACITY 512u
typedef struct {
    uint64_t frame, sample;
    float values[38];
    uint32_t proposal;
} record;
typedef struct {
    record rows[RECORD_CAPACITY];
    size_t count;
    uint64_t events, frame_base, sample_base;
} observation;

static float parameters[EN63_GRU_PARAMETERS], window[400], mean[32], deviation[32], mel[32];
static uint16_t starts[32], counts[32], offsets[33];
static en63_state stream_state;
static observation actual, expected;

static en63_model invented_model(unsigned kind)
{
    size_t i, size = kind == EN63_GRU ? EN63_GRU_PARAMETERS : EN63_DSC_PARAMETERS;
    memset(parameters, 0, sizeof(parameters));
    for (i = 0; i < size; ++i) parameters[i] = (float)((int)((i*17+3)%31)-15)*0.0005f;
    if (kind == EN63_DSC_TCN)
        for (i = 0; i < 32; ++i) parameters[1024+i] = 0.25f+(float)i*0.005f;
    /* Nonzero classifier/encoder weights expose pooling in every saved logit.
     * A dominant class bias also guarantees real proposal/cooldown observations.
     * These are invented test tensors and never an acoustic-quality assertion.
     */
    parameters[size-3] = 0; parameters[size-2] = 4; parameters[size-1] = -1;
    for (i = 0; i < 400; ++i) window[i] = 0.75f+(float)(i%9)*0.03125f;
    for (i = 0; i < 32; ++i) {
        mean[i] = (float)((int)i-16)*0.001f; deviation[i] = 0.75f+(float)i*0.01f;
        starts[i] = (uint16_t)((i*7)%257); counts[i] = 1; offsets[i] = (uint16_t)i;
        mel[i] = 0.5f+(float)i*0.01f;
    }
    offsets[32] = 32;
    { en63_model model = {kind, size, parameters, window, mean, deviation,
                          starts, counts, offsets, mel, 32, "invented-pool-fixture-only"}; return model; }
}

static void observe(void *opaque, uint64_t frame, uint64_t sample,
                    const float features[32], const float logits[3],
                    const float probabilities[3], uint32_t proposal)
{
    observation *out = opaque;
    record *row;
    unsigned i;
    require(out->count < RECORD_CAPACITY && frame > out->frame_base && sample >= out->sample_base,
            "invented trace capacity and monotonic offset");
    row = &out->rows[out->count++];
    memset(row, 0, sizeof(*row));
    row->frame = frame-out->frame_base; row->sample = sample-out->sample_base;
    require(row->sample == 400+(row->frame-1)*160, "supplied PCM clock including gap offset");
    memcpy(row->values, features, 32*sizeof(float));
    memcpy(row->values+32, logits, 3*sizeof(float));
    memcpy(row->values+35, probabilities, 3*sizeof(float));
    for (i = 0; i < 38; ++i) require(isfinite(row->values[i]), "invented stream finite output");
    row->proposal = proposal;
    if (proposal) ++out->events;
}

static void feed_invented(const implementation *api, size_t samples, int mixed)
{
    static const size_t pattern[6] = {1, 159, 161, 3199, 320, 7};
    int16_t pcm[EN63_MAX_CHUNK];
    size_t offset = 0, call = 0;
    while (offset < samples) {
        size_t i, take = mixed ? pattern[call++ % 6] : 160;
        if (take > samples-offset) take = samples-offset;
        for (i = 0; i < take; ++i) {
            size_t index = offset+i;
            pcm[i] = (int16_t)((int)((index*73+(index/17)*91)%8191)-4095);
        }
        require(api->feed(&stream_state, pcm, take, observe, &actual) == EN63_OK, "invented feed");
        offset += take;
    }
}

static void finish_and_compare(const implementation *api)
{
    size_t count = actual.count;
    uint64_t events = actual.events;
    require(api->finish(&stream_state) == EN63_OK && api->finish(&stream_state) == EN63_OK,
            "invented EOF without padding");
    require(actual.count == count && actual.events == events && count == 401 && events > 0,
            "invented full stream frame/event/EOF counts");
    require(actual.count == expected.count && actual.events == expected.events &&
            memcmp(actual.rows, expected.rows, count*sizeof(record)) == 0,
            "all feature/logit/probability bits and event/clock rows identical");
}

static void invented_streams(void)
{
    unsigned kind, variant, plan;
    require(variants[0].state_bytes() == sizeof(en63_state) &&
            variants[1].state_bytes() == sizeof(en63_state), "unchanged caller state size");
    for (kind = EN63_GRU; kind <= EN63_DSC_TCN; ++kind) {
        en63_model model = invented_model(kind);
        for (variant = 0; variant < 2; ++variant) {
            const implementation *api = &variants[variant];
            for (plan = 0; plan < 2; ++plan) {
                memset(&actual, 0, sizeof(actual));
                require(api->init(&stream_state, &model) == EN63_OK, "invented init");
                feed_invented(api, STREAM_SAMPLES, (int)plan);
                require(stream_state.pool_count == 200 && stream_state.pool_write == 1 &&
                        stream_state.samples == STREAM_SAMPLES, "invented ring fills and wraps twice");
                if (variant == 0 && plan == 0) expected = actual;
                finish_and_compare(api);
            }
            memset(&actual, 0, sizeof(actual));
            require(api->init(&stream_state, &model) == EN63_OK, "reset-prefix init");
            feed_invented(api, PREFIX_SAMPLES, 1);
            require(api->reset(&stream_state) == EN63_OK && stream_state.frames == 0 &&
                    stream_state.samples == 0 && stream_state.events == 0, "new-session reset");
            memset(&actual, 0, sizeof(actual));
            feed_invented(api, STREAM_SAMPLES, 1);
            finish_and_compare(api);

            memset(&actual, 0, sizeof(actual));
            require(api->init(&stream_state, &model) == EN63_OK, "gap-prefix init");
            feed_invented(api, PREFIX_SAMPLES, 1);
            {
                uint64_t frame_base = stream_state.frames, sample_base = stream_state.samples;
                uint64_t old_events = stream_state.events;
                require(api->gap(&stream_state) == EN63_OK && stream_state.pool_count == 0 &&
                        stream_state.pool_write == 0 && stream_state.frames == frame_base &&
                        stream_state.samples == sample_base && stream_state.events == old_events,
                        "gap discards pool and preserves session counters");
                memset(&actual, 0, sizeof(actual));
                actual.frame_base = frame_base; actual.sample_base = sample_base;
                feed_invented(api, STREAM_SAMPLES, 1);
                require(stream_state.frames == frame_base+401 &&
                        stream_state.samples == sample_base+STREAM_SAMPLES &&
                        stream_state.events == old_events+actual.events,
                        "gap suffix cumulative counters");
                finish_and_compare(api);
            }
        }
        printf("PASS invented model kind %u: both pool variants, 160/mixed chunks, reset, gap, complete 401-row byte/event/clock identity\n", kind);
    }
}

int main(void)
{
    all_pool_geometries();
    invented_streams();
    puts("PASS pooling source contracts; saved project weights/audio executions = 0");
    return 0;
}
#endif
