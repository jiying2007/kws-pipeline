#include "k1_fixed_slot_ctc.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* Written from the public contract without reading the owner's test oracle.
 * This enumerates raw six-label paths and ordinary CTC collapse, not DP states.
 * Integer numerators over powers of 8 are exact for this finite row family. */
static const unsigned weights[8][6] = {
    {8,0,0,0,0,0}, {0,8,0,0,0,0}, {0,0,8,0,0,0},
    {0,0,0,8,0,0}, {0,0,0,0,8,0}, {0,0,0,0,0,8},
    {1,1,1,1,2,2}, {4,0,0,0,3,1}
};

typedef struct {
    k1_row rows[260];
    uint8_t raw[260];
    k1_view view;
    k1_prefix prefix;
    k1_slot slot;
} fixture;

static unsigned success_calls, error_calls;
static uint64_t enumerated_paths;

static unsigned power(unsigned base, unsigned exponent)
{
    unsigned value = 1;
    while (exponent-- != 0) value *= base;
    return value;
}

static void binding(fixture *f, unsigned prefix_count, unsigned slot_count)
{
    k1_binding h = {0};
    h.source_id = 11; h.epoch = 12; h.generation = 13; h.range_id = 14;
    h.prefix_begin = 100; h.a = 100 + prefix_count; h.b = h.a + slot_count;
    h.evidence_horizon = 20;
    f->view.binding = h; f->prefix.binding = h; f->slot.binding = h;
    f->view.available_rows = prefix_count + slot_count;
    f->view.immutable_through = h.b;
    f->prefix.path_count = prefix_count;
}

static void initialize(fixture *f, unsigned slot_count)
{
    unsigned i;
    memset(f, 0, sizeof *f);
    f->view.rows = f->rows; f->view.capacity_rows = 260;
    f->view.first_row = 100; f->view.class_count = 6;
    f->view.append_only = 1; f->view.contiguous = 1;
    f->prefix.raw_path = f->raw; f->prefix.path_capacity = 260;
    f->prefix.immutable = 1; f->prefix.run_closed = 1;
    f->slot.sealed = 1;
    binding(f, 3, slot_count);
    for (i = 0; i < 260; ++i) f->rows[i].available_at = 20;
    for (i = 0; i < 3; ++i) {
        f->raw[i] = (uint8_t)(i + 1);
        f->rows[i].p[i + 1] = 1.0f;
    }
    for (i = 0; i < slot_count; ++i) f->rows[3 + i].p[K1_WO] = 1.0f;
}

static void expect_ok(fixture *f, k1_result *r)
{
    fixture before;
    memcpy(&before, f, sizeof before);
    assert(k1_score(&f->view, &f->prefix, &f->slot, r) == K1_OK);
    assert(memcmp(&before, f, sizeof before) == 0);
    assert(r->provenance == K1_CALLER_BOUND_UNVERIFIED);
    assert(r->boundaries == K1_FIXED_COORDINATES_ACOUSTIC_BOUNDARIES_UNPROVEN);
    ++success_calls;
}

static void expect_error(fixture *f, enum k1_error wanted)
{
    fixture before;
    k1_result r, prior;
    memcpy(&before, f, sizeof before);
    memset(&r, 0xa5, sizeof r);
    memcpy(&prior, &r, sizeof prior);
    assert(k1_score(&f->view, &f->prefix, &f->slot, &r) == wanted);
    assert(memcmp(&before, f, sizeof before) == 0);
    assert(memcmp(&r, &prior, sizeof r) == 0);
    ++error_calls;
}

static void expect_mass(double observed, uint64_t numerator, unsigned denominator)
{
    if (numerator == 0) assert(observed == -INFINITY);
    else {
        double exact_log = log((double)numerator) - log((double)denominator);
        assert(isfinite(observed));
        assert(fabs(observed - exact_log) < 2e-12);
    }
}

static void exhaustive_oracle(void)
{
    unsigned n, cases = 0;
    for (n = 0; n <= 3; ++n) {
        unsigned sequence;
        for (sequence = 0; sequence < power(8, n); ++sequence) {
            fixture f;
            k1_result result;
            unsigned row_family[3] = {0}, path_index, t, code = sequence;
            uint64_t numerator[2] = {0, 0};
            initialize(&f, n);
            for (t = 0; t < n; ++t) {
                unsigned label;
                row_family[t] = code % 8; code /= 8;
                for (label = 0; label < 6; ++label)
                    f.rows[3 + t].p[label] = (float)weights[row_family[t]][label] / 8.0f;
            }
            for (path_index = 0; path_index < power(6, n); ++path_index) {
                unsigned raw_code = path_index, previous = 0, emitted = 0, only = 0;
                uint64_t weight = 1;
                for (t = 0; t < n; ++t) {
                    unsigned raw = raw_code % 6; raw_code /= 6;
                    weight *= weights[row_family[t]][raw];
                    if (raw != previous && raw != 0) { ++emitted; only = raw; }
                    previous = raw;
                }
                if (emitted == 1 && (only == K1_WO || only == K1_WU))
                    numerator[only - K1_WO] += weight;
                ++enumerated_paths;
            }
            expect_ok(&f, &result);
            assert(result.log_prefix_mass == 0.0);
            for (t = 0; t < 2; ++t) {
                expect_mass(result.log_conditional_mass[t], numerator[t], power(8, n));
                assert(result.log_joint_mass[t] == result.log_conditional_mass[t]);
            }
            if (n == 0) assert(result.support == K1_SUPPORT_EMPTY_INTERVAL);
            else if (numerator[0] == 0 && numerator[1] == 0)
                assert(result.support == K1_SUPPORT_NEITHER);
            else if (numerator[1] == 0) assert(result.support == K1_SUPPORT_WO_ONLY);
            else if (numerator[0] == 0) assert(result.support == K1_SUPPORT_WU_ONLY);
            else assert(result.support == K1_SUPPORT_BOTH);
            ++cases;
        }
    }
    assert(cases == 585);
    assert(enumerated_paths == 112945);
    printf("exhaustive: %u row sequences, 1170 branch masses, %llu full six-label paths\n",
           cases, (unsigned long long)enumerated_paths);
}

static void directed_checks(void)
{
    fixture f;
    k1_result r, original;
    unsigned i;

    /* The prefix ends in raw Xiao. A continued Xiao in the CLOSED slot is excluded. */
    initialize(&f, 2); f.rows[3].p[K1_WO] = 0.0f; f.rows[3].p[K1_XIAO] = 1.0f;
    expect_ok(&f, &r); assert(r.support == K1_SUPPORT_NEITHER);
    initialize(&f, 3); f.rows[4].p[K1_WO] = 0.0f; f.rows[4].p[K1_BLANK] = 1.0f;
    expect_ok(&f, &r); assert(r.support == K1_SUPPORT_NEITHER); /* q blank q */

    /* A non-unit fixed W contributes its exact path product, independently of the slot. */
    initialize(&f, 1);
    for (i = 0; i < 3; ++i) { f.rows[i].p[i + 1] = 0.5f; f.rows[i].p[0] = 0.5f; }
    f.rows[3].p[K1_WO] = 0.25f; f.rows[3].p[K1_WU] = 0.5f; f.rows[3].p[K1_NI] = 0.25f;
    expect_ok(&f, &r);
    expect_mass(r.log_prefix_mass, 1, 8);
    expect_mass(r.log_conditional_mass[0], 1, 4);
    expect_mass(r.log_conditional_mass[1], 1, 2);
    expect_mass(r.log_joint_mass[0], 1, 32);
    expect_mass(r.log_joint_mass[1], 1, 16);

    /* Raw entry state is applied before prefix collapse. */
    initialize(&f, 1); f.prefix.entry_raw_label = K1_NI;
    expect_error(&f, K1_ERR_PREFIX_PATH);
    initialize(&f, 1); binding(&f, 5, 1); f.prefix.entry_raw_label = K1_NI;
    {
        const uint8_t path[6] = {K1_NI, K1_BLANK, K1_NI, K1_HAO, K1_XIAO, K1_WO};
        for (i = 0; i < 6; ++i) {
            memset(f.rows[i].p, 0, sizeof f.rows[i].p);
            f.rows[i].p[path[i]] = 1.0f; f.raw[i] = path[i];
        }
    }
    expect_ok(&f, &r); assert(r.support == K1_SUPPORT_WO_ONLY);

    initialize(&f, 1); binding(&f, 4, 1);
    memset(f.rows[3].p, 0, sizeof f.rows[3].p);
    f.rows[3].p[K1_BLANK] = 1.0f; f.raw[3] = K1_BLANK;
    f.rows[4].p[K1_WO] = 1.0f;
    expect_ok(&f, &r); assert(r.final_prefix_raw_label == K1_BLANK);

    /* Appended posterior rows only: deliberately late and invalid values remain unconsumed. */
    initialize(&f, 1); expect_ok(&f, &original);
    f.view.available_rows = 6;
    f.rows[4].available_at = UINT64_MAX; f.rows[4].p[K1_WU] = 1.0f;
    f.rows[5].available_at = UINT64_MAX; f.rows[5].p[K1_NI] = NAN;
    expect_ok(&f, &r);
    assert(r.log_conditional_mass[0] == original.log_conditional_mass[0]);
    assert(r.log_conditional_mass[1] == original.log_conditional_mass[1]);
    assert(r.log_joint_mass[0] == original.log_joint_mass[0]);
    assert(r.log_joint_mass[1] == original.log_joint_mass[1]);

    initialize(&f, 1); f.rows[3].p[K1_NI] = NAN; expect_error(&f, K1_ERR_PROBABILITY);
    initialize(&f, 1); f.rows[3].p[K1_HAO] = INFINITY; expect_error(&f, K1_ERR_PROBABILITY);
    initialize(&f, 1); f.rows[3].p[K1_XIAO] = -0.125f; expect_error(&f, K1_ERR_PROBABILITY);
    initialize(&f, 1); f.rows[3].p[K1_WO] = 1.125f; expect_error(&f, K1_ERR_PROBABILITY);
    initialize(&f, 1); f.rows[3].p[K1_WO] = 0.0f; expect_error(&f, K1_ERR_NORMALIZATION);
    initialize(&f, 1); f.rows[3].p[K1_BLANK] = 0.5f; expect_error(&f, K1_ERR_NORMALIZATION);
    initialize(&f, 1); ++f.prefix.binding.epoch; expect_error(&f, K1_ERR_BINDING);
    initialize(&f, 1); ++f.slot.binding.evidence_horizon; expect_error(&f, K1_ERR_BINDING);
    initialize(&f, 1); f.rows[0].available_at = 21; expect_error(&f, K1_ERR_LATE);
    initialize(&f, 1); f.rows[3].available_at = 21; expect_error(&f, K1_ERR_LATE);
    initialize(&f, 1); f.view.available_rows = 3; f.view.immutable_through = 103;
    expect_error(&f, K1_ERR_UNAVAILABLE);
    initialize(&f, 1); f.view.contiguous = 0; expect_error(&f, K1_ERR_GAPPED);
    initialize(&f, 1); f.prefix.run_closed = 0; expect_error(&f, K1_ERR_UNSEALED);
    initialize(&f, 1); f.view.immutable_through = 103; expect_error(&f, K1_ERR_UNSEALED);
    initialize(&f, 1); f.prefix.path_capacity = 2; expect_error(&f, K1_ERR_CAPACITY);
    initialize(&f, 1); f.view.capacity_rows = 3; expect_error(&f, K1_ERR_CAPACITY);
    initialize(&f, 1); f.view.capacity_rows = SIZE_MAX / sizeof(k1_row) + 1;
    expect_error(&f, K1_ERR_OVERFLOW);
    initialize(&f, 1); f.view.first_row = UINT64_MAX; expect_error(&f, K1_ERR_OVERFLOW);
    initialize(&f, 1); f.rows[0].p[K1_NI] = 0.0f; f.rows[0].p[K1_BLANK] = 1.0f;
    expect_error(&f, K1_ERR_PREFIX_UNSUPPORTED);
    initialize(&f, 254); expect_error(&f, K1_ERR_CAPACITY);
    initialize(&f, 1); f.prefix.raw_path = (const uint8_t *)f.rows;
    expect_error(&f, K1_ERR_OVERLAP);
    assert(success_calls == 592);
    assert(error_calls == 22);
    printf("directed: 7 successful calls and 22 atomic error checks\n");
}

int main(void)
{
    exhaustive_oracle();
    directed_checks();
    puts("independent fixed-slot CTC review tests: PASS");
    return 0;
}
