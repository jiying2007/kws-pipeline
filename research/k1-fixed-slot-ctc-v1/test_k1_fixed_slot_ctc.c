#include "k1_fixed_slot_ctc.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define FIXTURE_ROWS 300u

typedef struct {
    k1_row rows[FIXTURE_ROWS];
    uint8_t raw[FIXTURE_ROWS];
    k1_view view;
    k1_prefix prefix;
    k1_slot slot;
} fixture;

static uint64_t terminal_paths, prefix_paths, terminal_matrices;

static void bind_all(fixture *f, k1_binding h)
{
    f->view.binding = h;
    f->prefix.binding = h;
    f->slot.binding = h;
}

static void one_hot(k1_row *row, enum k1_label label)
{
    memset(row, 0, sizeof *row);
    row->p[label] = 1.0f;
}

static void init(fixture *f, size_t prefix_n, size_t slot_n)
{
    k1_binding h = {11, 22, 33, 44, 100, 100 + prefix_n,
                    100 + prefix_n + slot_n, 1000};
    size_t i;
    memset(f, 0, sizeof *f);
    f->view.rows = f->rows;
    f->view.available_rows = prefix_n + slot_n;
    f->view.capacity_rows = FIXTURE_ROWS;
    f->view.first_row = 100;
    f->view.immutable_through = h.b;
    f->view.class_count = K1_CLASS_COUNT;
    f->view.append_only = 1;
    f->view.contiguous = 1;
    f->prefix.raw_path = f->raw;
    f->prefix.path_count = prefix_n;
    f->prefix.path_capacity = FIXTURE_ROWS;
    f->prefix.entry_raw_label = K1_BLANK;
    f->prefix.immutable = 1;
    f->prefix.run_closed = 1;
    f->slot.sealed = 1;
    bind_all(f, h);
    for (i = 0; i < FIXTURE_ROWS; ++i) one_hot(&f->rows[i], K1_BLANK);
    if (prefix_n >= 3) {
        f->raw[0] = K1_NI;
        f->raw[1] = K1_HAO;
        f->raw[2] = K1_XIAO;
        for (i = 0; i < prefix_n; ++i)
            one_hot(&f->rows[i], (enum k1_label)f->raw[i]);
    }
}

static k1_result score(fixture *f)
{
    k1_result r;
    assert(k1_score(&f->view, &f->prefix, &f->slot, &r) == K1_OK);
    assert(r.provenance == K1_CALLER_BOUND_UNVERIFIED);
    assert(r.boundaries == K1_FIXED_COORDINATES_ACOUSTIC_BOUNDARIES_UNPROVEN);
    return r;
}

static void error_unchanged(fixture *f, enum k1_error error)
{
    k1_result r;
    unsigned char before[sizeof r];
    memset(&r, 0xa5, sizeof r);
    memcpy(before, &r, sizeof r);
    assert(k1_score(&f->view, &f->prefix, &f->slot, &r) == error);
    assert(memcmp(before, &r, sizeof r) == 0);
}

static void close_double(double x, double y)
{
    if (isinf(y)) assert(x == y);
    else assert(isfinite(x) && fabs(x - y) <= 2e-12 * (1.0 + fabs(y)));
}

static void same_scores(const k1_result *x, const k1_result *y)
{
    size_t j;
    assert(x->support == y->support);
    assert(x->log_prefix_mass == y->log_prefix_mass);
    assert(x->provenance == y->provenance && x->boundaries == y->boundaries);
    for (j = 0; j < 2; ++j) {
        assert(x->log_conditional_mass[j] == y->log_conditional_mass[j]);
        assert(x->log_joint_mass[j] == y->log_joint_mass[j]);
    }
}

/* Independent oracle: materialize every raw path over all SIX labels,
 * merge adjacent equal labels, then remove blanks. No scorer states or
 * recurrence are used. Zero-product paths are enumerated too.
 */
static size_t collapse(const uint8_t *raw, size_t n, uint8_t entry,
                       uint8_t *labels)
{
    size_t i, length = 0;
    uint8_t last = entry;
    for (i = 0; i < n; ++i) {
        if (raw[i] != last && raw[i] != K1_BLANK) labels[length++] = raw[i];
        last = raw[i];
    }
    return length;
}

static void enumerate_terminal(const k1_row *rows, size_t n, size_t at,
                                uint8_t *raw, double product, double sums[2])
{
    size_t j;
    if (at == n) {
        uint8_t labels[8];
        size_t count = collapse(raw, n, K1_BLANK, labels);
        ++terminal_paths;
        if (count == 1 && labels[0] == K1_WO) sums[0] += product;
        if (count == 1 && labels[0] == K1_WU) sums[1] += product;
        return;
    }
    for (j = 0; j < K1_CLASS_COUNT; ++j) {
        raw[at] = (uint8_t)j;
        enumerate_terminal(rows, n, at + 1, raw,
                           product * (double)rows[at].p[j], sums);
    }
}

static void oracle_case(fixture *f, size_t slot_n)
{
    uint8_t raw[8];
    double sums[2] = {0.0, 0.0};
    size_t j;
    k1_result r = score(f);
    assert(slot_n <= sizeof raw);
    enumerate_terminal(&f->rows[f->prefix.path_count], slot_n, 0, raw, 1.0, sums);
    ++terminal_matrices;
    for (j = 0; j < 2; ++j) {
        double expected = sums[j] == 0.0 ? -INFINITY : log(sums[j]);
        close_double(r.log_conditional_mass[j], expected);
        close_double(r.log_joint_mass[j], r.log_prefix_mass + expected);
    }
    if (slot_n == 0) assert(r.support == K1_SUPPORT_EMPTY_INTERVAL);
    else if (sums[0] == 0.0 && sums[1] == 0.0) assert(r.support == K1_SUPPORT_NEITHER);
    else if (sums[1] == 0.0) assert(r.support == K1_SUPPORT_WO_ONLY);
    else if (sums[0] == 0.0) assert(r.support == K1_SUPPORT_WU_ONLY);
    else assert(r.support == K1_SUPPORT_BOTH);
}

static void test_exhaustive_terminal_paths(void)
{
    static const k1_row palette[7] = {
        {.p = {1, 0, 0, 0, 0, 0}}, {.p = {0, 0, 0, 0, 1, 0}},
        {.p = {0, 0, 0, 0, 0, 1}}, {.p = {0, 0, 0, 1, 0, 0}},
        {.p = {0.5f, 0, 0, 0, 0.25f, 0.25f}},
        {.p = {0.25f, 0.25f, 0.25f, 0.25f, 0, 0}},
        {.p = {0.125f, 0.125f, 0.125f, 0.125f, 0.25f, 0.25f}}
    };
    fixture f;
    size_t n, combinations = 1;
    for (n = 0; n <= 4; ++n) {
        size_t matrix;
        for (matrix = 0; matrix < combinations; ++matrix) {
            size_t code = matrix, i;
            init(&f, 3, n);
            /* W mass = (1/2)*(1/4)*(1/8), kept separate from slot sums. */
            for (i = 0; i < 3; ++i) {
                float chosen = i == 0 ? 0.5f : (i == 1 ? 0.25f : 0.125f);
                f.rows[i].p[f.raw[i]] = chosen;
                f.rows[i].p[K1_BLANK] = 1.0f - chosen;
            }
            for (i = 0; i < n; ++i) {
                f.rows[3 + i] = palette[code % 7];
                code /= 7;
            }
            oracle_case(&f, n);
            close_double(score(&f).log_prefix_mass, log(1.0 / 64.0));
        }
        combinations *= 7;
    }
    assert(terminal_matrices == 2801);
    assert(terminal_paths == UINT64_C(3187591));
}

static void enumerate_prefix(fixture *f, size_t n, size_t at)
{
    size_t j;
    if (at == n) {
        uint8_t labels[8];
        size_t count = collapse(f->raw, n, f->prefix.entry_raw_label, labels);
        int valid = count == 3 && labels[0] == K1_NI &&
                    labels[1] == K1_HAO && labels[2] == K1_XIAO;
        ++prefix_paths;
        if (valid) {
            double expected = 0.0;
            k1_result r = score(f);
            assert(n != 0 && (f->raw[n - 1] == K1_BLANK || f->raw[n - 1] == K1_XIAO));
            for (j = 0; j < n; ++j) expected += log((double)f->rows[j].p[f->raw[j]]);
            close_double(r.log_prefix_mass, expected);
            close_double(r.log_conditional_mass[0], log(0.5));
            close_double(r.log_conditional_mass[1], log(0.5));
        } else error_unchanged(f, K1_ERR_PREFIX_PATH);
        return;
    }
    for (j = 0; j < K1_CLASS_COUNT; ++j) {
        f->raw[at] = (uint8_t)j;
        enumerate_prefix(f, n, at + 1);
    }
}

static void test_exhaustive_prefix_paths(void)
{
    fixture f;
    size_t n, entry, i;
    const k1_row full = {.p = {0.375f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f}};
    const k1_row terminal = {.p = {0, 0, 0, 0, 0.5f, 0.5f}};
    for (n = 0; n <= 5; ++n)
        for (entry = 0; entry < K1_CLASS_COUNT; ++entry) {
            init(&f, n, 1);
            f.prefix.entry_raw_label = (uint8_t)entry;
            for (i = 0; i < n; ++i) f.rows[i] = full;
            f.rows[n] = terminal;
            enumerate_prefix(&f, n, 0);
        }
    assert(prefix_paths == UINT64_C(55986));
}

static void test_named_language_cases(void)
{
    fixture f;
    k1_result r;
    init(&f, 3, 0);
    r = score(&f);
    assert(r.support == K1_SUPPORT_EMPTY_INTERVAL);
    assert(r.log_conditional_mass[0] == -INFINITY && r.log_joint_mass[1] == -INFINITY);
    init(&f, 3, 4);
    r = score(&f);
    assert(r.support == K1_SUPPORT_NEITHER); /* nonempty all blank */
    one_hot(&f.rows[3], K1_WO);
    one_hot(&f.rows[5], K1_WO); /* WO blank WO blank is excluded */
    r = score(&f);
    assert(r.support == K1_SUPPORT_NEITHER);
    one_hot(&f.rows[4], K1_WO); /* WO WO WO blank is one run */
    r = score(&f);
    assert(r.support == K1_SUPPORT_WO_ONLY && r.log_conditional_mass[0] == 0.0);
    one_hot(&f.rows[4], K1_WU); /* other branch anywhere excludes both */
    assert(score(&f).support == K1_SUPPORT_NEITHER);
    init(&f, 3, 1);
    one_hot(&f.rows[3], K1_XIAO); /* no continuation of the prefix run */
    assert(score(&f).support == K1_SUPPORT_NEITHER);
    f.rows[3] = (k1_row){.p = {0, 0, 0, 0.5f, 0.25f, 0.25f}};
    r = score(&f);
    assert(r.support == K1_SUPPORT_BOTH);
    assert(r.final_prefix_raw_label == K1_XIAO);
    assert(r.log_conditional_mass[0] == r.log_conditional_mass[1]);
    close_double(r.log_conditional_mass[0], log(0.25)); /* no subset normalization */
    init(&f, 4, 1); /* same prefix with trailing blank */
    f.rows[4] = (k1_row){.p = {0, 0, 0, 0, 0.25f, 0.75f}};
    r = score(&f);
    assert(r.final_prefix_raw_label == K1_BLANK);
    close_double(r.log_conditional_mass[1], log(0.75));
    /* Boundary entry NI suppresses first NI, so the naive three-label W fails. */
    init(&f, 3, 1);
    f.prefix.entry_raw_label = K1_NI;
    error_unchanged(&f, K1_ERR_PREFIX_PATH);
    init(&f, 4, 1);
    f.prefix.entry_raw_label = K1_NI;
    f.raw[0] = K1_BLANK; f.raw[1] = K1_NI; f.raw[2] = K1_HAO; f.raw[3] = K1_XIAO;
    for (size_t i = 0; i < 4; ++i) one_hot(&f.rows[i], (enum k1_label)f.raw[i]);
    assert(score(&f).entry_raw_label == K1_NI);
}

static void test_append_invariance_and_unproven_coordinates(void)
{
    fixture f;
    k1_result before, after;
    k1_binding h;
    init(&f, 3, 2);
    f.rows[3] = (k1_row){.p = {0.5f, 0, 0, 0, 0.25f, 0.25f}};
    f.rows[4] = f.rows[3];
    before = score(&f);
    /* Availability expands but b and all used bytes remain fixed. */
    f.view.available_rows = FIXTURE_ROWS;
    for (size_t i = 5; i < FIXTURE_ROWS; ++i) {
        for (size_t j = 0; j < K1_CLASS_COUNT; ++j) f.rows[i].p[j] = NAN;
        f.rows[i].available_at = UINT64_MAX;
    }
    after = score(&f);
    same_scores(&before, &after); /* proves appended values aren't scanned */
    one_hot(&f.rows[99], K1_WU); /* separately later WO/WU evidence has no veto */
    after = score(&f);
    same_scores(&before, &after);
    /* A separate, later candidate with its own W and range handle. */
    f.raw[0] = K1_NI; f.raw[1] = K1_HAO; f.raw[2] = K1_XIAO;
    one_hot(&f.rows[96], K1_NI); one_hot(&f.rows[97], K1_HAO);
    one_hot(&f.rows[98], K1_XIAO);
    h = f.view.binding;
    h.range_id = 45; h.prefix_begin = 196; h.a = 199; h.b = 200;
    bind_all(&f, h);
    f.view.immutable_through = 200;
    assert(score(&f).support == K1_SUPPORT_WU_ONLY);
    bind_all(&f, before.binding);
    after = score(&f);
    same_scores(&before, &after);
    /* A caller can consistently declare different coordinates; their acoustic
     * meaning remains unproven even when they produce nonzero masses. */
    h = f.view.binding; h.range_id = 46; h.a = 104; h.prefix_begin = 101;
    bind_all(&f, h);
    f.rows[1] = (k1_row){.p = {0, 0.5f, 0.5f, 0, 0, 0}};
    f.rows[2] = (k1_row){.p = {0, 0, 0.5f, 0.5f, 0, 0}};
    f.rows[3] = (k1_row){.p = {0, 0, 0, 0.5f, 0.25f, 0.25f}};
    f.rows[0].p[K1_NI] = NAN; /* before prefix_begin is also unconsumed */
    f.rows[0].available_at = UINT64_MAX;
    after = score(&f);
    assert(after.boundaries == K1_FIXED_COORDINATES_ACOUSTIC_BOUNDARIES_UNPROVEN);
    assert(after.provenance == K1_CALLER_BOUND_UNVERIFIED);
    h.source_id = 99;
    bind_all(&f, h);
    assert(after.binding.source_id == 11); /* output owns copied values */
}

static void test_probability_errors_and_precision(void)
{
    fixture f;
    const float bad[] = {NAN, INFINITY, -INFINITY, -0.125f, 1.125f};
    size_t i, j;
    for (i = 0; i < sizeof bad / sizeof bad[0]; ++i)
        for (j = 0; j < K1_CLASS_COUNT; ++j) {
            init(&f, 3, 1);
            f.rows[3].p[j] = bad[i];
            error_unchanged(&f, K1_ERR_PROBABILITY);
        }
    init(&f, 3, 1);
    f.rows[0].p[K1_WU] = NAN; /* unselected prefix class still validated */
    error_unchanged(&f, K1_ERR_PROBABILITY);
    init(&f, 3, 1);
    f.rows[3] = (k1_row){.p = {0, 0, 0, 0, 0.25f, 0.25f}};
    error_unchanged(&f, K1_ERR_NORMALIZATION);
    f.rows[3] = (k1_row){.p = {0.75f, 0, 0, 0, 0.25f, 0.25f}};
    error_unchanged(&f, K1_ERR_NORMALIZATION);
    f.rows[3] = (k1_row){.p = {0.5f, 0, 0, 0, 0.25f, 0.25f + 1e-6f}};
    close_double(score(&f).log_conditional_mass[1], log((double)f.rows[3].p[K1_WU]));
    f.rows[3].p[K1_WU] = 0.25f + 2e-5f;
    error_unchanged(&f, K1_ERR_NORMALIZATION);
    init(&f, 3, 2);
    f.rows[3] = (k1_row){.p = {0x1p-20f, 0, 0, 0, 1, 0}};
    f.rows[4] = f.rows[3];
    assert(score(&f).log_conditional_mass[0] > 0.0);
    close_double(score(&f).log_conditional_mass[0], log1p(0x1p-19));
    init(&f, 3, 1);
    one_hot(&f.rows[1], K1_BLANK);
    error_unchanged(&f, K1_ERR_PREFIX_UNSUPPORTED);
    init(&f, 3, 1);
    f.rows[3].p[K1_WO] = 0x1p-149f; /* binary32 minimum subnormal, no floor */
    close_double(score(&f).log_conditional_mass[0], log((double)0x1p-149f));
    f.rows[3].p[K1_WU] = -0.0f;
    assert(score(&f).log_conditional_mass[1] == -INFINITY);
    /* Product underflows even binary64, but the log mass remains finite. */
    init(&f, 3, K1_RESEARCH_MAX_USED_ROWS - 3);
    for (i = 3; i < K1_RESEARCH_MAX_USED_ROWS; ++i)
        f.rows[i] = (k1_row){.p = {0, 1, 0, 0, 0x1p-149f, 0}};
    close_double(score(&f).log_conditional_mass[0],
                 (K1_RESEARCH_MAX_USED_ROWS - 3) * log((double)0x1p-149f));
    assert(score(&f).support == K1_SUPPORT_WO_ONLY);
}

static void test_invalid_metadata_and_atomicity(void)
{
    fixture f;
    k1_binding h;
    k1_result r;
    unsigned char before[sizeof r];
    init(&f, 3, 1);
    memset(&r, 0xa5, sizeof r); memcpy(before, &r, sizeof r);
    assert(k1_score(NULL, &f.prefix, &f.slot, &r) == K1_ERR_ARGUMENT);
    assert(k1_score(&f.view, NULL, &f.slot, &r) == K1_ERR_ARGUMENT);
    assert(k1_score(&f.view, &f.prefix, NULL, &r) == K1_ERR_ARGUMENT);
    assert(k1_score(&f.view, &f.prefix, &f.slot, NULL) == K1_ERR_ARGUMENT);
    assert(memcmp(before, &r, sizeof r) == 0);
    f.view.rows = NULL; error_unchanged(&f, K1_ERR_ARGUMENT);
    init(&f, 3, 1); f.prefix.raw_path = NULL; error_unchanged(&f, K1_ERR_ARGUMENT);
    init(&f, 3, 1); f.view.class_count = 3; error_unchanged(&f, K1_ERR_ARGUMENT);
    init(&f, 3, 1); f.view.append_only = 0; error_unchanged(&f, K1_ERR_ARGUMENT);
    init(&f, 3, 1); f.view.contiguous = 0; error_unchanged(&f, K1_ERR_GAPPED);
    init(&f, 3, 1); f.prefix.binding.source_id++; error_unchanged(&f, K1_ERR_BINDING);
    init(&f, 3, 1); f.prefix.binding.epoch++; error_unchanged(&f, K1_ERR_BINDING);
    init(&f, 3, 1); f.slot.binding.generation++; error_unchanged(&f, K1_ERR_BINDING);
    init(&f, 3, 1); f.slot.binding.range_id++; error_unchanged(&f, K1_ERR_BINDING);
    init(&f, 3, 1); f.slot.binding.a++; error_unchanged(&f, K1_ERR_BINDING);
    init(&f, 3, 1); f.prefix.binding.prefix_begin++; error_unchanged(&f, K1_ERR_BINDING);
    init(&f, 3, 1); f.slot.binding.b++; error_unchanged(&f, K1_ERR_BINDING);
    init(&f, 3, 1); f.prefix.binding.evidence_horizon++;
    error_unchanged(&f, K1_ERR_BINDING);
    init(&f, 3, 1); h = f.view.binding; h.range_id = 0; bind_all(&f, h);
    error_unchanged(&f, K1_ERR_BINDING);
    init(&f, 3, 1); h = f.view.binding; h.a = h.prefix_begin - 1; bind_all(&f, h);
    error_unchanged(&f, K1_ERR_RANGE);
    init(&f, 3, 1); h = f.view.binding; h.b = h.a - 1; bind_all(&f, h);
    error_unchanged(&f, K1_ERR_RANGE);
    init(&f, 3, 1); f.view.capacity_rows = SIZE_MAX;
    error_unchanged(&f, K1_ERR_OVERFLOW);
    init(&f, 3, 1); f.view.first_row = UINT64_MAX - 1;
    error_unchanged(&f, K1_ERR_OVERFLOW);
    init(&f, 3, 1); h = f.view.binding;
    h.prefix_begin = UINT64_MAX - 4; h.a = UINT64_MAX - 1; h.b = UINT64_MAX;
    bind_all(&f, h); f.view.first_row = h.prefix_begin;
    f.view.immutable_through = UINT64_MAX; (void)score(&f);
    init(&f, 3, 1); f.view.capacity_rows = 3; error_unchanged(&f, K1_ERR_CAPACITY);
    init(&f, 3, 1); f.prefix.path_capacity = 2; error_unchanged(&f, K1_ERR_CAPACITY);
    init(&f, 3, 1); f.view.immutable_through = 105; error_unchanged(&f, K1_ERR_RANGE);
    init(&f, 3, 1); f.view.immutable_through = 99; error_unchanged(&f, K1_ERR_RANGE);
    init(&f, 3, 1); f.view.first_row = 101; error_unchanged(&f, K1_ERR_UNAVAILABLE);
    init(&f, 3, 1); f.view.available_rows = 3; f.view.immutable_through = 103;
    error_unchanged(&f, K1_ERR_UNAVAILABLE); /* terminal evidence has not arrived */
    init(&f, 3, 1); f.rows[3].available_at = 1001; error_unchanged(&f, K1_ERR_LATE);
    init(&f, 3, 1); f.rows[0].available_at = 1001; error_unchanged(&f, K1_ERR_LATE);
    init(&f, 3, 1); f.rows[3].available_at = 1000; (void)score(&f);
    init(&f, 3, 1); f.view.immutable_through = 103; error_unchanged(&f, K1_ERR_UNSEALED);
    init(&f, 3, 1); f.slot.sealed = 0; error_unchanged(&f, K1_ERR_UNSEALED);
    init(&f, 3, 1); f.prefix.immutable = 0; error_unchanged(&f, K1_ERR_UNSEALED);
    init(&f, 3, 1); f.prefix.run_closed = 0; error_unchanged(&f, K1_ERR_UNSEALED);
    init(&f, 3, 1); f.prefix.path_count = 4; error_unchanged(&f, K1_ERR_PREFIX_PATH);
    init(&f, 3, 1); f.prefix.entry_raw_label = 6; error_unchanged(&f, K1_ERR_PREFIX_PATH);
    init(&f, 3, 1); f.raw[2] = 6; error_unchanged(&f, K1_ERR_PREFIX_PATH);
    init(&f, 3, 1); f.raw[2] = K1_BLANK; one_hot(&f.rows[2], K1_BLANK);
    error_unchanged(&f, K1_ERR_PREFIX_PATH);
    init(&f, 3, K1_RESEARCH_MAX_USED_ROWS - 2); error_unchanged(&f, K1_ERR_CAPACITY);
    /* Actual buffers stay valid; deliberately declare overlapping extents. */
    init(&f, 3, 1); f.prefix.raw_path = (const uint8_t *)(const void *)&f.rows[0];
    error_unchanged(&f, K1_ERR_OVERLAP);
    init(&f, 3, 1); f.prefix.path_capacity = SIZE_MAX;
    error_unchanged(&f, K1_ERR_OVERFLOW);
    init(&f, 3, 1);
    {
        unsigned char snapshot[sizeof f.rows];
        memcpy(snapshot, f.rows, sizeof snapshot);
        /* rows alignment suffices for result on our verified ABI; avoid
         * asserting any portable permission to dereference this cast. */
        _Static_assert(_Alignof(fixture) >= _Alignof(k1_result), "test alignment");
        assert(k1_score(&f.view, &f.prefix, &f.slot,
                        (k1_result *)(void *)f.rows) == K1_ERR_OVERLAP);
        assert(memcmp(snapshot, f.rows, sizeof snapshot) == 0);
    }
}

int main(void)
{
    test_exhaustive_terminal_paths();
    test_exhaustive_prefix_paths();
    test_named_language_cases();
    test_append_invariance_and_unproven_coordinates();
    test_probability_errors_and_precision();
    test_invalid_metadata_and_atomicity();
    printf("PASS: 6 named semantic test groups; %llu terminal matrices; "
           "%llu exhaustive terminal raw paths; %llu exhaustive prefix/entry paths\n",
           (unsigned long long)terminal_matrices,
           (unsigned long long)terminal_paths, (unsigned long long)prefix_paths);
    return 0;
}
