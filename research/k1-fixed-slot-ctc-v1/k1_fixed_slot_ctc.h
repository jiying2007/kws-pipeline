#ifndef K1_FIXED_SLOT_CTC_H
#define K1_FIXED_SLOT_CTC_H

#include <stddef.h>
#include <stdint.h>

/* RESEARCH capacity, not a latency, window-selection, or deployment default. */
#define K1_RESEARCH_MAX_USED_ROWS 256u
#define K1_CLASS_COUNT 6u
#define K1_FP32_SUM_TOLERANCE 1e-5

enum k1_label {
    K1_BLANK = 0, K1_NI = 1, K1_HAO = 2,
    K1_XIAO = 3, K1_WO = 4, K1_WU = 5
};

typedef struct {
    float p[K1_CLASS_COUNT];
    /* Actual row availability tick in the bound source/epoch/generation's
     * clock domain. Zero is a valid tick, not an unknown-time sentinel.
     * This is caller-supplied timing provenance, not verified clock evidence.
     */
    uint64_t available_at;
} k1_row;

/* All three objects must carry an EXACT copy of this caller-assigned handle.
 * Nonzero IDs identify an append-only source incarnation and one fixed range.
 * Equality is checked; truth, authenticity, and occurrence identity are not.
 * W occupies [prefix_begin, a); the terminal slot occupies [a, b).
 */
typedef struct {
    uint64_t source_id, epoch, generation, range_id;
    uint64_t prefix_begin, a, b;
    uint64_t evidence_horizon; /* fixed availability cutoff in the same clock */
} k1_binding;

typedef struct {
    k1_binding binding;
    const k1_row *rows;
    size_t available_rows, capacity_rows;
    uint64_t first_row;
    uint64_t immutable_through; /* exclusive absolute row coordinate */
    unsigned class_count;      /* must equal K1_CLASS_COUNT */
    unsigned append_only;      /* must equal 1: caller declaration */
    unsigned contiguous;       /* must equal 1: caller declaration */
} k1_view;

typedef struct {
    k1_binding binding;
    const uint8_t *raw_path;
    size_t path_count, path_capacity;
    uint8_t entry_raw_label;   /* preceding raw label, blank at true reset */
    unsigned immutable;       /* must equal 1: caller declaration */
    unsigned run_closed;      /* must equal 1 at a: caller declaration */
} k1_prefix;

typedef struct {
    k1_binding binding;
    unsigned sealed;          /* must equal 1: fixed interval [a,b) */
} k1_slot;

enum k1_error {
    K1_OK = 0,
    K1_ERR_ARGUMENT,
    K1_ERR_BINDING,
    K1_ERR_RANGE,
    K1_ERR_OVERFLOW,
    K1_ERR_CAPACITY,
    K1_ERR_GAPPED,
    K1_ERR_UNAVAILABLE,
    K1_ERR_LATE,
    K1_ERR_UNSEALED,
    K1_ERR_OVERLAP,
    K1_ERR_PROBABILITY,
    K1_ERR_NORMALIZATION,
    K1_ERR_PREFIX_PATH,
    K1_ERR_PREFIX_UNSUPPORTED
};

enum k1_support {
    K1_SUPPORT_EMPTY_INTERVAL = 0,
    K1_SUPPORT_NEITHER,
    K1_SUPPORT_WO_ONLY,
    K1_SUPPORT_WU_ONLY,
    K1_SUPPORT_BOTH
};

enum k1_provenance {
    K1_CALLER_BOUND_UNVERIFIED = 1
};
enum k1_boundary_status {
    K1_FIXED_COORDINATES_ACOUSTIC_BOUNDARIES_UNPROVEN = 1
};

typedef struct {
    k1_binding binding;        /* copied values; no borrowed pointers */
    enum k1_provenance provenance;
    enum k1_boundary_status boundaries;
    enum k1_support support;
    uint8_t entry_raw_label, final_prefix_raw_label;
    double log_prefix_mass;    /* log product for the ONE supplied W */
    /* Index 0 is WO (窝), index 1 is WU (屋).
     * Conditional means conditioned on this exact W, not normalized over q.
     * log_conditional_mass[q] = log sum_{v in blank* q+ blank*} P(v).
     * log_joint_mass[q] = log_prefix_mass + log_conditional_mass[q].
     * Exact zero mass is -INFINITY, never floored.
     */
    double log_conditional_mass[2];
    double log_joint_mass[2];
} k1_result;

/* No allocation, I/O, model/decoder calls, threshold, or decision policy.
 * Inputs and output are genuine, stable, pairwise disjoint C objects/buffers;
 * inputs remain immutable during the call. Declared full buffer extents are
 * checked for overlap; actual allocation sizes/lifetimes and races cannot be
 * established by this C API. A nonnull output must be writable and disjoint.
 * On EVERY error, output bytes are unchanged. NULL output returns ARGUMENT.
 * Only rows [prefix_begin,b) are inspected; appended values outside it are
 * ignored. Metadata and buffer bounds are always validated.
 */
enum k1_error k1_score(const k1_view *view, const k1_prefix *prefix,
                       const k1_slot *slot, k1_result *output);

#endif
