/* SPDX-License-Identifier: Apache-2.0
 * Independent ACTIVE113 research core; six scores plus OTHER original ID.
 * No full-vocabulary, probability, CTC, or acoustic qualification claim. */
#ifndef R1ACTIVE113_H
#define R1ACTIVE113_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define R1ACTIVE113_PAYLOAD_BYTES 1582624u
#define R1ACTIVE113_INPUTS 400u
#define R1ACTIVE113_OUTPUTS 6u
#define R1ACTIVE113_CACHE_FLOATS 5632u
#define R1ACTIVE113_TRACE_FLOATS 5006u
#define R1ACTIVE113_TRACE_FINAL_OFFSET 5000u
#define R1ACTIVE113_PAYLOAD_LITTLE_ENDIAN 1u
#define R1ACTIVE113_VIEW_ALIGNMENT 8u
typedef enum {
    R1ACTIVE113_OK=0,R1ACTIVE113_ARGUMENT=1,R1ACTIVE113_SIZE=2,
    R1ACTIVE113_IDENTITY=3,R1ACTIVE113_NONFINITE=4,R1ACTIVE113_STATE=5,
    R1ACTIVE113_ENVIRONMENT=6,R1ACTIVE113_OVERLAP=7
} r1active113_status;
typedef struct {
    const unsigned char *payload;
    const float *weights;
    const uint32_t *other_ids;
    const int8_t *qhead;
    const double *scales;
    const float *biases;
    const float *precise_head;
    const float *precise_biases;
    double cache[R1ACTIVE113_CACHE_FLOATS];
    uint32_t magic,other_winner_id;
    int fault;
} r1active113_model;
/* Trusted expected_sha256 is 64 lowercase hex bytes plus NUL. It must come
 * from the export manifest, not be derived from an untrusted payload.
 * Copy destination is caller-owned untyped allocated storage, 8-byte aligned.
 * The view payload is immutable aligned malloc/read or read-only mmap storage;
 * it must outlive the model. Do not originate it from a declared byte array.
 * Copy destination remains alive after load; source may then be released.
 * All buffers are disjoint. Failure leaves model and copy storage unchanged.
 * RNE/little-endian/gradual underflow required; caller masks FP traps.
 * Compile with the fixed no-fast-math/no-contraction build flags. */
r1active113_status r1active113_load(r1active113_model *,void *storage,size_t storage_bytes,
    const unsigned char *payload,size_t bytes,unsigned byte_order,const char expected_sha256[65]);
r1active113_status r1active113_load_view(r1active113_model *,const unsigned char *payload,
    size_t bytes,unsigned byte_order,const char expected_sha256[65]);
r1active113_status r1active113_reset(r1active113_model *);
r1active113_status r1active113_copy_cache(const r1active113_model *,double *,size_t);
/* Scores0..4 retain original IDs0,1027,1401,1462,2093; score5 is the maximum
 * of113 unchanged OTHER rows. Getter supplies its original winner ID.
 * Global winner compares score, then smaller ORIGINAL ID. Never treat index5
 * or a fixed sentinel as that ID. Trace stages0..19 are unchanged, stage20
 * contains only six scores. Capacities are exact except trace>=5006.
 * No implicit reset. Arithmetic failure faults state until reset. */
r1active113_status r1active113_step(r1active113_model *,const double *,size_t,
    double *,size_t,double *,size_t);
/* UINT32_MAX until a successful step or whenever state is invalid/faulted. */
uint32_t r1active113_other_winner(const r1active113_model *);
int r1active113_environment_ok(void);
#ifdef __cplusplus
}
#endif
#endif
