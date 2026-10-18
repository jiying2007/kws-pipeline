/* SPDX-License-Identifier: Apache-2.0
 * Modified research port of WeKws FSMN, Yueyue Nyy (2022), Jing Du (2023).
 * See NOTICE.md. Not an audio frontend, decoder, or qualified board runtime. */
#ifndef FSMN2599_H
#define FSMN2599_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define FSMN2599_FLOATS 756933u
#define FSMN2599_PAYLOAD_BYTES (FSMN2599_FLOATS * 4u)
#define FSMN2599_INPUTS 400u
#define FSMN2599_OUTPUTS 2599u
#define FSMN2599_CACHE_FLOATS 5632u
#define FSMN2599_TRACE_FLOATS 7599u
#define FSMN2599_TRACE_FINAL_OFFSET 5000u
#define FSMN2599_HEAD_WEIGHT_OFFSET 390474u
#define FSMN2599_HEAD_BIAS_OFFSET 754334u
#define FSMN2599_PAYLOAD_LITTLE_ENDIAN 1u

typedef enum {
    FSMN2599_OK = 0, FSMN2599_ARGUMENT = 1, FSMN2599_SIZE = 2,
    FSMN2599_IDENTITY = 3, FSMN2599_NONFINITE = 4, FSMN2599_STATE = 5,
    FSMN2599_ENVIRONMENT = 6, FSMN2599_OVERLAP = 7
} fsmn2599_status;

typedef struct {
    const float *weights; /* caller-owned, immutable throughout model lifetime */
    float cache[FSMN2599_CACHE_FLOATS]; /* [1][128][11][4], layer contiguous */
    uint32_t magic;
    int fault;
} fsmn2599_model;

/* Caller must provide the stated accessible capacities. No allocation or I/O.
 * Headerless payload is little-endian FP32; endian selection is explicit.
 * expected_sha256 must be a readable 65-byte NUL-terminated lowercase hex
 * identity from a trusted export manifest, not computed from untrusted input.
 * Host is IEEE FP32 little-endian, RNE, gradual-underflow; never changes FP mode.
 * Floating-point exception traps MUST be masked by the caller. The portable
 * environment check does not verify trap masks; an unmasked trap can SIGFPE
 * before a NONFINITE return, including during sNaN validation or overflow.
 * Build with the fixed no-contraction/no-fast-math flags in build_synthetic.py;
 * environment_ok cannot detect compiler reassociation or fused instructions.
 * On loader failure model and decoded_weights are unchanged. Buffers disjoint.
 * The payload can be released after success; decoded_weights must remain alive. */
fsmn2599_status fsmn2599_load(fsmn2599_model *model,
    float *decoded_weights, size_t weight_floats,
    const unsigned char *payload, size_t payload_bytes, unsigned payload_byte_order,
    const char expected_sha256[65]);

fsmn2599_status fsmn2599_reset(fsmn2599_model *model);
fsmn2599_status fsmn2599_copy_cache(const fsmn2599_model *model,
    float *output, size_t output_floats);

/* One pre-spliced 400D row, including CMVN internally, to 2599 raw logits.
 * input/output sizes must match exactly. Optional trace requires >=7599 floats:
 * stages 0..19 have stride250 (unused slots zero); final2599 begins at5000.
 * All buffers must be disjoint from state, weights and each other.
 * Invalid arguments leave state/output/trace unchanged. Arithmetic failure
 * faults the stream; cache/trace/output are invalid and must be discarded until
 * reset. No softmax, class reduction, quantization or implicit state reset.
 * Single-threaded and non-reentrant. */
fsmn2599_status fsmn2599_step(fsmn2599_model *model,
    const float *input, size_t input_floats, float *output, size_t output_floats,
    float *trace, size_t trace_floats);

/* Checks little-endian/RNE/gradual-underflow, not exception-trap mask state. */
int fsmn2599_environment_ok(void);
#ifdef __cplusplus
}
#endif
#endif
