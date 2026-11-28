#ifndef ENGLISH_V7_CTC_H
#define ENGLISH_V7_CTC_H
#include <stdint.h>
/* Research-only candidate-specific CTC token event decoder, not a calibrated
 * complete PCM/frontend/FSMN-CTC model. Candidate-relative blank+token-1+token-2+token-3. */
#define EN_V7_CTC_TOKEN_COUNT 4u
#define EN_V7_CTC_KEYWORD_ID_MIN 1u
#define EN_V7_CTC_KEYWORD_ID_MAX 2u
enum en_v7_ctc_status {
    EN_V7_CTC_OK=0, EN_V7_CTC_INVALID=-1, EN_V7_CTC_CLOCK=-2,
    EN_V7_CTC_NONFINITE=-3
};
typedef struct {
    uint32_t keyword_id;
    uint32_t progress;
    uint32_t previous_argmax;
    uint32_t clock_initialized;
    uint64_t last_sample;
    uint64_t last_progress_sample;
    uint64_t cooldown_until;
    uint64_t max_token_gap_samples;
    uint64_t cooldown_samples;
    uint64_t event_count;
} en_v7_ctc_state;
typedef struct {
    uint32_t detected; /* 1 means one fully ordered candidate keyword */
    uint32_t keyword_id; /* candidate identity on detection, otherwise 0 */
    uint64_t available_after_sample; /* evidence availability, NOT word end */
} en_v7_ctc_event;

/* Caller owns one state per stream. No allocation, I/O or lookahead. */
int en_v7_ctc_init(en_v7_ctc_state *s, uint32_t keyword_id,
                    uint64_t max_gap_samples, uint64_t cooldown_samples);
int en_v7_ctc_reset(en_v7_ctc_state *s);
/* On a real discontinuity declare an absolute future sample after the gap.
 * Reset both FSMN frontend state (outside this decoder) and this token state. */
int en_v7_ctc_discontinuity(en_v7_ctc_state *s,
                             uint64_t next_sample);
/* Supply 4 finite FRAME logits in vocabulary order blank/token-1/token-2/token-3.
 * sample clock strictly increases. Ties choose lower token index.
 * Detect only a complete greedy CTC sequence, never any partial word.
 * Outputs are uncalibrated research events; no FAR/FRR claims. */
int en_v7_ctc_accept(en_v7_ctc_state *s, const float logits[4],
                       uint64_t available_after_sample,
                       en_v7_ctc_event *event);
#endif
