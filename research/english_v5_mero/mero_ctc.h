#ifndef ENGLISH_V5_MERO_CTC_H
#define ENGLISH_V5_MERO_CTC_H
#include <stdint.h>
/* Research-only one-keyword CTC token event decoder, not a calibrated
 * complete PCM/frontend/FSMN-CTC model. Exactly blank+HEY+MEH+ROH. */
#define EN_V5_MERO_TOKEN_COUNT 4u
#define EN_V5_MERO_KEYWORD_ID 1u
enum en_v5_mono_status {
    EN_V5_OK=0, EN_V5_INVALID=-1, EN_V5_CLOCK=-2,
    EN_V5_NONFINITE=-3
};
typedef struct {
    uint32_t progress;
    uint32_t previous_argmax;
    uint32_t clock_initialized;
    uint64_t last_sample;
    uint64_t last_progress_sample;
    uint64_t cooldown_until;
    uint64_t max_token_gap_samples;
    uint64_t cooldown_samples;
    uint64_t event_count;
} en_v5_mero_state;
typedef struct {
    uint32_t detected; /* 1 means exactly one fully ordered Hey Mero */
    uint32_t keyword_id; /* 1 on detection, otherwise 0 */
    uint64_t available_after_sample; /* evidence availability, NOT word end */
} en_v5_mero_event;

/* Caller owns one state per stream. No allocation, I/O or lookahead. */
int en_v5_mero_init(en_v5_mero_state *s, uint64_t max_gap_samples,
                    uint64_t cooldown_samples);
int en_v5_mero_reset(en_v5_mero_state *s);
/* On a real discontinuity declare an absolute future sample after the gap.
 * Reset both FSMN frontend state (outside this decoder) and this token state. */
int en_v5_mero_discontinuity(en_v5_mero_state *s,
                             uint64_t next_sample);
/* Supply 4 finite FRAME logits in vocabulary order blank/HEY/MEH/ROH.
 * sample clock strictly increases. Ties choose lower token index.
 * Detect only a complete greedy CTC sequence, never any partial word.
 * Outputs are uncalibrated research events; no FAR/FRR claims. */
int en_v5_mero_accept(en_v5_mero_state *s, const float logits[4],
                       uint64_t available_after_sample,
                       en_v5_mero_event *event);
#endif
