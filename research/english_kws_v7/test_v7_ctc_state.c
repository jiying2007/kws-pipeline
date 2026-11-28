#include "v7_ctc.h"

#include <math.h>
#include <stdint.h>

static int emit_sequence(uint32_t candidate_id)
{
    en_v7_ctc_state state;
    en_v7_ctc_event event;
    float logits[4] = {-4.0f, -4.0f, -4.0f, -4.0f};

    if (en_v7_ctc_init(&state, candidate_id, 16000u, 0u) != EN_V7_CTC_OK) return 1;
    if (state.keyword_id != candidate_id) return 2;

    logits[1] = 4.0f;
    if (en_v7_ctc_accept(&state, logits, 160u, &event) != EN_V7_CTC_OK ||
        event.detected != 0u) return 3;
    logits[1] = -4.0f;
    logits[2] = 4.0f;
    if (en_v7_ctc_accept(&state, logits, 320u, &event) != EN_V7_CTC_OK ||
        event.detected != 0u) return 4;
    logits[2] = -4.0f;
    logits[3] = 4.0f;
    if (en_v7_ctc_accept(&state, logits, 480u, &event) != EN_V7_CTC_OK ||
        event.detected != 1u || event.keyword_id != candidate_id ||
        state.event_count != 1u) return 5;

    if (en_v7_ctc_reset(&state) != EN_V7_CTC_OK ||
        state.keyword_id != candidate_id || state.event_count != 1u) return 6;
    if (en_v7_ctc_discontinuity(&state, 1000u) != EN_V7_CTC_OK ||
        state.keyword_id != candidate_id) return 7;
    if (en_v7_ctc_accept(&state, logits, 1000u, &event) != EN_V7_CTC_CLOCK) return 8;
    return 0;
}

int main(void)
{
    en_v7_ctc_state state;
    if (emit_sequence(1u) != 0) return 1;
    if (emit_sequence(2u) != 0) return 2;
    if (en_v7_ctc_init(&state, 3u, 16000u, 0u) != EN_V7_CTC_INVALID) return 3;
    return 0;
}
