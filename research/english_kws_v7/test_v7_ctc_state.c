#include "v7_ctc.h"

int main(void)
{
    en_v7_ctc_state state;
    if (en_v7_ctc_init(&state, 1u, 16000u, 800u) != EN_V7_CTC_OK) return 1;
    if (state.keyword_id != 1u) return 2;
    if (en_v7_ctc_reset(&state) != EN_V7_CTC_OK) return 3;
    if (state.keyword_id != 1u) return 4;
    if (en_v7_ctc_discontinuity(&state) != EN_V7_CTC_OK) return 5;
    if (state.keyword_id != 1u) return 6;
    if (en_v7_ctc_init(&state, 2u, 16000u, 800u) != EN_V7_CTC_OK) return 7;
    if (state.keyword_id != 2u) return 8;
    return 0;
}
