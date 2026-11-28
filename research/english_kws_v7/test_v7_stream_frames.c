#include "v7_kws_stream.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t frames;
    uint32_t events;
    uint32_t candidate_id;
} callback_state;

static void on_frame(void *user, const float logits[4], uint64_t frame_index,
                     uint64_t available_after_sample, const en_v7_ctc_event *event)
{
    callback_state *state = (callback_state *)user;
    (void)logits;
    (void)frame_index;
    (void)available_after_sample;
    ++state->frames;
    if (event->detected != 0u) {
        ++state->events;
        if (event->keyword_id != state->candidate_id) state->events += 1000u;
    }
}

static int run_candidate(uint32_t candidate_id)
{
    uint8_t *pack = (uint8_t *)calloc(EN_V7_KWS_PACK_BYTES, 1u);
    en_v7_kws_model *model = (en_v7_kws_model *)calloc(1u, en_v7_kws_model_bytes());
    en_v7_kws_stream *stream = (en_v7_kws_stream *)calloc(1u, en_v7_kws_stream_bytes());
    int16_t *pcm = (int16_t *)calloc(6400u, sizeof(int16_t));
    callback_state state = {0u, 0u, candidate_id};
    float *values;
    size_t index;
    int status;
    if (pack == NULL || model == NULL || stream == NULL || pcm == NULL) return 1;

    memcpy(pack, "KWSV7C01", 8u);
    pack[8] = 1u;
    pack[12] = (uint8_t)(EN_V7_KWS_FLOATS & 0xffu);
    pack[13] = (uint8_t)((EN_V7_KWS_FLOATS >> 8) & 0xffu);
    pack[14] = (uint8_t)((EN_V7_KWS_FLOATS >> 16) & 0xffu);
    pack[15] = (uint8_t)((EN_V7_KWS_FLOATS >> 24) & 0xffu);
    values = (float *)(void *)(pack + 16u);
    /* Zero weights, blank-biased output, nonempty Mel bands, unit scales. */
    values[EN_V7_KWS_PARAMS - 4u] = 1.0f;
    for (index = EN_V7_KWS_PARAMS;
         index < EN_V7_KWS_PARAMS + EN_V7_KWS_MEL_WEIGHTS; ++index) {
        values[index] = 0.01f;
    }
    for (index = EN_V7_KWS_PARAMS + EN_V7_KWS_MEL_WEIGHTS + 40u;
         index < EN_V7_KWS_FLOATS; ++index) {
        values[index] = 1.0f;
    }

    status = en_v7_kws_model_load(model, pack, EN_V7_KWS_PACK_BYTES, candidate_id);
    if (status != EN_V7_KWS_OK || model->candidate_id != candidate_id) return 2;
    status = en_v7_kws_stream_init(stream, model, 16000u, 0u);
    if (status != EN_V7_KWS_OK) return 3;
    status = en_v7_kws_feed(stream, pcm, 6400u, 0u, on_frame, &state);
    if (status != EN_V7_KWS_OK || state.frames != 37u || state.events != 0u) return 4;

    free(pcm);
    free(stream);
    free(model);
    free(pack);
    return 0;
}

int main(void)
{
    if (run_candidate(1u) != 0) return 1;
    if (run_candidate(2u) != 0) return 2;
    return 0;
}
