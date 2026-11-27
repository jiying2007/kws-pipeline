#include "v7_kws_stream.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static void put_u32(uint8_t *dst, uint32_t value)
{
    dst[0] = (uint8_t)(value & 0xffu);
    dst[1] = (uint8_t)((value >> 8) & 0xffu);
    dst[2] = (uint8_t)((value >> 16) & 0xffu);
    dst[3] = (uint8_t)((value >> 24) & 0xffu);
}

int main(void)
{
    uint8_t *pack = (uint8_t *)calloc(EN_V7_KWS_PACK_BYTES, 1u);
    en_v7_kws_model *model = (en_v7_kws_model *)calloc(1u, en_v7_kws_model_bytes());
    en_v7_kws_stream *stream = (en_v7_kws_stream *)calloc(1u, en_v7_kws_stream_bytes());
    float *values;
    size_t index;
    int status;
    if (pack == NULL || model == NULL || stream == NULL) return 1;
    memcpy(pack, "KWSV7C01", 8u);
    put_u32(pack + 8u, 1u);
    put_u32(pack + 12u, EN_V7_KWS_FLOATS);
    values = (float *)(void *)(pack + 16u);
    for (index = 43876u; index < 43876u + 257u * 40u; ++index) values[index] = 0.01f;
    for (index = 43876u + 257u * 40u + 40u; index < EN_V7_KWS_FLOATS; ++index) values[index] = 1.0f;
    status = en_v7_kws_model_load(model, pack, EN_V7_KWS_PACK_BYTES, 1u);
    if (status != EN_V7_KWS_OK || model->candidate_id != 1u) return 2;
    status = en_v7_kws_stream_init(stream, model, 16000u, 800u);
    if (status != EN_V7_KWS_OK) return 3;
    status = en_v7_kws_model_load(model, pack, EN_V7_KWS_PACK_BYTES, 2u);
    if (status != EN_V7_KWS_OK || model->candidate_id != 2u) return 4;
    status = en_v7_kws_model_load(model, pack, EN_V7_KWS_PACK_BYTES, 3u);
    if (status == EN_V7_KWS_OK) return 5;
    free(stream);
    free(model);
    free(pack);
    return 0;
}
