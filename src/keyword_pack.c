#include "kws_pipeline/kws.h"

#include <stdint.h>
#include <string.h>

#define KWS_KEYWORD_PACK_HEADER_BYTES 16u
#define KWS_KEYWORD_PACK_RECORD_BYTES 44u

static uint16_t rd16(const uint8_t *p) {
  return (uint16_t)((uint16_t)p[0] | (uint16_t)((uint16_t)p[1] << 8u));
}

static uint32_t rd32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8u) |
         ((uint32_t)p[2] << 16u) | ((uint32_t)p[3] << 24u);
}

static float rdf32(const uint8_t *p) {
  uint32_t u = rd32(p);
  float f = 0.0f;
  memcpy(&f, &u, sizeof(f));
  return f;
}

kws_status_t kws_keyword_pack_open(const void *blob,
                                   size_t blob_bytes,
                                   const kws_model_t *model,
                                   kws_keyword_pack_t *out_pack) {
  const uint8_t *p = (const uint8_t *)blob;
  uint16_t count;
  uint16_t vocab_size;
  uint32_t total_bytes;
  size_t expected_bytes;

  if (p == NULL || model == NULL || out_pack == NULL ||
      blob_bytes < KWS_KEYWORD_PACK_HEADER_BYTES) {
    return KWS_EINVAL;
  }
  if (memcmp(p, "KWKP", 4u) != 0 ||
      rd16(p + 4u) != KWS_KEYWORD_PACK_VERSION ||
      rd16(p + 6u) != KWS_KEYWORD_PACK_HEADER_BYTES) {
    return KWS_EFORMAT;
  }

  count = rd16(p + 8u);
  vocab_size = rd16(p + 10u);
  total_bytes = rd32(p + 12u);
  expected_bytes = KWS_KEYWORD_PACK_HEADER_BYTES +
                   (size_t)count * KWS_KEYWORD_PACK_RECORD_BYTES;

  if (count > KWS_MAX_KEYWORDS || vocab_size != model->vocab_size ||
      total_bytes != blob_bytes || expected_bytes != blob_bytes) {
    return KWS_EFORMAT;
  }

  memset(out_pack, 0, sizeof(*out_pack));
  out_pack->keyword_count = count;

  for (uint16_t k = 0u; k < count; ++k) {
    const uint8_t *record =
        p + KWS_KEYWORD_PACK_HEADER_BYTES +
        (size_t)k * KWS_KEYWORD_PACK_RECORD_BYTES;
    uint32_t id = rd32(record + 0u);
    float threshold = rdf32(record + 4u);
    uint16_t num_tokens = rd16(record + 8u);

    if (num_tokens == 0u || num_tokens > KWS_MAX_TOKENS_PER_KEYWORD ||
        threshold <= 0.0f || threshold >= 1.0f) {
      return KWS_EFORMAT;
    }

    for (uint16_t prior = 0u; prior < k; ++prior) {
      if (out_pack->keywords[prior].id == id) {
        return KWS_EFORMAT;
      }
    }

    for (uint16_t i = 0u; i < num_tokens; ++i) {
      uint16_t token = rd16(record + 12u + (size_t)i * 2u);
      if (token == 0u || token >= model->vocab_size) {
        return KWS_EBOUNDS;
      }
      out_pack->token_storage[k][i] = token;
    }

    out_pack->keywords[k].id = id;
    out_pack->keywords[k].tokens = out_pack->token_storage[k];
    out_pack->keywords[k].num_tokens = num_tokens;
    out_pack->keywords[k].threshold = threshold;
  }

  return KWS_OK;
}
