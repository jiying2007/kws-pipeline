#include "kws_pipeline/kws.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

static uint16_t rd16(const uint8_t *p) {
  return (uint16_t)((uint16_t)p[0] | (uint16_t)((uint16_t)p[1] << 8u));
}

static uint32_t rd32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8u) |
         ((uint32_t)p[2] << 16u) | ((uint32_t)p[3] << 24u);
}

static uint64_t rd64(const uint8_t *p) {
  return (uint64_t)rd32(p) | ((uint64_t)rd32(p + 4u) << 32u);
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  kws_model_t model;
  kws_keyword_pack_t pack;

  memset(&model, 0, sizeof(model));
  if (size >= 24u) {
    model.vocab_size = rd16(data + 10u);
    model.vocab_fingerprint = rd64(data + 16u);
  } else {
    model.vocab_size = KWS_MAX_VOCAB_SIZE;
    model.vocab_fingerprint = UINT64_C(1);
  }
  (void)kws_keyword_pack_open(data, size, &model, &pack);
  return 0;
}
