#include "kws_pipeline/kws.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  kws_model_t model;
  kws_keyword_pack_t pack;

  memset(&model, 0, sizeof(model));
  model.vocab_size = KWS_MAX_VOCAB_SIZE;
  model.vocab_fingerprint = UINT64_C(0x1122334455667788);
  (void)kws_keyword_pack_open(data, size, &model, &pack);
  return 0;
}
