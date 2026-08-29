#include "kws_pipeline/kws.h"

#include <stddef.h>
#include <stdint.h>

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  kws_model_t model;

  if (kws_model_open(data, size, &model) == KWS_OK) {
    (void)kws_engine_required_bytes(&model);
  }
  return 0;
}
