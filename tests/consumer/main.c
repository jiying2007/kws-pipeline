#include "kws_pipeline/kws.h"

#include <stdio.h>

int main(void) {
  kws_config_t config = kws_default_config();
  if (KWS_MODEL_VERSION != 2u || KWS_KEYWORD_PACK_VERSION != 2u ||
      KWS_SAMPLE_RATE_HZ != 16000u || config.refractory_ms == 0u) {
    return 1;
  }
  printf("KwsPipeline consumer: model_abi=%u pack_abi=%u sample_rate=%u\n",
         (unsigned)KWS_MODEL_VERSION, (unsigned)KWS_KEYWORD_PACK_VERSION,
         (unsigned)KWS_SAMPLE_RATE_HZ);
  return 0;
}
