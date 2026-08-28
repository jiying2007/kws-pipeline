#include "kws_pipeline/kws.h"
#include <stdio.h>
int main(void) {
  printf("kws_engine ABI ready; sample_rate=%u max_keywords=%u max_vocab=%u\n", (unsigned)KWS_SAMPLE_RATE_HZ, (unsigned)KWS_MAX_KEYWORDS, (unsigned)KWS_MAX_VOCAB_SIZE);
  return 0;
}
