#include "decoder.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(x)                                                              \
  do {                                                                        \
    if (!(x)) {                                                               \
      fprintf(stderr, "CHECK failed: %s:%d: %s\n", __FILE__, __LINE__, #x); \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

static void set_logits(float logits[4],
                       float blank,
                       float token1,
                       float token2,
                       float other) {
  logits[0] = blank;
  logits[1] = token1;
  logits[2] = token2;
  logits[3] = other;
}

static void verify_stale_prefix_expires(int gap_speech_active, int gap_frames) {
  kws_decoder_t decoder;
  const uint16_t tokens[] = {1u, 2u};
  kws_keyword_t item = {0};
  float logits[4];
  uint32_t keyword_id = 0u;
  float confidence = 0.0f;

  item.id = 77u;
  item.tokens = tokens;
  item.num_tokens = 2u;
  item.threshold = 0.50f;
  item.prefix_policy = (uint8_t)KWS_PREFIX_IMMEDIATE;

  kws_decoder_init(&decoder, 0.0f, 0.94f);
  CHECK(kws_decoder_set_keywords(&decoder, &item, 1u, 4u) == KWS_OK);

  /* Start a valid prefix, then leave it blank-dominant long enough that its
   * cumulative retention cost exceeds the decoder path budget. */
  set_logits(logits, -8.0f, 8.0f, -8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);
  set_logits(logits, 8.0f, -8.0f, -8.0f, -8.0f);
  for (int frame = 0; frame < gap_frames; ++frame) {
    CHECK(kws_decoder_step(&decoder, logits, 4u, gap_speech_active, &keyword_id,
                           &confidence) == 0);
  }
  set_logits(logits, -8.0f, -8.0f, 8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);

  /* The stale path must not poison recovery: a fresh contiguous sequence still
   * qualifies immediately with the same threshold. */
  set_logits(logits, -8.0f, 8.0f, -8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);
  set_logits(logits, -8.0f, -8.0f, 8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 1);
  CHECK(keyword_id == 77u);
  CHECK(confidence > 0.50f);
}

int main(void) {
  /* state_retention=0.94 gives log-retention ~= -0.0619 per speech frame, so
   * 320 frames exceed the -16 path budget. Silence uses the stronger fixed
   * retention decay and expires well within 80 frames. */
  verify_stale_prefix_expires(1, 320);
  verify_stale_prefix_expires(0, 80);

  puts("test_decoder_retention: ok");
  return 0;
}
