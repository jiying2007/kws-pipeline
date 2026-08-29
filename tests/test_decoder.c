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

static void test_non_repeated_path_is_unchanged(void) {
  kws_decoder_t decoder;
  const uint16_t tokens[] = {1u, 2u};
  const kws_keyword_t keyword = {7u, tokens, 2u, 0.50f};
  float logits[4];
  uint32_t keyword_id = 0u;
  float confidence = 0.0f;

  kws_decoder_init(&decoder, 0.0f, 0.94f);
  CHECK(kws_decoder_set_keywords(&decoder, &keyword, 1u, 4u) == KWS_OK);

  set_logits(logits, -8.0f, 8.0f, -8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);
  set_logits(logits, -8.0f, -8.0f, 8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 1);
  CHECK(keyword_id == 7u);
  CHECK(confidence > 0.50f);
}

static void test_repeated_token_requires_blank_separator(void) {
  kws_decoder_t decoder;
  const uint16_t tokens[] = {1u, 1u};
  const kws_keyword_t keyword = {42u, tokens, 2u, 0.50f};
  float logits[4];
  uint32_t keyword_id = 0u;
  float confidence = 0.0f;

  kws_decoder_init(&decoder, 0.0f, 0.94f);
  CHECK(kws_decoder_set_keywords(&decoder, &keyword, 1u, 4u) == KWS_OK);

  set_logits(logits, -8.0f, 8.0f, -8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);

  /* A second consecutive token-1 frame is still the same CTC label. */
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);

  /* A blank-dominant frame arms the repeated-token transition. */
  set_logits(logits, 8.0f, -8.0f, -8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);

  set_logits(logits, -8.0f, 8.0f, -8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 1);
  CHECK(keyword_id == 42u);
  CHECK(confidence > 0.50f);
}

static void test_blank_readiness_does_not_leak_after_new_token(void) {
  kws_decoder_t decoder;
  const uint16_t tokens[] = {1u, 2u, 2u};
  const kws_keyword_t keyword = {99u, tokens, 3u, 0.50f};
  float logits[4];
  uint32_t keyword_id = 0u;
  float confidence = 0.0f;

  kws_decoder_init(&decoder, 0.0f, 0.94f);
  CHECK(kws_decoder_set_keywords(&decoder, &keyword, 1u, 4u) == KWS_OK);

  set_logits(logits, -8.0f, 8.0f, -8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);
  set_logits(logits, 8.0f, -8.0f, -8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);
  set_logits(logits, -8.0f, -8.0f, 8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);

  /* The earlier blank separated 1->2, not the repeated 2->2 transition. */
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);
  set_logits(logits, 8.0f, -8.0f, -8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);
  set_logits(logits, -8.0f, -8.0f, 8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 1);
  CHECK(keyword_id == 99u);
}

static void test_blank_dominant_token_cannot_advance(void) {
  kws_decoder_t decoder;
  const uint16_t tokens[] = {1u, 2u};
  const kws_keyword_t keyword = {123u, tokens, 2u, 0.50f};
  float logits[4];
  uint32_t keyword_id = 0u;
  float confidence = 0.0f;

  kws_decoder_init(&decoder, 0.0f, 0.94f);
  CHECK(kws_decoder_set_keywords(&decoder, &keyword, 1u, 4u) == KWS_OK);

  set_logits(logits, -8.0f, 8.0f, -8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);

  /* Token 2 has material posterior, but blank is still the greedy CTC label. */
  set_logits(logits, 8.0f, -8.0f, 7.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);

  set_logits(logits, -8.0f, -8.0f, 8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 1);
  CHECK(keyword_id == 123u);
}

static void test_nonbest_nonblank_token_cannot_advance(void) {
  kws_decoder_t decoder;
  const uint16_t tokens[] = {1u, 2u};
  const kws_keyword_t keyword = {321u, tokens, 2u, 0.50f};
  float logits[4];
  uint32_t keyword_id = 0u;
  float confidence = 0.0f;

  kws_decoder_init(&decoder, 0.0f, 0.94f);
  CHECK(kws_decoder_set_keywords(&decoder, &keyword, 1u, 4u) == KWS_OK);

  set_logits(logits, -8.0f, 8.0f, -8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);

  /* Token 2 is plausible, but token 3 is the greedy nonblank label. */
  set_logits(logits, -8.0f, -8.0f, 7.0f, 8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);

  /* Prefix retention permits the genuine token 2 to complete the path later. */
  set_logits(logits, -8.0f, -8.0f, 8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 1);
  CHECK(keyword_id == 321u);
}

int main(void) {
  test_non_repeated_path_is_unchanged();
  test_repeated_token_requires_blank_separator();
  test_blank_readiness_does_not_leak_after_new_token();
  test_blank_dominant_token_cannot_advance();
  test_nonbest_nonblank_token_cannot_advance();
  puts("kws_decoder_tests: ok");
  return 0;
}
