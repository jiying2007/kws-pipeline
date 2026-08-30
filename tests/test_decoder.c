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

static kws_keyword_t keyword(uint32_t id,
                             const uint16_t *tokens,
                             uint16_t count,
                             float threshold) {
  kws_keyword_t result = {0};
  result.id = id;
  result.tokens = tokens;
  result.num_tokens = count;
  result.threshold = threshold;
  result.prefix_policy = (uint8_t)KWS_PREFIX_IMMEDIATE;
  return result;
}

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
  kws_keyword_t item = keyword(7u, tokens, 2u, 0.50f);
  float logits[4];
  uint32_t keyword_id = 0u;
  float confidence = 0.0f;

  kws_decoder_init(&decoder, 0.0f, 0.94f);
  CHECK(kws_decoder_set_keywords(&decoder, &item, 1u, 4u) == KWS_OK);

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
  kws_keyword_t item = keyword(42u, tokens, 2u, 0.50f);
  float logits[4];
  uint32_t keyword_id = 0u;
  float confidence = 0.0f;

  kws_decoder_init(&decoder, 0.0f, 0.94f);
  CHECK(kws_decoder_set_keywords(&decoder, &item, 1u, 4u) == KWS_OK);
  set_logits(logits, -8.0f, 8.0f, -8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);
  set_logits(logits, 8.0f, -8.0f, -8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);
  set_logits(logits, -8.0f, 8.0f, -8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 1);
  CHECK(keyword_id == 42u);
}

static void test_blank_readiness_does_not_leak_after_new_token(void) {
  kws_decoder_t decoder;
  const uint16_t tokens[] = {1u, 2u, 2u};
  kws_keyword_t item = keyword(99u, tokens, 3u, 0.50f);
  float logits[4];
  uint32_t keyword_id = 0u;
  float confidence = 0.0f;

  kws_decoder_init(&decoder, 0.0f, 0.94f);
  CHECK(kws_decoder_set_keywords(&decoder, &item, 1u, 4u) == KWS_OK);
  set_logits(logits, -8.0f, 8.0f, -8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);
  set_logits(logits, 8.0f, -8.0f, -8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);
  set_logits(logits, -8.0f, -8.0f, 8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);
  set_logits(logits, 8.0f, -8.0f, -8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);
  set_logits(logits, -8.0f, -8.0f, 8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 1);
  CHECK(keyword_id == 99u);
}

static void test_blank_dominant_child_can_compete(void) {
  kws_decoder_t decoder;
  const uint16_t tokens[] = {1u, 2u};
  kws_keyword_t item = keyword(123u, tokens, 2u, 0.50f);
  float logits[4];
  uint32_t keyword_id = 0u;
  float confidence = 0.0f;

  kws_decoder_init(&decoder, 0.0f, 0.94f);
  CHECK(kws_decoder_set_keywords(&decoder, &item, 1u, 4u) == KWS_OK);
  set_logits(logits, -8.0f, 8.0f, -8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);
  set_logits(logits, 8.0f, -8.0f, 7.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 1);
  CHECK(keyword_id == 123u);
  CHECK(confidence > 0.50f);
}

static void test_trie_child_competes_with_global_nonblank(void) {
  kws_decoder_t decoder;
  const uint16_t tokens[] = {1u, 2u};
  kws_keyword_t item = keyword(124u, tokens, 2u, 0.50f);
  float logits[4];
  uint32_t keyword_id = 0u;
  float confidence = 0.0f;

  kws_decoder_init(&decoder, 0.0f, 0.94f);
  CHECK(kws_decoder_set_keywords(&decoder, &item, 1u, 4u) == KWS_OK);
  set_logits(logits, -8.0f, 8.0f, -8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);
  set_logits(logits, -8.0f, -8.0f, 7.0f, 8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 1);
  CHECK(keyword_id == 124u);
  CHECK(confidence > 0.50f);
}

static void test_longest_prefix_waits_for_longer_keyword(void) {
  kws_decoder_t decoder;
  const uint16_t short_tokens[] = {1u};
  const uint16_t long_tokens[] = {1u, 2u};
  kws_keyword_t items[2] = {
      {10u, short_tokens, 1u, 0.50f, 1u, 0u, (uint8_t)KWS_PREFIX_LONGEST, 0u},
      {11u, long_tokens, 2u, 0.50f, 0u, 0u, (uint8_t)KWS_PREFIX_IMMEDIATE, 0u},
  };
  float logits[4];
  uint32_t keyword_id = 0u;
  float confidence = 0.0f;

  kws_decoder_init(&decoder, 0.0f, 0.94f);
  CHECK(kws_decoder_set_keywords(&decoder, items, 2u, 4u) == KWS_OK);
  set_logits(logits, -8.0f, 8.0f, -8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);
  set_logits(logits, -8.0f, -8.0f, 8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 1);
  CHECK(keyword_id == 11u);
}

static void test_longest_prefix_emits_after_blank(void) {
  kws_decoder_t decoder;
  const uint16_t short_tokens[] = {1u};
  const uint16_t long_tokens[] = {1u, 2u};
  kws_keyword_t items[2] = {
      {10u, short_tokens, 1u, 0.50f, 1u, 0u, (uint8_t)KWS_PREFIX_LONGEST, 0u},
      {11u, long_tokens, 2u, 0.50f, 0u, 0u, (uint8_t)KWS_PREFIX_IMMEDIATE, 0u},
  };
  float logits[4];
  uint32_t keyword_id = 0u;
  float confidence = 0.0f;

  kws_decoder_init(&decoder, 0.0f, 0.94f);
  CHECK(kws_decoder_set_keywords(&decoder, items, 2u, 4u) == KWS_OK);
  set_logits(logits, -8.0f, 8.0f, -8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);
  set_logits(logits, 8.0f, -8.0f, -8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 1);
  CHECK(keyword_id == 10u);
}

static void test_grace_policy_holds_then_emits(void) {
  kws_decoder_t decoder;
  const uint16_t tokens[] = {1u};
  kws_keyword_t item = {20u, tokens, 1u, 0.50f, 0u, 0u,
                        (uint8_t)KWS_PREFIX_GRACE, 2u};
  float logits[4];
  uint32_t keyword_id = 0u;
  float confidence = 0.0f;

  kws_decoder_init(&decoder, 0.0f, 0.94f);
  CHECK(kws_decoder_set_keywords(&decoder, &item, 1u, 4u) == KWS_OK);
  set_logits(logits, -8.0f, 8.0f, -8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);
  set_logits(logits, 8.0f, -8.0f, -8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 1);
  CHECK(keyword_id == 20u);
}

int main(void) {
  test_non_repeated_path_is_unchanged();
  test_repeated_token_requires_blank_separator();
  test_blank_readiness_does_not_leak_after_new_token();
  test_blank_dominant_child_can_compete();
  test_trie_child_competes_with_global_nonblank();
  test_longest_prefix_waits_for_longer_keyword();
  test_longest_prefix_emits_after_blank();
  test_grace_policy_holds_then_emits();
  puts("kws_decoder_tests: ok");
  return 0;
}
