#include "decoder.h"

#include <float.h>
#include <math.h>
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

static void test_blank_dominant_root_can_start_within_margin(void) {
  kws_decoder_t decoder;
  const uint16_t tokens[] = {1u, 2u};
  kws_keyword_t item = keyword(122u, tokens, 2u, 0.50f);
  float logits[4];
  uint32_t keyword_id = 0u;
  float confidence = 0.0f;

  kws_decoder_init(&decoder, 0.0f, 0.94f);
  CHECK(kws_decoder_set_keywords(&decoder, &item, 1u, 4u) == KWS_OK);

  set_logits(logits, 8.0f, 7.75f, -8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);
  set_logits(logits, -8.0f, -8.0f, 8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 1);
  CHECK(keyword_id == 122u);
  CHECK(confidence > 0.50f);
}

static void test_blank_dominant_root_outside_margin_does_not_start(void) {
  kws_decoder_t decoder;
  const uint16_t tokens[] = {1u, 2u};
  kws_keyword_t item = keyword(121u, tokens, 2u, 0.50f);
  float logits[4];
  uint32_t keyword_id = 0u;
  float confidence = 0.0f;

  kws_decoder_init(&decoder, 0.0f, 0.94f);
  CHECK(kws_decoder_set_keywords(&decoder, &item, 1u, 4u) == KWS_OK);

  set_logits(logits, 8.0f, 7.0f, -8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);
  set_logits(logits, -8.0f, -8.0f, 8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);
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

static void test_blank_retention_does_not_change_acoustic_confidence(void) {
  kws_decoder_t decoder;
  const uint16_t tokens[] = {1u, 2u};
  kws_keyword_t item = keyword(125u, tokens, 2u, 0.50f);
  float logits[4];
  float direct_confidence = 0.0f;
  float speech_blank_confidence = 0.0f;
  float silence_blank_confidence = 0.0f;
  uint32_t keyword_id = 0u;

  kws_decoder_init(&decoder, 1.5f, 0.94f);
  CHECK(kws_decoder_set_keywords(&decoder, &item, 1u, 4u) == KWS_OK);
  set_logits(logits, -8.0f, 8.0f, -8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id,
                         &direct_confidence) == 0);
  set_logits(logits, -8.0f, -8.0f, 8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id,
                         &direct_confidence) == 1);
  CHECK(keyword_id == 125u);
  CHECK(direct_confidence > 0.50f);

  kws_decoder_reset(&decoder);
  keyword_id = 0u;
  set_logits(logits, -8.0f, 8.0f, -8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id,
                         &speech_blank_confidence) == 0);
  set_logits(logits, 8.0f, -8.0f, -8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id,
                         &speech_blank_confidence) == 0);
  set_logits(logits, -8.0f, -8.0f, 8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id,
                         &speech_blank_confidence) == 1);
  CHECK(keyword_id == 125u);

  kws_decoder_reset(&decoder);
  keyword_id = 0u;
  set_logits(logits, -8.0f, 8.0f, -8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id,
                         &silence_blank_confidence) == 0);
  set_logits(logits, 8.0f, -8.0f, -8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 0, &keyword_id,
                         &silence_blank_confidence) == 0);
  set_logits(logits, -8.0f, -8.0f, 8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id,
                         &silence_blank_confidence) == 1);
  CHECK(keyword_id == 125u);

  CHECK(fabsf(direct_confidence - speech_blank_confidence) < 1.0e-5f);
  CHECK(fabsf(direct_confidence - silence_blank_confidence) < 1.0e-5f);
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


static void test_debug_blank_retention_changes_long_blank_gap_survival(void) {
  kws_decoder_t decoder;
  const uint16_t tokens[] = {1u, 2u};
  kws_keyword_t item = keyword(126u, tokens, 2u, 0.10f);
  float logits[4];
  uint32_t keyword_id = 0u;
  float confidence = 0.0f;

  kws_decoder_init(&decoder, 0.0f, 0.94f);
  CHECK(kws_decoder_set_keywords(&decoder, &item, 1u, 4u) == KWS_OK);
  CHECK(kws_decoder_debug_set_search_policy(&decoder, 0.50f, -8.25f) == KWS_OK);
  set_logits(logits, -8.0f, 8.0f, -8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);
  for (int i = 0; i < 24; ++i) {
    set_logits(logits, 8.0f, -8.0f, -8.0f, -8.0f);
    CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);
  }
  set_logits(logits, -8.0f, -8.0f, 8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);

  CHECK(kws_decoder_debug_set_search_policy(&decoder, 0.95f, -8.25f) == KWS_OK);
  set_logits(logits, -8.0f, 8.0f, -8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);
  for (int i = 0; i < 24; ++i) {
    set_logits(logits, 8.0f, -8.0f, -8.0f, -8.0f);
    CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);
  }
  set_logits(logits, -8.0f, -8.0f, 8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 1);
  CHECK(keyword_id == 126u);
}

static void test_debug_fuzzy_cost_changes_two_nontop_advances(void) {
  kws_decoder_t decoder;
  const uint16_t tokens[] = {1u, 2u, 3u};
  kws_keyword_t item = keyword(127u, tokens, 3u, 0.10f);
  float logits[5];
  uint32_t keyword_id = 0u;
  float confidence = 0.0f;

  kws_decoder_init(&decoder, 0.0f, 0.94f);
  CHECK(kws_decoder_set_keywords(&decoder, &item, 1u, 5u) == KWS_OK);
  CHECK(kws_decoder_debug_set_search_policy(&decoder, 0.70f, -8.25f) == KWS_OK);

  logits[0] = -8.0f; logits[1] = 8.0f; logits[2] = -8.0f;
  logits[3] = -8.0f; logits[4] = -8.0f;
  CHECK(kws_decoder_step(&decoder, logits, 5u, 1, &keyword_id, &confidence) == 0);
  logits[0] = -8.0f; logits[1] = -8.0f; logits[2] = 7.0f;
  logits[3] = -8.0f; logits[4] = 8.0f;
  CHECK(kws_decoder_step(&decoder, logits, 5u, 1, &keyword_id, &confidence) == 0);
  logits[2] = -8.0f; logits[3] = 7.0f; logits[4] = 8.0f;
  CHECK(kws_decoder_step(&decoder, logits, 5u, 1, &keyword_id, &confidence) == 0);

  CHECK(kws_decoder_debug_set_search_policy(&decoder, 0.70f, -4.0f) == KWS_OK);
  logits[0] = -8.0f; logits[1] = 8.0f; logits[2] = -8.0f;
  logits[3] = -8.0f; logits[4] = -8.0f;
  CHECK(kws_decoder_step(&decoder, logits, 5u, 1, &keyword_id, &confidence) == 0);
  logits[0] = -8.0f; logits[1] = -8.0f; logits[2] = 7.0f;
  logits[3] = -8.0f; logits[4] = 8.0f;
  CHECK(kws_decoder_step(&decoder, logits, 5u, 1, &keyword_id, &confidence) == 0);
  logits[2] = -8.0f; logits[3] = 7.0f; logits[4] = 8.0f;
  CHECK(kws_decoder_step(&decoder, logits, 5u, 1, &keyword_id, &confidence) == 1);
  CHECK(keyword_id == 127u);
}

/* token_boost is retained in the public configuration, but no accepted finite
 * value may alter path scores, retention admission, timing or confidence. */
static void test_deprecated_boost_has_no_search_effect(void) {
  const float boosts[] = {0.0f, 1.5f, 10.0f, 1.0e9f, FLT_MAX};
  const uint16_t tokens[] = {1u, 2u, 3u};
  kws_keyword_t item = keyword(150u, tokens, 3u, 0.10f);
  const float frames[][4] = {
      {-8.0f, 8.0f, -8.0f, -8.0f},
      {8.0f, -8.0f, 7.0f, -8.0f},
      {8.0f, -8.0f, -8.0f, 7.0f},
  };

  for (size_t b = 0u; b < sizeof(boosts) / sizeof(boosts[0]); ++b) {
    kws_decoder_t decoder;
    uint32_t id = 0u;
    float confidence = 0.0f;
    kws_decoder_init(&decoder, boosts[b], 0.94f);
    CHECK(kws_decoder_set_keywords(&decoder, &item, 1u, 4u) == KWS_OK);
    for (size_t f = 0u; f < sizeof(frames) / sizeof(frames[0]); ++f) {
      /* Two fuzzy advances cost 2 * -8.25, exceeding the -16 budget even
       * though their acoustic confidence exceeds this keyword's threshold. */
      CHECK(kws_decoder_step(&decoder, frames[f], 4u, 1, &id, &confidence) == 0);
      for (uint16_t n = 0u; n < decoder.node_count; ++n) {
        CHECK(isfinite(decoder.nodes[n].score));
        CHECK(isfinite(decoder.nodes[n].blank_score));
      }
    }
    CHECK(decoder.nodes[3].score - decoder.nodes[3].acoustic_score < -16.0f);
    kws_decoder_reset(&decoder);
    for (uint16_t token = 1u; token <= 3u; ++token) {
      float exact[4] = {-8.0f, -8.0f, -8.0f, -8.0f};
      exact[token] = 8.0f;
      CHECK(kws_decoder_step(&decoder, exact, 4u, 1, &id, &confidence) ==
            (token == 3u));
    }
    CHECK(id == 150u);
    CHECK(confidence == 1.0f);

    /* A finite compatibility boost cannot revive an expired blank prefix. */
    CHECK(kws_decoder_step(&decoder, frames[0], 4u, 1, &id, &confidence) == 0);
    for (int f = 0; f < 80; ++f) {
      const float blank[4] = {8.0f, -8.0f, -8.0f, -8.0f};
      CHECK(kws_decoder_step(&decoder, blank, 4u, 1, &id, &confidence) == 0);
    }
    for (uint16_t token = 2u; token <= 3u; ++token) {
      float exact[4] = {-8.0f, -8.0f, -8.0f, -8.0f};
      exact[token] = 8.0f;
      CHECK(kws_decoder_step(&decoder, exact, 4u, 1, &id, &confidence) == 0);
    }
  }

  for (size_t b = 1u; b < sizeof(boosts) / sizeof(boosts[0]); ++b) {
    kws_decoder_t reference;
    kws_decoder_t candidate;
    uint32_t state = 13u;
    kws_decoder_init(&reference, 0.0f, 0.94f);
    kws_decoder_init(&candidate, boosts[b], 0.94f);
    CHECK(kws_decoder_set_keywords(&reference, &item, 1u, 4u) == KWS_OK);
    CHECK(kws_decoder_set_keywords(&candidate, &item, 1u, 4u) == KWS_OK);
    for (size_t f = 0u; f < 512u; ++f) {
      float logits[4];
      uint32_t reference_id = 0u;
      uint32_t candidate_id = 0u;
      float reference_confidence = 0.0f;
      float candidate_confidence = 0.0f;
      int speech_active = f % 5u != 0u;
      int reference_hit;
      int candidate_hit;
      for (size_t t = 0u; t < 4u; ++t) {
        state = state * UINT32_C(1664525) + UINT32_C(1013904223);
        logits[t] = (float)(state % 65u) * 0.25f - 8.0f;
      }
      reference_hit = kws_decoder_step(&reference, logits, 4u, speech_active,
                                       &reference_id, &reference_confidence);
      candidate_hit = kws_decoder_step(&candidate, logits, 4u, speech_active,
                                       &candidate_id, &candidate_confidence);
      CHECK(reference_hit == candidate_hit);
      CHECK(reference_id == candidate_id);
      CHECK(reference_confidence == candidate_confidence);
      for (uint16_t n = 0u; n < reference.node_count; ++n) {
        CHECK(reference.nodes[n].score == candidate.nodes[n].score);
        CHECK(reference.nodes[n].blank_score == candidate.nodes[n].blank_score);
        CHECK(reference.nodes[n].acoustic_score == candidate.nodes[n].acoustic_score);
        CHECK(reference.nodes[n].blank_acoustic_score ==
              candidate.nodes[n].blank_acoustic_score);
      }
    }
  }
}

static void test_all_policies_obey_trailing_blank_gate(void) {
  const uint16_t tokens[] = {1u};
  const float token[4] = {-8.0f, 8.0f, -8.0f, -8.0f};
  const float blank[4] = {8.0f, -8.0f, -8.0f, -8.0f};
  for (uint8_t policy = 0u; policy <= (uint8_t)KWS_PREFIX_GRACE; ++policy) {
    for (uint8_t min_blanks = 0u; min_blanks <= 8u; ++min_blanks) {
      kws_decoder_t decoder;
      kws_keyword_t item = keyword(151u, tokens, 1u, 0.50f);
      uint32_t id = 0u;
      float confidence = 0.0f;
      unsigned int wait;
      if (policy == (uint8_t)KWS_PREFIX_LONGEST && min_blanks == 0u) {
        continue;
      }
      item.prefix_policy = policy;
      item.min_trailing_blanks = min_blanks;
      item.grace_frames = policy == (uint8_t)KWS_PREFIX_GRACE ? 2u : 0u;
      wait = min_blanks > item.grace_frames ? min_blanks : item.grace_frames;
      kws_decoder_init(&decoder, 1.5f, 0.94f);
      CHECK(kws_decoder_set_keywords(&decoder, &item, 1u, 4u) == KWS_OK);
      CHECK(kws_decoder_step(&decoder, token, 4u, 1, &id, &confidence) ==
            (wait == 0u));
      for (unsigned int f = 1u; f <= wait; ++f) {
        CHECK(kws_decoder_step(&decoder, blank, 4u, 1, &id, &confidence) ==
              (f == wait));
      }
      CHECK(id == 151u);
      CHECK(confidence > 0.50f);
    }
  }
}

static void test_trailing_blanks_restart_after_nonblank(void) {
  const uint16_t tokens[] = {1u};
  const float token[4] = {-8.0f, 8.0f, -8.0f, -8.0f};
  const float blank[4] = {8.0f, -8.0f, -8.0f, -8.0f};
  for (uint8_t policy = 0u; policy <= (uint8_t)KWS_PREFIX_GRACE; ++policy) {
    kws_decoder_t decoder;
    kws_keyword_t item = keyword(152u, tokens, 1u, 0.50f);
    uint32_t id = 0u;
    float confidence = 0.0f;
    item.prefix_policy = policy;
    item.min_trailing_blanks = 3u;
    item.grace_frames = policy == (uint8_t)KWS_PREFIX_GRACE ? 2u : 0u;
    kws_decoder_init(&decoder, 1.5f, 0.94f);
    CHECK(kws_decoder_set_keywords(&decoder, &item, 1u, 4u) == KWS_OK);
    CHECK(kws_decoder_step(&decoder, token, 4u, 1, &id, &confidence) == 0);
    CHECK(kws_decoder_step(&decoder, blank, 4u, 1, &id, &confidence) == 0);
    CHECK(kws_decoder_step(&decoder, blank, 4u, 1, &id, &confidence) == 0);
    CHECK(kws_decoder_step(&decoder, token, 4u, 1, &id, &confidence) == 0);
    CHECK(kws_decoder_step(&decoder, blank, 4u, 1, &id, &confidence) == 0);
    CHECK(kws_decoder_step(&decoder, blank, 4u, 1, &id, &confidence) == 0);
    CHECK(kws_decoder_step(&decoder, blank, 4u, 1, &id, &confidence) == 1);
    CHECK(id == 152u);
  }
}

static void test_immediate_wait_does_not_outlive_qualified_terminal(void) {
  kws_decoder_t decoder;
  const uint16_t tokens[] = {1u};
  kws_keyword_t item = keyword(153u, tokens, 1u, 0.50f);
  const float token[4] = {-8.0f, 8.0f, -8.0f, -8.0f};
  const float other[4] = {-8.0f, -8.0f, -8.0f, 8.0f};
  const float blank[4] = {8.0f, -8.0f, -8.0f, -8.0f};
  uint32_t id = 0u;
  float confidence = 0.0f;
  item.min_trailing_blanks = 2u;
  kws_decoder_init(&decoder, 1.5f, 0.94f);
  CHECK(kws_decoder_set_keywords(&decoder, &item, 1u, 4u) == KWS_OK);
  CHECK(kws_decoder_step(&decoder, token, 4u, 1, &id, &confidence) == 0);
  CHECK(kws_decoder_step(&decoder, blank, 4u, 1, &id, &confidence) == 0);
  CHECK(kws_decoder_step(&decoder, other, 4u, 1, &id, &confidence) == 0);
  for (int f = 0; f < 80; ++f) {
    CHECK(kws_decoder_step(&decoder, blank, 4u, 1, &id, &confidence) == 0);
  }
  /* Inactive speech may discharge a previously qualified terminal, but must
   * not independently admit a new terminal. */
  CHECK(kws_decoder_step(&decoder, token, 4u, 0, &id, &confidence) == 0);
  CHECK(kws_decoder_step(&decoder, blank, 4u, 0, &id, &confidence) == 0);
  CHECK(kws_decoder_step(&decoder, blank, 4u, 0, &id, &confidence) == 0);
  CHECK(kws_decoder_step(&decoder, token, 4u, 1, &id, &confidence) == 0);
  CHECK(kws_decoder_step(&decoder, blank, 4u, 0, &id, &confidence) == 0);
  CHECK(kws_decoder_step(&decoder, blank, 4u, 0, &id, &confidence) == 1);
  CHECK(id == 153u);

  CHECK(kws_decoder_step(&decoder, token, 4u, 1, &id, &confidence) == 0);
  CHECK(kws_decoder_step(&decoder, blank, 4u, 1, &id, &confidence) == 0);
  kws_decoder_reset(&decoder);
  CHECK(kws_decoder_step(&decoder, blank, 4u, 1, &id, &confidence) == 0);
  CHECK(kws_decoder_step(&decoder, blank, 4u, 1, &id, &confidence) == 0);

  /* Waiting never grants a terminal an exemption from its path budget. */
  CHECK(kws_decoder_debug_set_search_policy(&decoder, 0.0001f, -8.25f) == KWS_OK);
  CHECK(kws_decoder_step(&decoder, token, 4u, 1, &id, &confidence) == 0);
  CHECK(kws_decoder_step(&decoder, blank, 4u, 1, &id, &confidence) == 0);
  CHECK(kws_decoder_step(&decoder, blank, 4u, 1, &id, &confidence) == 0);
}

static void test_held_immediate_uses_priority_before_depth(void) {
  kws_decoder_t decoder;
  const uint16_t short_tokens[] = {2u};
  const uint16_t long_tokens[] = {1u, 2u};
  kws_keyword_t items[2] = {
      {154u, short_tokens, 1u, 0.50f, 2u, 15u, (uint8_t)KWS_PREFIX_IMMEDIATE, 0u},
      {155u, long_tokens, 2u, 0.50f, 2u, 10u, (uint8_t)KWS_PREFIX_IMMEDIATE, 0u},
  };
  const float first[4] = {-8.0f, 8.0f, -8.0f, -8.0f};
  const float second[4] = {-8.0f, -8.0f, 8.0f, -8.0f};
  const float blank[4] = {8.0f, -8.0f, -8.0f, -8.0f};
  uint32_t id = 0u;
  float confidence = 0.0f;
  kws_decoder_init(&decoder, 1.5f, 0.94f);
  CHECK(kws_decoder_set_keywords(&decoder, items, 2u, 4u) == KWS_OK);
  CHECK(kws_decoder_step(&decoder, first, 4u, 1, &id, &confidence) == 0);
  CHECK(kws_decoder_step(&decoder, second, 4u, 1, &id, &confidence) == 0);
  CHECK(kws_decoder_step(&decoder, blank, 4u, 1, &id, &confidence) == 0);
  CHECK(kws_decoder_step(&decoder, blank, 4u, 1, &id, &confidence) == 1);
  CHECK(id == 154u);
}

int main(void) {
  test_deprecated_boost_has_no_search_effect();
  test_all_policies_obey_trailing_blank_gate();
  test_trailing_blanks_restart_after_nonblank();
  test_immediate_wait_does_not_outlive_qualified_terminal();
  test_held_immediate_uses_priority_before_depth();
  test_non_repeated_path_is_unchanged();
  test_repeated_token_requires_blank_separator();
  test_blank_readiness_does_not_leak_after_new_token();
  test_blank_dominant_root_can_start_within_margin();
  test_blank_dominant_root_outside_margin_does_not_start();
  test_blank_dominant_child_can_compete();
  test_trie_child_competes_with_global_nonblank();
  test_blank_retention_does_not_change_acoustic_confidence();
  test_debug_blank_retention_changes_long_blank_gap_survival();
  test_debug_fuzzy_cost_changes_two_nontop_advances();
  test_longest_prefix_waits_for_longer_keyword();
  test_longest_prefix_emits_after_blank();
  test_grace_policy_holds_then_emits();
  puts("kws_decoder_tests: ok");
  return 0;
}
