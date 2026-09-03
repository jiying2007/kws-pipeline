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

static void configure_two_token_keyword(kws_decoder_t *decoder,
                                        kws_keyword_t *item,
                                        const uint16_t tokens[2]) {
  item->id = 77u;
  item->tokens = tokens;
  item->num_tokens = 2u;
  item->threshold = 0.50f;
  item->prefix_policy = (uint8_t)KWS_PREFIX_IMMEDIATE;

  kws_decoder_init(decoder, 0.0f, 0.94f);
  CHECK(kws_decoder_set_keywords(decoder, item, 1u, 4u) == KWS_OK);
}

static void configure_competing_root_keywords(kws_decoder_t *decoder,
                                               kws_keyword_t items[2],
                                               const uint16_t first[2],
                                               const uint16_t second[2]) {
  items[0].id = 77u;
  items[0].tokens = first;
  items[0].num_tokens = 2u;
  items[0].threshold = 0.50f;
  items[0].prefix_policy = (uint8_t)KWS_PREFIX_IMMEDIATE;

  items[1].id = 88u;
  items[1].tokens = second;
  items[1].num_tokens = 2u;
  items[1].threshold = 0.50f;
  items[1].prefix_policy = (uint8_t)KWS_PREFIX_IMMEDIATE;

  kws_decoder_init(decoder, 0.0f, 0.94f);
  CHECK(kws_decoder_set_keywords(decoder, items, 2u, 4u) == KWS_OK);
}

static void verify_fresh_sequence_recovers(kws_decoder_t *decoder) {
  float logits[4];
  uint32_t keyword_id = 0u;
  float confidence = 0.0f;

  set_logits(logits, -8.0f, 8.0f, -8.0f, -8.0f);
  CHECK(kws_decoder_step(decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);
  set_logits(logits, -8.0f, -8.0f, 8.0f, -8.0f);
  CHECK(kws_decoder_step(decoder, logits, 4u, 1, &keyword_id, &confidence) == 1);
  CHECK(keyword_id == 77u);
  CHECK(confidence > 0.50f);
}

static void verify_stale_prefix_expires(int gap_speech_active, int gap_frames) {
  kws_decoder_t decoder;
  const uint16_t tokens[] = {1u, 2u};
  kws_keyword_t item = {0};
  float logits[4];
  uint32_t keyword_id = 0u;
  float confidence = 0.0f;

  configure_two_token_keyword(&decoder, &item, tokens);

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

  /* The stale path must not poison recovery. */
  verify_fresh_sequence_recovers(&decoder);
}

static void verify_unrelated_nonblank_breaks_prefix(void) {
  kws_decoder_t decoder;
  const uint16_t tokens[] = {1u, 2u};
  kws_keyword_t item = {0};
  float logits[4];
  uint32_t keyword_id = 0u;
  float confidence = 0.0f;

  configure_two_token_keyword(&decoder, &item, tokens);

  /* A strong unrelated nonblank token must invalidate the old prefix instead
   * of letting token 1 survive and combine with a later token 2. This is the
   * deterministic analogue of the continuous hard-negative failure. */
  set_logits(logits, -8.0f, 8.0f, -8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);
  set_logits(logits, -8.0f, -8.0f, -8.0f, 8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);
  set_logits(logits, -8.0f, -8.0f, 8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);

  verify_fresh_sequence_recovers(&decoder);
}

static void verify_subdominant_root_cannot_start(void) {
  kws_decoder_t decoder;
  const uint16_t tokens[] = {1u, 2u};
  kws_keyword_t item = {0};
  float logits[4];
  uint32_t keyword_id = 0u;
  float confidence = 0.0f;

  configure_two_token_keyword(&decoder, &item, tokens);

  /* token 1 is acoustically strong enough that unrestricted fuzzy transition
   * would start the keyword, but unrelated token 3 is a full logit stronger
   * and must block that root start. */
  set_logits(logits, -8.0f, 7.0f, -8.0f, 8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);
  set_logits(logits, -8.0f, -8.0f, 8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);

  verify_fresh_sequence_recovers(&decoder);
}

static void verify_near_tied_nonroot_cannot_start(void) {
  kws_decoder_t decoder;
  const uint16_t tokens[] = {1u, 2u};
  kws_keyword_t item = {0};
  float logits[4];
  uint32_t keyword_id = 0u;
  float confidence = 0.0f;

  configure_two_token_keyword(&decoder, &item, tokens);

  /* A near-tied arbitrary non-root peak must still be contradictory evidence.
   * This directly prevents h02-style hard negatives from shadow-starting a
   * wake path via a strong secondary keyword-root posterior. */
  set_logits(logits, -8.0f, 7.6f, -8.0f, 8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);
  set_logits(logits, -8.0f, -8.0f, 8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);

  verify_fresh_sequence_recovers(&decoder);
}

static void verify_near_tied_keyword_root_can_start(void) {
  kws_decoder_t decoder;
  const uint16_t first[] = {1u, 2u};
  const uint16_t second[] = {3u, 3u};
  kws_keyword_t items[2] = {{0}, {0}};
  float logits[4];
  uint32_t keyword_id = 0u;
  float confidence = 0.0f;

  configure_competing_root_keywords(&decoder, items, first, second);

  /* A 0.4-logit ambiguity between two configured keyword roots is plausible
   * first-token competition. Preserve the secondary root so noisy/far-field
   * evidence can recover without admitting arbitrary non-root shadow starts. */
  set_logits(logits, -8.0f, 7.6f, -8.0f, 8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);
  set_logits(logits, -8.0f, -8.0f, 8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 1);
  CHECK(keyword_id == 77u);
  CHECK(confidence > 0.50f);
}

static void verify_strong_keyword_root_competitor_blocks(void) {
  kws_decoder_t decoder;
  const uint16_t first[] = {1u, 2u};
  const uint16_t second[] = {3u, 3u};
  kws_keyword_t items[2] = {{0}, {0}};
  float logits[4];
  uint32_t keyword_id = 0u;
  float confidence = 0.0f;

  configure_competing_root_keywords(&decoder, items, first, second);

  /* Even a configured competing root cannot authorize a secondary start when
   * it is clearly stronger than the candidate root. */
  set_logits(logits, -8.0f, 7.0f, -8.0f, 8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);
  set_logits(logits, -8.0f, -8.0f, 8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);
}

static void verify_two_fuzzy_children_cannot_complete(void) {
  kws_decoder_t decoder;
  const uint16_t tokens[] = {1u, 2u, 1u};
  kws_keyword_t item = {0};
  float logits[4];
  uint32_t keyword_id = 0u;
  float confidence = 0.0f;

  item.id = 99u;
  item.tokens = tokens;
  item.num_tokens = 3u;
  item.threshold = 0.50f;
  item.prefix_policy = (uint8_t)KWS_PREFIX_IMMEDIATE;
  kws_decoder_init(&decoder, 0.0f, 0.94f);
  CHECK(kws_decoder_set_keywords(&decoder, &item, 1u, 4u) == KWS_OK);

  /* Preserve one fuzzy child transition (the general decoder contract), but a
   * second unrelated top-1 competitor must push the accumulated search-only
   * path cost beyond the terminal retention budget. */
  set_logits(logits, -8.0f, 8.0f, -8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);
  set_logits(logits, -8.0f, -8.0f, 7.0f, 8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);
  set_logits(logits, -8.0f, 7.0f, -8.0f, 8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);

  /* The budget only suppresses the synthetic fuzzy path, not a fresh dominant
   * sequence. */
  kws_decoder_reset(&decoder);
  set_logits(logits, -8.0f, 8.0f, -8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);
  set_logits(logits, -8.0f, -8.0f, 8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 0);
  set_logits(logits, -8.0f, 8.0f, -8.0f, -8.0f);
  CHECK(kws_decoder_step(&decoder, logits, 4u, 1, &keyword_id, &confidence) == 1);
  CHECK(keyword_id == 99u);
  CHECK(confidence > 0.50f);
}

int main(void) {
  /* Blank-dominant posterior evidence is an utterance boundary even when an
   * upstream VAD remains active because of noise. Both VAD states must expire
   * a stale prefix on the same bounded silence-retention timescale. */
  verify_stale_prefix_expires(1, 80);
  verify_stale_prefix_expires(0, 80);
  verify_unrelated_nonblank_breaks_prefix();
  verify_subdominant_root_cannot_start();
  verify_near_tied_nonroot_cannot_start();
  verify_near_tied_keyword_root_can_start();
  verify_strong_keyword_root_competitor_blocks();
  verify_two_fuzzy_children_cannot_complete();

  puts("test_decoder_retention: ok");
  return 0;
}
