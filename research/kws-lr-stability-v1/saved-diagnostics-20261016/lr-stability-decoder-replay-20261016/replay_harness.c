/* Controlled saved-LP diagnostic. Links the pinned, unmodified C decoder only.
 * No model loader, frontend, RNN, audio input, KWTRACE or product engine calls. */
#include "decoder.h"
#include "kws_parameter_limits.h"
#include "refractory_mirror.h"

#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REQUIRE(condition) do { if (!(condition)) { \
  fprintf(stderr, "diagnostic invariant failed line %d: %s\n", __LINE__, #condition); \
  exit(3); } } while (0)

_Static_assert(sizeof(float) == 4, "requires binary32 float");
_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128,
               "requires IEEE binary32 geometry");

static const uint16_t kw1[] = {1u, 2u, 3u, 4u};
static const uint16_t kw2[] = {3u, 4u, 3u, 4u};
static const uint32_t expected_lengths[12] = {75u, 83u, 83u, 71u, 83u, 191u,
                                            255u, 111u, 95u, 83u, 83u, 63u};

static void initialize(kws_decoder_t *decoder) {
  const kws_keyword_t keywords[2] = {
      {1u, kw1, 4u, 0.55f, 0u, 0u, (uint8_t)KWS_PREFIX_IMMEDIATE, 0u},
      {2u, kw2, 4u, 0.55f, 0u, 0u, (uint8_t)KWS_PREFIX_IMMEDIATE, 0u}};
  kws_decoder_init(decoder, KWS_PARAM_TOKEN_BOOST_DEFAULT,
                   KWS_PARAM_STATE_RETENTION_DEFAULT);
  REQUIRE(kws_decoder_set_keywords(decoder, keywords, 2u, 5u) == KWS_OK);
  REQUIRE(decoder->keyword_ids[0] == 1u && decoder->keyword_ids[1] == 2u);
}

static void require_reset_state(const kws_decoder_t *decoder) {
  REQUIRE(decoder->inactive_frames == 0u && decoder->pending_keyword == -1);
  for (uint16_t i = 0u; i < decoder->node_count; ++i) {
    REQUIRE(decoder->nodes[i].score == (i == 0u ? 0.0f : -1.0e30f));
    REQUIRE(decoder->nodes[i].blank_score == -1.0e30f);
  }
  for (uint16_t i = 0u; i < decoder->keyword_count; ++i) {
    REQUIRE(decoder->immediate_confidences[i] == 0.0f);
    REQUIRE(decoder->immediate_blank_frames[i] == 0u);
  }
}

static void self_test(void) {
  suppression_mirror_t mirror = {0u};
  REQUIRE(suppression_apply(&mirror, 0, 400u) == MIRROR_NO_HIT);
  REQUIRE(mirror.suppress_until_sample == 0u);
  REQUIRE(suppression_apply(&mirror, 1, 400u) == MIRROR_ACCEPTED);
  REQUIRE(mirror.suppress_until_sample == 19600u);
  REQUIRE(suppression_apply(&mirror, 1, 1000u) == MIRROR_SUPPRESSED);
  REQUIRE(mirror.suppress_until_sample == 19600u);
  REQUIRE(suppression_apply(&mirror, 1, 19599u) == MIRROR_SUPPRESSED);
  REQUIRE(mirror.suppress_until_sample == 19600u);
  REQUIRE(suppression_apply(&mirror, 1, 19600u) == MIRROR_ACCEPTED);
  REQUIRE(mirror.suppress_until_sample == 38800u);
  REQUIRE(suppression_apply(&mirror, 1, 19601u) == MIRROR_SUPPRESSED);
  REQUIRE(mirror.suppress_until_sample == 38800u);
  mirror.suppress_until_sample = 0u;
  REQUIRE(suppression_apply(&mirror, 1, 400u) == MIRROR_ACCEPTED);

  /* Same clip: second native decoder hit resets even when mirror suppresses it.
   * New clip: the first native hit is accepted again. No saved-LP inputs. */
  for (unsigned clip = 0u; clip < 2u; ++clip) {
    kws_decoder_t decoder;
    initialize(&decoder);
    mirror.suppress_until_sample = 0u;
    unsigned hits = 0u, accepted = 0u, suppressed = 0u;
    for (unsigned frame = 0u; frame < 8u; ++frame) {
      float logits[5] = {-20.f, -20.f, -20.f, -20.f, -20.f};
      uint32_t id = 0u;
      float confidence = 0.f;
      logits[1u + frame % 4u] = 0.f;
      int hit = kws_decoder_step(&decoder, logits, 5u, 1, &id, &confidence);
      enum mirror_outcome outcome = suppression_apply(&mirror, hit, 400u + 320u * frame);
      if (hit != 0) {
        REQUIRE((frame == 3u || frame == 7u) && id == 1u);
        REQUIRE(confidence >= .55f && confidence <= 1.f);
        require_reset_state(&decoder);
        ++hits;
      }
      accepted += (unsigned)(outcome == MIRROR_ACCEPTED);
      suppressed += (unsigned)(outcome == MIRROR_SUPPRESSED);
    }
    REQUIRE(hits == 2u && accepted == 1u && suppressed == 1u);
  }
  puts("{\"status\":\"PASS\",\"scope\":\"synthetic-fixtures-only\","
       "\"checks\":[\"no-hit\",\"first-hit\",\"within-window-no-extension\","
       "\"one-sample-before-expiry\",\"exact-expiry\",\"new-window\","
       "\"per-clip-reset\",\"native-reset-on-suppressed-hit\"]}");
}

int main(int argc, char **argv) {
  if (argc == 2 && strcmp(argv[1], "--self-test") == 0) {
    self_test();
    return 0;
  }
  REQUIRE(argc == 1);
  char magic[40];
  uint32_t clips = 0u;
  REQUIRE(scanf("%39s %" SCNu32, magic, &clips) == 2);
  REQUIRE(strcmp(magic, "KWSLP_DIAGNOSTIC_V1") == 0 && clips == 12u);
  uint32_t total_frames = 0u, total_hits = 0u, total_accepted = 0u, total_suppressed = 0u;
  for (uint32_t clip = 0u; clip < clips; ++clip) {
    char tag[16];
    uint32_t index = 0u, frames = 0u;
    REQUIRE(scanf("%15s %" SCNu32 " %" SCNu32, tag, &index, &frames) == 3);
    REQUIRE(strcmp(tag, "CLIP") == 0 && index == clip && frames == expected_lengths[clip]);
    kws_decoder_t decoder;
    initialize(&decoder);
    suppression_mirror_t mirror = {0u};
    for (uint32_t frame = 0u; frame < frames; ++frame) {
      uint32_t input_frame = 0u, speech_active = 0u;
      uint64_t end_sample = 0u;
      float lp[5];
      REQUIRE(scanf("%" SCNu32 " %" SCNu64 " %" SCNu32 " %a %a %a %a %a",
                    &input_frame, &end_sample, &speech_active,
                    &lp[0], &lp[1], &lp[2], &lp[3], &lp[4]) == 8);
      REQUIRE(input_frame == frame && end_sample == 400u + 320u * (uint64_t)frame);
      REQUIRE(speech_active == 1u && end_sample <= UINT64_MAX - 19200u);
      double normalization = 0.;
      uint32_t bits[5];
      unsigned top = 0u;
      for (unsigned token = 0u; token < 5u; ++token) {
        REQUIRE(isfinite(lp[token]));
        normalization += exp((double)lp[token]);
        if (lp[token] > lp[top]) top = token;
        memcpy(&bits[token], &lp[token], sizeof(float));
      }
      REQUIRE(fabs(normalization - 1.) < 2.1e-6);
      uint32_t id = 0u;
      float confidence = 0.f;
      uint64_t before = mirror.suppress_until_sample;
      int hit = kws_decoder_step(&decoder, lp, 5u, 1, &id, &confidence);
      REQUIRE(hit == 0 || hit == 1);
      if (hit != 0) {
        REQUIRE(id == 1u || id == 2u);
        REQUIRE(isfinite(confidence) && confidence >= .55f && confidence <= 1.f);
        require_reset_state(&decoder);
      }
      enum mirror_outcome outcome = suppression_apply(&mirror, hit, end_sample);
      printf("{\"kind\":\"frame\",\"clip_index\":%" PRIu32 ",\"frame\":%" PRIu32
             ",\"end_sample\":%" PRIu64 ",\"speech_active\":1,\"raw_argmax\":%u,"
             "\"lp_float32_bits\":[\"%08" PRIx32 "\",\"%08" PRIx32 "\",\"%08" PRIx32
             "\",\"%08" PRIx32 "\",\"%08" PRIx32 "\"],\"decoder_hit\":%d,"
             "\"keyword_id\":%" PRIu32 ",\"confidence\":%.9g,\"mirror_outcome\":%d,"
             "\"suppress_until_before\":%" PRIu64 ",\"suppress_until_after\":%" PRIu64 "}\n",
             clip, frame, end_sample, top, bits[0], bits[1], bits[2], bits[3], bits[4],
             hit, id, (double)confidence, (int)outcome, before, mirror.suppress_until_sample);
      ++total_frames;
      total_hits += (uint32_t)hit;
      total_accepted += (uint32_t)(outcome == MIRROR_ACCEPTED);
      total_suppressed += (uint32_t)(outcome == MIRROR_SUPPRESSED);
    }
  }
  char extra;
  REQUIRE(scanf(" %c", &extra) == EOF);
  REQUIRE(total_frames == 1276u && total_hits == total_accepted + total_suppressed);
  printf("{\"kind\":\"terminal\",\"status\":\"completed\",\"clips\":12,"
         "\"frames\":%" PRIu32 ",\"native_decoder_hits\":%" PRIu32
         ",\"mirror_accepted\":%" PRIu32 ",\"mirror_suppressed\":%" PRIu32 "}\n",
         total_frames, total_hits, total_accepted, total_suppressed);
  REQUIRE(fflush(stdout) == 0 && ferror(stdout) == 0);
  return 0;
}
