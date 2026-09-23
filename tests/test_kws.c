#include "kws_pipeline/kws.h"
#include "kws_debug.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x)                                                               \
  do {                                                                         \
    if (!(x)) {                                                                \
      fprintf(stderr, "CHECK failed: %s:%d: %s\n", __FILE__, __LINE__, #x);  \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define TEST_VOCAB_FINGERPRINT UINT64_C(0x1122334455667788)
#define TEST_BH_OFFSET 216u
#define TEST_BO_OFFSET 248u
#define TEST_BLOCK_SAMPLES 160u

static kws_keyword_t make_keyword(uint32_t id,
                                  const uint16_t *tokens,
                                  uint16_t count,
                                  float threshold) {
  kws_keyword_t value = {0};
  value.id = id;
  value.tokens = tokens;
  value.num_tokens = count;
  value.threshold = threshold;
  value.prefix_policy = (uint8_t)KWS_PREFIX_IMMEDIATE;
  return value;
}

static void put16(uint8_t *p, uint16_t v) {
  p[0] = (uint8_t)(v & 0xffu);
  p[1] = (uint8_t)(v >> 8u);
}

static void put32(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)(v & 0xffu);
  p[1] = (uint8_t)((v >> 8u) & 0xffu);
  p[2] = (uint8_t)((v >> 16u) & 0xffu);
  p[3] = (uint8_t)(v >> 24u);
}

static void put64(uint8_t *p, uint64_t v) {
  put32(p, (uint32_t)(v & UINT64_C(0xffffffff)));
  put32(p + 4u, (uint32_t)(v >> 32u));
}

static void putf(uint8_t *p, float v) {
  uint32_t u = 0u;
  memcpy(&u, &v, sizeof(u));
  put32(p, u);
}

static size_t make_test_model(uint8_t *blob, size_t cap) {
  const uint16_t f = 32u;
  const uint16_t h = 4u;
  const uint16_t v = 4u;
  const uint32_t wx = 72u;
  const uint32_t wh = wx + (uint32_t)f * (uint32_t)h;
  const uint32_t bh = wh + (uint32_t)h * (uint32_t)h;
  const uint32_t wo = bh + (uint32_t)h * 4u;
  const uint32_t bo = wo + (uint32_t)v * (uint32_t)h;
  const uint32_t total = bo + (uint32_t)v * 4u;

  CHECK(bh == TEST_BH_OFFSET);
  CHECK(bo == TEST_BO_OFFSET);
  CHECK(cap >= total);
  memset(blob, 0, total);
  memcpy(blob, "KWSP", 4u);
  put16(blob + 4u, KWS_MODEL_VERSION);
  put16(blob + 6u, 72u);
  put16(blob + 8u, f);
  put16(blob + 10u, h);
  put16(blob + 12u, v);
  put16(blob + 14u, KWS_FRONTEND_LOGMEL);
  put32(blob + 16u, KWS_SAMPLE_RATE_HZ);
  put32(blob + 20u, KWS_FRAME_LENGTH_SAMPLES);
  put32(blob + 24u, KWS_FRAME_HOP_SAMPLES);
  putf(blob + 28u, 0.01f);
  putf(blob + 32u, 0.01f);
  putf(blob + 36u, 0.01f);
  put64(blob + 40u, TEST_VOCAB_FINGERPRINT);
  put32(blob + 48u, wx);
  put32(blob + 52u, wh);
  put32(blob + 56u, bh);
  put32(blob + 60u, wo);
  put32(blob + 64u, bo);
  put32(blob + 68u, total);
  putf(blob + bo + 0u, -4.0f);
  putf(blob + bo + 4u, 4.0f);
  putf(blob + bo + 8u, 3.0f);
  putf(blob + bo + 12u, -4.0f);
  return total;
}

/* make_test_model() leaves every int8 weight at zero, so a build whose int8
 * dot product is wrong still passes: 0 * x is 0 in any lane order, with any
 * sign handling. This fixture loads the in-projection with a deterministic
 * pseudo-random byte pattern and the recurrent projection with an identity
 * diagonal, so the logits actually depend on the weight bytes and on their
 * sign interpretation. `sign` negates every in-projection weight, which
 * negates the top logit and therefore moves the winning token between
 * "token 1" and "blank".
 *
 * What this establishes: the weights are consumed, and negating them negates
 * the result.
 * What this cannot establish: the value of a dot product. An assertion of this
 * shape is blind to sign-extension bugs, because feature normalisation makes
 * every frame exactly zero-mean: reinterpreting the weights as unsigned maps
 * the dot product to 256 * sum(features) - dot, which is its own negation.
 * Value-level verification of the NEON kernel belongs to
 * tests/test_arm_parity.py, which compares the confidence the hosted and
 * Cortex-A32 builds print for identical weights and does fail under that
 * mutation. */
static size_t make_weighted_test_model(uint8_t *blob, size_t cap, int sign) {
  const uint16_t f = 32u;
  const uint16_t h = 4u;
  const uint16_t v = 4u;
  const uint32_t wx = 72u;
  const uint32_t wh = wx + (uint32_t)f * (uint32_t)h;
  const uint32_t bh = wh + (uint32_t)h * (uint32_t)h;
  const uint32_t wo = bh + (uint32_t)h * 4u;
  const uint32_t bo = wo + (uint32_t)v * (uint32_t)h;
  const uint32_t total = bo + (uint32_t)v * 4u;
  uint32_t state = UINT32_C(0x2545f491);
  size_t i;

  CHECK(cap >= total);
  memset(blob, 0, total);
  memcpy(blob, "KWSP", 4u);
  put16(blob + 4u, KWS_MODEL_VERSION);
  put16(blob + 6u, 72u);
  put16(blob + 8u, f);
  put16(blob + 10u, h);
  put16(blob + 12u, v);
  put16(blob + 14u, KWS_FRONTEND_LOGMEL);
  put32(blob + 16u, KWS_SAMPLE_RATE_HZ);
  put32(blob + 20u, KWS_FRAME_LENGTH_SAMPLES);
  put32(blob + 24u, KWS_FRAME_HOP_SAMPLES);
  putf(blob + 28u, 0.01f);
  putf(blob + 32u, 0.01f);
  putf(blob + 36u, 0.01f);
  put64(blob + 40u, TEST_VOCAB_FINGERPRINT);
  put32(blob + 48u, wx);
  put32(blob + 52u, wh);
  put32(blob + 56u, bh);
  put32(blob + 60u, wo);
  put32(blob + 64u, bo);
  put32(blob + 68u, total);

  for (i = 0u; i < (size_t)f; ++i) {
    int8_t weight;
    state = state * UINT32_C(1103515245) + UINT32_C(12345);
    weight = (int8_t)((int32_t)((state >> 16u) % 255u) - 127);
    blob[wx + i] = (uint8_t)(sign != 0 ? (int8_t)-weight : weight);
  }
  for (i = 0u; i < (size_t)h; ++i) {
    blob[wh + i * (size_t)h + i] = (uint8_t)127;
  }
  for (i = 0u; i < (size_t)h; ++i) {
    blob[wo + (size_t)h + i] = (uint8_t)127;
  }
  return total;
}

static void test_model_and_engine(void) {
  _Alignas(max_align_t) uint8_t blob[512];
  _Alignas(max_align_t) uint8_t arena[65536];
  kws_model_t model;
  kws_model_t invalid_model;
  kws_engine_t *engine = NULL;
  kws_config_t config = kws_default_config();
  kws_config_t invalid_config;
  const uint16_t sequence[] = {1u};
  const uint16_t alternate_sequence[] = {2u};
  kws_keyword_t keyword = make_keyword(42u, sequence, 1u, 0.30f);
  kws_keyword_t nan_keyword = make_keyword(43u, sequence, 1u, NAN);
  kws_keyword_t duplicate_ids[] = {
      make_keyword(100u, sequence, 1u, 0.30f),
      make_keyword(100u, alternate_sequence, 1u, 0.40f),
  };
  int16_t pcm[1200];
  int detected_any = 0;
  kws_detection_t first_detection = {0u, 0.0f, 0u};
  kws_engine_stats_t stats;
  size_t bytes = make_test_model(blob, sizeof(blob));
  const size_t sample_count = sizeof(pcm) / sizeof(pcm[0]);

  CHECK(kws_model_open(blob, bytes, &model) == KWS_OK);
  CHECK(model.vocab_fingerprint == TEST_VOCAB_FINGERPRINT);
  CHECK(model.frontend_kind == KWS_FRONTEND_LOGMEL);
  CHECK(kws_engine_required_alignment() >= _Alignof(uint64_t));
  CHECK(kws_engine_required_bytes(&model) <= sizeof(arena));
  /* A rejected init must not hand back a pointer into a partially written
   * arena: the caller cannot tell that apart from a usable engine. */
  engine = (kws_engine_t *)arena;
  CHECK(kws_engine_init(arena + 1u, sizeof(arena) - 1u, &model, NULL, &engine) == KWS_EINVAL);
  CHECK(engine == NULL);
  CHECK(kws_engine_get_stats(NULL, &stats) == KWS_EINVAL);

  invalid_model = model;
  invalid_model.feature_dim = (uint16_t)(KWS_MAX_FEATURE_DIM + 1u);
  CHECK(kws_engine_required_bytes(&invalid_model) == 0u);
  CHECK(kws_engine_init(arena, sizeof(arena), &invalid_model, NULL, &engine) == KWS_EINVAL);

  invalid_model = model;
  invalid_model.frame_hop_samples = KWS_FRAME_HOP_SAMPLES / 2u;
  CHECK(kws_engine_required_bytes(&invalid_model) == 0u);
  invalid_model = model;
  invalid_model.wx_scale = NAN;
  CHECK(kws_engine_required_bytes(&invalid_model) == 0u);

  invalid_config = kws_default_config();
  invalid_config.token_boost = NAN;
  CHECK(kws_engine_init(arena, sizeof(arena), &model, &invalid_config, &engine) == KWS_EINVAL);
  invalid_config = kws_default_config();
  invalid_config.min_speech_dbfs = INFINITY;
  engine = (kws_engine_t *)arena;
  CHECK(kws_engine_init(arena, sizeof(arena), &model, &invalid_config, &engine) == KWS_EINVAL);
  CHECK(engine == NULL);

  config.min_speech_dbfs = -80.0f;
  config.refractory_ms = 100u;
  CHECK(kws_engine_init(arena, sizeof(arena), &model, &config, &engine) == KWS_OK);
  CHECK(kws_engine_get_stats(engine, NULL) == KWS_EINVAL);
  CHECK(kws_engine_get_stats(engine, &stats) == KWS_OK);
  CHECK(stats.processed_samples == 0u);
  CHECK(stats.processed_frames == 0u);
  CHECK(stats.discontinuities == 0u);
  CHECK(stats.last_discontinuity_reason == 0u);
  CHECK(stats.keyword_count == 0u);
  CHECK(stats.trie_nodes == 1u);
  CHECK(stats.pending_keyword_index == -1);

  CHECK(kws_engine_set_keywords(engine, &keyword, 1u, TEST_VOCAB_FINGERPRINT) == KWS_OK);
  CHECK(kws_engine_set_keywords(engine, &keyword, 1u, UINT64_C(0x8877665544332211)) == KWS_EFORMAT);
  CHECK(kws_engine_set_keywords(engine, &nan_keyword, 1u, TEST_VOCAB_FINGERPRINT) == KWS_EINVAL);
  CHECK(kws_engine_set_keywords(engine, duplicate_ids, 2u, TEST_VOCAB_FINGERPRINT) == KWS_EINVAL);

  for (size_t i = 0u; i < sample_count; ++i) {
    pcm[i] = ((i / 20u) & 1u) != 0u ? 12000 : -12000;
  }

  CHECK(kws_engine_notify_discontinuity(NULL, KWS_DISCONTINUITY_XRUN) == KWS_EINVAL);
  CHECK(kws_engine_notify_discontinuity(engine, (kws_discontinuity_reason_t)0) == KWS_EINVAL);
  {
    int detected = 0;
    CHECK(kws_engine_accept_pcm16(engine, pcm, 100u, NULL, &detected) == KWS_OK);
    CHECK(detected == 0);
    CHECK(kws_engine_notify_discontinuity(engine, KWS_DISCONTINUITY_XRUN) == KWS_OK);
    CHECK(kws_engine_accept_pcm16(engine, pcm + 100u, 300u, NULL, &detected) == KWS_OK);
    CHECK(kws_engine_get_stats(engine, &stats) == KWS_OK);
    CHECK(stats.processed_samples == 400u);
    CHECK(stats.processed_frames == 0u);
    CHECK(stats.discontinuities == 1u);
    CHECK(stats.last_discontinuity_reason == (uint32_t)KWS_DISCONTINUITY_XRUN);
    CHECK(stats.keyword_count == 1u);
    CHECK(stats.trie_nodes == 2u);
    CHECK(kws_engine_notify_discontinuity(engine, KWS_DISCONTINUITY_ROUTE_CHANGE) == KWS_OK);
  }

  {
    int detected = 7;
    uint64_t before = kws_engine_processed_samples(engine);
    CHECK(kws_engine_accept_pcm16(engine, pcm, KWS_MAX_PCM_BLOCK_SAMPLES + 1u, NULL, &detected) == KWS_EBOUNDS);
    CHECK(detected == 0);
    CHECK(kws_engine_processed_samples(engine) == before);
  }
  for (size_t offset = 0u; offset < sample_count; offset += TEST_BLOCK_SAMPLES) {
    size_t count = sample_count - offset;
    int detected = 0;
    kws_detection_t detection;
    if (count > TEST_BLOCK_SAMPLES) {
      count = TEST_BLOCK_SAMPLES;
    }
    CHECK(kws_engine_accept_pcm16(engine, pcm + offset, count, &detection, &detected) == KWS_OK);
    if (detected != 0 && detected_any == 0) {
      first_detection = detection;
      detected_any = 1;
    }
  }
  CHECK(detected_any == 1);
  CHECK(first_detection.keyword_id == 42u);
  CHECK(first_detection.confidence > 0.30f);
  CHECK(first_detection.end_sample > 400u);
  CHECK(kws_engine_processed_samples(engine) == sample_count + 400u);

  CHECK(kws_engine_get_stats(engine, &stats) == KWS_OK);
  CHECK(stats.processed_samples == sample_count + 400u);
  CHECK(stats.processed_frames == 3u);
  CHECK(stats.speech_frames == 3u);
  CHECK(stats.blank_top1_frames == 0u);
  CHECK(stats.discontinuities == 2u);
  CHECK(stats.last_discontinuity_reason == (uint32_t)KWS_DISCONTINUITY_ROUTE_CHANGE);
  CHECK(stats.keyword_count == 1u);
  CHECK(stats.trie_nodes == 2u);
  CHECK(stats.decoder_hits >= stats.detections);
  CHECK(stats.detections >= 1u);
  CHECK(stats.max_detection_confidence >= first_detection.confidence);
}

static void test_debug_frame_replay(void) {
  _Alignas(max_align_t) uint8_t blob[512];
  _Alignas(max_align_t) uint8_t live_arena[65536];
  _Alignas(max_align_t) uint8_t replay_arena[65536];
  kws_model_t model;
  kws_engine_t *live = NULL;
  kws_engine_t *replay = NULL;
  kws_config_t config = kws_default_config();
  const uint16_t sequence[] = {1u};
  const kws_keyword_t keyword = make_keyword(42u, sequence, 1u, 0.10f);
  int16_t pcm[KWS_FRAME_LENGTH_SAMPLES];
  float logits[KWS_MAX_VOCAB_SIZE];
  uint64_t frame_number = 0u;
  uint64_t end_sample = 0u;
  uint16_t vocab_size = 0u;
  int speech_active = 0;
  kws_detection_t live_hit = {0u, 0.0f, 0u};
  kws_detection_t replay_hit = {0u, 0.0f, 0u};
  int live_detected = 0;
  int replay_detected = 0;
  size_t bytes = make_test_model(blob, sizeof(blob));

  CHECK(kws_model_open(blob, bytes, &model) == KWS_OK);
  config.min_speech_dbfs = -80.0f;
  config.refractory_ms = 100u;
  CHECK(kws_engine_init(live_arena, sizeof(live_arena), &model, &config,
                        &live) == KWS_OK);
  CHECK(kws_engine_init(replay_arena, sizeof(replay_arena), &model, &config,
                        &replay) == KWS_OK);
  CHECK(kws_engine_set_keywords(live, &keyword, 1u,
                                TEST_VOCAB_FINGERPRINT) == KWS_OK);
  CHECK(kws_engine_set_keywords(replay, &keyword, 1u,
                                TEST_VOCAB_FINGERPRINT) == KWS_OK);

  CHECK(kws_engine_debug_copy_last_frame(live, &frame_number, &end_sample,
                                         &speech_active, logits,
                                         KWS_MAX_VOCAB_SIZE,
                                         &vocab_size) == 0);
  for (size_t i = 0u; i < KWS_FRAME_LENGTH_SAMPLES; ++i) {
    pcm[i] = ((i / 20u) & 1u) != 0u ? 12000 : -12000;
  }
  for (size_t offset = 0u; offset < KWS_FRAME_LENGTH_SAMPLES;
       offset += TEST_BLOCK_SAMPLES) {
    size_t count = KWS_FRAME_LENGTH_SAMPLES - offset;
    int detected = 0;
    kws_detection_t hit = {0u, 0.0f, 0u};
    if (count > TEST_BLOCK_SAMPLES) {
      count = TEST_BLOCK_SAMPLES;
    }
    CHECK(kws_engine_accept_pcm16(live, pcm + offset, count, &hit,
                                  &detected) == KWS_OK);
    if (detected != 0) {
      live_detected = 1;
      live_hit = hit;
    }
  }

  CHECK(kws_engine_debug_copy_last_frame(live, &frame_number, &end_sample,
                                         &speech_active, logits,
                                         KWS_MAX_VOCAB_SIZE,
                                         &vocab_size) == 1);
  CHECK(frame_number == 1u);
  CHECK(end_sample == KWS_FRAME_LENGTH_SAMPLES);
  CHECK(speech_active == 1);
  CHECK(vocab_size == model.vocab_size);
  CHECK(kws_engine_debug_copy_last_frame(live, &frame_number, &end_sample,
                                         &speech_active, logits, 1u,
                                         &vocab_size) == -1);

  CHECK(kws_engine_debug_replay_frame(replay, logits, vocab_size,
                                      speech_active, end_sample,
                                      &replay_hit, &replay_detected) == KWS_OK);
  CHECK(replay_detected == live_detected);
  if (live_detected != 0) {
    CHECK(replay_hit.keyword_id == live_hit.keyword_id);
    CHECK(replay_hit.end_sample == live_hit.end_sample);
    CHECK(fabsf(replay_hit.confidence - live_hit.confidence) < 1.0e-7f);
  }
  replay_detected = 7;
  CHECK(kws_engine_debug_replay_frame(replay, logits, vocab_size,
                                      speech_active, end_sample,
                                      &replay_hit, &replay_detected) == KWS_EINVAL);
  CHECK(replay_detected == 0);

  kws_engine_reset(live);
  CHECK(kws_engine_debug_copy_last_frame(live, &frame_number, &end_sample,
                                         &speech_active, logits,
                                         KWS_MAX_VOCAB_SIZE,
                                         &vocab_size) == 0);
}


static void test_validation(void) {
  _Alignas(max_align_t) uint8_t blob[512];
  _Alignas(max_align_t) uint8_t arena[65536];
  kws_model_t model;
  kws_engine_t *engine = NULL;
  size_t bytes = make_test_model(blob, sizeof(blob));
  const uint16_t invalid_sequence[] = {99u};
  const uint16_t seq_a[] = {1u};
  const uint16_t seq_b[] = {2u};
  kws_keyword_t invalid_keyword = make_keyword(1u, invalid_sequence, 1u, 0.5f);
  kws_keyword_t duplicate_ids[] = {
      make_keyword(7u, seq_a, 1u, 0.5f),
      make_keyword(7u, seq_b, 1u, 0.5f),
  };

  CHECK(kws_model_open(blob, bytes - 1u, &model) == KWS_EFORMAT);
  put16(blob + 4u, 1u);
  CHECK(kws_model_open(blob, bytes, &model) == KWS_EFORMAT);
  put16(blob + 4u, KWS_MODEL_VERSION);
  put16(blob + 14u, 9u);
  CHECK(kws_model_open(blob, bytes, &model) == KWS_EFORMAT);
  put16(blob + 14u, KWS_FRONTEND_PCEN_LITE);
  CHECK(kws_model_open(blob, bytes, &model) == KWS_OK);
  CHECK(model.frontend_kind == KWS_FRONTEND_PCEN_LITE);
  put16(blob + 14u, KWS_FRONTEND_LOGMEL);
  put64(blob + 40u, 0u);
  CHECK(kws_model_open(blob, bytes, &model) == KWS_EFORMAT);
  put64(blob + 40u, TEST_VOCAB_FINGERPRINT);
  put32(blob + 20u, KWS_FRAME_LENGTH_SAMPLES - 1u);
  CHECK(kws_model_open(blob, bytes, &model) == KWS_EFORMAT);
  put32(blob + 20u, KWS_FRAME_LENGTH_SAMPLES);
  put32(blob + 24u, KWS_FRAME_HOP_SAMPLES - 1u);
  CHECK(kws_model_open(blob, bytes, &model) == KWS_EFORMAT);
  put32(blob + 24u, KWS_FRAME_HOP_SAMPLES);
  putf(blob + 28u, NAN);
  CHECK(kws_model_open(blob, bytes, &model) == KWS_EFORMAT);
  putf(blob + 28u, 0.01f);
  putf(blob + TEST_BH_OFFSET, INFINITY);
  CHECK(kws_model_open(blob, bytes, &model) == KWS_EFORMAT);
  putf(blob + TEST_BH_OFFSET, 0.0f);
  putf(blob + TEST_BO_OFFSET, NAN);
  CHECK(kws_model_open(blob, bytes, &model) == KWS_EFORMAT);
  putf(blob + TEST_BO_OFFSET, -4.0f);

  CHECK(kws_model_open(blob, bytes, &model) == KWS_OK);
  CHECK(kws_engine_init(arena, sizeof(arena), &model, NULL, &engine) == KWS_OK);
  CHECK(kws_engine_set_keywords(engine, &invalid_keyword, 1u, TEST_VOCAB_FINGERPRINT) == KWS_EBOUNDS);
  CHECK(kws_engine_set_keywords(engine, duplicate_ids, 2u, TEST_VOCAB_FINGERPRINT) == KWS_EINVAL);
}

static void test_metadata_and_build_identity(void) {
  _Alignas(max_align_t) uint8_t blob[512];
  _Alignas(max_align_t) uint8_t arena[65536];
  kws_model_t model;
  kws_engine_t *engine = NULL;
  kws_frame_metadata_t metadata;
  kws_engine_stats_v2_t stats;
  const kws_build_info_t *build = kws_build_info();
  int16_t pcm[320] = {0};
  int detected = 0;
  size_t bytes = make_test_model(blob, sizeof(blob));

  CHECK(build != NULL);
  CHECK(build->struct_size == sizeof(*build));
  CHECK(build->api_version == KWS_BUILD_INFO_API_VERSION);
  CHECK(build->version != NULL && build->version[0] != '\0');
  CHECK(build->source_revision != NULL && build->source_revision[0] != '\0');
  CHECK(build->config_digest != NULL && strlen(build->config_digest) == 64u);

  CHECK(kws_model_open(blob, bytes, &model) == KWS_OK);
  CHECK(kws_engine_init(arena, sizeof(arena), &model, NULL, &engine) == KWS_OK);

  memset(&metadata, 0, sizeof(metadata));
  metadata.struct_size = sizeof(metadata);
  metadata.api_version = KWS_FRAME_METADATA_API_VERSION;
  metadata.flags = KWS_FRAME_EXTERNAL_VAD_VALID;
  metadata.external_vad_probability = 0.0f;
  CHECK(kws_engine_accept_pcm16_ex(engine, pcm, 320u, &metadata, NULL, &detected) == KWS_OK);
  CHECK(kws_engine_accept_pcm16_ex(engine, pcm, 80u, &metadata, NULL, &detected) == KWS_OK);

  memset(&metadata, 0, sizeof(metadata));
  metadata.struct_size = sizeof(metadata);
  metadata.api_version = KWS_FRAME_METADATA_API_VERSION;
  metadata.flags = KWS_FRAME_DISCONTINUITY;
  metadata.lost_samples = 160u;
  metadata.stream_sequence = 7u;
  metadata.capture_timestamp_ns = UINT64_C(123456789);
  metadata.afe_latency_samples = 320u;
  memset(metadata.afe_config_sha256, 0x5a, sizeof(metadata.afe_config_sha256));
  CHECK(kws_engine_accept_pcm16_ex(engine, NULL, 0u, &metadata, NULL, &detected) == KWS_OK);

  memset(&stats, 0, sizeof(stats));
  stats.struct_size = sizeof(stats);
  stats.api_version = KWS_ENGINE_STATS_V2_API_VERSION;
  CHECK(kws_engine_get_stats_v2(engine, &stats) == KWS_OK);
  CHECK(stats.processed_samples == 400u);
  CHECK(stats.processed_frames == 1u);
  CHECK(stats.speech_frames == 0u);
  CHECK(stats.external_vad_frames == 1u);
  CHECK(stats.discontinuities == 1u);
  CHECK(stats.lost_samples == 160u);
  CHECK(stats.last_stream_sequence == 7u);
  CHECK(stats.last_capture_timestamp_ns == UINT64_C(123456789));
  CHECK(stats.afe_latency_samples == 320u);
  CHECK(stats.afe_config_sha256[0] == 0x5au);

  metadata.api_version = 0u;
  CHECK(kws_engine_accept_pcm16_ex(engine, NULL, 0u, &metadata, NULL, &detected) == KWS_EINVAL);

  /* A non-zero reserved word is the forward-compatibility signal: a caller
   * built against a newer ABI must be rejected, not silently stripped of the
   * field it set. */
  metadata.api_version = KWS_FRAME_METADATA_API_VERSION;
  metadata.reserved[0] = 1u;
  CHECK(kws_engine_accept_pcm16_ex(engine, pcm, TEST_BLOCK_SAMPLES, &metadata, NULL,
                                   &detected) == KWS_EINVAL);
  metadata.reserved[0] = 0u;
  metadata.reserved[7] = 1u;
  CHECK(kws_engine_accept_pcm16_ex(engine, pcm, TEST_BLOCK_SAMPLES, &metadata, NULL,
                                   &detected) == KWS_EINVAL);
  metadata.reserved[7] = 0u;
  CHECK(kws_engine_accept_pcm16_ex(engine, pcm, TEST_BLOCK_SAMPLES, &metadata, NULL,
                                   &detected) == KWS_OK);
}

static void test_external_vad_threshold(void) {
  _Alignas(max_align_t) uint8_t blob[512];
  _Alignas(max_align_t) uint8_t arena[65536];
  int16_t pcm[KWS_FRAME_LENGTH_SAMPLES];
  kws_model_t model;
  kws_engine_t *engine = NULL;
  kws_config_t config = kws_default_config();
  kws_config_t invalid_config;
  kws_frame_metadata_t metadata;
  kws_engine_stats_t stats;
  size_t bytes = make_test_model(blob, sizeof(blob));
  int detected = 0;

  CHECK(kws_model_open(blob, bytes, &model) == KWS_OK);

  /* The contract default must itself be a legal probability. */
  CHECK(config.external_vad_threshold > 0.0f);
  CHECK(config.external_vad_threshold < 1.0f);

  invalid_config = kws_default_config();
  invalid_config.external_vad_threshold = 0.0f;
  CHECK(kws_engine_init(arena, sizeof(arena), &model, &invalid_config, &engine) ==
        KWS_EINVAL);
  invalid_config.external_vad_threshold = 1.0f;
  CHECK(kws_engine_init(arena, sizeof(arena), &model, &invalid_config, &engine) ==
        KWS_EINVAL);
  invalid_config.external_vad_threshold = NAN;
  CHECK(kws_engine_init(arena, sizeof(arena), &model, &invalid_config, &engine) ==
        KWS_EINVAL);

  /* Every L2 bound comes from configs/parameter-contract.json. */
  invalid_config = kws_default_config();
  invalid_config.min_speech_dbfs = -121.0f;
  CHECK(kws_engine_init(arena, sizeof(arena), &model, &invalid_config, &engine) ==
        KWS_EINVAL);
  invalid_config = kws_default_config();
  invalid_config.refractory_ms = 10001u;
  CHECK(kws_engine_init(arena, sizeof(arena), &model, &invalid_config, &engine) ==
        KWS_EINVAL);
  invalid_config = kws_default_config();
  invalid_config.state_retention = 1.0f;
  CHECK(kws_engine_init(arena, sizeof(arena), &model, &invalid_config, &engine) ==
        KWS_EINVAL);

  for (size_t i = 0u; i < KWS_FRAME_LENGTH_SAMPLES; ++i) {
    pcm[i] = ((i / 20u) & 1u) != 0u ? 12000 : -12000;
  }
  memset(&metadata, 0, sizeof(metadata));
  metadata.struct_size = sizeof(metadata);
  metadata.api_version = KWS_FRAME_METADATA_API_VERSION;
  metadata.flags = KWS_FRAME_EXTERNAL_VAD_VALID;
  metadata.external_vad_probability = 0.5f;

  /* 0.5 clears the default threshold, so the frame counts as speech. */
  CHECK(kws_engine_init(arena, sizeof(arena), &model, &config, &engine) == KWS_OK);
  CHECK(kws_engine_accept_pcm16_ex(engine, pcm, TEST_BLOCK_SAMPLES, &metadata, NULL,
                                   &detected) == KWS_OK);
  CHECK(kws_engine_accept_pcm16_ex(engine, pcm + TEST_BLOCK_SAMPLES,
                                   KWS_FRAME_LENGTH_SAMPLES - TEST_BLOCK_SAMPLES,
                                   &metadata, NULL, &detected) == KWS_OK);
  CHECK(kws_engine_get_stats(engine, &stats) == KWS_OK);
  CHECK(stats.processed_frames == 1u);
  CHECK(stats.speech_frames == 1u);

  /* Raising the threshold above the supplied probability must flip the gate;
   * the hard-coded 0.45 this replaced could never do that. */
  config.external_vad_threshold = 0.6f;
  CHECK(kws_engine_init(arena, sizeof(arena), &model, &config, &engine) == KWS_OK);
  CHECK(kws_engine_accept_pcm16_ex(engine, pcm, TEST_BLOCK_SAMPLES, &metadata, NULL,
                                   &detected) == KWS_OK);
  CHECK(kws_engine_accept_pcm16_ex(engine, pcm + TEST_BLOCK_SAMPLES,
                                   KWS_FRAME_LENGTH_SAMPLES - TEST_BLOCK_SAMPLES,
                                   &metadata, NULL, &detected) == KWS_OK);
  CHECK(kws_engine_get_stats(engine, &stats) == KWS_OK);
  CHECK(stats.processed_frames == 1u);
  CHECK(stats.speech_frames == 0u);
}

/* The int8 dot product is the only place the Cortex-A32 build takes a different
 * code path (NEON) from the hosted build. The zero-weight fixtures used
 * elsewhere cannot see it at all: every lane computes 0 * x. This test runs a
 * non-zero-weight model through the full engine and checks that the weight bytes
 * decide the outcome, by negating every in-projection weight and requiring the
 * detection to move to the other phase. Value-level parity between the scalar
 * and NEON kernels is asserted separately by tests/test_arm_parity.py. */
static void test_weighted_kernel_inference(void) {
  _Alignas(max_align_t) uint8_t blob[512];
  _Alignas(max_align_t) uint8_t arena[65536];
  kws_model_t model;
  kws_engine_t *engine = NULL;
  kws_config_t config = kws_default_config();
  const uint16_t sequence[] = {1u};
  const kws_keyword_t keyword = make_keyword(42u, sequence, 1u, 0.10f);
  int16_t pcm[1200];
  const size_t sample_count = sizeof(pcm) / sizeof(pcm[0]);
  int fired[2] = {0, 0};
  float best[2] = {0.0f, 0.0f};

  config.min_speech_dbfs = -80.0f;
  config.refractory_ms = 100u;
  for (size_t i = 0u; i < sample_count; ++i) {
    pcm[i] = ((i / 20u) & 1u) != 0u ? 12000 : -12000;
  }

  for (int variant = 0; variant < 2; ++variant) {
    size_t bytes = make_weighted_test_model(blob, sizeof(blob), variant);

    CHECK(kws_model_open(blob, bytes, &model) == KWS_OK);
    CHECK(kws_engine_init(arena, sizeof(arena), &model, &config, &engine) ==
          KWS_OK);
    CHECK(kws_engine_set_keywords(engine, &keyword, 1u,
                                  TEST_VOCAB_FINGERPRINT) == KWS_OK);
    for (size_t offset = 0u; offset < sample_count; offset += TEST_BLOCK_SAMPLES) {
      size_t count = sample_count - offset;
      kws_detection_t hit = {0u, 0.0f, 0u};
      int detected = 0;

      if (count > TEST_BLOCK_SAMPLES) {
        count = TEST_BLOCK_SAMPLES;
      }
      CHECK(kws_engine_accept_pcm16(engine, pcm + offset, count, &hit,
                                    &detected) == KWS_OK);
      if (detected != 0) {
        CHECK(hit.keyword_id == 42u);
        fired[variant] += 1;
        if (hit.confidence > best[variant]) {
          best[variant] = hit.confidence;
        }
      }
    }
  }

  /* Exactly one phase can win: the two fixtures differ only by an exact
   * negation of every in-projection weight, so their top logits are exact
   * negatives and cannot both clear the threshold. A kernel that loses the
   * sign of the weight bytes stops producing exact negatives, which fails
   * here. */
  CHECK(fired[0] + fired[1] == 1);
  CHECK((fired[0] == 1 ? best[0] : best[1]) > 0.10f);
}

int main(void) {
  test_model_and_engine();
  test_debug_frame_replay();
  test_validation();
  test_metadata_and_build_identity();
  test_external_vad_threshold();
  test_weighted_kernel_inference();
  puts("kws_tests: ok");
  return 0;
}
