#include "kws_pipeline/kws.h"

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

static void put16(uint8_t *p, uint16_t v) {
  p[0] = (uint8_t)(v & 0xffu);
  p[1] = (uint8_t)(v >> 8u);
}

static void put32(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)(v & 0xffu);
  p[1] = (uint8_t)((v >> 8u) & 0xffu);
  p[2] = (uint8_t)((v >> 16u) & 0xffu);
  p[3] = (uint8_t)((v >> 24u) & 0xffu);
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
  putf(blob + bo + 8u, 4.0f);
  putf(blob + bo + 12u, -4.0f);
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
  const uint16_t sequence[] = {1u, 2u};
  const kws_keyword_t keyword = {42u, sequence, 2u, 0.30f};
  const kws_keyword_t nan_keyword = {43u, sequence, 2u, NAN};
  const kws_keyword_t ambiguous[] = {
      {100u, sequence, 2u, 0.30f},
      {101u, sequence, 2u, 0.40f},
  };
  int16_t pcm[1200];
  int detected_any = 0;
  kws_detection_t first_detection = {0u, 0.0f, 0u};
  size_t bytes = make_test_model(blob, sizeof(blob));
  const size_t sample_count = sizeof(pcm) / sizeof(pcm[0]);

  CHECK(kws_model_open(blob, bytes, &model) == KWS_OK);
  CHECK(model.vocab_fingerprint == TEST_VOCAB_FINGERPRINT);
  CHECK(kws_engine_required_alignment() >= _Alignof(uint64_t));
  CHECK(kws_engine_required_bytes(&model) <= sizeof(arena));
  CHECK(kws_engine_init(arena + 1u, sizeof(arena) - 1u, &model, NULL, &engine) ==
        KWS_EINVAL);

  invalid_model = model;
  invalid_model.feature_dim = (uint16_t)(KWS_MAX_FEATURE_DIM + 1u);
  CHECK(kws_engine_required_bytes(&invalid_model) == 0u);
  CHECK(kws_engine_init(arena, sizeof(arena), &invalid_model, NULL, &engine) ==
        KWS_EINVAL);

  invalid_model = model;
  invalid_model.frame_hop_samples = KWS_FRAME_HOP_SAMPLES / 2u;
  CHECK(kws_engine_required_bytes(&invalid_model) == 0u);

  invalid_model = model;
  invalid_model.wx_scale = NAN;
  CHECK(kws_engine_required_bytes(&invalid_model) == 0u);
  CHECK(kws_engine_init(arena, sizeof(arena), &invalid_model, NULL, &engine) ==
        KWS_EINVAL);

  invalid_config = kws_default_config();
  invalid_config.token_boost = NAN;
  CHECK(kws_engine_init(arena, sizeof(arena), &model, &invalid_config, &engine) ==
        KWS_EINVAL);
  invalid_config = kws_default_config();
  invalid_config.min_speech_dbfs = INFINITY;
  CHECK(kws_engine_init(arena, sizeof(arena), &model, &invalid_config, &engine) ==
        KWS_EINVAL);

  config.min_speech_dbfs = -80.0f;
  config.refractory_ms = 100u;
  CHECK(kws_engine_init(arena, sizeof(arena), &model, &config, &engine) ==
        KWS_OK);
  CHECK(kws_engine_set_keywords(engine, &keyword, 1u,
                                TEST_VOCAB_FINGERPRINT) == KWS_OK);
  CHECK(kws_engine_set_keywords(engine, &keyword, 1u,
                                UINT64_C(0x8877665544332211)) == KWS_EFORMAT);
  CHECK(kws_engine_set_keywords(engine, &nan_keyword, 1u,
                                TEST_VOCAB_FINGERPRINT) == KWS_EINVAL);
  CHECK(kws_engine_set_keywords(engine, ambiguous, 2u,
                                TEST_VOCAB_FINGERPRINT) == KWS_EINVAL);

  for (size_t i = 0u; i < sample_count; ++i) {
    pcm[i] = ((i / 20u) & 1u) != 0u ? 12000 : -12000;
  }

  {
    int detected = 7;
    uint64_t before = kws_engine_processed_samples(engine);
    CHECK(kws_engine_accept_pcm16(engine, pcm, KWS_MAX_PCM_BLOCK_SAMPLES + 1u,
                                  NULL, &detected) == KWS_EBOUNDS);
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
    CHECK(kws_engine_accept_pcm16(engine, pcm + offset, count, &detection,
                                  &detected) == KWS_OK);
    if (detected != 0 && detected_any == 0) {
      first_detection = detection;
      detected_any = 1;
    }
  }

  CHECK(detected_any == 1);
  CHECK(first_detection.keyword_id == 42u);
  CHECK(first_detection.confidence > 0.30f);
  CHECK(first_detection.end_sample > 0u);
  CHECK(kws_engine_processed_samples(engine) == sample_count);
}

static void test_validation(void) {
  _Alignas(max_align_t) uint8_t blob[512];
  _Alignas(max_align_t) uint8_t arena[65536];
  kws_model_t model;
  kws_engine_t *engine = NULL;
  size_t bytes = make_test_model(blob, sizeof(blob));
  const uint16_t invalid_sequence[] = {99u};
  const kws_keyword_t invalid_keyword = {1u, invalid_sequence, 1u, 0.5f};
  const uint16_t seq_a[] = {1u};
  const uint16_t seq_b[] = {2u};
  const kws_keyword_t duplicate_ids[] = {
      {7u, seq_a, 1u, 0.5f},
      {7u, seq_b, 1u, 0.5f},
  };

  CHECK(kws_model_open(blob, bytes - 1u, &model) == KWS_EFORMAT);

  put16(blob + 4u, 1u);
  CHECK(kws_model_open(blob, bytes, &model) == KWS_EFORMAT);
  put16(blob + 4u, KWS_MODEL_VERSION);

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
  CHECK(kws_engine_set_keywords(engine, &invalid_keyword, 1u,
                                TEST_VOCAB_FINGERPRINT) == KWS_EBOUNDS);
  CHECK(kws_engine_set_keywords(engine, duplicate_ids, 2u,
                                TEST_VOCAB_FINGERPRINT) == KWS_EINVAL);
}

int main(void) {
  test_model_and_engine();
  test_validation();
  puts("kws_tests: ok");
  return 0;
}
