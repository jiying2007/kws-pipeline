#include "kws_pipeline/kws.h"

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

static void putf(uint8_t *p, float v) {
  uint32_t u = 0u;
  memcpy(&u, &v, sizeof(u));
  put32(p, u);
}

static size_t make_test_model(uint8_t *blob, size_t cap) {
  const uint16_t f = 32u;
  const uint16_t h = 4u;
  const uint16_t v = 4u;
  const uint32_t wx = 64u;
  const uint32_t wh = wx + (uint32_t)f * (uint32_t)h;
  const uint32_t bh = wh + (uint32_t)h * (uint32_t)h;
  const uint32_t wo = bh + (uint32_t)h * 4u;
  const uint32_t bo = wo + (uint32_t)v * (uint32_t)h;
  const uint32_t total = bo + (uint32_t)v * 4u;

  CHECK(cap >= total);
  memset(blob, 0, total);
  memcpy(blob, "KWSP", 4u);
  put16(blob + 4u, 1u);
  put16(blob + 6u, 64u);
  put16(blob + 8u, f);
  put16(blob + 10u, h);
  put16(blob + 12u, v);
  put32(blob + 16u, 16000u);
  put32(blob + 20u, 400u);
  put32(blob + 24u, 320u);
  putf(blob + 28u, 0.01f);
  putf(blob + 32u, 0.01f);
  putf(blob + 36u, 0.01f);
  put32(blob + 40u, wx);
  put32(blob + 44u, wh);
  put32(blob + 48u, bh);
  put32(blob + 52u, wo);
  put32(blob + 56u, bo);
  put32(blob + 60u, total);
  putf(blob + bo + 0u, -4.0f);
  putf(blob + bo + 4u, 4.0f);
  putf(blob + bo + 8u, 4.0f);
  putf(blob + bo + 12u, -4.0f);
  return total;
}

static void test_model_and_engine(void) {
  _Alignas(8) uint8_t blob[512];
  _Alignas(8) uint8_t arena[65536];
  kws_model_t model;
  kws_engine_t *engine = NULL;
  kws_config_t config = kws_default_config();
  const uint16_t sequence[] = {1u, 2u};
  const kws_keyword_t keyword = {42u, sequence, 2u, 0.30f};
  const kws_keyword_t ambiguous[] = {
      {100u, sequence, 2u, 0.30f},
      {101u, sequence, 2u, 0.40f},
  };
  int16_t pcm[1200];
  int detected = 0;
  kws_detection_t detection;
  size_t bytes = make_test_model(blob, sizeof(blob));
  const size_t sample_count = sizeof(pcm) / sizeof(pcm[0]);

  CHECK(kws_model_open(blob, bytes, &model) == KWS_OK);
  CHECK(kws_engine_required_bytes(&model) <= sizeof(arena));
  config.min_speech_dbfs = -80.0f;
  config.refractory_ms = 100u;
  CHECK(kws_engine_init(arena, sizeof(arena), &model, &config, &engine) ==
        KWS_OK);
  CHECK(kws_engine_set_keywords(engine, &keyword, 1u) == KWS_OK);
  CHECK(kws_engine_set_keywords(engine, ambiguous, 2u) == KWS_EINVAL);

  for (size_t i = 0u; i < sample_count; ++i) {
    pcm[i] = ((i / 20u) & 1u) != 0u ? 12000 : -12000;
  }

  CHECK(kws_engine_accept_pcm16(engine, pcm, sample_count, &detection,
                                &detected) == KWS_OK);
  CHECK(detected == 1);
  CHECK(detection.keyword_id == 42u);
  CHECK(detection.confidence > 0.30f);
  CHECK(detection.end_sample > 0u);
  CHECK(kws_engine_processed_samples(engine) == sample_count);
}

static void test_validation(void) {
  _Alignas(8) uint8_t blob[512];
  _Alignas(8) uint8_t arena[65536];
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
  put32(blob + 20u, 1u);
  CHECK(kws_model_open(blob, bytes, &model) == KWS_EFORMAT);
  put32(blob + 20u, 400u);
  CHECK(kws_model_open(blob, bytes, &model) == KWS_OK);
  CHECK(kws_engine_init(arena, sizeof(arena), &model, NULL, &engine) == KWS_OK);
  CHECK(kws_engine_set_keywords(engine, &invalid_keyword, 1u) == KWS_EBOUNDS);
  CHECK(kws_engine_set_keywords(engine, duplicate_ids, 2u) == KWS_EINVAL);
}

int main(void) {
  test_model_and_engine();
  test_validation();
  puts("kws_tests: ok");
  return 0;
}
