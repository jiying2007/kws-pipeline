#include "kws_pipeline/kws.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CHECK(x)                                                               \
  do {                                                                         \
    if (!(x)) {                                                                \
      fprintf(stderr, "bench failure: %s\n", #x);                            \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#define BENCH_FEATURE_DIM 32u
#define BENCH_HIDDEN_DIM 48u
#define BENCH_VOCAB_SIZE 420u
#define BENCH_SECONDS 10u
#define BENCH_BLOCK_SAMPLES 160u
#define BENCH_VOCAB_FINGERPRINT UINT64_C(0x123456789abcdef0)
#define BENCH_MODEL_BUDGET_BYTES 30000u
#define BENCH_ENGINE_BUDGET_BYTES 65536u

typedef struct bench_case {
  const char *name;
  uint16_t frontend_kind;
  size_t keyword_count;
} bench_case_t;

static void put16(uint8_t *p, uint16_t value) {
  p[0] = (uint8_t)(value & 0xffu);
  p[1] = (uint8_t)(value >> 8u);
}

static void put32(uint8_t *p, uint32_t value) {
  p[0] = (uint8_t)(value & 0xffu);
  p[1] = (uint8_t)((value >> 8u) & 0xffu);
  p[2] = (uint8_t)((value >> 16u) & 0xffu);
  p[3] = (uint8_t)(value >> 24u);
}

static void put64(uint8_t *p, uint64_t value) {
  put32(p, (uint32_t)(value & UINT64_C(0xffffffff)));
  put32(p + 4u, (uint32_t)(value >> 32u));
}

static void putf(uint8_t *p, float value) {
  uint32_t bits = 0u;
  memcpy(&bits, &value, sizeof(bits));
  put32(p, bits);
}

static uint32_t align4(uint32_t value) {
  return (value + 3u) & ~UINT32_C(3);
}

static size_t make_model(uint8_t *blob,
                         size_t capacity,
                         uint16_t frontend_kind) {
  const uint32_t wx = 72u;
  const uint32_t wx_bytes = BENCH_HIDDEN_DIM * BENCH_FEATURE_DIM;
  const uint32_t wh = align4(wx + wx_bytes);
  const uint32_t wh_bytes = BENCH_HIDDEN_DIM * BENCH_HIDDEN_DIM;
  const uint32_t bh = align4(wh + wh_bytes);
  const uint32_t bh_bytes = BENCH_HIDDEN_DIM * 4u;
  const uint32_t wo = align4(bh + bh_bytes);
  const uint32_t wo_bytes = BENCH_VOCAB_SIZE * BENCH_HIDDEN_DIM;
  const uint32_t bo = align4(wo + wo_bytes);
  const uint32_t total = bo + BENCH_VOCAB_SIZE * 4u;

  CHECK(frontend_kind == KWS_FRONTEND_LOGMEL ||
        frontend_kind == KWS_FRONTEND_PCEN_LITE);
  CHECK(capacity >= total);
  memset(blob, 0, total);
  memcpy(blob, "KWSP", 4u);
  put16(blob + 4u, KWS_MODEL_VERSION);
  put16(blob + 6u, 72u);
  put16(blob + 8u, BENCH_FEATURE_DIM);
  put16(blob + 10u, BENCH_HIDDEN_DIM);
  put16(blob + 12u, BENCH_VOCAB_SIZE);
  put16(blob + 14u, frontend_kind);
  put32(blob + 16u, KWS_SAMPLE_RATE_HZ);
  put32(blob + 20u, KWS_FRAME_LENGTH_SAMPLES);
  put32(blob + 24u, KWS_FRAME_HOP_SAMPLES);
  putf(blob + 28u, 0.01f);
  putf(blob + 32u, 0.01f);
  putf(blob + 36u, 0.01f);
  put64(blob + 40u, BENCH_VOCAB_FINGERPRINT);
  put32(blob + 48u, wx);
  put32(blob + 52u, wh);
  put32(blob + 56u, bh);
  put32(blob + 60u, wo);
  put32(blob + 64u, bo);
  put32(blob + 68u, total);
  return (size_t)total;
}

static void make_keywords(kws_keyword_t *keywords,
                          uint16_t token_storage[KWS_MAX_KEYWORDS][4],
                          size_t count) {
  CHECK(count > 0u && count <= KWS_MAX_KEYWORDS);
  for (size_t i = 0u; i < count; ++i) {
    token_storage[i][0] = 1u;
    token_storage[i][1] = 2u;
    token_storage[i][2] = (uint16_t)(1u + ((i / 4u) % 4u));
    token_storage[i][3] = (uint16_t)(1u + (i % 4u));
    keywords[i].id = (uint32_t)(i + 1u);
    keywords[i].tokens = token_storage[i];
    keywords[i].num_tokens = 4u;
    keywords[i].threshold = 0.99f;
    keywords[i].min_trailing_blanks = 1u;
    keywords[i].priority = (uint8_t)i;
    keywords[i].prefix_policy = (uint8_t)KWS_PREFIX_GRACE;
    keywords[i].grace_frames = 2u;
  }
}

static void run_case(const bench_case_t *bench_case) {
  _Alignas(8) static uint8_t model_blob[BENCH_MODEL_BUDGET_BYTES];
  _Alignas(8) static uint8_t arena[BENCH_ENGINE_BUDGET_BYTES];
  int16_t pcm[BENCH_BLOCK_SAMPLES];
  kws_model_t model;
  kws_engine_t *engine = NULL;
  kws_config_t config = kws_default_config();
  kws_keyword_t keywords[KWS_MAX_KEYWORDS];
  uint16_t token_storage[KWS_MAX_KEYWORDS][4];
  kws_engine_stats_t stats;
  const uint64_t total_samples =
      (uint64_t)BENCH_SECONDS * (uint64_t)KWS_SAMPLE_RATE_HZ;
  const uint64_t estimated_macs_per_frame =
      (uint64_t)BENCH_HIDDEN_DIM * (uint64_t)BENCH_FEATURE_DIM +
      (uint64_t)BENCH_HIDDEN_DIM * (uint64_t)BENCH_HIDDEN_DIM +
      (uint64_t)BENCH_VOCAB_SIZE * (uint64_t)BENCH_HIDDEN_DIM;
  uint64_t consumed = 0u;
  size_t model_bytes =
      make_model(model_blob, sizeof(model_blob), bench_case->frontend_kind);
  clock_t begin;
  clock_t end;
  double elapsed;
  double rtf;

  CHECK(kws_model_open(model_blob, model_bytes, &model) == KWS_OK);
  CHECK(model_bytes <= BENCH_MODEL_BUDGET_BYTES);
  CHECK(kws_engine_required_bytes(&model) <= sizeof(arena));
  config.min_speech_dbfs = -120.0f;
  CHECK(kws_engine_init(arena, sizeof(arena), &model, &config, &engine) ==
        KWS_OK);
  make_keywords(keywords, token_storage, bench_case->keyword_count);
  CHECK(kws_engine_set_keywords(engine, keywords, bench_case->keyword_count,
                                BENCH_VOCAB_FINGERPRINT) == KWS_OK);

  for (size_t i = 0u; i < BENCH_BLOCK_SAMPLES; ++i) {
    pcm[i] = ((i / 8u) & 1u) != 0u ? 12000 : -12000;
  }

  begin = clock();
  while (consumed < total_samples) {
    size_t count = BENCH_BLOCK_SAMPLES;
    int detected = 0;
    if (total_samples - consumed < (uint64_t)count) {
      count = (size_t)(total_samples - consumed);
    }
    CHECK(kws_engine_accept_pcm16(engine, pcm, count, NULL, &detected) ==
          KWS_OK);
    consumed += count;
  }
  end = clock();
  CHECK(end >= begin);
  CHECK(kws_engine_get_stats(engine, &stats) == KWS_OK);
  CHECK(stats.keyword_count == bench_case->keyword_count);
  CHECK(stats.trie_nodes <= KWS_MAX_TRIE_NODES);
  CHECK(stats.processed_frames > 0u);

  elapsed = (double)(end - begin) / (double)CLOCKS_PER_SEC;
  rtf = elapsed / (double)BENCH_SECONDS;
  printf("case=%s frontend=%s keywords=%zu trie_nodes=%u geometry=%ux%ux%u "
         "model_bytes=%zu engine_bytes=%zu estimated_macs_per_frame=%llu "
         "audio_s=%u cpu_s=%.6f rtf=%.6f us_per_audio_s=%.1f\n",
         bench_case->name,
         bench_case->frontend_kind == KWS_FRONTEND_PCEN_LITE ? "pcen-lite"
                                                              : "logmel",
         bench_case->keyword_count, (unsigned)stats.trie_nodes,
         (unsigned)BENCH_FEATURE_DIM, (unsigned)BENCH_HIDDEN_DIM,
         (unsigned)BENCH_VOCAB_SIZE, model_bytes,
         kws_engine_required_bytes(&model),
         (unsigned long long)estimated_macs_per_frame,
         (unsigned)BENCH_SECONDS, elapsed, rtf,
         elapsed * 1000000.0 / (double)BENCH_SECONDS);
}

int main(void) {
  const bench_case_t cases[] = {
      {"baseline", KWS_FRONTEND_LOGMEL, 1u},
      {"product", KWS_FRONTEND_LOGMEL, 4u},
      {"pcen-product", KWS_FRONTEND_PCEN_LITE, 4u},
      {"worst-case", KWS_FRONTEND_PCEN_LITE, KWS_MAX_KEYWORDS},
  };

  for (size_t i = 0u; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    run_case(&cases[i]);
  }
  return 0;
}
