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

static size_t make_pack(uint8_t *blob, size_t cap) {
  const uint16_t count = 2u;
  const uint32_t total = 24u + 48u * (uint32_t)count;
  uint8_t *r0;
  uint8_t *r1;

  CHECK(cap >= total);
  memset(blob, 0, total);
  memcpy(blob, "KWKP", 4u);
  put16(blob + 4u, KWS_KEYWORD_PACK_VERSION);
  put16(blob + 6u, 24u);
  put16(blob + 8u, count);
  put16(blob + 10u, 8u);
  put32(blob + 12u, total);
  put64(blob + 16u, TEST_VOCAB_FINGERPRINT);

  r0 = blob + 24u;
  put32(r0 + 0u, 100u);
  putf(r0 + 4u, 0.55f);
  put16(r0 + 8u, 4u);
  r0[10u] = 1u;
  r0[11u] = 3u;
  r0[12u] = (uint8_t)KWS_PREFIX_LONGEST;
  put16(r0 + 16u, 1u);
  put16(r0 + 18u, 2u);
  put16(r0 + 20u, 3u);
  put16(r0 + 22u, 4u);

  r1 = blob + 72u;
  put32(r1 + 0u, 101u);
  putf(r1 + 4u, 0.65f);
  put16(r1 + 8u, 2u);
  r1[12u] = (uint8_t)KWS_PREFIX_GRACE;
  r1[13u] = 3u;
  put16(r1 + 16u, 4u);
  put16(r1 + 18u, 4u);
  return (size_t)total;
}

int main(void) {
  _Alignas(max_align_t) uint8_t arena[65536];
  uint8_t blob[160];
  int8_t wx[32] = {0};
  int8_t wh[1] = {0};
  int8_t wo[8] = {0};
  float bh[1] = {0.0f};
  float bo[8] = {0.0f};
  kws_model_t model;
  kws_keyword_pack_t pack;
  kws_engine_t *engine = NULL;
  size_t bytes;

  memset(&model, 0, sizeof(model));
  model.vocab_size = 8u;
  model.vocab_fingerprint = TEST_VOCAB_FINGERPRINT;
  model.feature_dim = 32u;
  model.hidden_dim = 1u;
  model.frontend_kind = KWS_FRONTEND_LOGMEL;
  model.sample_rate_hz = KWS_SAMPLE_RATE_HZ;
  model.frame_length_samples = 400u;
  model.frame_hop_samples = 320u;
  model.wx_scale = 0.01f;
  model.wh_scale = 0.01f;
  model.wo_scale = 0.01f;
  model.wx = wx;
  model.wh = wh;
  model.bh = bh;
  model.wo = wo;
  model.bo = bo;
  bytes = make_pack(blob, sizeof(blob));

  CHECK(kws_keyword_pack_open(blob, bytes, &model, &pack) == KWS_OK);
  CHECK(pack.keyword_count == 2u);
  CHECK(pack.keywords[0].id == 100u);
  CHECK(pack.keywords[0].tokens[3] == 4u);
  CHECK(pack.keywords[0].min_trailing_blanks == 1u);
  CHECK(pack.keywords[0].priority == 3u);
  CHECK(pack.keywords[0].prefix_policy == (uint8_t)KWS_PREFIX_LONGEST);
  CHECK(pack.keywords[1].prefix_policy == (uint8_t)KWS_PREFIX_GRACE);
  CHECK(pack.keywords[1].grace_frames == 3u);
  CHECK(kws_engine_required_bytes(&model) <= sizeof(arena));
  CHECK(kws_engine_init(arena, sizeof(arena), &model, NULL, &engine) == KWS_OK);
  CHECK(kws_engine_set_keyword_pack(engine, &pack) == KWS_OK);

  CHECK(kws_keyword_pack_open(blob, bytes - 1u, &model, &pack) == KWS_EFORMAT);

  put16(blob + 4u, 2u);
  CHECK(kws_keyword_pack_open(blob, bytes, &model, &pack) == KWS_EFORMAT);
  put16(blob + 4u, KWS_KEYWORD_PACK_VERSION);

  blob[36u] = 9u; /* prefix policy */
  CHECK(kws_keyword_pack_open(blob, bytes, &model, &pack) == KWS_EFORMAT);
  blob[36u] = (uint8_t)KWS_PREFIX_LONGEST;

  blob[34u] = 0u; /* longest requires at least one trailing blank */
  CHECK(kws_keyword_pack_open(blob, bytes, &model, &pack) == KWS_EFORMAT);
  blob[34u] = 1u;

  blob[85u] = 0u; /* grace policy requires grace frames */
  CHECK(kws_keyword_pack_open(blob, bytes, &model, &pack) == KWS_EFORMAT);
  blob[85u] = 3u;

  putf(blob + 28u, NAN);
  CHECK(kws_keyword_pack_open(blob, bytes, &model, &pack) == KWS_EFORMAT);
  putf(blob + 28u, 0.55f);

  put16(blob + 40u, 8u);
  CHECK(kws_keyword_pack_open(blob, bytes, &model, &pack) == KWS_EBOUNDS);
  put16(blob + 40u, 1u);

  put32(blob + 72u, 100u);
  CHECK(kws_keyword_pack_open(blob, bytes, &model, &pack) == KWS_EFORMAT);
  put32(blob + 72u, 101u);

  puts("kws_keyword_pack_tests: ok");
  return 0;
}
