#include "kws_pipeline/kws.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "CHECK failed: %s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)

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

static size_t make_pack(uint8_t *blob, size_t cap) {
  const uint16_t count = 2u;
  const uint32_t total = 16u + 44u * (uint32_t)count;
  uint8_t *r0;
  uint8_t *r1;
  CHECK(cap >= total);
  memset(blob, 0, total);
  memcpy(blob, "KWKP", 4u);
  put16(blob + 4u, KWS_KEYWORD_PACK_VERSION);
  put16(blob + 6u, 16u);
  put16(blob + 8u, count);
  put16(blob + 10u, 8u);
  put32(blob + 12u, total);

  r0 = blob + 16u;
  put32(r0 + 0u, 100u);
  putf(r0 + 4u, 0.55f);
  put16(r0 + 8u, 4u);
  put16(r0 + 12u, 1u);
  put16(r0 + 14u, 2u);
  put16(r0 + 16u, 3u);
  put16(r0 + 18u, 4u);

  r1 = blob + 60u;
  put32(r1 + 0u, 101u);
  putf(r1 + 4u, 0.65f);
  put16(r1 + 8u, 2u);
  put16(r1 + 12u, 4u);
  put16(r1 + 14u, 4u);
  return (size_t)total;
}

int main(void) {
  uint8_t blob[128];
  kws_model_t model;
  kws_keyword_pack_t pack;
  size_t bytes;

  memset(&model, 0, sizeof(model));
  model.vocab_size = 8u;
  bytes = make_pack(blob, sizeof(blob));

  CHECK(kws_keyword_pack_open(blob, bytes, &model, &pack) == KWS_OK);
  CHECK(pack.keyword_count == 2u);
  CHECK(pack.keywords[0].id == 100u);
  CHECK(pack.keywords[0].num_tokens == 4u);
  CHECK(pack.keywords[0].tokens[3] == 4u);
  CHECK(pack.keywords[1].id == 101u);
  CHECK(pack.keywords[1].tokens[0] == 4u);

  CHECK(kws_keyword_pack_open(blob, bytes - 1u, &model, &pack) == KWS_EFORMAT);

  put16(blob + 10u, 7u);
  CHECK(kws_keyword_pack_open(blob, bytes, &model, &pack) == KWS_EFORMAT);
  put16(blob + 10u, 8u);

  put32(blob + 60u, 100u);
  CHECK(kws_keyword_pack_open(blob, bytes, &model, &pack) == KWS_EFORMAT);
  put32(blob + 60u, 101u);

  put16(blob + 12u + 32u, 8u);
  CHECK(kws_keyword_pack_open(blob, bytes, &model, &pack) == KWS_EBOUNDS);

  puts("kws_keyword_pack_tests: ok");
  return 0;
}
