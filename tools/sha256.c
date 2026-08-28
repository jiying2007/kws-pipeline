#include "sha256.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef struct sha256_ctx {
  uint32_t state[8];
  uint64_t total_bytes;
  uint8_t buffer[64];
  size_t buffer_bytes;
} sha256_ctx_t;

static const uint32_t k_table[64] = {
    UINT32_C(0x428a2f98), UINT32_C(0x71374491), UINT32_C(0xb5c0fbcf),
    UINT32_C(0xe9b5dba5), UINT32_C(0x3956c25b), UINT32_C(0x59f111f1),
    UINT32_C(0x923f82a4), UINT32_C(0xab1c5ed5), UINT32_C(0xd807aa98),
    UINT32_C(0x12835b01), UINT32_C(0x243185be), UINT32_C(0x550c7dc3),
    UINT32_C(0x72be5d74), UINT32_C(0x80deb1fe), UINT32_C(0x9bdc06a7),
    UINT32_C(0xc19bf174), UINT32_C(0xe49b69c1), UINT32_C(0xefbe4786),
    UINT32_C(0x0fc19dc6), UINT32_C(0x240ca1cc), UINT32_C(0x2de92c6f),
    UINT32_C(0x4a7484aa), UINT32_C(0x5cb0a9dc), UINT32_C(0x76f988da),
    UINT32_C(0x983e5152), UINT32_C(0xa831c66d), UINT32_C(0xb00327c8),
    UINT32_C(0xbf597fc7), UINT32_C(0xc6e00bf3), UINT32_C(0xd5a79147),
    UINT32_C(0x06ca6351), UINT32_C(0x14292967), UINT32_C(0x27b70a85),
    UINT32_C(0x2e1b2138), UINT32_C(0x4d2c6dfc), UINT32_C(0x53380d13),
    UINT32_C(0x650a7354), UINT32_C(0x766a0abb), UINT32_C(0x81c2c92e),
    UINT32_C(0x92722c85), UINT32_C(0xa2bfe8a1), UINT32_C(0xa81a664b),
    UINT32_C(0xc24b8b70), UINT32_C(0xc76c51a3), UINT32_C(0xd192e819),
    UINT32_C(0xd6990624), UINT32_C(0xf40e3585), UINT32_C(0x106aa070),
    UINT32_C(0x19a4c116), UINT32_C(0x1e376c08), UINT32_C(0x2748774c),
    UINT32_C(0x34b0bcb5), UINT32_C(0x391c0cb3), UINT32_C(0x4ed8aa4a),
    UINT32_C(0x5b9cca4f), UINT32_C(0x682e6ff3), UINT32_C(0x748f82ee),
    UINT32_C(0x78a5636f), UINT32_C(0x84c87814), UINT32_C(0x8cc70208),
    UINT32_C(0x90befffa), UINT32_C(0xa4506ceb), UINT32_C(0xbef9a3f7),
    UINT32_C(0xc67178f2),
};

static uint32_t rotr32(uint32_t value, unsigned bits) {
  return (value >> bits) | (value << (32u - bits));
}

static uint32_t read_be32(const uint8_t *p) {
  return ((uint32_t)p[0] << 24u) | ((uint32_t)p[1] << 16u) |
         ((uint32_t)p[2] << 8u) | (uint32_t)p[3];
}

static void write_be32(uint8_t *p, uint32_t value) {
  p[0] = (uint8_t)(value >> 24u);
  p[1] = (uint8_t)(value >> 16u);
  p[2] = (uint8_t)(value >> 8u);
  p[3] = (uint8_t)value;
}

static void write_be64(uint8_t *p, uint64_t value) {
  for (unsigned i = 0u; i < 8u; ++i) {
    p[7u - i] = (uint8_t)(value >> (i * 8u));
  }
}

static void sha256_transform(sha256_ctx_t *ctx, const uint8_t block[64]) {
  uint32_t w[64];
  uint32_t a;
  uint32_t b;
  uint32_t c;
  uint32_t d;
  uint32_t e;
  uint32_t f;
  uint32_t g;
  uint32_t h;

  for (unsigned i = 0u; i < 16u; ++i) {
    w[i] = read_be32(block + i * 4u);
  }
  for (unsigned i = 16u; i < 64u; ++i) {
    uint32_t s0 = rotr32(w[i - 15u], 7u) ^ rotr32(w[i - 15u], 18u) ^
                  (w[i - 15u] >> 3u);
    uint32_t s1 = rotr32(w[i - 2u], 17u) ^ rotr32(w[i - 2u], 19u) ^
                  (w[i - 2u] >> 10u);
    w[i] = w[i - 16u] + s0 + w[i - 7u] + s1;
  }

  a = ctx->state[0];
  b = ctx->state[1];
  c = ctx->state[2];
  d = ctx->state[3];
  e = ctx->state[4];
  f = ctx->state[5];
  g = ctx->state[6];
  h = ctx->state[7];

  for (unsigned i = 0u; i < 64u; ++i) {
    uint32_t sum1 = rotr32(e, 6u) ^ rotr32(e, 11u) ^ rotr32(e, 25u);
    uint32_t choose = (e & f) ^ ((~e) & g);
    uint32_t temp1 = h + sum1 + choose + k_table[i] + w[i];
    uint32_t sum0 = rotr32(a, 2u) ^ rotr32(a, 13u) ^ rotr32(a, 22u);
    uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
    uint32_t temp2 = sum0 + majority;
    h = g;
    g = f;
    f = e;
    e = d + temp1;
    d = c;
    c = b;
    b = a;
    a = temp1 + temp2;
  }

  ctx->state[0] += a;
  ctx->state[1] += b;
  ctx->state[2] += c;
  ctx->state[3] += d;
  ctx->state[4] += e;
  ctx->state[5] += f;
  ctx->state[6] += g;
  ctx->state[7] += h;
}

static void sha256_init(sha256_ctx_t *ctx) {
  memset(ctx, 0, sizeof(*ctx));
  ctx->state[0] = UINT32_C(0x6a09e667);
  ctx->state[1] = UINT32_C(0xbb67ae85);
  ctx->state[2] = UINT32_C(0x3c6ef372);
  ctx->state[3] = UINT32_C(0xa54ff53a);
  ctx->state[4] = UINT32_C(0x510e527f);
  ctx->state[5] = UINT32_C(0x9b05688c);
  ctx->state[6] = UINT32_C(0x1f83d9ab);
  ctx->state[7] = UINT32_C(0x5be0cd19);
}

static void sha256_update(sha256_ctx_t *ctx, const uint8_t *data, size_t bytes) {
  size_t offset = 0u;
  ctx->total_bytes += (uint64_t)bytes;

  if (ctx->buffer_bytes != 0u) {
    size_t room = 64u - ctx->buffer_bytes;
    size_t take = bytes < room ? bytes : room;
    memcpy(ctx->buffer + ctx->buffer_bytes, data, take);
    ctx->buffer_bytes += take;
    offset += take;
    if (ctx->buffer_bytes == 64u) {
      sha256_transform(ctx, ctx->buffer);
      ctx->buffer_bytes = 0u;
    }
  }

  while (bytes - offset >= 64u) {
    sha256_transform(ctx, data + offset);
    offset += 64u;
  }
  if (offset < bytes) {
    ctx->buffer_bytes = bytes - offset;
    memcpy(ctx->buffer, data + offset, ctx->buffer_bytes);
  }
}

static void sha256_final(sha256_ctx_t *ctx, uint8_t digest[32]) {
  uint64_t total_bits = ctx->total_bytes * UINT64_C(8);
  size_t used = ctx->buffer_bytes;

  ctx->buffer[used++] = UINT8_C(0x80);
  if (used > 56u) {
    memset(ctx->buffer + used, 0, 64u - used);
    sha256_transform(ctx, ctx->buffer);
    used = 0u;
  }
  memset(ctx->buffer + used, 0, 56u - used);
  write_be64(ctx->buffer + 56u, total_bits);
  sha256_transform(ctx, ctx->buffer);

  for (unsigned i = 0u; i < 8u; ++i) {
    write_be32(digest + i * 4u, ctx->state[i]);
  }
}

static void digest_to_hex(const uint8_t digest[32], char out_hex[65]) {
  static const char hex[] = "0123456789abcdef";
  for (unsigned i = 0u; i < 32u; ++i) {
    out_hex[i * 2u] = hex[digest[i] >> 4u];
    out_hex[i * 2u + 1u] = hex[digest[i] & UINT8_C(0x0f)];
  }
  out_hex[64] = '\0';
}

void kws_sha256_memory_hex(const uint8_t *data, size_t bytes, char out_hex[65]) {
  sha256_ctx_t ctx;
  uint8_t digest[32];
  sha256_init(&ctx);
  if (bytes != 0u) {
    sha256_update(&ctx, data, bytes);
  }
  sha256_final(&ctx, digest);
  digest_to_hex(digest, out_hex);
}

int kws_sha256_file_hex(const char *path, char out_hex[65]) {
  sha256_ctx_t ctx;
  uint8_t digest[32];
  uint8_t buffer[8192];
  FILE *stream;

  if (path == NULL || out_hex == NULL) {
    return 0;
  }
  stream = fopen(path, "rb");
  if (stream == NULL) {
    return 0;
  }
  sha256_init(&ctx);
  for (;;) {
    size_t got = fread(buffer, 1u, sizeof(buffer), stream);
    if (got != 0u) {
      sha256_update(&ctx, buffer, got);
    }
    if (got < sizeof(buffer)) {
      if (ferror(stream) != 0) {
        fclose(stream);
        return 0;
      }
      break;
    }
  }
  fclose(stream);
  sha256_final(&ctx, digest);
  digest_to_hex(digest, out_hex);
  return 1;
}
