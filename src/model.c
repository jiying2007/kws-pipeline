#include "model.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#define KWS_HEADER_BYTES 72u

_Static_assert(sizeof(float) == 4u, "kws model ABI requires 32-bit float");
_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128,
               "kws model ABI requires IEEE-754 binary32 float");

static uint16_t rd16(const uint8_t *p) {
  return (uint16_t)((uint16_t)p[0] | (uint16_t)((uint16_t)p[1] << 8u));
}

static uint32_t rd32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8u) |
         ((uint32_t)p[2] << 16u) | ((uint32_t)p[3] << 24u);
}

static uint64_t rd64(const uint8_t *p) {
  return (uint64_t)rd32(p) | ((uint64_t)rd32(p + 4u) << 32u);
}

static float rdf32(const uint8_t *p) {
  uint32_t u = rd32(p);
  float f = 0.0f;
  memcpy(&f, &u, sizeof(f));
  return f;
}

static size_t align4(size_t value) {
  return (value + 3u) & ~(size_t)3u;
}

static int float_array_finite(const uint8_t *p, uint32_t offset, uint16_t count) {
  for (uint16_t i = 0u; i < count; ++i) {
    if (!isfinite(rdf32(p + offset + (size_t)i * sizeof(float)))) {
      return 0;
    }
  }
  return 1;
}

kws_status_t kws_model_open(const void *blob,
                            size_t blob_bytes,
                            kws_model_t *out_model) {
  const uint8_t *p = (const uint8_t *)blob;
  uint32_t wx_off;
  uint32_t wh_off;
  uint32_t bh_off;
  uint32_t wo_off;
  uint32_t bo_off;
  uint32_t total_bytes;
  size_t wx_bytes;
  size_t wh_bytes;
  size_t bh_bytes;
  size_t wo_bytes;
  size_t bo_bytes;
  size_t expected_wx;
  size_t expected_wh;
  size_t expected_bh;
  size_t expected_wo;
  size_t expected_bo;
  size_t expected_total;

  if (p == NULL || out_model == NULL || blob_bytes < KWS_HEADER_BYTES) {
    return KWS_EINVAL;
  }
  if (memcmp(p, "KWSP", 4u) != 0 ||
      rd16(p + 4u) != KWS_MODEL_VERSION ||
      rd16(p + 6u) != KWS_HEADER_BYTES) {
    return KWS_EFORMAT;
  }

  memset(out_model, 0, sizeof(*out_model));
  out_model->feature_dim = rd16(p + 8u);
  out_model->hidden_dim = rd16(p + 10u);
  out_model->vocab_size = rd16(p + 12u);
  out_model->frontend_kind = rd16(p + 14u);
  out_model->sample_rate_hz = rd32(p + 16u);
  out_model->frame_length_samples = rd32(p + 20u);
  out_model->frame_hop_samples = rd32(p + 24u);
  out_model->wx_scale = rdf32(p + 28u);
  out_model->wh_scale = rdf32(p + 32u);
  out_model->wo_scale = rdf32(p + 36u);
  out_model->vocab_fingerprint = rd64(p + 40u);
  wx_off = rd32(p + 48u);
  wh_off = rd32(p + 52u);
  bh_off = rd32(p + 56u);
  wo_off = rd32(p + 60u);
  bo_off = rd32(p + 64u);
  total_bytes = rd32(p + 68u);

  if (total_bytes != blob_bytes ||
      out_model->sample_rate_hz != KWS_SAMPLE_RATE_HZ ||
      out_model->frame_length_samples != KWS_FRAME_LENGTH_SAMPLES ||
      out_model->frame_hop_samples != KWS_FRAME_HOP_SAMPLES ||
      (out_model->frontend_kind != KWS_FRONTEND_LOGMEL &&
       out_model->frontend_kind != KWS_FRONTEND_PCEN_LITE) ||
      out_model->feature_dim == 0u ||
      out_model->feature_dim > KWS_MAX_FEATURE_DIM ||
      out_model->hidden_dim == 0u ||
      out_model->hidden_dim > KWS_MAX_HIDDEN_DIM ||
      out_model->vocab_size < 2u ||
      out_model->vocab_size > KWS_MAX_VOCAB_SIZE ||
      out_model->vocab_fingerprint == 0u ||
      !isfinite(out_model->wx_scale) || !isfinite(out_model->wh_scale) ||
      !isfinite(out_model->wo_scale) || out_model->wx_scale <= 0.0f ||
      out_model->wh_scale <= 0.0f || out_model->wo_scale <= 0.0f) {
    return KWS_EFORMAT;
  }

  wx_bytes = (size_t)out_model->hidden_dim * (size_t)out_model->feature_dim;
  wh_bytes = (size_t)out_model->hidden_dim * (size_t)out_model->hidden_dim;
  bh_bytes = (size_t)out_model->hidden_dim * sizeof(float);
  wo_bytes = (size_t)out_model->vocab_size * (size_t)out_model->hidden_dim;
  bo_bytes = (size_t)out_model->vocab_size * sizeof(float);

  expected_wx = align4(KWS_HEADER_BYTES);
  expected_wh = align4(expected_wx + wx_bytes);
  expected_bh = align4(expected_wh + wh_bytes);
  expected_wo = align4(expected_bh + bh_bytes);
  expected_bo = align4(expected_wo + wo_bytes);
  expected_total = expected_bo + bo_bytes;

  if ((size_t)wx_off != expected_wx || (size_t)wh_off != expected_wh ||
      (size_t)bh_off != expected_bh || (size_t)wo_off != expected_wo ||
      (size_t)bo_off != expected_bo || (size_t)total_bytes != expected_total ||
      (((uintptr_t)(p + bh_off)) % _Alignof(float)) != 0u ||
      (((uintptr_t)(p + bo_off)) % _Alignof(float)) != 0u ||
      float_array_finite(p, bh_off, out_model->hidden_dim) == 0 ||
      float_array_finite(p, bo_off, out_model->vocab_size) == 0) {
    return KWS_EFORMAT;
  }

  out_model->wx = (const int8_t *)(const void *)(p + wx_off);
  out_model->wh = (const int8_t *)(const void *)(p + wh_off);
  out_model->bh = (const float *)(const void *)(p + bh_off);
  out_model->wo = (const int8_t *)(const void *)(p + wo_off);
  out_model->bo = (const float *)(const void *)(p + bo_off);
  return KWS_OK;
}
