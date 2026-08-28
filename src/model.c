#include "model.h"

#include <stdint.h>
#include <string.h>

#define KWS_HEADER_BYTES 64u

static uint16_t rd16(const uint8_t *p) {
  return (uint16_t)((uint16_t)p[0] | (uint16_t)((uint16_t)p[1] << 8u));
}

static uint32_t rd32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8u) |
         ((uint32_t)p[2] << 16u) | ((uint32_t)p[3] << 24u);
}

static float rdf32(const uint8_t *p) {
  uint32_t u = rd32(p);
  float f = 0.0f;
  memcpy(&f, &u, sizeof(f));
  return f;
}

static int range_ok(uint32_t off, size_t bytes, size_t total) {
  return (size_t)off <= total && bytes <= total - (size_t)off;
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
  out_model->sample_rate_hz = rd32(p + 16u);
  out_model->frame_length_samples = rd32(p + 20u);
  out_model->frame_hop_samples = rd32(p + 24u);
  out_model->wx_scale = rdf32(p + 28u);
  out_model->wh_scale = rdf32(p + 32u);
  out_model->wo_scale = rdf32(p + 36u);
  wx_off = rd32(p + 40u);
  wh_off = rd32(p + 44u);
  bh_off = rd32(p + 48u);
  wo_off = rd32(p + 52u);
  bo_off = rd32(p + 56u);
  total_bytes = rd32(p + 60u);

  if (total_bytes != blob_bytes ||
      out_model->sample_rate_hz != KWS_SAMPLE_RATE_HZ ||
      out_model->feature_dim == 0u ||
      out_model->feature_dim > KWS_MAX_FEATURE_DIM ||
      out_model->hidden_dim == 0u ||
      out_model->hidden_dim > KWS_MAX_HIDDEN_DIM ||
      out_model->vocab_size < 2u ||
      out_model->vocab_size > KWS_MAX_VOCAB_SIZE ||
      out_model->frame_length_samples < 2u ||
      out_model->frame_length_samples > 512u ||
      out_model->frame_hop_samples == 0u ||
      out_model->frame_hop_samples > out_model->frame_length_samples ||
      out_model->wx_scale <= 0.0f || out_model->wh_scale <= 0.0f ||
      out_model->wo_scale <= 0.0f) {
    return KWS_EFORMAT;
  }

  wx_bytes = (size_t)out_model->hidden_dim * (size_t)out_model->feature_dim;
  wh_bytes = (size_t)out_model->hidden_dim * (size_t)out_model->hidden_dim;
  bh_bytes = (size_t)out_model->hidden_dim * sizeof(float);
  wo_bytes = (size_t)out_model->vocab_size * (size_t)out_model->hidden_dim;
  bo_bytes = (size_t)out_model->vocab_size * sizeof(float);

  if (!range_ok(wx_off, wx_bytes, blob_bytes) ||
      !range_ok(wh_off, wh_bytes, blob_bytes) ||
      !range_ok(bh_off, bh_bytes, blob_bytes) ||
      !range_ok(wo_off, wo_bytes, blob_bytes) ||
      !range_ok(bo_off, bo_bytes, blob_bytes) ||
      (bh_off & 3u) != 0u || (bo_off & 3u) != 0u ||
      (((uintptr_t)(p + bh_off)) & 3u) != 0u ||
      (((uintptr_t)(p + bo_off)) & 3u) != 0u) {
    return KWS_EFORMAT;
  }

  out_model->wx = (const int8_t *)(const void *)(p + wx_off);
  out_model->wh = (const int8_t *)(const void *)(p + wh_off);
  out_model->bh = (const float *)(const void *)(p + bh_off);
  out_model->wo = (const int8_t *)(const void *)(p + wo_off);
  out_model->bo = (const float *)(const void *)(p + bo_off);
  return KWS_OK;
}
