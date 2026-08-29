#include "kws_pipeline/kws.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

#include "decoder.h"
#include "frontend.h"

struct kws_engine {
  kws_model_t model;
  kws_config_t config;
  kws_frontend_t frontend;
  kws_decoder_t decoder;
  float hidden[KWS_MAX_HIDDEN_DIM];
  float next_hidden[KWS_MAX_HIDDEN_DIM];
  float features[KWS_MAX_FEATURE_DIM];
  float logits[KWS_MAX_VOCAB_SIZE];
  uint64_t processed_samples;
  uint64_t suppress_until_sample;
  uint64_t processed_frames;
  uint64_t speech_frames;
  uint64_t blank_top1_frames;
  uint64_t decoder_hits;
  uint64_t refractory_suppressed;
  uint64_t detections;
  float max_detection_confidence;
};

static int model_contract_valid(const kws_model_t *model) {
  if (model == NULL || model->sample_rate_hz != KWS_SAMPLE_RATE_HZ ||
      model->frame_length_samples != KWS_FRAME_LENGTH_SAMPLES ||
      model->frame_hop_samples != KWS_FRAME_HOP_SAMPLES ||
      model->feature_dim == 0u || model->feature_dim > KWS_MAX_FEATURE_DIM ||
      model->hidden_dim == 0u || model->hidden_dim > KWS_MAX_HIDDEN_DIM ||
      model->vocab_size < 2u || model->vocab_size > KWS_MAX_VOCAB_SIZE ||
      model->vocab_fingerprint == 0u || !isfinite(model->wx_scale) ||
      !isfinite(model->wh_scale) || !isfinite(model->wo_scale) ||
      model->wx_scale <= 0.0f || model->wh_scale <= 0.0f ||
      model->wo_scale <= 0.0f || model->wx == NULL || model->wh == NULL ||
      model->bh == NULL || model->wo == NULL || model->bo == NULL) {
    return 0;
  }

  if (((uintptr_t)model->bh % _Alignof(float)) != 0u ||
      ((uintptr_t)model->bo % _Alignof(float)) != 0u) {
    return 0;
  }

  for (uint16_t h = 0u; h < model->hidden_dim; ++h) {
    if (!isfinite(model->bh[h])) {
      return 0;
    }
  }
  for (uint16_t v = 0u; v < model->vocab_size; ++v) {
    if (!isfinite(model->bo[v])) {
      return 0;
    }
  }
  return 1;
}

kws_config_t kws_default_config(void) {
  kws_config_t c = {-55.0f, 1.5f, 0.94f, 1200u};
  return c;
}

size_t kws_engine_required_bytes(const kws_model_t *model) {
  return model_contract_valid(model) != 0 ? sizeof(kws_engine_t) : 0u;
}

size_t kws_engine_required_alignment(void) {
  return _Alignof(kws_engine_t);
}

kws_status_t kws_engine_init(void *arena,
                             size_t arena_bytes,
                             const kws_model_t *model,
                             const kws_config_t *config,
                             kws_engine_t **out_engine) {
  kws_config_t c;
  kws_engine_t *e;

  if (arena == NULL || out_engine == NULL || model_contract_valid(model) == 0 ||
      arena_bytes < sizeof(kws_engine_t)) {
    return KWS_EINVAL;
  }
  if (((uintptr_t)arena % _Alignof(kws_engine_t)) != 0u) {
    return KWS_EINVAL;
  }

  c = config != NULL ? *config : kws_default_config();
  if (!isfinite(c.min_speech_dbfs) || !isfinite(c.token_boost) ||
      !isfinite(c.state_retention) || c.state_retention <= 0.0f ||
      c.state_retention >= 1.0f || c.token_boost < 0.0f ||
      c.refractory_ms > 10000u) {
    return KWS_EINVAL;
  }

  e = (kws_engine_t *)arena;
  memset(e, 0, sizeof(*e));
  e->model = *model;
  e->config = c;
  kws_frontend_init(&e->frontend, model);
  kws_decoder_init(&e->decoder, c.token_boost, c.state_retention);
  *out_engine = e;
  return KWS_OK;
}

kws_status_t kws_engine_set_keywords(kws_engine_t *engine,
                                     const kws_keyword_t *keywords,
                                     size_t keyword_count,
                                     uint64_t vocab_fingerprint) {
  if (engine == NULL) {
    return KWS_EINVAL;
  }
  if (vocab_fingerprint == 0u ||
      vocab_fingerprint != engine->model.vocab_fingerprint) {
    return KWS_EFORMAT;
  }
  return kws_decoder_set_keywords(&engine->decoder, keywords, keyword_count,
                                  engine->model.vocab_size);
}

kws_status_t kws_engine_set_keyword_pack(kws_engine_t *engine,
                                         const kws_keyword_pack_t *pack) {
  if (engine == NULL || pack == NULL) {
    return KWS_EINVAL;
  }
  return kws_engine_set_keywords(engine, pack->keywords, pack->keyword_count,
                                 pack->vocab_fingerprint);
}

void kws_engine_reset(kws_engine_t *engine) {
  if (engine == NULL) {
    return;
  }
  memset(engine->hidden, 0, sizeof(engine->hidden));
  memset(engine->next_hidden, 0, sizeof(engine->next_hidden));
  kws_frontend_reset(&engine->frontend);
  kws_decoder_reset(&engine->decoder);
  engine->suppress_until_sample = 0u;
}

static float dot_i8_f32(const int8_t *weights,
                        const float *values,
                        size_t count) {
  size_t i = 0u;
  float sum = 0.0f;

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  float32x4_t acc = vdupq_n_f32(0.0f);
  for (; i + 8u <= count; i += 8u) {
    int8x8_t w8 = vld1_s8(weights + i);
    int16x8_t w16 = vmovl_s8(w8);
    int32x4_t w32_lo = vmovl_s16(vget_low_s16(w16));
    int32x4_t w32_hi = vmovl_s16(vget_high_s16(w16));
    float32x4_t wf_lo = vcvtq_f32_s32(w32_lo);
    float32x4_t wf_hi = vcvtq_f32_s32(w32_hi);
    float32x4_t x_lo = vld1q_f32(values + i);
    float32x4_t x_hi = vld1q_f32(values + i + 4u);
    acc = vmlaq_f32(acc, wf_lo, x_lo);
    acc = vmlaq_f32(acc, wf_hi, x_hi);
  }
  {
    float32x2_t pair = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
    pair = vpadd_f32(pair, pair);
    sum = vget_lane_f32(pair, 0);
  }
#endif

  for (; i < count; ++i) {
    sum += (float)weights[i] * values[i];
  }
  return sum;
}

static uint16_t infer_frame(kws_engine_t *e) {
  const kws_model_t *m = &e->model;
  uint16_t top_index = 0u;

  for (uint16_t h = 0u; h < m->hidden_dim; ++h) {
    float acc = m->bh[h];
    size_t wx_base = (size_t)h * (size_t)m->feature_dim;
    size_t wh_base = (size_t)h * (size_t)m->hidden_dim;
    float in_sum = dot_i8_f32(m->wx + wx_base, e->features,
                              (size_t)m->feature_dim);
    float rec_sum = dot_i8_f32(m->wh + wh_base, e->hidden,
                               (size_t)m->hidden_dim);
    e->next_hidden[h] =
        tanhf(acc + m->wx_scale * in_sum + m->wh_scale * rec_sum);
  }

  memcpy(e->hidden, e->next_hidden,
         (size_t)m->hidden_dim * sizeof(float));

  for (uint16_t v = 0u; v < m->vocab_size; ++v) {
    size_t base = (size_t)v * (size_t)m->hidden_dim;
    float sum = dot_i8_f32(m->wo + base, e->hidden, (size_t)m->hidden_dim);
    e->logits[v] = m->bo[v] + m->wo_scale * sum;
    if (v == 0u || e->logits[v] > e->logits[top_index]) {
      top_index = v;
    }
  }
  return top_index;
}

kws_status_t kws_engine_accept_pcm16(kws_engine_t *engine,
                                     const int16_t *samples,
                                     size_t sample_count,
                                     kws_detection_t *out_detection,
                                     int *out_detected) {
  if (engine == NULL || (samples == NULL && sample_count != 0u) ||
      out_detected == NULL) {
    return KWS_EINVAL;
  }

  *out_detected = 0;
  if (sample_count > KWS_MAX_PCM_BLOCK_SAMPLES) {
    return KWS_EBOUNDS;
  }

  for (size_t i = 0u; i < sample_count; ++i) {
    engine->processed_samples++;
    if (kws_frontend_push(&engine->frontend, samples[i],
                          engine->features) != 0) {
      uint32_t keyword_id = 0u;
      float confidence = 0.0f;
      int speech_active =
          kws_frontend_last_dbfs(&engine->frontend) >=
          engine->config.min_speech_dbfs;
      int decoder_hit;
      uint16_t top_index;

      engine->processed_frames++;
      if (speech_active != 0) {
        engine->speech_frames++;
      }
      top_index = infer_frame(engine);
      if (top_index == 0u) {
        engine->blank_top1_frames++;
      }
      decoder_hit = kws_decoder_step(&engine->decoder, engine->logits,
                                     engine->model.vocab_size, speech_active,
                                     &keyword_id, &confidence);

      if (decoder_hit != 0) {
        engine->decoder_hits++;
        if (engine->processed_samples >= engine->suppress_until_sample) {
          uint64_t refractory_samples =
              ((uint64_t)engine->config.refractory_ms *
               (uint64_t)KWS_SAMPLE_RATE_HZ) /
              1000u;
          engine->suppress_until_sample =
              engine->processed_samples + refractory_samples;
          engine->detections++;
          if (confidence > engine->max_detection_confidence) {
            engine->max_detection_confidence = confidence;
          }

          *out_detected = 1;
          if (out_detection != NULL) {
            out_detection->keyword_id = keyword_id;
            out_detection->confidence = confidence;
            out_detection->end_sample = engine->processed_samples;
          }
        } else {
          engine->refractory_suppressed++;
        }
      }
    }
  }

  return KWS_OK;
}

uint64_t kws_engine_processed_samples(const kws_engine_t *engine) {
  return engine != NULL ? engine->processed_samples : 0u;
}

kws_status_t kws_engine_get_stats(const kws_engine_t *engine,
                                  kws_engine_stats_t *out_stats) {
  if (engine == NULL || out_stats == NULL) {
    return KWS_EINVAL;
  }

  out_stats->processed_samples = engine->processed_samples;
  out_stats->processed_frames = engine->processed_frames;
  out_stats->speech_frames = engine->speech_frames;
  out_stats->blank_top1_frames = engine->blank_top1_frames;
  out_stats->decoder_hits = engine->decoder_hits;
  out_stats->refractory_suppressed = engine->refractory_suppressed;
  out_stats->detections = engine->detections;
  out_stats->keyword_count = engine->decoder.keyword_count;
  out_stats->trie_nodes = engine->decoder.node_count;
  out_stats->pending_keyword_index = engine->decoder.pending_keyword;
  out_stats->pending_age_frames = engine->decoder.pending_age_frames;
  out_stats->max_detection_confidence = engine->max_detection_confidence;
  return KWS_OK;
}
