#include "kws_pipeline/kws.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

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
};

kws_config_t kws_default_config(void) {
  kws_config_t c = {-55.0f, 1.5f, 0.94f, 1200u};
  return c;
}

size_t kws_engine_required_bytes(const kws_model_t *model) {
  return model == NULL ? 0u : sizeof(kws_engine_t);
}

kws_status_t kws_engine_init(void *arena,
                             size_t arena_bytes,
                             const kws_model_t *model,
                             const kws_config_t *config,
                             kws_engine_t **out_engine) {
  kws_config_t c;
  kws_engine_t *e;

  if (arena == NULL || model == NULL || out_engine == NULL ||
      arena_bytes < sizeof(kws_engine_t)) {
    return KWS_EINVAL;
  }
  if ((((uintptr_t)arena) & (sizeof(void *) - 1u)) != 0u) {
    return KWS_EINVAL;
  }

  c = config != NULL ? *config : kws_default_config();
  if (c.state_retention <= 0.0f || c.state_retention >= 1.0f ||
      c.token_boost < 0.0f || c.refractory_ms > 10000u) {
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
                                     size_t keyword_count) {
  if (engine == NULL) {
    return KWS_EINVAL;
  }
  return kws_decoder_set_keywords(&engine->decoder, keywords, keyword_count,
                                  engine->model.vocab_size);
}

kws_status_t kws_engine_set_keyword_pack(kws_engine_t *engine,
                                         const kws_keyword_pack_t *pack) {
  if (engine == NULL || pack == NULL) {
    return KWS_EINVAL;
  }
  return kws_engine_set_keywords(engine, pack->keywords, pack->keyword_count);
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

static void infer_frame(kws_engine_t *e) {
  const kws_model_t *m = &e->model;

  for (uint16_t h = 0u; h < m->hidden_dim; ++h) {
    float acc = m->bh[h];
    float in_sum = 0.0f;
    float rec_sum = 0.0f;
    size_t wx_base = (size_t)h * (size_t)m->feature_dim;
    size_t wh_base = (size_t)h * (size_t)m->hidden_dim;

    for (uint16_t i = 0u; i < m->feature_dim; ++i) {
      in_sum += (float)m->wx[wx_base + i] * e->features[i];
    }
    for (uint16_t i = 0u; i < m->hidden_dim; ++i) {
      rec_sum += (float)m->wh[wh_base + i] * e->hidden[i];
    }
    e->next_hidden[h] =
        tanhf(acc + m->wx_scale * in_sum + m->wh_scale * rec_sum);
  }

  memcpy(e->hidden, e->next_hidden,
         (size_t)m->hidden_dim * sizeof(float));

  for (uint16_t v = 0u; v < m->vocab_size; ++v) {
    float sum = 0.0f;
    size_t base = (size_t)v * (size_t)m->hidden_dim;
    for (uint16_t h = 0u; h < m->hidden_dim; ++h) {
      sum += (float)m->wo[base + h] * e->hidden[h];
    }
    e->logits[v] = m->bo[v] + m->wo_scale * sum;
  }
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

      infer_frame(engine);
      decoder_hit = kws_decoder_step(&engine->decoder, engine->logits,
                                     engine->model.vocab_size, speech_active,
                                     &keyword_id, &confidence);

      if (decoder_hit != 0 &&
          engine->processed_samples >= engine->suppress_until_sample) {
        uint64_t refractory_samples =
            ((uint64_t)engine->config.refractory_ms *
             (uint64_t)KWS_SAMPLE_RATE_HZ) /
            1000u;
        engine->suppress_until_sample =
            engine->processed_samples + refractory_samples;

        if (*out_detected == 0) {
          *out_detected = 1;
          if (out_detection != NULL) {
            out_detection->keyword_id = keyword_id;
            out_detection->confidence = confidence;
            out_detection->end_sample = engine->processed_samples;
          }
        }
      }
    }
  }

  return KWS_OK;
}

uint64_t kws_engine_processed_samples(const kws_engine_t *engine) {
  return engine != NULL ? engine->processed_samples : 0u;
}
