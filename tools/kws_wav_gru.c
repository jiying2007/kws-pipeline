#include "kws_pipeline/kws.h"
#include "decoder.h"
#include "frontend.h"
#include "tool_io.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GRU_MAGIC "KWG1"
#define GRU_FORMAT_VERSION 1u
#define GRU_HEADER_BYTES 80u
#define BLOCK_SAMPLES 160u

typedef struct gru_model {
  uint16_t feature_dim;
  uint16_t hidden_dim;
  uint16_t vocab_size;
  uint16_t frontend_kind;
  uint32_t sample_rate_hz;
  uint32_t frame_length_samples;
  uint32_t frame_hop_samples;
  float input_scale;
  float recurrent_scale;
  float output_scale;
  uint64_t vocab_fingerprint;
  const int8_t *weight_ih;
  const int8_t *weight_hh;
  const float *bias_ih;
  const float *bias_hh;
  const int8_t *weight_out;
  const float *bias_out;
} gru_model_t;

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

static int float_array_finite(const uint8_t *p, uint32_t offset, size_t count) {
  for (size_t i = 0u; i < count; ++i) {
    if (!isfinite(rdf32(p + offset + i * sizeof(float)))) {
      return 0;
    }
  }
  return 1;
}

static int gru_model_open(const void *blob, size_t blob_bytes, gru_model_t *out) {
  const uint8_t *p = (const uint8_t *)blob;
  uint32_t wih_off;
  uint32_t whh_off;
  uint32_t bih_off;
  uint32_t bhh_off;
  uint32_t wo_off;
  uint32_t bo_off;
  uint32_t total_bytes;
  size_t wih_bytes;
  size_t whh_bytes;
  size_t bias_bytes;
  size_t wo_bytes;
  size_t bo_bytes;
  size_t expected_wih;
  size_t expected_whh;
  size_t expected_bih;
  size_t expected_bhh;
  size_t expected_wo;
  size_t expected_bo;
  size_t expected_total;

  if (p == NULL || out == NULL || blob_bytes < GRU_HEADER_BYTES) {
    return 0;
  }
  if (memcmp(p, GRU_MAGIC, 4u) != 0 || rd16(p + 4u) != GRU_FORMAT_VERSION ||
      rd16(p + 6u) != GRU_HEADER_BYTES || rd32(p + 48u) != 0u) {
    return 0;
  }
  memset(out, 0, sizeof(*out));
  out->feature_dim = rd16(p + 8u);
  out->hidden_dim = rd16(p + 10u);
  out->vocab_size = rd16(p + 12u);
  out->frontend_kind = rd16(p + 14u);
  out->sample_rate_hz = rd32(p + 16u);
  out->frame_length_samples = rd32(p + 20u);
  out->frame_hop_samples = rd32(p + 24u);
  out->input_scale = rdf32(p + 28u);
  out->recurrent_scale = rdf32(p + 32u);
  out->output_scale = rdf32(p + 36u);
  out->vocab_fingerprint = rd64(p + 40u);
  wih_off = rd32(p + 52u);
  whh_off = rd32(p + 56u);
  bih_off = rd32(p + 60u);
  bhh_off = rd32(p + 64u);
  wo_off = rd32(p + 68u);
  bo_off = rd32(p + 72u);
  total_bytes = rd32(p + 76u);

  if (total_bytes != blob_bytes || out->sample_rate_hz != KWS_SAMPLE_RATE_HZ ||
      out->frame_length_samples != KWS_FRAME_LENGTH_SAMPLES ||
      out->frame_hop_samples != KWS_FRAME_HOP_SAMPLES ||
      (out->frontend_kind != KWS_FRONTEND_LOGMEL &&
       out->frontend_kind != KWS_FRONTEND_PCEN_LITE) ||
      out->feature_dim == 0u || out->feature_dim > KWS_MAX_FEATURE_DIM ||
      out->hidden_dim == 0u || out->hidden_dim > KWS_MAX_HIDDEN_DIM ||
      out->vocab_size < 2u || out->vocab_size > KWS_MAX_VOCAB_SIZE ||
      out->vocab_fingerprint == 0u || !isfinite(out->input_scale) ||
      !isfinite(out->recurrent_scale) || !isfinite(out->output_scale) ||
      out->input_scale <= 0.0f || out->recurrent_scale <= 0.0f ||
      out->output_scale <= 0.0f) {
    return 0;
  }

  wih_bytes = (size_t)3u * (size_t)out->hidden_dim * (size_t)out->feature_dim;
  whh_bytes = (size_t)3u * (size_t)out->hidden_dim * (size_t)out->hidden_dim;
  bias_bytes = (size_t)3u * (size_t)out->hidden_dim * sizeof(float);
  wo_bytes = (size_t)out->vocab_size * (size_t)out->hidden_dim;
  bo_bytes = (size_t)out->vocab_size * sizeof(float);
  expected_wih = align4(GRU_HEADER_BYTES);
  expected_whh = align4(expected_wih + wih_bytes);
  expected_bih = align4(expected_whh + whh_bytes);
  expected_bhh = align4(expected_bih + bias_bytes);
  expected_wo = align4(expected_bhh + bias_bytes);
  expected_bo = align4(expected_wo + wo_bytes);
  expected_total = expected_bo + bo_bytes;

  if ((size_t)wih_off != expected_wih || (size_t)whh_off != expected_whh ||
      (size_t)bih_off != expected_bih || (size_t)bhh_off != expected_bhh ||
      (size_t)wo_off != expected_wo || (size_t)bo_off != expected_bo ||
      (size_t)total_bytes != expected_total ||
      (((uintptr_t)(p + bih_off)) % _Alignof(float)) != 0u ||
      (((uintptr_t)(p + bhh_off)) % _Alignof(float)) != 0u ||
      (((uintptr_t)(p + bo_off)) % _Alignof(float)) != 0u ||
      float_array_finite(p, bih_off, (size_t)3u * out->hidden_dim) == 0 ||
      float_array_finite(p, bhh_off, (size_t)3u * out->hidden_dim) == 0 ||
      float_array_finite(p, bo_off, out->vocab_size) == 0) {
    return 0;
  }

  out->weight_ih = (const int8_t *)(const void *)(p + wih_off);
  out->weight_hh = (const int8_t *)(const void *)(p + whh_off);
  out->bias_ih = (const float *)(const void *)(p + bih_off);
  out->bias_hh = (const float *)(const void *)(p + bhh_off);
  out->weight_out = (const int8_t *)(const void *)(p + wo_off);
  out->bias_out = (const float *)(const void *)(p + bo_off);
  return 1;
}

static float dot_i8_f32(const int8_t *weights, const float *values, size_t count) {
  float sum = 0.0f;
  for (size_t i = 0u; i < count; ++i) {
    sum += (float)weights[i] * values[i];
  }
  return sum;
}

static float sigmoidf_stable(float value) {
  if (value >= 0.0f) {
    float z = expf(-value);
    return 1.0f / (1.0f + z);
  }
  {
    float z = expf(value);
    return z / (1.0f + z);
  }
}

static void infer_gru(const gru_model_t *m,
                      const float *features,
                      float *hidden,
                      float *next_hidden,
                      float *logits) {
  const size_t hdim = (size_t)m->hidden_dim;
  const size_t fdim = (size_t)m->feature_dim;
  for (uint16_t h = 0u; h < m->hidden_dim; ++h) {
    size_t r = (size_t)h;
    size_t z = hdim + (size_t)h;
    size_t n = 2u * hdim + (size_t)h;
    float r_in = dot_i8_f32(m->weight_ih + r * fdim, features, fdim);
    float r_rec = dot_i8_f32(m->weight_hh + r * hdim, hidden, hdim);
    float z_in = dot_i8_f32(m->weight_ih + z * fdim, features, fdim);
    float z_rec = dot_i8_f32(m->weight_hh + z * hdim, hidden, hdim);
    float n_in = dot_i8_f32(m->weight_ih + n * fdim, features, fdim);
    float n_rec = dot_i8_f32(m->weight_hh + n * hdim, hidden, hdim);
    float reset = sigmoidf_stable(
        m->bias_ih[r] + m->bias_hh[r] + m->input_scale * r_in +
        m->recurrent_scale * r_rec);
    float update = sigmoidf_stable(
        m->bias_ih[z] + m->bias_hh[z] + m->input_scale * z_in +
        m->recurrent_scale * z_rec);
    float candidate = tanhf(
        m->bias_ih[n] + m->input_scale * n_in +
        reset * (m->bias_hh[n] + m->recurrent_scale * n_rec));
    next_hidden[h] = (1.0f - update) * candidate + update * hidden[h];
  }
  memcpy(hidden, next_hidden, hdim * sizeof(float));
  for (uint16_t v = 0u; v < m->vocab_size; ++v) {
    size_t base = (size_t)v * hdim;
    logits[v] = m->bias_out[v] +
                m->output_scale *
                    dot_i8_f32(m->weight_out + base, hidden, hdim);
  }
}

int main(int argc, char **argv) {
  uint8_t *model_blob = NULL;
  uint8_t *pack_blob = NULL;
  size_t model_bytes = 0u;
  size_t pack_bytes = 0u;
  gru_model_t gru;
  kws_model_t frontend_model;
  kws_keyword_pack_t pack;
  kws_frontend_t frontend;
  kws_decoder_t decoder;
  kws_config_t config;
  float hidden[KWS_MAX_HIDDEN_DIM] = {0.0f};
  float next_hidden[KWS_MAX_HIDDEN_DIM] = {0.0f};
  float features[KWS_MAX_FEATURE_DIM] = {0.0f};
  float logits[KWS_MAX_VOCAB_SIZE] = {0.0f};
  FILE *wav = NULL;
  uint32_t wav_bytes = 0u;
  long wav_data_offset = 0L;
  uint32_t remaining;
  int16_t pcm[BLOCK_SAMPLES];
  uint64_t processed_samples = 0u;
  uint64_t suppress_until_sample = 0u;
  int exit_code = 1;

  if (argc != 5) {
    fprintf(stderr, "usage: %s model.kwg keywords.kwk audio.wav recording-id\n", argv[0]);
    return 2;
  }
  if (kws_tool_read_file(argv[1], &model_blob, &model_bytes) == 0 ||
      kws_tool_read_file(argv[2], &pack_blob, &pack_bytes) == 0) {
    fprintf(stderr, "cannot read experimental GRU model or keyword pack\n");
    goto cleanup;
  }
  if (gru_model_open(model_blob, model_bytes, &gru) == 0) {
    fprintf(stderr, "invalid experimental GRU model: %s\n", argv[1]);
    goto cleanup;
  }

  memset(&frontend_model, 0, sizeof(frontend_model));
  frontend_model.feature_dim = gru.feature_dim;
  frontend_model.hidden_dim = gru.hidden_dim;
  frontend_model.vocab_size = gru.vocab_size;
  frontend_model.frontend_kind = gru.frontend_kind;
  frontend_model.sample_rate_hz = gru.sample_rate_hz;
  frontend_model.frame_length_samples = gru.frame_length_samples;
  frontend_model.frame_hop_samples = gru.frame_hop_samples;
  frontend_model.vocab_fingerprint = gru.vocab_fingerprint;
  if (kws_keyword_pack_open(pack_blob, pack_bytes, &frontend_model, &pack) != KWS_OK) {
    fprintf(stderr, "invalid keyword pack for experimental GRU model: %s\n", argv[2]);
    goto cleanup;
  }
  config = kws_default_config();
  kws_frontend_init(&frontend, &frontend_model);
  kws_decoder_init(&decoder, config.token_boost, config.state_retention);
  if (kws_decoder_set_keywords(&decoder, pack.keywords, pack.keyword_count,
                               gru.vocab_size) != KWS_OK) {
    fprintf(stderr, "cannot initialize decoder for experimental GRU model\n");
    goto cleanup;
  }

  wav = fopen(argv[3], "rb");
  if (wav == NULL || kws_tool_open_wav(wav, &wav_bytes, &wav_data_offset) == 0) {
    fprintf(stderr, "expected mono 16-kHz PCM16 WAV: %s\n", argv[3]);
    goto cleanup;
  }
  if (fseek(wav, wav_data_offset, SEEK_SET) != 0) {
    fprintf(stderr, "cannot seek WAV data: %s\n", argv[3]);
    goto cleanup;
  }

  remaining = wav_bytes;
  while (remaining != 0u) {
    size_t want_samples = (size_t)(remaining / 2u);
    size_t got_samples;
    if (want_samples > BLOCK_SAMPLES) {
      want_samples = BLOCK_SAMPLES;
    }
    got_samples = fread(pcm, sizeof(pcm[0]), want_samples, wav);
    if (got_samples != want_samples) {
      fprintf(stderr, "truncated WAV data: %s\n", argv[3]);
      goto cleanup;
    }
    remaining -= (uint32_t)(got_samples * sizeof(pcm[0]));
    for (size_t i = 0u; i < got_samples; ++i) {
      processed_samples++;
      if (kws_frontend_push(&frontend, pcm[i], features) != 0) {
        uint32_t keyword_id = 0u;
        float confidence = 0.0f;
        int speech_active =
            kws_frontend_last_dbfs(&frontend) >= config.min_speech_dbfs;
        int decoder_hit;
        infer_gru(&gru, features, hidden, next_hidden, logits);
        decoder_hit = kws_decoder_step(&decoder, logits, gru.vocab_size,
                                       speech_active, &keyword_id, &confidence);
        if (decoder_hit != 0 && processed_samples >= suppress_until_sample) {
          uint64_t refractory_samples =
              ((uint64_t)config.refractory_ms * (uint64_t)KWS_SAMPLE_RATE_HZ) /
              1000u;
          suppress_until_sample = processed_samples + refractory_samples;
          fputs("{\"recording\":", stdout);
          kws_tool_print_json_string(stdout, argv[4]);
          fprintf(stdout,
                  ",\"keyword_id\":%u,\"time_s\":%.6f,\"confidence\":%.6f}\n",
                  keyword_id,
                  (double)processed_samples / (double)KWS_SAMPLE_RATE_HZ,
                  (double)confidence);
        }
      }
    }
  }
  exit_code = ferror(stdout) != 0 ? 1 : 0;

cleanup:
  if (wav != NULL) {
    fclose(wav);
  }
  free(pack_blob);
  free(model_blob);
  return exit_code;
}
