#include "frontend.h"

#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define PCEN_SMOOTHING 0.025f
#define PCEN_ALPHA 0.98f
#define PCEN_DELTA 2.0f
#define PCEN_EPSILON 1.0e-6f
#define PCEN_DELTA_ROOT 1.4142135623730950488f

static float hz_to_mel(float hz) {
  return 2595.0f * log10f(1.0f + hz / 700.0f);
}

static float mel_to_hz(float mel) {
  return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

static void fft512(kws_frontend_t *fe) {
  unsigned i;
  unsigned j = 0u;
  unsigned stage = 0u;

  for (i = 1u; i < 512u; ++i) {
    unsigned bit = 256u;
    while ((j & bit) != 0u) {
      j ^= bit;
      bit >>= 1u;
    }
    j ^= bit;
    if (i < j) {
      float tr = fe->re[i];
      float ti = fe->im[i];
      fe->re[i] = fe->re[j];
      fe->im[i] = fe->im[j];
      fe->re[j] = tr;
      fe->im[j] = ti;
    }
  }

  for (unsigned len = 2u; len <= 512u; len <<= 1u) {
    float wlen_r = fe->fft_wlen_re[stage];
    float wlen_i = fe->fft_wlen_im[stage];
    for (i = 0u; i < 512u; i += len) {
      float wr = 1.0f;
      float wi = 0.0f;
      for (unsigned k = 0u; k < len / 2u; ++k) {
        unsigned a = i + k;
        unsigned b = a + len / 2u;
        float vr = fe->re[b] * wr - fe->im[b] * wi;
        float vi = fe->re[b] * wi + fe->im[b] * wr;
        float ur = fe->re[a];
        float ui = fe->im[a];
        fe->re[a] = ur + vr;
        fe->im[a] = ui + vi;
        fe->re[b] = ur - vr;
        fe->im[b] = ui - vi;
        {
          float nwr = wr * wlen_r - wi * wlen_i;
          wi = wr * wlen_i + wi * wlen_r;
          wr = nwr;
        }
      }
    }
    stage++;
  }
}

void kws_frontend_init(kws_frontend_t *fe, const kws_model_t *model) {
  float mel_lo = hz_to_mel(80.0f);
  float mel_hi = hz_to_mel(7600.0f);
  unsigned stage = 0u;

  memset(fe, 0, sizeof(*fe));
  fe->frame_len = model->frame_length_samples;
  fe->hop = model->frame_hop_samples;
  fe->feature_dim = model->feature_dim;
  fe->frontend_kind = model->frontend_kind;

  for (uint32_t i = 0u; i < fe->frame_len; ++i) {
    fe->window[i] = 0.5f - 0.5f * cosf((2.0f * (float)M_PI * (float)i) /
                                      (float)(fe->frame_len - 1u));
  }

  for (unsigned len = 2u; len <= 512u; len <<= 1u) {
    float angle = -2.0f * (float)M_PI / (float)len;
    fe->fft_wlen_re[stage] = cosf(angle);
    fe->fft_wlen_im[stage] = sinf(angle);
    stage++;
  }

  for (uint16_t m = 0u; m < (uint16_t)(fe->feature_dim + 2u); ++m) {
    float t = (float)m / (float)(fe->feature_dim + 1u);
    float hz = mel_to_hz(mel_lo + t * (mel_hi - mel_lo));
    uint32_t bin = (uint32_t)floorf((513.0f * hz) / 16000.0f);
    if (bin > 256u) {
      bin = 256u;
    }
    fe->mel_bins[m] = (uint16_t)bin;
  }
}

void kws_frontend_reset(kws_frontend_t *fe) {
  fe->fill = 0u;
  fe->last_dbfs = -120.0f;
  fe->pcen_initialized = 0u;
  memset(fe->pcm, 0, sizeof(fe->pcm));
  memset(fe->pcen_smooth, 0, sizeof(fe->pcen_smooth));
}

static float transform_energy(kws_frontend_t *fe, uint16_t channel, float energy) {
  if (fe->frontend_kind == KWS_FRONTEND_PCEN_LITE) {
    float smooth;
    float normalized;
    if (fe->pcen_initialized == 0u) {
      fe->pcen_smooth[channel] = energy;
    } else {
      fe->pcen_smooth[channel] =
          (1.0f - PCEN_SMOOTHING) * fe->pcen_smooth[channel] +
          PCEN_SMOOTHING * energy;
    }
    smooth = fe->pcen_smooth[channel];
    normalized = energy / powf(PCEN_EPSILON + smooth, PCEN_ALPHA);
    return sqrtf(normalized + PCEN_DELTA) - PCEN_DELTA_ROOT;
  }
  return log1pf(32.0f * energy);
}

static void make_features(kws_frontend_t *fe, float *out) {
  double sumsq = 0.0;
  float mean = 0.0f;

  for (uint32_t i = 0u; i < 512u; ++i) {
    fe->re[i] = 0.0f;
    fe->im[i] = 0.0f;
  }

  for (uint32_t i = 0u; i < fe->frame_len; ++i) {
    float x = (float)fe->pcm[i] / 32768.0f;
    sumsq += (double)x * (double)x;
    fe->re[i] = x * fe->window[i];
  }

  fe->last_dbfs = 10.0f * log10f((float)(sumsq / (double)fe->frame_len) + 1.0e-12f);
  fft512(fe);

  for (uint16_t m = 0u; m < fe->feature_dim; ++m) {
    uint16_t l = fe->mel_bins[m];
    uint16_t c = fe->mel_bins[m + 1u];
    uint16_t r = fe->mel_bins[m + 2u];
    float e = 0.0f;

    if (c <= l) {
      c = (uint16_t)(l + 1u);
    }
    if (r <= c) {
      r = (uint16_t)(c + 1u);
    }
    if (r > 257u) {
      r = 257u;
    }

    for (uint16_t k = l; k < c && k < 257u; ++k) {
      float w = (float)(k - l) / (float)(c - l);
      float p = fe->re[k] * fe->re[k] + fe->im[k] * fe->im[k];
      e += w * p;
    }
    for (uint16_t k = c; k < r; ++k) {
      float w = (float)(r - k) / (float)(r - c);
      float p = fe->re[k] * fe->re[k] + fe->im[k] * fe->im[k];
      e += w * p;
    }

    out[m] = transform_energy(fe, m, e);
    mean += out[m];
  }
  if (fe->frontend_kind == KWS_FRONTEND_PCEN_LITE) {
    fe->pcen_initialized = 1u;
  }

  mean /= (float)fe->feature_dim;
  for (uint16_t m = 0u; m < fe->feature_dim; ++m) {
    out[m] = (out[m] - mean) * 0.25f;
  }
}

int kws_frontend_push(kws_frontend_t *fe, int16_t sample, float *out_features) {
  fe->pcm[fe->fill++] = sample;
  if (fe->fill < fe->frame_len) {
    return 0;
  }

  make_features(fe, out_features);
  {
    uint32_t keep = fe->frame_len - fe->hop;
    if (keep > 0u) {
      memmove(fe->pcm, fe->pcm + fe->hop,
              (size_t)keep * sizeof(fe->pcm[0]));
    }
    fe->fill = keep;
  }
  return 1;
}

float kws_frontend_last_dbfs(const kws_frontend_t *fe) {
  return fe->last_dbfs;
}
