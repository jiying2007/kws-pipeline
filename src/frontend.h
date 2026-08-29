#ifndef KWS_PIPELINE_FRONTEND_H
#define KWS_PIPELINE_FRONTEND_H

#include <stdint.h>

#include "kws_pipeline/kws.h"

#define KWS_FFT_STAGES 9u

typedef struct kws_frontend {
  uint32_t frame_len;
  uint32_t hop;
  uint32_t fill;
  uint16_t feature_dim;
  uint16_t frontend_kind;
  int16_t pcm[512];
  float window[512];
  float re[512];
  float im[512];
  float fft_wlen_re[KWS_FFT_STAGES];
  float fft_wlen_im[KWS_FFT_STAGES];
  uint16_t mel_bins[KWS_MAX_FEATURE_DIM + 2u];
  float pcen_smooth[KWS_MAX_FEATURE_DIM];
  uint8_t pcen_initialized;
  float last_dbfs;
} kws_frontend_t;

void kws_frontend_init(kws_frontend_t *fe, const kws_model_t *model);
void kws_frontend_reset(kws_frontend_t *fe);
int kws_frontend_push(kws_frontend_t *fe, int16_t sample, float *out_features);
float kws_frontend_last_dbfs(const kws_frontend_t *fe);

#endif
