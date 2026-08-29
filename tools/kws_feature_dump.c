#include "frontend.h"
#include "tool_io.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int parse_feature_dim(const char *text, uint16_t *out_value) {
  char *end = NULL;
  unsigned long value;

  errno = 0;
  value = strtoul(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || value == 0ul ||
      value > (unsigned long)KWS_MAX_FEATURE_DIM) {
    return 0;
  }
  *out_value = (uint16_t)value;
  return 1;
}

static int parse_frontend(const char *text, uint16_t *out_value) {
  if (strcmp(text, "logmel") == 0) {
    *out_value = KWS_FRONTEND_LOGMEL;
    return 1;
  }
  if (strcmp(text, "pcen-lite") == 0) {
    *out_value = KWS_FRONTEND_PCEN_LITE;
    return 1;
  }
  return 0;
}

int main(int argc, char **argv) {
  FILE *wav = NULL;
  uint32_t wav_bytes = 0u;
  long wav_data_offset = 0L;
  kws_model_t model;
  kws_frontend_t frontend;
  uint16_t feature_dim = 32u;
  uint16_t frontend_kind = KWS_FRONTEND_LOGMEL;
  float features[KWS_MAX_FEATURE_DIM];
  uint32_t remaining;
  size_t frame_index = 0u;
  int exit_code = 1;

  if (argc < 2 || argc > 4) {
    fprintf(stderr, "usage: %s audio.wav [feature-dim] [logmel|pcen-lite]\n", argv[0]);
    return 2;
  }
  if (argc >= 3 && parse_feature_dim(argv[2], &feature_dim) == 0) {
    fprintf(stderr, "feature-dim must be 1..%u\n", KWS_MAX_FEATURE_DIM);
    return 2;
  }
  if (argc == 4 && parse_frontend(argv[3], &frontend_kind) == 0) {
    fprintf(stderr, "frontend must be logmel or pcen-lite\n");
    return 2;
  }

  memset(&model, 0, sizeof(model));
  model.feature_dim = feature_dim;
  model.frontend_kind = frontend_kind;
  model.frame_length_samples = KWS_FRAME_LENGTH_SAMPLES;
  model.frame_hop_samples = KWS_FRAME_HOP_SAMPLES;
  kws_frontend_init(&frontend, &model);
  kws_frontend_reset(&frontend);

  wav = fopen(argv[1], "rb");
  if (wav == NULL ||
      kws_tool_open_wav(wav, &wav_bytes, &wav_data_offset) == 0 ||
      wav_bytes == 0u) {
    fprintf(stderr, "expected non-empty mono 16-kHz PCM16 WAV: %s\n", argv[1]);
    goto cleanup;
  }
  if (fseek(wav, wav_data_offset, SEEK_SET) != 0) {
    fprintf(stderr, "cannot seek WAV data\n");
    goto cleanup;
  }

  remaining = wav_bytes;
  while (remaining >= sizeof(int16_t)) {
    int16_t sample;
    if (fread(&sample, sizeof(sample), 1u, wav) != 1u) {
      fprintf(stderr, "truncated WAV data\n");
      goto cleanup;
    }
    remaining -= (uint32_t)sizeof(sample);
    if (kws_frontend_push(&frontend, sample, features) != 0) {
      fprintf(stdout,
              "{\"frame\":%zu,\"dbfs\":%.9g,\"features\":[",
              frame_index, (double)kws_frontend_last_dbfs(&frontend));
      for (uint16_t i = 0u; i < feature_dim; ++i) {
        if (i != 0u) {
          fputc(',', stdout);
        }
        fprintf(stdout, "%.9g", (double)features[i]);
      }
      fputs("]}\n", stdout);
      frame_index++;
    }
  }
  if (remaining != 0u) {
    fprintf(stderr, "WAV data byte count is not PCM16 aligned\n");
    goto cleanup;
  }
  exit_code = ferror(stdout) != 0 ? 1 : 0;

cleanup:
  if (wav != NULL) {
    fclose(wav);
  }
  return exit_code;
}
