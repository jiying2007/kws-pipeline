#include "kws_pipeline/kws.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BLOCK_SAMPLES 160u

static uint16_t rd16(const uint8_t *p) {
  return (uint16_t)((uint16_t)p[0] | (uint16_t)((uint16_t)p[1] << 8u));
}

static uint32_t rd32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8u) |
         ((uint32_t)p[2] << 16u) | ((uint32_t)p[3] << 24u);
}

static int read_file(const char *path, uint8_t **out_data, size_t *out_bytes) {
  FILE *stream = fopen(path, "rb");
  long size;
  uint8_t *data;
  if (stream == NULL) {
    fprintf(stderr, "cannot open %s\n", path);
    return 0;
  }
  if (fseek(stream, 0, SEEK_END) != 0) {
    fclose(stream);
    return 0;
  }
  size = ftell(stream);
  if (size <= 0 || fseek(stream, 0, SEEK_SET) != 0) {
    fclose(stream);
    return 0;
  }
  data = (uint8_t *)malloc((size_t)size);
  if (data == NULL) {
    fclose(stream);
    return 0;
  }
  if (fread(data, 1u, (size_t)size, stream) != (size_t)size) {
    free(data);
    fclose(stream);
    return 0;
  }
  fclose(stream);
  *out_data = data;
  *out_bytes = (size_t)size;
  return 1;
}

static int open_wav(FILE *stream, uint32_t *out_data_bytes) {
  uint8_t header[12];
  int have_fmt = 0;
  if (fread(header, 1u, sizeof(header), stream) != sizeof(header) ||
      memcmp(header, "RIFF", 4u) != 0 || memcmp(header + 8u, "WAVE", 4u) != 0) {
    return 0;
  }

  for (;;) {
    uint8_t chunk[8];
    uint32_t size;
    long skip;
    if (fread(chunk, 1u, sizeof(chunk), stream) != sizeof(chunk)) {
      return 0;
    }
    size = rd32(chunk + 4u);
    if (memcmp(chunk, "fmt ", 4u) == 0) {
      uint8_t fmt[16];
      if (size < sizeof(fmt) || fread(fmt, 1u, sizeof(fmt), stream) != sizeof(fmt)) {
        return 0;
      }
      if (rd16(fmt + 0u) != 1u || rd16(fmt + 2u) != 1u ||
          rd32(fmt + 4u) != KWS_SAMPLE_RATE_HZ || rd16(fmt + 14u) != 16u) {
        return 0;
      }
      have_fmt = 1;
      skip = (long)(size - (uint32_t)sizeof(fmt));
      if ((size & 1u) != 0u) {
        skip += 1L;
      }
      if (skip > 0L && fseek(stream, skip, SEEK_CUR) != 0) {
        return 0;
      }
    } else if (memcmp(chunk, "data", 4u) == 0) {
      if (!have_fmt || (size & 1u) != 0u) {
        return 0;
      }
      *out_data_bytes = size;
      return 1;
    } else {
      skip = (long)size + (((size & 1u) != 0u) ? 1L : 0L);
      if (skip > 0L && fseek(stream, skip, SEEK_CUR) != 0) {
        return 0;
      }
    }
  }
}

int main(int argc, char **argv) {
  uint8_t *model_blob = NULL;
  uint8_t *pack_blob = NULL;
  size_t model_bytes = 0u;
  size_t pack_bytes = 0u;
  kws_model_t model;
  kws_keyword_pack_t pack;
  kws_engine_t *engine = NULL;
  void *arena = NULL;
  FILE *wav = NULL;
  uint32_t wav_bytes = 0u;
  uint32_t remaining;
  int16_t pcm[BLOCK_SAMPLES];
  int exit_code = 1;

  if (argc != 5) {
    fprintf(stderr, "usage: %s model.kwm keywords.kwk audio.wav recording-id\n", argv[0]);
    return 2;
  }
  if (!read_file(argv[1], &model_blob, &model_bytes) ||
      !read_file(argv[2], &pack_blob, &pack_bytes)) {
    goto cleanup;
  }
  if (kws_model_open(model_blob, model_bytes, &model) != KWS_OK) {
    fprintf(stderr, "invalid model: %s\n", argv[1]);
    goto cleanup;
  }
  if (kws_keyword_pack_open(pack_blob, pack_bytes, &model, &pack) != KWS_OK) {
    fprintf(stderr, "invalid keyword pack: %s\n", argv[2]);
    goto cleanup;
  }

  arena = malloc(kws_engine_required_bytes(&model));
  if (arena == NULL ||
      kws_engine_init(arena, kws_engine_required_bytes(&model), &model, NULL, &engine) != KWS_OK ||
      kws_engine_set_keyword_pack(engine, &pack) != KWS_OK) {
    fprintf(stderr, "cannot initialize KWS engine\n");
    goto cleanup;
  }

  wav = fopen(argv[3], "rb");
  if (wav == NULL || !open_wav(wav, &wav_bytes)) {
    fprintf(stderr, "expected mono 16-kHz PCM16 WAV: %s\n", argv[3]);
    goto cleanup;
  }

  remaining = wav_bytes;
  while (remaining != 0u) {
    size_t want_samples = remaining / 2u;
    size_t got_samples;
    kws_detection_t hit;
    int detected = 0;
    if (want_samples > BLOCK_SAMPLES) {
      want_samples = BLOCK_SAMPLES;
    }
    got_samples = fread(pcm, sizeof(pcm[0]), want_samples, wav);
    if (got_samples != want_samples) {
      fprintf(stderr, "truncated WAV data: %s\n", argv[3]);
      goto cleanup;
    }
    remaining -= (uint32_t)(got_samples * sizeof(pcm[0]));
    if (kws_engine_accept_pcm16(engine, pcm, got_samples, &hit, &detected) != KWS_OK) {
      fprintf(stderr, "KWS runtime error\n");
      goto cleanup;
    }
    if (detected != 0) {
      printf("{\"recording\":\"%s\",\"keyword_id\":%u,\"time_s\":%.6f,\"confidence\":%.6f}\n",
             argv[4], hit.keyword_id,
             (double)hit.end_sample / (double)KWS_SAMPLE_RATE_HZ,
             (double)hit.confidence);
    }
  }
  exit_code = 0;

cleanup:
  if (wav != NULL) {
    fclose(wav);
  }
  free(arena);
  free(pack_blob);
  free(model_blob);
  return exit_code;
}
