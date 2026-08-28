#include "kws_pipeline/kws.h"
#include "tool_io.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SAMPLES 160u

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
  long wav_data_offset = 0L;
  uint32_t remaining;
  int16_t pcm[BLOCK_SAMPLES];
  int exit_code = 1;

  if (argc != 5) {
    fprintf(stderr, "usage: %s model.kwm keywords.kwk audio.wav recording-id\n", argv[0]);
    return 2;
  }
  if (kws_tool_read_file(argv[1], &model_blob, &model_bytes) == 0 ||
      kws_tool_read_file(argv[2], &pack_blob, &pack_bytes) == 0) {
    fprintf(stderr, "cannot read model or keyword pack\n");
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
      kws_engine_init(arena, kws_engine_required_bytes(&model), &model, NULL,
                      &engine) != KWS_OK ||
      kws_engine_set_keyword_pack(engine, &pack) != KWS_OK) {
    fprintf(stderr, "cannot initialize KWS engine\n");
    goto cleanup;
  }

  wav = fopen(argv[3], "rb");
  if (wav == NULL ||
      kws_tool_open_wav(wav, &wav_bytes, &wav_data_offset) == 0) {
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
    if (kws_engine_accept_pcm16(engine, pcm, got_samples, &hit, &detected) !=
        KWS_OK) {
      fprintf(stderr, "KWS runtime error\n");
      goto cleanup;
    }
    if (detected != 0) {
      fputs("{\"recording\":", stdout);
      kws_tool_print_json_string(stdout, argv[4]);
      fprintf(stdout,
              ",\"keyword_id\":%u,\"time_s\":%.6f,\"confidence\":%.6f}\n",
              hit.keyword_id,
              (double)hit.end_sample / (double)KWS_SAMPLE_RATE_HZ,
              (double)hit.confidence);
    }
  }
  exit_code = ferror(stdout) != 0 ? 1 : 0;

cleanup:
  if (wav != NULL) {
    fclose(wav);
  }
  free(arena);
  free(pack_blob);
  free(model_blob);
  return exit_code;
}
