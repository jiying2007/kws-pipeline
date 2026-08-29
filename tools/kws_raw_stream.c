#include "kws_pipeline/kws.h"
#include "tool_io.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SAMPLES 320u

int main(int argc, char **argv) {
  uint8_t *model_blob = NULL;
  uint8_t *pack_blob = NULL;
  size_t model_bytes = 0u;
  size_t pack_bytes = 0u;
  kws_model_t model;
  kws_keyword_pack_t pack;
  kws_engine_t *engine = NULL;
  void *arena = NULL;
  int16_t pcm[BLOCK_SAMPLES];
  int exit_code = 1;

  if (argc != 4) {
    fprintf(stderr, "usage: %s model.kwm keywords.kwk recording-id < pcm16le.raw\n", argv[0]);
    return 2;
  }
  if (kws_tool_read_file(argv[1], &model_blob, &model_bytes) == 0 ||
      kws_tool_read_file(argv[2], &pack_blob, &pack_bytes) == 0) {
    fprintf(stderr, "cannot read model or keyword pack\n");
    goto cleanup;
  }
  if (kws_model_open(model_blob, model_bytes, &model) != KWS_OK ||
      kws_keyword_pack_open(pack_blob, pack_bytes, &model, &pack) != KWS_OK) {
    fprintf(stderr, "invalid model or keyword pack\n");
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

  for (;;) {
    size_t got = fread(pcm, sizeof(pcm[0]), BLOCK_SAMPLES, stdin);
    if (got == 0u) {
      if (ferror(stdin) != 0) {
        fprintf(stderr, "raw PCM input error\n");
        goto cleanup;
      }
      break;
    }
    {
      kws_detection_t hit;
      int detected = 0;
      if (kws_engine_accept_pcm16(engine, pcm, got, &hit, &detected) != KWS_OK) {
        fprintf(stderr, "KWS runtime error\n");
        goto cleanup;
      }
      if (detected != 0) {
        fputs("{\"recording\":", stdout);
        kws_tool_print_json_string(stdout, argv[3]);
        fprintf(stdout,
                ",\"keyword_id\":%u,\"time_s\":%.6f,\"confidence\":%.6f}\n",
                hit.keyword_id,
                (double)hit.end_sample / (double)KWS_SAMPLE_RATE_HZ,
                (double)hit.confidence);
        if (fflush(stdout) != 0) {
          goto cleanup;
        }
      }
    }
    if (got != BLOCK_SAMPLES) {
      if (ferror(stdin) != 0) {
        fprintf(stderr, "raw PCM input error\n");
        goto cleanup;
      }
      break;
    }
  }
  exit_code = 0;

cleanup:
  free(arena);
  free(pack_blob);
  free(model_blob);
  return exit_code;
}
