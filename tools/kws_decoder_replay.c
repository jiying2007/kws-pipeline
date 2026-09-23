#include "kws_pipeline/kws.h"
#include "kws_debug.h"
#include "kws_trace_io.h"
#include "sha256.h"
#include "tool_io.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int parse_float(const char *text, float *out) {
  char *end = NULL;
  float value;
  errno = 0;
  value = strtof(text, &end);
  if (errno != 0 || end == text || *end != '\0') {
    return 0;
  }
  *out = value;
  return 1;
}

static int parse_u32(const char *text, uint32_t *out) {
  char *end = NULL;
  unsigned long value;
  errno = 0;
  value = strtoul(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || value > UINT32_MAX) {
    return 0;
  }
  *out = (uint32_t)value;
  return 1;
}

int main(int argc, char **argv) {
  uint8_t *model_blob = NULL;
  uint8_t *pack_blob = NULL;
  size_t model_bytes = 0u;
  size_t pack_bytes = 0u;
  kws_model_t model;
  kws_keyword_pack_t pack;
  kws_config_t config = kws_default_config();
  kws_engine_t *engine = NULL;
  void *arena = NULL;
  kws_trace_reader_t reader = {0};
  kws_trace_header_t trace = {0};
  float logits[KWS_MAX_VOCAB_SIZE];
  char model_sha256[65];
  int reader_open = 0;
  int exit_code = 1;

  if (argc < 5 || ((argc - 5) % 2) != 0) {
    fprintf(stderr,
            "usage: %s model.kwm keywords.kwk trace.kwtr recording-id "
            "[--state-retention value] [--refractory-ms value]\n",
            argv[0]);
    return 2;
  }
  for (int i = 5; i < argc; i += 2) {
    if (strcmp(argv[i], "--state-retention") == 0) {
      if (!parse_float(argv[i + 1], &config.state_retention)) {
        fprintf(stderr, "invalid --state-retention\n");
        return 2;
      }
    } else if (strcmp(argv[i], "--refractory-ms") == 0) {
      if (!parse_u32(argv[i + 1], &config.refractory_ms)) {
        fprintf(stderr, "invalid --refractory-ms\n");
        return 2;
      }
    } else {
      fprintf(stderr, "unknown option: %s\n", argv[i]);
      return 2;
    }
  }

  if (kws_tool_read_file(argv[1], &model_blob, &model_bytes) == 0 ||
      kws_tool_read_file(argv[2], &pack_blob, &pack_bytes) == 0 ||
      kws_model_open(model_blob, model_bytes, &model) != KWS_OK ||
      kws_keyword_pack_open(pack_blob, pack_bytes, &model, &pack) != KWS_OK ||
      kws_sha256_file_hex(argv[1], model_sha256) == 0) {
    fprintf(stderr, "cannot open model/keyword pack\n");
    goto cleanup;
  }
  if (!kws_trace_reader_open(&reader, argv[3], &trace)) {
    fprintf(stderr, "cannot open acoustic trace: %s\n", argv[3]);
    goto cleanup;
  }
  reader_open = 1;
  if (trace.vocab_size != model.vocab_size ||
      trace.frontend_kind != model.frontend_kind ||
      trace.sample_rate_hz != model.sample_rate_hz ||
      trace.frame_length_samples != model.frame_length_samples ||
      trace.frame_hop_samples != model.frame_hop_samples ||
      trace.vocab_fingerprint != model.vocab_fingerprint ||
      strcmp(trace.model_sha256, model_sha256) != 0) {
    fprintf(stderr, "acoustic trace does not match model\n");
    goto cleanup;
  }

  arena = malloc(kws_engine_required_bytes(&model));
  if (arena == NULL ||
      kws_engine_init(arena, kws_engine_required_bytes(&model), &model, &config,
                      &engine) != KWS_OK ||
      kws_engine_set_keyword_pack(engine, &pack) != KWS_OK) {
    fprintf(stderr, "cannot initialize replay engine\n");
    goto cleanup;
  }

  for (;;) {
    uint64_t end_sample = 0u;
    int speech_active = 0;
    int detected = 0;
    kws_detection_t hit = {0u, 0.0f, 0u};
    int status = kws_trace_reader_next(&reader, &end_sample, &speech_active,
                                       logits, KWS_MAX_VOCAB_SIZE);
    if (status == 0) {
      break;
    }
    if (status < 0 ||
        kws_engine_debug_replay_frame(engine, logits, trace.vocab_size,
                                      speech_active, end_sample, &hit,
                                      &detected) != KWS_OK) {
      fprintf(stderr, "invalid acoustic trace frame\n");
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
  if (!kws_trace_reader_close(&reader)) {
    reader_open = 0;
    fprintf(stderr, "acoustic trace length/trailer is invalid\n");
    goto cleanup;
  }
  reader_open = 0;
  exit_code = ferror(stdout) != 0 ? 1 : 0;

cleanup:
  if (reader_open != 0) {
    (void)kws_trace_reader_close(&reader);
  }
  free(arena);
  free(pack_blob);
  free(model_blob);
  return exit_code;
}
