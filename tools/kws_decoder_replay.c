#if defined(__linux__)
#define _POSIX_C_SOURCE 200809L
#endif

#include "kws_pipeline/kws.h"
#include "kws_debug.h"
#include "kws_trace_io.h"
#include "kws_parameter_limits.h"
#include "sha256.h"
#include "tool_io.h"

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#if defined(__linux__)
#include <unistd.h>
#endif
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define COMPLETION_TRACE_LIMIT (64u * 1024u * 1024u)

/* Hash and replay exactly the same private byte snapshot. The original path
 * may change afterwards; the digest describes the bytes copied, not a lock. */
static FILE *snapshot_trace(const char *path, char digest[65]) {
  FILE *input = fopen(path, "rb");
  FILE *snapshot = NULL;
  uint8_t *bytes = NULL;
  long length;
  if (input == NULL) {
    return NULL;
  }
  if (fseek(input, 0, SEEK_END) != 0 || (length = ftell(input)) < 0 ||
      (unsigned long)length > COMPLETION_TRACE_LIMIT ||
      fseek(input, 0, SEEK_SET) != 0) {
    goto done;
  }
  bytes = (uint8_t *)malloc((size_t)length + 1u);
  if (bytes == NULL || fread(bytes, 1u, (size_t)length, input) != (size_t)length ||
      fgetc(input) != EOF || ferror(input) != 0) {
    goto done;
  }
  kws_sha256_memory_hex(bytes, (size_t)length, digest);
  snapshot = tmpfile();
  if (snapshot != NULL &&
      (fwrite(bytes, 1u, (size_t)length, snapshot) != (size_t)length ||
       fflush(snapshot) != 0 || fseek(snapshot, 0, SEEK_SET) != 0)) {
    fclose(snapshot);
    snapshot = NULL;
  }
done:
  if (fclose(input) != 0 && snapshot != NULL) {
    fclose(snapshot);
    snapshot = NULL;
  }
  free(bytes);
  return snapshot;
}

static int publish_completion(const char *path, const char *recording,
                              const kws_engine_stats_t *stats,
                              const kws_config_t *config,
                              float blank_retention, float fuzzy_child_cost_log,
                              int search_policy_override,
                              const char model_sha[65], const char pack_sha[65],
                              const char trace_sha[65], const char tool_sha[65]) {
#if defined(__linux__)
  size_t length = strlen(path);
  char *temporary;
  FILE *stream;
  int descriptor;
  int ok = 0;
  if (length > SIZE_MAX - 16u) {
    return 0;
  }
  temporary = (char *)malloc(length + 16u);
  if (temporary == NULL) {
    return 0;
  }
  (void)snprintf(temporary, length + 16u, "%s.tmp.XXXXXX", path);
  descriptor = mkstemp(temporary);
  if (descriptor < 0) {
    free(temporary);
    return 0;
  }
  stream = fdopen(descriptor, "w");
  if (stream == NULL) {
    close(descriptor);
    goto done;
  }
  fprintf(stream,
          "{\"schema\":\"kws-decoder-replay-completion-v1\","
          "\"execution_kind\":\"native-linux-decoder-replay\","
          "\"production_qualified\":false,\"completion\":{"
          "\"exit_code\":0,\"frames_consumed\":%" PRIu64 ","
          "\"decoder_hits\":%" PRIu64 ",\"detections\":%" PRIu64 ","
          "\"refractory_suppressed\":%" PRIu64 "},"
          "\"identity\":{\"model_sha256\":\"%s\","
          "\"keyword_pack_sha256\":\"%s\",\"trace_sha256\":\"%s\","
          "\"tool_sha256\":\"%s\",\"parameter_contract_sha256\":\"%s\"},"
          "\"context\":{\"recording_id\":",
          stats->processed_frames, stats->decoder_hits, stats->detections,
          stats->refractory_suppressed, model_sha, pack_sha, trace_sha, tool_sha,
          KWS_PARAMETER_CONTRACT_SHA256);
  kws_tool_print_json_string(stream, recording);
  fprintf(stream,
          ",\"mode\":\"per-clip-fresh-engine\",\"vad\":\"saved-frame-flags\","
          "\"reset\":\"fresh-engine-and-decoder-inactive-boundary\","
          "\"clock\":\"saved-end-sample\"},\"runtime\":{"
          "\"min_speech_dbfs\":%.9g,\"token_boost\":%.9g,"
          "\"state_retention\":%.9g,\"refractory_ms\":%u,"
          "\"external_vad_threshold\":%.9g,\"blank_retention\":%.9g,"
          "\"fuzzy_child_cost_log\":%.9g,\"search_policy_override\":%s}}\n",
          (double)config->min_speech_dbfs, (double)config->token_boost,
          (double)config->state_retention, config->refractory_ms,
          (double)config->external_vad_threshold, (double)blank_retention,
          (double)fuzzy_child_cost_log,
          search_policy_override != 0 ? "true" : "false");
  ok = ferror(stream) == 0 && fflush(stream) == 0 && fsync(descriptor) == 0;
  if (fclose(stream) != 0) {
    ok = 0;
  }
  /* link is an atomic no-replace commit, unlike exists+rename. Once linked,
   * cleanup of the temporary alias must not turn completion into a failure. */
  if (ok != 0) {
    ok = link(temporary, path) == 0;
  }
done:
  (void)unlink(temporary);
  free(temporary);
  return ok;
#else
  (void)path; (void)recording; (void)stats; (void)config;
  (void)blank_retention; (void)fuzzy_child_cost_log;
  (void)search_policy_override; (void)model_sha; (void)pack_sha;
  (void)trace_sha; (void)tool_sha;
  return 0;
#endif
}

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
  const char *first = text;
  while (isspace((unsigned char)*first)) {
    ++first;
  }
  if (*first == '-') {
    return 0;
  }
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
  float blank_retention = expf(KWS_SILENCE_RETENTION_LOG);
  float fuzzy_child_cost_log = KWS_FUZZY_CHILD_RETENTION_COST_LOG;
  int search_policy_override = 0;
  kws_engine_t *engine = NULL;
  void *arena = NULL;
  kws_trace_reader_t reader = {0};
  kws_trace_header_t trace = {0};
  float logits[KWS_MAX_VOCAB_SIZE];
  char model_sha256[65];
  char pack_sha256[65];
  char trace_sha256[65];
  char tool_sha256[65];
  const char *completion_path = NULL;
  kws_engine_stats_t completion_stats;
  int reader_open = 0;
  int exit_code = 1;

  if (argc < 5 || ((argc - 5) % 2) != 0) {
    fprintf(stderr,
            "usage: %s model.kwm keywords.kwk trace.kwtr recording-id "
            "[--state-retention value] [--refractory-ms value] "
            "[--blank-retention value] [--fuzzy-child-cost-log value] "
            "[--completion new-path]\n",
            argv[0]);
    return 2;
  }
  for (int i = 5; i < argc; i += 2) {
    if (strcmp(argv[i], "--completion") == 0) {
      if (completion_path != NULL || argv[i + 1][0] == '\0') {
        fprintf(stderr, "invalid --completion\n");
        return 2;
      }
      completion_path = argv[i + 1];
    } else if (strcmp(argv[i], "--state-retention") == 0) {
      if (!parse_float(argv[i + 1], &config.state_retention)) {
        fprintf(stderr, "invalid --state-retention\n");
        return 2;
      }
    } else if (strcmp(argv[i], "--refractory-ms") == 0) {
      if (!parse_u32(argv[i + 1], &config.refractory_ms)) {
        fprintf(stderr, "invalid --refractory-ms\n");
        return 2;
      }
    } else if (strcmp(argv[i], "--blank-retention") == 0) {
      if (!parse_float(argv[i + 1], &blank_retention)) {
        fprintf(stderr, "invalid --blank-retention\n");
        return 2;
      }
      search_policy_override = 1;
    } else if (strcmp(argv[i], "--fuzzy-child-cost-log") == 0) {
      if (!parse_float(argv[i + 1], &fuzzy_child_cost_log)) {
        fprintf(stderr, "invalid --fuzzy-child-cost-log\n");
        return 2;
      }
      search_policy_override = 1;
    } else {
      fprintf(stderr, "unknown option: %s\n", argv[i]);
      return 2;
    }
  }

  if (kws_tool_read_file(argv[1], &model_blob, &model_bytes) == 0 ||
      kws_tool_read_file(argv[2], &pack_blob, &pack_bytes) == 0 ||
      kws_model_open(model_blob, model_bytes, &model) != KWS_OK ||
      kws_keyword_pack_open(pack_blob, pack_bytes, &model, &pack) != KWS_OK) {
    fprintf(stderr, "cannot open model/keyword pack\n");
    goto cleanup;
  }
  if (completion_path != NULL) {
    kws_sha256_memory_hex(model_blob, model_bytes, model_sha256);
    kws_sha256_memory_hex(pack_blob, pack_bytes, pack_sha256);
#if defined(__linux__)
    char invocation_sha256[65];
    if (kws_sha256_file_hex("/proc/self/exe", tool_sha256) == 0 ||
        kws_sha256_file_hex(argv[0], invocation_sha256) == 0 ||
        strcmp(tool_sha256, invocation_sha256) != 0) {
      fprintf(stderr, "completion requires matching native Linux executable identity\n");
      goto cleanup;
    }
#else
    fprintf(stderr, "completion requires native Linux\n");
    goto cleanup;
#endif
  } else if (kws_sha256_file_hex(argv[1], model_sha256) == 0) {
    fprintf(stderr, "cannot hash model\n");
    goto cleanup;
  }
  if (!(completion_path != NULL
          ? kws_trace_reader_open_stream(&reader, snapshot_trace(argv[3], trace_sha256), &trace)
          : kws_trace_reader_open(&reader, argv[3], &trace))) {
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
  if (search_policy_override != 0 &&
      kws_engine_debug_set_decoder_search_policy(
          engine, blank_retention, fuzzy_child_cost_log) != KWS_OK) {
    fprintf(stderr, "invalid decoder search-policy override\n");
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
  if (completion_path != NULL &&
      (kws_engine_get_stats(engine, &completion_stats) != KWS_OK ||
       completion_stats.processed_frames != reader.frames_read)) {
    fprintf(stderr, "cannot obtain replay completion stats\n");
    goto cleanup;
  }
  if (!kws_trace_reader_close(&reader)) {
    reader_open = 0;
    fprintf(stderr, "acoustic trace length/trailer is invalid\n");
    goto cleanup;
  }
  reader_open = 0;
  if (fflush(stdout) != 0 || ferror(stdout) != 0) {
    fprintf(stderr, "cannot write stdout\n");
    goto cleanup;
  }
  exit_code = 0;
  if (completion_path != NULL &&
      !publish_completion(completion_path, argv[4], &completion_stats, &config,
                          blank_retention, fuzzy_child_cost_log,
                          search_policy_override, model_sha256, pack_sha256,
                          trace_sha256, tool_sha256)) {
    fprintf(stderr, "cannot publish replay completion (destination must be new)\n");
    exit_code = 1;
    goto cleanup;
  }

cleanup:
  if (reader_open != 0) {
    (void)kws_trace_reader_close(&reader);
  }
  free(arena);
  free(pack_blob);
  free(model_blob);
  return exit_code;
}
