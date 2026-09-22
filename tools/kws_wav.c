#include "kws_pipeline/kws.h"
#include "tool_io.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
  const char *stats_path = NULL;
  int exit_code = 1;

  if (argc != 5 && argc != 7) {
    fprintf(stderr,
            "usage: %s model.kwm keywords.kwk audio.wav recording-id "
            "[--stats-json path]\n",
            argv[0]);
    return 2;
  }
  if (argc == 7) {
    if (strcmp(argv[5], "--stats-json") != 0 || argv[6][0] == '\0') {
      fprintf(stderr, "expected optional --stats-json path\n");
      return 2;
    }
    stats_path = argv[6];
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
  if (stats_path != NULL) {
    kws_engine_stats_v2_t stats = {0};
    FILE *stats_file = NULL;

    stats.struct_size = (uint32_t)sizeof(stats);
    stats.api_version = KWS_ENGINE_STATS_V2_API_VERSION;
    if (kws_engine_get_stats_v2(engine, &stats) != KWS_OK) {
      fprintf(stderr, "cannot read KWS runtime stats\n");
      goto cleanup;
    }
    stats_file = fopen(stats_path, "wb");
    if (stats_file == NULL) {
      fprintf(stderr, "cannot open stats output: %s\n", stats_path);
      goto cleanup;
    }
    fprintf(stats_file,
            "{\"schema_version\":1,"
            "\"processed_samples\":%" PRIu64 ","
            "\"processed_frames\":%" PRIu64 ","
            "\"speech_frames\":%" PRIu64 ","
            "\"blank_top1_frames\":%" PRIu64 ","
            "\"decoder_hits\":%" PRIu64 ","
            "\"refractory_suppressed\":%" PRIu64 ","
            "\"detections\":%" PRIu64 ","
            "\"pending_keyword_index\":%d,"
            "\"pending_age_frames\":%u,"
            "\"max_detection_confidence\":%.9g}\n",
            stats.processed_samples,
            stats.processed_frames,
            stats.speech_frames,
            stats.blank_top1_frames,
            stats.decoder_hits,
            stats.refractory_suppressed,
            stats.detections,
            (int)stats.pending_keyword_index,
            (unsigned)stats.pending_age_frames,
            (double)stats.max_detection_confidence);
    {
      int write_failed = ferror(stats_file) != 0;
      int close_failed = fclose(stats_file) != 0;
      if (write_failed || close_failed) {
        fprintf(stderr, "cannot write stats output: %s\n", stats_path);
        goto cleanup;
      }
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
