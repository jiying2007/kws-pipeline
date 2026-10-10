#define _POSIX_C_SOURCE 200809L

#include "kws_pipeline/kws.h"
#include "kws_bench_statistics.h"
#include "sha256.h"
#include "tool_io.h"

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BENCH_BLOCK_SAMPLES KWS_FRAME_HOP_SAMPLES
#define MAX_REPEATS 1000u

static int compare_double(const void *lhs, const void *rhs) {
  double a = *(const double *)lhs;
  double b = *(const double *)rhs;
  return (a > b) - (a < b);
}

static double elapsed_us(const struct timespec *begin,
                         const struct timespec *end) {
  int64_t sec = (int64_t)end->tv_sec - (int64_t)begin->tv_sec;
  int64_t nsec = (int64_t)end->tv_nsec - (int64_t)begin->tv_nsec;
  return (double)sec * 1000000.0 + (double)nsec / 1000.0;
}

static int parse_repeats(const char *text, unsigned *out_repeats) {
  char *end = NULL;
  unsigned long value;

  errno = 0;
  value = strtoul(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || value == 0ul ||
      value > (unsigned long)MAX_REPEATS) {
    return 0;
  }
  *out_repeats = (unsigned)value;
  return 1;
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
  long wav_data_offset = 0L;
  size_t blocks_per_repeat;
  size_t total_blocks;
  uint64_t total_processed_frames = 0u;
  uint64_t total_processed_samples = 0u;
  uint64_t frames_per_repeat;
  double *block_us = NULL;
  int16_t pcm[BENCH_BLOCK_SAMPLES];
  unsigned repeats = 1u;
  size_t sample_index = 0u;
  double total_process_us = 0.0;
  double max_process_us = 0.0;
  char runner_sha256[65];
  char model_sha256[65];
  char pack_sha256[65];
  char audio_sha256[65];
  int exit_code = 1;
  const kws_build_info_t *build_info = kws_build_info();

  if (argc != 4 && argc != 5) {
    fprintf(stderr,
            "usage: %s model.kwm keywords.kwk audio.wav [repeats]\n",
            argv[0]);
    return 2;
  }
  if (argc == 5 && parse_repeats(argv[4], &repeats) == 0) {
    fprintf(stderr, "repeats must be 1..%u\n", MAX_REPEATS);
    return 2;
  }
  if (kws_tool_read_file(argv[1], &model_blob, &model_bytes) == 0 ||
      kws_tool_read_file(argv[2], &pack_blob, &pack_bytes) == 0) {
    fprintf(stderr, "cannot read model or keyword pack\n");
    goto cleanup;
  }
  if (kws_model_open(model_blob, model_bytes, &model) != KWS_OK ||
      kws_keyword_pack_open(pack_blob, pack_bytes, &model, &pack) != KWS_OK) {
    fprintf(stderr, "model/keyword pack validation failed\n");
    goto cleanup;
  }
  if (kws_sha256_file_hex(argv[0], runner_sha256) == 0 ||
      kws_sha256_file_hex(argv[3], audio_sha256) == 0) {
    fprintf(stderr, "cannot hash benchmark runner or WAV input\n");
    goto cleanup;
  }
  kws_sha256_memory_hex(model_blob, model_bytes, model_sha256);
  kws_sha256_memory_hex(pack_blob, pack_bytes, pack_sha256);

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
      kws_tool_open_wav(wav, &wav_bytes, &wav_data_offset) == 0 ||
      wav_bytes < KWS_FRAME_LENGTH_SAMPLES * sizeof(int16_t)) {
    fprintf(stderr, "expected mono 16-kHz PCM16 WAV with at least one 400-sample frame: %s\n", argv[3]);
    goto cleanup;
  }

  frames_per_repeat = 1u + ((uint64_t)wav_bytes / sizeof(int16_t) -
                             KWS_FRAME_LENGTH_SAMPLES) / KWS_FRAME_HOP_SAMPLES;
  blocks_per_repeat =
      ((size_t)wav_bytes / sizeof(int16_t) + BENCH_BLOCK_SAMPLES - 1u) /
      BENCH_BLOCK_SAMPLES;
  if (blocks_per_repeat == 0u ||
      blocks_per_repeat > SIZE_MAX / (size_t)repeats) {
    fprintf(stderr, "benchmark input is too large\n");
    goto cleanup;
  }
  total_blocks = blocks_per_repeat * (size_t)repeats;
  if (total_blocks > SIZE_MAX / sizeof(*block_us)) {
    fprintf(stderr, "benchmark sample buffer is too large\n");
    goto cleanup;
  }
  block_us = (double *)malloc(total_blocks * sizeof(*block_us));
  if (block_us == NULL) {
    fprintf(stderr, "cannot allocate timing samples\n");
    goto cleanup;
  }

  for (unsigned repeat = 0u; repeat < repeats; ++repeat) {
    uint32_t remaining = wav_bytes;
    kws_engine_stats_v2_t stats = {0};
    stats.struct_size = sizeof(stats);
    stats.api_version = KWS_ENGINE_STATS_V2_API_VERSION;
    kws_engine_reset(engine);
    if (fseek(wav, wav_data_offset, SEEK_SET) != 0) {
      fprintf(stderr, "cannot seek WAV data\n");
      goto cleanup;
    }

    while (remaining != 0u) {
      size_t want_samples = (size_t)(remaining / 2u);
      size_t got_samples;
      struct timespec begin;
      struct timespec end;
      double duration_us;
      int detected = 0;

      if (want_samples > BENCH_BLOCK_SAMPLES) {
        want_samples = BENCH_BLOCK_SAMPLES;
      }
      got_samples = fread(pcm, sizeof(pcm[0]), want_samples, wav);
      if (got_samples != want_samples) {
        fprintf(stderr, "truncated WAV data\n");
        goto cleanup;
      }
      remaining -= (uint32_t)(got_samples * sizeof(pcm[0]));

      if (clock_gettime(CLOCK_MONOTONIC, &begin) != 0 ||
          kws_engine_accept_pcm16(engine, pcm, got_samples, NULL, &detected) !=
              KWS_OK ||
          clock_gettime(CLOCK_MONOTONIC, &end) != 0) {
        fprintf(stderr, "benchmark timing/runtime failure\n");
        goto cleanup;
      }
      duration_us = elapsed_us(&begin, &end);
      if (duration_us < 0.0 || sample_index >= total_blocks) {
        fprintf(stderr, "invalid benchmark timing sample\n");
        goto cleanup;
      }
      block_us[sample_index++] = duration_us;
      total_process_us += duration_us;
      if (duration_us > max_process_us) {
        max_process_us = duration_us;
      }
    }
    /* Query actual completed model/decoder frames outside the timed region.
     * Each repeat resets frontend state, so short streams never accumulate
     * enough input merely by increasing repeats. */
    if (kws_engine_get_stats_v2(engine, &stats) != KWS_OK ||
        stats.processed_frames != frames_per_repeat ||
        stats.processed_samples != (uint64_t)wav_bytes / sizeof(int16_t)) {
      fprintf(stderr, "benchmark effective workload mismatch\n");
      goto cleanup;
    }
    total_processed_frames += stats.processed_frames;
    total_processed_samples += stats.processed_samples;
  }

  if (sample_index != total_blocks) {
    fprintf(stderr, "benchmark block count mismatch\n");
    goto cleanup;
  }
  qsort(block_us, total_blocks, sizeof(*block_us), compare_double);

  {
    const double audio_seconds =
        (double)(wav_bytes / 2u) / (double)KWS_SAMPLE_RATE_HZ;
    const double total_audio_seconds = audio_seconds * (double)repeats;
    const double mean_us = total_process_us / (double)total_blocks;
    const double p50_us = kws_bench_percentile_nearest_rank(block_us, total_blocks, 0.50);
    const double p95_us = kws_bench_percentile_nearest_rank(block_us, total_blocks, 0.95);
    const double p99_us = kws_bench_percentile_nearest_rank(block_us, total_blocks, 0.99);
    const double rtf =
        total_audio_seconds > 0.0
            ? total_process_us / (total_audio_seconds * 1000000.0)
            : 0.0;
    /* The nominal 320-sample call deadline is not a product 160-sample
     * callback deadline. A shorter final call is included without padding. */
    const unsigned final_block_samples =
        (unsigned)(((size_t)wav_bytes / sizeof(int16_t) - 1u) %
                   BENCH_BLOCK_SAMPLES + 1u);
    const double deadline_us =
        (double)BENCH_BLOCK_SAMPLES * 1000000.0 /
        (double)KWS_SAMPLE_RATE_HZ;
    const double p99_headroom = p99_us > 0.0 ? deadline_us / p99_us : 0.0;

    if (total_process_us <= 0.0 || p99_us <= 0.0) {
      fprintf(stderr, "benchmark has no positive measured runtime\n");
      goto cleanup;
    }
    fprintf(stdout,
            "{\"schema_version\":2,\"runner_sha256\":\"%s\","
            "\"runtime_version\":\"%s\",\"runtime_source_revision\":\"%s\","
            "\"runtime_config_digest\":\"%s\",\"runtime_target\":\"%s\","
            "\"model_sha256\":\"%s\",\"keyword_pack_sha256\":\"%s\","
            "\"audio_sha256\":\"%s\",\"block_samples\":%u,"
            "\"audio_samples\":%u,\"blocks_per_repeat\":%zu,"
            "\"processed_frames_per_repeat\":%" PRIu64 ","
            "\"processed_frames\":%" PRIu64 ",\"processed_samples\":%" PRIu64 ","
            "\"sample_rate_hz\":%u,\"frame_length_samples\":%u,"
            "\"frame_hop_samples\":%u,\"final_block_samples\":%u,"
            "\"repeat_policy\":\"reset-before-each-repeat\","
            "\"workload_contract\":\"pcm16-400-window-320-hop-v1\","
            "\"timing_unit\":\"input-call\",\"deadline_basis\":\"nominal-block\","
            "\"tail_policy\":\"short-final-call-included-no-padding\","
            "\"percentile_estimator\":\"" KWS_BENCH_PERCENTILE_ESTIMATOR "\","
            "\"block_deadline_us\":%.3f,\"audio_seconds\":%.6f,"
            "\"repeats\":%u,\"blocks\":%zu,\"model_bytes\":%zu,"
            "\"keyword_pack_bytes\":%zu,\"arena_bytes\":%zu,"
            "\"total_process_us\":%.3f,\"mean_process_us\":%.3f,"
            "\"p50_process_us\":%.3f,\"p95_process_us\":%.3f,"
            "\"p99_process_us\":%.3f,\"max_process_us\":%.3f,"
            "\"rtf\":%.9f,\"p99_headroom\":%.6f}\n",
            runner_sha256, build_info->version, build_info->source_revision,
            build_info->config_digest, build_info->target_triple,
            model_sha256, pack_sha256, audio_sha256,
            (unsigned)BENCH_BLOCK_SAMPLES, (unsigned)(wav_bytes / 2u),
            blocks_per_repeat, frames_per_repeat, total_processed_frames,
            total_processed_samples, (unsigned)KWS_SAMPLE_RATE_HZ,
            (unsigned)KWS_FRAME_LENGTH_SAMPLES, (unsigned)KWS_FRAME_HOP_SAMPLES,
            final_block_samples, deadline_us, audio_seconds, repeats,
            total_blocks, model_bytes, pack_bytes,
            kws_engine_required_bytes(&model), total_process_us, mean_us,
            p50_us, p95_us, p99_us, max_process_us, rtf, p99_headroom);
  }
  exit_code = ferror(stdout) != 0 ? 1 : 0;

cleanup:
  if (wav != NULL) {
    fclose(wav);
  }
  free(block_us);
  free(arena);
  free(pack_blob);
  free(model_blob);
  return exit_code;
}
