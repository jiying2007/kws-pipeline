#include "kws_pipeline/kws.h"
#include "tool_io.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kws_timeline.h"

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
  int16_t pcm[KWS_MAX_PCM_BLOCK_SAMPLES];
  const char *metadata_path = NULL;
  FILE *timeline = NULL;
  kws_timeline_span_t span = {0};
  uint64_t output_samples = 0u;
  uint64_t epoch = 0u;
  size_t block_samples = BLOCK_SAMPLES;
  const char *stats_path = NULL;
  int exit_code = 1;

  if (argc < 5 || (argc - 5) % 2 != 0) {
    fprintf(stderr, "usage: %s model.kwm keywords.kwk audio.wav recording-id "
            "[--stats-json path] [--metadata-tsv path] [--block-samples 1..320]\n", argv[0]);
    return 2;
  }
  for (int i = 5; i < argc; i += 2) {
    if (strcmp(argv[i], "--stats-json") == 0 && stats_path == NULL) {
      stats_path = argv[i + 1];
    } else if (strcmp(argv[i], "--metadata-tsv") == 0 && metadata_path == NULL) {
      metadata_path = argv[i + 1];
    } else if (strcmp(argv[i], "--block-samples") == 0) {
      uint64_t value;
      if (!timeline_uint(argv[i + 1], &value) || value == 0u ||
          value > KWS_MAX_PCM_BLOCK_SAMPLES) return 2;
      block_samples = (size_t)value;
    } else {
      fprintf(stderr, "unknown or duplicate option: %s\n", argv[i]);
      return 2;
    }
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

  if (metadata_path != NULL) {
    timeline = fopen(metadata_path, "rb");
    if (timeline == NULL || !timeline_validate(timeline, wav_bytes / 2u)) {
      fprintf(stderr, "invalid AFE timeline (coverage/sequence/time/config): %s\n", metadata_path);
      goto cleanup;
    }
  }
  remaining = wav_bytes;
  while (remaining != 0u) {
    size_t want_samples = (size_t)(remaining / 2u);
    size_t got_samples;
    kws_detection_t hit;
    int detected = 0;

    if (want_samples > block_samples) want_samples = block_samples;
    if (timeline != NULL) {
      if (output_samples == span.start + span.count) {
        if (timeline_read(timeline, &span) != 1) goto cleanup;
        if ((span.metadata.flags & KWS_FRAME_CLOCK_RESET) != 0u) epoch++;
      }
      if ((uint64_t)want_samples > span.start + span.count - output_samples)
        want_samples = (size_t)(span.start + span.count - output_samples);
    }
    got_samples = fread(pcm, sizeof(pcm[0]), want_samples, wav);
    if (got_samples != want_samples) {
      fprintf(stderr, "truncated WAV data: %s\n", argv[3]);
      goto cleanup;
    }
    remaining -= (uint32_t)(got_samples * sizeof(pcm[0]));
    {
      kws_frame_metadata_t metadata = span.metadata;
      if (timeline != NULL && output_samples != span.start) {
        metadata.flags &= KWS_FRAME_EXTERNAL_VAD_VALID;
        metadata.lost_samples = 0u;
        metadata.capture_timestamp_ns += (output_samples - span.start) * UINT64_C(62500);
      }
      if (kws_engine_accept_pcm16_ex(engine, pcm, got_samples,
              timeline != NULL ? &metadata : NULL, &hit, &detected) != KWS_OK) {
        fprintf(stderr, "KWS runtime error\n");
        goto cleanup;
      }
    }
    output_samples += got_samples;
    if (detected != 0) {
      fputs("{\"recording\":", stdout);
      kws_tool_print_json_string(stdout, argv[4]);
      fprintf(stdout,
              ",\"keyword_id\":%u,\"time_s\":%.6f,\"confidence\":%.6f",
              hit.keyword_id,
              (double)hit.end_sample / (double)KWS_SAMPLE_RATE_HZ,
              (double)hit.confidence);
      if (timeline != NULL) {
        int64_t decision_ns = (int64_t)(span.metadata.capture_timestamp_ns +
            (hit.end_sample - span.start) * UINT64_C(62500));
        int64_t acoustic_ns = decision_ns -
            (int64_t)span.metadata.afe_latency_samples * INT64_C(62500);
        fprintf(stdout, ",\"timing_contract\":\"afe-timeline-v1\","
                "\"output_end_sample\":%" PRIu64 ",\"capture_epoch\":%" PRIu64
                ",\"decision_capture_ns\":%" PRId64 ",\"raw_acoustic_end_ns\":%" PRId64,
                hit.end_sample, epoch, decision_ns, acoustic_ns);
      }
      fputs("}\n", stdout);
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
            "\"max_detection_confidence\":%.9g,"
            "\"external_vad_frames\":%" PRIu64 ","
            "\"discontinuities\":%" PRIu64 ","
            "\"lost_samples\":%" PRIu64 "}\n",
            stats.processed_samples,
            stats.processed_frames,
            stats.speech_frames,
            stats.blank_top1_frames,
            stats.decoder_hits,
            stats.refractory_suppressed,
            stats.detections,
            (int)stats.pending_keyword_index,
            (unsigned)stats.pending_age_frames,
            (double)stats.max_detection_confidence,
            stats.external_vad_frames, stats.discontinuities, stats.lost_samples);
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
  if (timeline != NULL) fclose(timeline);
  if (wav != NULL) {
    fclose(wav);
  }
  free(arena);
  free(pack_blob);
  free(model_blob);
  return exit_code;
}
