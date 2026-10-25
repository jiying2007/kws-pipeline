#include "kws_pipeline/kws.h"
#include "kws_debug.h"
#include "kws_trace_io.h"
#include "sha256.h"
#include "tool_io.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kws_timeline.h"

#define BLOCK_SAMPLES 160u

/* KWTRACE1 stores acoustic frames, not discontinuity/reset events. Reject an
 * unsupported timeline before inference; preserve the legacy no-trace path. */
static int trace_span_supported(const kws_timeline_span_t *span) {
  const uint32_t reset_flags = KWS_FRAME_DISCONTINUITY | KWS_FRAME_XRUN |
      KWS_FRAME_CODEC_REOPEN | KWS_FRAME_CLOCK_RESET;
  return (span->metadata.flags & reset_flags) == 0u &&
         span->metadata.lost_samples == 0u;
}

static int trace_timeline_supported(FILE *timeline) {
  kws_timeline_span_t span;
  int status;
  while ((status = timeline_read(timeline, &span)) == 1) {
    if (!trace_span_supported(&span)) return 0;
  }
  if (status != 0) return 0;
  rewind(timeline);
  return timeline_header(timeline);
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
  uint32_t remaining;
  int16_t pcm[KWS_MAX_PCM_BLOCK_SAMPLES];
  const char *metadata_path = NULL;
  FILE *timeline = NULL;
  kws_timeline_span_t span = {0};
  uint64_t output_samples = 0u;
  uint64_t epoch = 0u;
  size_t block_samples = BLOCK_SAMPLES;
  const char *stats_path = NULL;
  const char *trace_path = NULL;
  kws_trace_writer_t trace_writer = {0};
  uint64_t trace_frames = 0u;
  int trace_open = 0;
  int trace_created = 0;
  FILE *reserved_stats = NULL;
  int stats_created = 0;
  int block_samples_seen = 0;
  int exit_code = 1;

  if (argc < 5 || (argc - 5) % 2 != 0) {
    fprintf(stderr, "usage: %s model.kwm keywords.kwk audio.wav recording-id "
            "[--stats-json path] [--metadata-tsv path] [--block-samples 1..320] "
            "[--trace-out new-path.kwtr]\n", argv[0]);
    return 2;
  }
  for (int i = 5; i < argc; i += 2) {
    if (strcmp(argv[i], "--stats-json") == 0 && stats_path == NULL) {
      stats_path = argv[i + 1];
    } else if (strcmp(argv[i], "--metadata-tsv") == 0 && metadata_path == NULL) {
      metadata_path = argv[i + 1];
    } else if (strcmp(argv[i], "--trace-out") == 0 && trace_path == NULL) {
      trace_path = argv[i + 1];
    } else if (strcmp(argv[i], "--block-samples") == 0 && block_samples_seen == 0) {
      uint64_t value;
      if (!timeline_uint(argv[i + 1], &value) || value == 0u ||
          value > KWS_MAX_PCM_BLOCK_SAMPLES) return 2;
      block_samples = (size_t)value;
      block_samples_seen = 1;
    } else {
      fprintf(stderr, "unknown or duplicate option: %s\n", argv[i]);
      return 2;
    }
  }
  if (trace_path != NULL && stats_path != NULL &&
      strcmp(trace_path, stats_path) == 0) {
    fprintf(stderr, "trace and stats must use separate new destinations\n");
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

  if (metadata_path != NULL) {
    timeline = fopen(metadata_path, "rb");
    if (timeline == NULL || !timeline_validate(timeline, wav_bytes / 2u)) {
      fprintf(stderr, "invalid AFE timeline (coverage/sequence/time/config): %s\n", metadata_path);
      goto cleanup;
    }
    if (trace_path != NULL && !trace_timeline_supported(timeline)) {
      fprintf(stderr, "KWTRACE1 cannot record AFE discontinuities/resets/loss\n");
      goto cleanup;
    }
  }
  if (trace_path != NULL) {
    kws_trace_header_t header = {0};
    /* Reserve both outputs exclusively before PCM execution. This also stops
     * aliased/symlinked stats paths from overwriting the new trace or inputs. */
    if (stats_path != NULL) {
      reserved_stats = fopen(stats_path, "wbx");
      if (reserved_stats == NULL) {
        fprintf(stderr, "trace capture requires a new stats destination\n");
        goto cleanup;
      }
      stats_created = 1;
    }
    header.schema_version = KWS_TRACE_FORMAT_VERSION;
    header.vocab_size = model.vocab_size;
    header.frontend_kind = model.frontend_kind;
    header.sample_rate_hz = model.sample_rate_hz;
    header.frame_length_samples = model.frame_length_samples;
    header.frame_hop_samples = model.frame_hop_samples;
    header.vocab_fingerprint = model.vocab_fingerprint;
    /* Bind the exact blob loaded by this engine, not a second file read. */
    kws_sha256_memory_hex(model_blob, model_bytes, header.model_sha256);
    if (!kws_trace_writer_open_new(&trace_writer, trace_path, &header)) {
      fprintf(stderr, "cannot create new acoustic trace: %s\n", trace_path);
      goto cleanup;
    }
    trace_open = 1;
    trace_created = 1;
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
        if (trace_path != NULL && !trace_span_supported(&span)) goto cleanup;
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
    if (trace_path != NULL) {
      float logits[KWS_MAX_VOCAB_SIZE];
      uint64_t frame_number = 0u;
      uint64_t end_sample = 0u;
      uint16_t vocab_size = 0u;
      int speech_active = 0;
      int copied = kws_engine_debug_copy_last_frame(
          engine, &frame_number, &end_sample, &speech_active, logits,
          KWS_MAX_VOCAB_SIZE, &vocab_size);
      /* Caller chunks are at most one model hop, so no frame may be skipped. */
      if (copied < 0 || (copied != 0 && frame_number != trace_frames &&
          (frame_number != trace_frames + 1u ||
           !kws_trace_writer_append(&trace_writer, end_sample, speech_active,
                                    logits, vocab_size)))) {
        fprintf(stderr, "cannot append complete acoustic trace\n");
        goto cleanup;
      }
      if (copied != 0) trace_frames = frame_number;
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
    stats_file = reserved_stats != NULL ? reserved_stats : fopen(stats_path, "wb");
    reserved_stats = NULL;
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
  if (trace_path != NULL) {
    uint64_t samples = (uint64_t)wav_bytes / 2u;
    uint64_t expected_frames = samples < model.frame_length_samples ? 0u :
        1u + (samples - model.frame_length_samples) / model.frame_hop_samples;
    if (trace_frames != expected_frames) {
      fprintf(stderr, "acoustic trace frame coverage mismatch\n");
      goto cleanup;
    }
    trace_open = 0;
    if (!kws_trace_writer_close(&trace_writer)) {
      fprintf(stderr, "cannot finalize acoustic trace\n");
      goto cleanup;
    }
  }
  if (fflush(stdout) != 0 || ferror(stdout) != 0) {
    fprintf(stderr, "cannot write stdout\n");
    goto cleanup;
  }
  exit_code = 0;

cleanup:
  if (trace_open != 0) (void)kws_trace_writer_close(&trace_writer);
  if (reserved_stats != NULL) (void)fclose(reserved_stats);
  if (exit_code != 0) {
    if (trace_created != 0) (void)remove(trace_path);
    if (stats_created != 0) (void)remove(stats_path);
  }
  if (timeline != NULL) fclose(timeline);
  if (wav != NULL) {
    fclose(wav);
  }
  free(arena);
  free(pack_blob);
  free(model_blob);
  return exit_code;
}
