#include "kws_pipeline/kws.h"
#include "kws_debug.h"
#include "kws_trace_io.h"
#include "sha256.h"
#include "tool_io.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BLOCK_SAMPLES 160u

int main(int argc, char **argv) {
  uint8_t *model_blob = NULL;
  size_t model_bytes = 0u;
  kws_model_t model;
  kws_engine_t *engine = NULL;
  void *arena = NULL;
  FILE *wav = NULL;
  uint32_t wav_bytes = 0u;
  long wav_data_offset = 0L;
  uint32_t remaining;
  int16_t pcm[BLOCK_SAMPLES];
  float logits[KWS_MAX_VOCAB_SIZE];
  uint64_t last_frame_number = 0u;
  char model_sha256[65];
  kws_trace_header_t header = {0};
  kws_trace_writer_t writer = {0};
  int writer_open = 0;
  int exit_code = 1;

  if (argc != 4) {
    fprintf(stderr, "usage: %s model.kwm audio.wav trace.kwtr\n", argv[0]);
    return 2;
  }
  if (kws_tool_read_file(argv[1], &model_blob, &model_bytes) == 0 ||
      kws_model_open(model_blob, model_bytes, &model) != KWS_OK ||
      kws_sha256_file_hex(argv[1], model_sha256) == 0) {
    fprintf(stderr, "cannot open/hash model: %s\n", argv[1]);
    goto cleanup;
  }

  arena = malloc(kws_engine_required_bytes(&model));
  if (arena == NULL ||
      kws_engine_init(arena, kws_engine_required_bytes(&model), &model, NULL,
                      &engine) != KWS_OK) {
    fprintf(stderr, "cannot initialize KWS engine\n");
    goto cleanup;
  }

  header.schema_version = KWS_TRACE_FORMAT_VERSION;
  header.vocab_size = model.vocab_size;
  header.frontend_kind = model.frontend_kind;
  header.sample_rate_hz = model.sample_rate_hz;
  header.frame_length_samples = model.frame_length_samples;
  header.frame_hop_samples = model.frame_hop_samples;
  header.vocab_fingerprint = model.vocab_fingerprint;
  memcpy(header.model_sha256, model_sha256, sizeof(model_sha256));
  if (!kws_trace_writer_open(&writer, argv[3], &header)) {
    fprintf(stderr, "cannot create trace: %s\n", argv[3]);
    goto cleanup;
  }
  writer_open = 1;

  wav = fopen(argv[2], "rb");
  if (wav == NULL ||
      kws_tool_open_wav(wav, &wav_bytes, &wav_data_offset) == 0 ||
      fseek(wav, wav_data_offset, SEEK_SET) != 0) {
    fprintf(stderr, "expected mono 16-kHz PCM16 WAV: %s\n", argv[2]);
    goto cleanup;
  }

  remaining = wav_bytes;
  while (remaining != 0u) {
    size_t want_samples = (size_t)(remaining / 2u);
    size_t got_samples;
    int detected = 0;
    uint64_t frame_number = 0u;
    uint64_t end_sample = 0u;
    uint16_t vocab_size = 0u;
    int speech_active = 0;
    int copied;

    if (want_samples > BLOCK_SAMPLES) {
      want_samples = BLOCK_SAMPLES;
    }
    got_samples = fread(pcm, sizeof(pcm[0]), want_samples, wav);
    if (got_samples != want_samples) {
      fprintf(stderr, "truncated WAV data: %s\n", argv[2]);
      goto cleanup;
    }
    remaining -= (uint32_t)(got_samples * sizeof(pcm[0]));
    if (kws_engine_accept_pcm16(engine, pcm, got_samples, NULL, &detected) !=
        KWS_OK) {
      fprintf(stderr, "KWS acoustic runtime error\n");
      goto cleanup;
    }

    copied = kws_engine_debug_copy_last_frame(
        engine, &frame_number, &end_sample, &speech_active, logits,
        KWS_MAX_VOCAB_SIZE, &vocab_size);
    if (copied < 0) {
      fprintf(stderr, "cannot read KWS acoustic frame\n");
      goto cleanup;
    }
    if (copied != 0 && frame_number != last_frame_number) {
      if (frame_number != last_frame_number + 1u ||
          !kws_trace_writer_append(&writer, end_sample, speech_active, logits,
                                   vocab_size)) {
        fprintf(stderr, "cannot append acoustic trace frame\n");
        goto cleanup;
      }
      last_frame_number = frame_number;
    }
  }

  if (last_frame_number == 0u || !kws_trace_writer_close(&writer)) {
    writer_open = 0;
    fprintf(stderr, "cannot finalize non-empty acoustic trace\n");
    goto cleanup;
  }
  writer_open = 0;
  {
    char trace_sha256[65];
    if (kws_sha256_file_hex(argv[3], trace_sha256) == 0) {
      fprintf(stderr, "cannot hash finalized trace: %s\n", argv[3]);
      goto cleanup;
    }
    printf(
        "{\"schema_version\":1,\"evidence_class\":"
        "\"kws-posterior-trace-v1\",\"model_sha256\":\"%s\","
        "\"trace_sha256\":\"%s\",\"frames\":%llu,"
        "\"vocab_size\":%u}\n",
        model_sha256, trace_sha256, (unsigned long long)last_frame_number,
        (unsigned)model.vocab_size);
  }
  exit_code = ferror(stdout) != 0 ? 1 : 0;

cleanup:
  if (writer_open != 0) {
    (void)kws_trace_writer_close(&writer);
  }
  if (wav != NULL) {
    fclose(wav);
  }
  if (exit_code != 0 && argc >= 4) {
    (void)remove(argv[3]);
  }
  free(arena);
  free(model_blob);
  return exit_code;
}
