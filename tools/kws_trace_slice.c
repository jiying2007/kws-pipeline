#include "kws_trace_io.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int parse_u64(const char *text, uint64_t *out) {
  char *end = NULL;
  unsigned long long value;
  errno = 0;
  value = strtoull(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0') {
    return 0;
  }
  *out = (uint64_t)value;
  return 1;
}

static int slice_trace(const char *input,
                       const char *output,
                       uint64_t start_sample_exclusive,
                       uint64_t end_sample_inclusive,
                       uint64_t *out_frames) {
  kws_trace_reader_t reader = {0};
  kws_trace_writer_t writer = {0};
  kws_trace_header_t header = {0};
  float logits[256];
  uint64_t end_sample = 0u;
  uint64_t selected = 0u;
  int speech_active = 0;
  int reader_open = 0;
  int writer_open = 0;
  int ok = 0;

  if (input == NULL || output == NULL || out_frames == NULL ||
      start_sample_exclusive >= end_sample_inclusive) {
    return 0;
  }
  if (!kws_trace_reader_open(&reader, input, &header)) {
    return 0;
  }
  reader_open = 1;
  if (header.vocab_size > (uint16_t)(sizeof(logits) / sizeof(logits[0])) ||
      !kws_trace_writer_open(&writer, output, &header)) {
    goto cleanup;
  }
  writer_open = 1;

  for (;;) {
    int status = kws_trace_reader_next(&reader, &end_sample, &speech_active,
                                       logits,
                                       sizeof(logits) / sizeof(logits[0]));
    if (status == 0) {
      break;
    }
    if (status < 0) {
      goto cleanup;
    }
    if (end_sample > start_sample_exclusive &&
        end_sample <= end_sample_inclusive) {
      if (!kws_trace_writer_append(&writer, end_sample, speech_active, logits,
                                   header.vocab_size)) {
        goto cleanup;
      }
      selected++;
    }
  }
  if (!kws_trace_reader_close(&reader)) {
    reader_open = 0;
    goto cleanup;
  }
  reader_open = 0;
  if (selected == 0u || !kws_trace_writer_close(&writer)) {
    writer_open = 0;
    goto cleanup;
  }
  writer_open = 0;
  *out_frames = selected;
  ok = 1;

cleanup:
  if (reader_open != 0) {
    (void)kws_trace_reader_close(&reader);
  }
  if (writer_open != 0) {
    (void)kws_trace_writer_close(&writer);
  }
  if (ok == 0) {
    (void)remove(output);
  }
  return ok;
}

static int self_test(void) {
  const char *source = "kws_trace_slice_source.kwtr";
  const char *sliced = "kws_trace_slice_output.kwtr";
  kws_trace_header_t header = {0};
  kws_trace_writer_t writer = {0};
  kws_trace_reader_t reader = {0};
  kws_trace_header_t sliced_header = {0};
  const float logits[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float read_logits[4] = {0};
  const uint64_t ends[4] = {160u, 320u, 480u, 640u};
  uint64_t frames = 0u;
  uint64_t end_sample = 0u;
  int speech_active = 0;
  int ok = 0;

  header.schema_version = KWS_TRACE_FORMAT_VERSION;
  header.vocab_size = 4u;
  header.frontend_kind = 1u;
  header.sample_rate_hz = 16000u;
  header.frame_length_samples = 512u;
  header.frame_hop_samples = 160u;
  header.vocab_fingerprint = UINT64_C(0x1020304050607080);
  memset(header.model_sha256, 'a', KWS_TRACE_MODEL_SHA256_HEX_LENGTH);
  header.model_sha256[KWS_TRACE_MODEL_SHA256_HEX_LENGTH] = '\0';

  if (!kws_trace_writer_open(&writer, source, &header)) {
    goto cleanup;
  }
  for (size_t i = 0u; i < 4u; ++i) {
    if (!kws_trace_writer_append(&writer, ends[i], (int)(i & 1u), logits, 4u)) {
      goto cleanup_writer;
    }
  }
  if (!kws_trace_writer_close(&writer)) {
    goto cleanup;
  }
  memset(&writer, 0, sizeof(writer));

  if (!slice_trace(source, sliced, 200u, 500u, &frames) || frames != 2u) {
    goto cleanup;
  }
  if (!kws_trace_reader_open(&reader, sliced, &sliced_header) ||
      sliced_header.frame_count != 2u) {
    goto cleanup;
  }
  if (kws_trace_reader_next(&reader, &end_sample, &speech_active,
                            read_logits, 4u) != 1 ||
      end_sample != 320u || speech_active != 1) {
    goto cleanup_reader;
  }
  if (kws_trace_reader_next(&reader, &end_sample, &speech_active,
                            read_logits, 4u) != 1 ||
      end_sample != 480u || speech_active != 0) {
    goto cleanup_reader;
  }
  if (kws_trace_reader_next(&reader, &end_sample, &speech_active,
                            read_logits, 4u) != 0 ||
      !kws_trace_reader_close(&reader)) {
    memset(&reader, 0, sizeof(reader));
    goto cleanup;
  }
  memset(&reader, 0, sizeof(reader));
  ok = 1;
  goto cleanup;

cleanup_writer:
  (void)kws_trace_writer_close(&writer);
  memset(&writer, 0, sizeof(writer));
  goto cleanup;

cleanup_reader:
  (void)kws_trace_reader_close(&reader);
  memset(&reader, 0, sizeof(reader));

cleanup:
  (void)remove(source);
  (void)remove(sliced);
  return ok;
}

int main(int argc, char **argv) {
  uint64_t start_sample = 0u;
  uint64_t end_sample = 0u;
  uint64_t frames = 0u;

  if (argc == 2 && strcmp(argv[1], "--self-test") == 0) {
    if (!self_test()) {
      fprintf(stderr, "kws_trace_slice self-test failed\n");
      return 1;
    }
    puts("kws_trace_slice self-test: PASS");
    return 0;
  }
  if (argc != 5 || !parse_u64(argv[3], &start_sample) ||
      !parse_u64(argv[4], &end_sample) || start_sample >= end_sample) {
    fprintf(stderr,
            "usage: %s input.kwtr output.kwtr "
            "start_sample_exclusive end_sample_inclusive\n",
            argv[0]);
    return 2;
  }
  if (!slice_trace(argv[1], argv[2], start_sample, end_sample, &frames)) {
    fprintf(stderr, "cannot slice acoustic trace\n");
    return 1;
  }
  printf("{\"frames\":%llu,\"start_sample_exclusive\":%llu,"
         "\"end_sample_inclusive\":%llu}\n",
         (unsigned long long)frames,
         (unsigned long long)start_sample,
         (unsigned long long)end_sample);
  return ferror(stdout) != 0 ? 1 : 0;
}
