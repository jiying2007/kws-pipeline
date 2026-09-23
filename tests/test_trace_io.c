#include "kws_trace_io.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x)                                                               \
  do {                                                                         \
    if (!(x)) {                                                                \
      fprintf(stderr, "CHECK failed: %s:%d: %s\n", __FILE__, __LINE__, #x); \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

int main(void) {
  const char *path = "kws_trace_io_test.kwtr";
  kws_trace_header_t header = {0};
  kws_trace_header_t read_header = {0};
  kws_trace_writer_t writer = {0};
  kws_trace_reader_t reader = {0};
  const float first[] = {1.0f, -2.5f, 3.25f, 0.0f};
  const float second[] = {-1.0f, 2.0f, 0.125f, 9.0f};
  float logits[4] = {0};
  uint64_t end_sample = 0u;
  int speech_active = 0;

  header.schema_version = KWS_TRACE_FORMAT_VERSION;
  header.vocab_size = 4u;
  header.frontend_kind = 1u;
  header.sample_rate_hz = 16000u;
  header.frame_length_samples = 512u;
  header.frame_hop_samples = 160u;
  header.vocab_fingerprint = UINT64_C(0x1122334455667788);
  memset(header.model_sha256, 'a', KWS_TRACE_MODEL_SHA256_HEX_LENGTH);
  header.model_sha256[KWS_TRACE_MODEL_SHA256_HEX_LENGTH] = '\0';

  CHECK(kws_trace_writer_open(&writer, path, &header) == 1);
  CHECK(kws_trace_writer_append(&writer, 512u, 1, first, 4u) == 1);
  CHECK(kws_trace_writer_append(&writer, 512u, 1, first, 4u) == 0);
  CHECK(kws_trace_writer_append(&writer, 672u, 0, second, 4u) == 1);
  CHECK(kws_trace_writer_close(&writer) == 1);

  CHECK(kws_trace_reader_open(&reader, path, &read_header) == 1);
  CHECK(read_header.schema_version == KWS_TRACE_FORMAT_VERSION);
  CHECK(read_header.vocab_size == 4u);
  CHECK(read_header.frontend_kind == 1u);
  CHECK(read_header.sample_rate_hz == 16000u);
  CHECK(read_header.frame_length_samples == 512u);
  CHECK(read_header.frame_hop_samples == 160u);
  CHECK(read_header.vocab_fingerprint == UINT64_C(0x1122334455667788));
  CHECK(read_header.frame_count == 2u);
  CHECK(strcmp(read_header.model_sha256, header.model_sha256) == 0);

  CHECK(kws_trace_reader_next(&reader, &end_sample, &speech_active, logits,
                              4u) == 1);
  CHECK(end_sample == 512u);
  CHECK(speech_active == 1);
  for (size_t i = 0u; i < 4u; ++i) {
    CHECK(memcmp(&logits[i], &first[i], sizeof(float)) == 0);
  }

  CHECK(kws_trace_reader_next(&reader, &end_sample, &speech_active, logits,
                              4u) == 1);
  CHECK(end_sample == 672u);
  CHECK(speech_active == 0);
  for (size_t i = 0u; i < 4u; ++i) {
    CHECK(memcmp(&logits[i], &second[i], sizeof(float)) == 0);
  }
  CHECK(kws_trace_reader_next(&reader, &end_sample, &speech_active, logits,
                              4u) == 0);
  CHECK(kws_trace_reader_close(&reader) == 1);
  CHECK(remove(path) == 0);

  puts("kws_trace_io_tests: ok");
  return 0;
}
