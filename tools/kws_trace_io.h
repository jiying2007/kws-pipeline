#ifndef KWS_PIPELINE_TRACE_IO_H
#define KWS_PIPELINE_TRACE_IO_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define KWS_TRACE_FORMAT_VERSION 1u
#define KWS_TRACE_MODEL_SHA256_HEX_LENGTH 64u

typedef struct kws_trace_header {
  uint32_t schema_version;
  uint16_t vocab_size;
  uint16_t frontend_kind;
  uint32_t sample_rate_hz;
  uint32_t frame_length_samples;
  uint32_t frame_hop_samples;
  uint64_t vocab_fingerprint;
  uint64_t frame_count;
  char model_sha256[KWS_TRACE_MODEL_SHA256_HEX_LENGTH + 1u];
} kws_trace_header_t;

typedef struct kws_trace_writer {
  FILE *stream;
  kws_trace_header_t header;
  uint64_t frames_written;
} kws_trace_writer_t;

typedef struct kws_trace_reader {
  FILE *stream;
  kws_trace_header_t header;
  uint64_t frames_read;
} kws_trace_reader_t;

int kws_trace_writer_open(kws_trace_writer_t *writer,
                          const char *path,
                          const kws_trace_header_t *header);
int kws_trace_writer_append(kws_trace_writer_t *writer,
                            uint64_t end_sample,
                            int speech_active,
                            const float *logits,
                            uint16_t vocab_size);
int kws_trace_writer_close(kws_trace_writer_t *writer);

int kws_trace_reader_open(kws_trace_reader_t *reader,
                          const char *path,
                          kws_trace_header_t *out_header);
int kws_trace_reader_next(kws_trace_reader_t *reader,
                          uint64_t *out_end_sample,
                          int *out_speech_active,
                          float *out_logits,
                          size_t logits_capacity);
int kws_trace_reader_close(kws_trace_reader_t *reader);

#endif
