#include "kws_trace_io.h"

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define KWS_TRACE_HEADER_BYTES 112L
#define KWS_TRACE_FRAME_COUNT_OFFSET 40L

static const uint8_t KWS_TRACE_MAGIC[8] = {
    'K', 'W', 'T', 'R', 'A', 'C', 'E', '1',
};

static int write_bytes(FILE *stream, const void *data, size_t bytes) {
  return fwrite(data, 1u, bytes, stream) == bytes;
}

static int read_bytes(FILE *stream, void *data, size_t bytes) {
  return fread(data, 1u, bytes, stream) == bytes;
}

static int write_u16_le(FILE *stream, uint16_t value) {
  uint8_t data[2];
  data[0] = (uint8_t)(value & 0xffu);
  data[1] = (uint8_t)(value >> 8u);
  return write_bytes(stream, data, sizeof(data));
}

static int write_u32_le(FILE *stream, uint32_t value) {
  uint8_t data[4];
  data[0] = (uint8_t)(value & 0xffu);
  data[1] = (uint8_t)((value >> 8u) & 0xffu);
  data[2] = (uint8_t)((value >> 16u) & 0xffu);
  data[3] = (uint8_t)(value >> 24u);
  return write_bytes(stream, data, sizeof(data));
}

static int write_u64_le(FILE *stream, uint64_t value) {
  return write_u32_le(stream, (uint32_t)(value & UINT64_C(0xffffffff))) &&
         write_u32_le(stream, (uint32_t)(value >> 32u));
}

static int write_f32_le(FILE *stream, float value) {
  uint32_t bits = 0u;
  memcpy(&bits, &value, sizeof(bits));
  return write_u32_le(stream, bits);
}

static int read_u16_le(FILE *stream, uint16_t *out) {
  uint8_t data[2];
  if (!read_bytes(stream, data, sizeof(data))) {
    return 0;
  }
  *out = (uint16_t)((uint16_t)data[0] | ((uint16_t)data[1] << 8u));
  return 1;
}

static int read_u32_le(FILE *stream, uint32_t *out) {
  uint8_t data[4];
  if (!read_bytes(stream, data, sizeof(data))) {
    return 0;
  }
  *out = (uint32_t)data[0] | ((uint32_t)data[1] << 8u) |
         ((uint32_t)data[2] << 16u) | ((uint32_t)data[3] << 24u);
  return 1;
}

static int read_u64_le(FILE *stream, uint64_t *out) {
  uint32_t low = 0u;
  uint32_t high = 0u;
  if (!read_u32_le(stream, &low) || !read_u32_le(stream, &high)) {
    return 0;
  }
  *out = (uint64_t)low | ((uint64_t)high << 32u);
  return 1;
}

static int read_f32_le(FILE *stream, float *out) {
  uint32_t bits = 0u;
  if (!read_u32_le(stream, &bits)) {
    return 0;
  }
  memcpy(out, &bits, sizeof(bits));
  return 1;
}

static int sha256_hex_valid(const char *value) {
  if (value == NULL || strlen(value) != KWS_TRACE_MODEL_SHA256_HEX_LENGTH) {
    return 0;
  }
  for (size_t i = 0u; i < KWS_TRACE_MODEL_SHA256_HEX_LENGTH; ++i) {
    unsigned char ch = (unsigned char)value[i];
    if (isdigit(ch) == 0 && (ch < (unsigned char)'a' || ch > (unsigned char)'f')) {
      return 0;
    }
  }
  return 1;
}

static int header_valid(const kws_trace_header_t *header) {
  return header != NULL && header->schema_version == KWS_TRACE_FORMAT_VERSION &&
         header->vocab_size >= 2u && header->sample_rate_hz != 0u &&
         header->frame_length_samples != 0u && header->frame_hop_samples != 0u &&
         header->vocab_fingerprint != 0u &&
         sha256_hex_valid(header->model_sha256) != 0;
}

static int write_header(FILE *stream, const kws_trace_header_t *header) {
  uint8_t reserved[8] = {0};
  if (!write_bytes(stream, KWS_TRACE_MAGIC, sizeof(KWS_TRACE_MAGIC)) ||
      !write_u32_le(stream, header->schema_version) ||
      !write_u16_le(stream, header->vocab_size) ||
      !write_u16_le(stream, header->frontend_kind) ||
      !write_u32_le(stream, header->sample_rate_hz) ||
      !write_u32_le(stream, header->frame_length_samples) ||
      !write_u32_le(stream, header->frame_hop_samples) ||
      !write_u32_le(stream, 0u) ||
      !write_u64_le(stream, header->vocab_fingerprint) ||
      !write_u64_le(stream, header->frame_count) ||
      !write_bytes(stream, header->model_sha256,
                   KWS_TRACE_MODEL_SHA256_HEX_LENGTH) ||
      !write_bytes(stream, reserved, sizeof(reserved))) {
    return 0;
  }
  return ftell(stream) == KWS_TRACE_HEADER_BYTES;
}

static int read_header(FILE *stream, kws_trace_header_t *header) {
  uint8_t magic[8];
  uint32_t reserved32 = 0u;
  uint8_t reserved[8];
  if (!read_bytes(stream, magic, sizeof(magic)) ||
      memcmp(magic, KWS_TRACE_MAGIC, sizeof(magic)) != 0 ||
      !read_u32_le(stream, &header->schema_version) ||
      !read_u16_le(stream, &header->vocab_size) ||
      !read_u16_le(stream, &header->frontend_kind) ||
      !read_u32_le(stream, &header->sample_rate_hz) ||
      !read_u32_le(stream, &header->frame_length_samples) ||
      !read_u32_le(stream, &header->frame_hop_samples) ||
      !read_u32_le(stream, &reserved32) ||
      !read_u64_le(stream, &header->vocab_fingerprint) ||
      !read_u64_le(stream, &header->frame_count) ||
      !read_bytes(stream, header->model_sha256,
                  KWS_TRACE_MODEL_SHA256_HEX_LENGTH) ||
      !read_bytes(stream, reserved, sizeof(reserved))) {
    return 0;
  }
  header->model_sha256[KWS_TRACE_MODEL_SHA256_HEX_LENGTH] = '\0';
  if (reserved32 != 0u) {
    return 0;
  }
  for (size_t i = 0u; i < sizeof(reserved); ++i) {
    if (reserved[i] != 0u) {
      return 0;
    }
  }
  return ftell(stream) == KWS_TRACE_HEADER_BYTES && header_valid(header);
}

int kws_trace_writer_open(kws_trace_writer_t *writer,
                          const char *path,
                          const kws_trace_header_t *header) {
  if (writer == NULL || path == NULL || !header_valid(header)) {
    return 0;
  }
  memset(writer, 0, sizeof(*writer));
  writer->header = *header;
  writer->header.frame_count = 0u;
  writer->stream = fopen(path, "wb+");
  if (writer->stream == NULL || !write_header(writer->stream, &writer->header)) {
    if (writer->stream != NULL) {
      fclose(writer->stream);
    }
    memset(writer, 0, sizeof(*writer));
    return 0;
  }
  return 1;
}

int kws_trace_writer_append(kws_trace_writer_t *writer,
                            uint64_t end_sample,
                            int speech_active,
                            const float *logits,
                            uint16_t vocab_size) {
  uint8_t flags[8] = {0};
  if (writer == NULL || writer->stream == NULL || logits == NULL ||
      vocab_size != writer->header.vocab_size ||
      (speech_active != 0 && speech_active != 1) ||
      (writer->frames_written != 0u &&
       end_sample <= writer->header.frame_hop_samples) ) {
    return 0;
  }
  if (writer->frames_written != 0u) {
    long record_bytes =
        16L + (long)writer->header.vocab_size * (long)sizeof(float);
    long expected =
        KWS_TRACE_HEADER_BYTES + (long)writer->frames_written * record_bytes;
    if (ftell(writer->stream) != expected) {
      return 0;
    }
  }
  flags[0] = (uint8_t)speech_active;
  if (!write_u64_le(writer->stream, end_sample) ||
      !write_bytes(writer->stream, flags, sizeof(flags))) {
    return 0;
  }
  for (uint16_t v = 0u; v < vocab_size; ++v) {
    if (!isfinite(logits[v]) || !write_f32_le(writer->stream, logits[v])) {
      return 0;
    }
  }
  writer->frames_written++;
  return 1;
}

int kws_trace_writer_close(kws_trace_writer_t *writer) {
  int ok = 1;
  if (writer == NULL || writer->stream == NULL) {
    return 0;
  }
  if (fseek(writer->stream, KWS_TRACE_FRAME_COUNT_OFFSET, SEEK_SET) != 0 ||
      !write_u64_le(writer->stream, writer->frames_written) ||
      fflush(writer->stream) != 0 || ferror(writer->stream) != 0) {
    ok = 0;
  }
  if (fclose(writer->stream) != 0) {
    ok = 0;
  }
  writer->stream = NULL;
  writer->header.frame_count = writer->frames_written;
  return ok;
}

int kws_trace_reader_open(kws_trace_reader_t *reader,
                          const char *path,
                          kws_trace_header_t *out_header) {
  if (reader == NULL || path == NULL) {
    return 0;
  }
  memset(reader, 0, sizeof(*reader));
  reader->stream = fopen(path, "rb");
  if (reader->stream == NULL || !read_header(reader->stream, &reader->header)) {
    if (reader->stream != NULL) {
      fclose(reader->stream);
    }
    memset(reader, 0, sizeof(*reader));
    return 0;
  }
  if (out_header != NULL) {
    *out_header = reader->header;
  }
  return 1;
}

int kws_trace_reader_next(kws_trace_reader_t *reader,
                          uint64_t *out_end_sample,
                          int *out_speech_active,
                          float *out_logits,
                          size_t logits_capacity) {
  uint8_t flags[8];
  uint64_t end_sample = 0u;
  if (reader == NULL || reader->stream == NULL || out_end_sample == NULL ||
      out_speech_active == NULL || out_logits == NULL ||
      logits_capacity < (size_t)reader->header.vocab_size) {
    return -1;
  }
  if (reader->frames_read == reader->header.frame_count) {
    return 0;
  }
  if (!read_u64_le(reader->stream, &end_sample) ||
      !read_bytes(reader->stream, flags, sizeof(flags)) ||
      (flags[0] != 0u && flags[0] != 1u)) {
    return -1;
  }
  for (size_t i = 1u; i < sizeof(flags); ++i) {
    if (flags[i] != 0u) {
      return -1;
    }
  }
  for (uint16_t v = 0u; v < reader->header.vocab_size; ++v) {
    if (!read_f32_le(reader->stream, &out_logits[v]) ||
        !isfinite(out_logits[v])) {
      return -1;
    }
  }
  *out_end_sample = end_sample;
  *out_speech_active = (int)flags[0];
  reader->frames_read++;
  return 1;
}

int kws_trace_reader_close(kws_trace_reader_t *reader) {
  int ok = 1;
  if (reader == NULL || reader->stream == NULL) {
    return 0;
  }
  if (reader->frames_read != reader->header.frame_count) {
    ok = 0;
  }
  if (fgetc(reader->stream) != EOF) {
    ok = 0;
  }
  if (fclose(reader->stream) != 0) {
    ok = 0;
  }
  reader->stream = NULL;
  return ok;
}
