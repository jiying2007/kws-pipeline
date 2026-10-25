#include "tool_io.h"

#include "kws_pipeline/kws.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static uint16_t rd16(const uint8_t *p) {
  return (uint16_t)((uint16_t)p[0] | (uint16_t)((uint16_t)p[1] << 8u));
}

static uint32_t rd32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8u) |
         ((uint32_t)p[2] << 16u) | ((uint32_t)p[3] << 24u);
}

int kws_tool_read_file(const char *path, uint8_t **out_data, size_t *out_bytes) {
  FILE *stream;
  long size;
  uint8_t *data;

  if (path == NULL || out_data == NULL || out_bytes == NULL) {
    return 0;
  }
  *out_data = NULL;
  *out_bytes = 0u;

  stream = fopen(path, "rb");
  if (stream == NULL) {
    return 0;
  }
  if (fseek(stream, 0, SEEK_END) != 0) {
    fclose(stream);
    return 0;
  }
  size = ftell(stream);
  if (size <= 0 || fseek(stream, 0, SEEK_SET) != 0) {
    fclose(stream);
    return 0;
  }
  data = (uint8_t *)malloc((size_t)size);
  if (data == NULL) {
    fclose(stream);
    return 0;
  }
  if (fread(data, 1u, (size_t)size, stream) != (size_t)size) {
    free(data);
    fclose(stream);
    return 0;
  }
  fclose(stream);
  *out_data = data;
  *out_bytes = (size_t)size;
  return 1;
}

int kws_tool_open_wav(FILE *stream,
                      uint32_t *out_data_bytes,
                      long *out_data_offset) {
  uint8_t header[12];
  int have_fmt = 0;

  if (stream == NULL || out_data_bytes == NULL || out_data_offset == NULL) {
    return 0;
  }
  if (fread(header, 1u, sizeof(header), stream) != sizeof(header) ||
      memcmp(header, "RIFF", 4u) != 0 || memcmp(header + 8u, "WAVE", 4u) != 0) {
    return 0;
  }

  for (;;) {
    uint8_t chunk[8];
    uint32_t size;
    long skip;

    if (fread(chunk, 1u, sizeof(chunk), stream) != sizeof(chunk)) {
      return 0;
    }
    size = rd32(chunk + 4u);
    if (memcmp(chunk, "fmt ", 4u) == 0) {
      uint8_t fmt[16];
      if (size < sizeof(fmt) || fread(fmt, 1u, sizeof(fmt), stream) != sizeof(fmt)) {
        return 0;
      }
      if (rd16(fmt + 0u) != 1u || rd16(fmt + 2u) != 1u ||
          rd32(fmt + 4u) != KWS_SAMPLE_RATE_HZ || rd16(fmt + 14u) != 16u) {
        return 0;
      }
      have_fmt = 1;
      skip = (long)(size - (uint32_t)sizeof(fmt));
      if ((size & 1u) != 0u) {
        skip += 1L;
      }
      if (skip > 0L && fseek(stream, skip, SEEK_CUR) != 0) {
        return 0;
      }
    } else if (memcmp(chunk, "data", 4u) == 0) {
      long offset;
      if (have_fmt == 0 || (size & 1u) != 0u) {
        return 0;
      }
      offset = ftell(stream);
      if (offset < 0L) {
        return 0;
      }
      *out_data_bytes = size;
      *out_data_offset = offset;
      return 1;
    } else {
      skip = (long)size + (((size & 1u) != 0u) ? 1L : 0L);
      if (skip > 0L && fseek(stream, skip, SEEK_CUR) != 0) {
        return 0;
      }
    }
  }
}

void kws_tool_print_json_string(FILE *stream, const char *text) {
  const unsigned char *p;

  if (stream == NULL || text == NULL) {
    return;
  }
  p = (const unsigned char *)(const void *)text;
  fputc('"', stream);
  while (*p != 0u) {
    unsigned char ch = *p++;
    if (ch == (unsigned char)'"' || ch == (unsigned char)'\\') {
      fputc('\\', stream);
      fputc((int)ch, stream);
    } else if (ch == (unsigned char)'\b') {
      fputs("\\b", stream);
    } else if (ch == (unsigned char)'\f') {
      fputs("\\f", stream);
    } else if (ch == (unsigned char)'\n') {
      fputs("\\n", stream);
    } else if (ch == (unsigned char)'\r') {
      fputs("\\r", stream);
    } else if (ch == (unsigned char)'\t') {
      fputs("\\t", stream);
    } else if (ch < 0x20u) {
      fprintf(stream, "\\u%04x", (unsigned)ch);
    } else {
      fputc((int)ch, stream);
    }
  }
  fputc('"', stream);
}
