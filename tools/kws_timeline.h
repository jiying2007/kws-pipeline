#ifndef KWS_TOOL_TIMELINE_H
#define KWS_TOOL_TIMELINE_H

/* Host-only, opt-in replay adapter. No parser or I/O enters the SDK data plane. */
#include <errno.h>
#include <limits.h>
#include <math.h>

typedef struct kws_timeline_span {
  uint64_t start;
  uint64_t count;
  kws_frame_metadata_t metadata;
} kws_timeline_span_t;

static int timeline_uint(const char *text, uint64_t *value) {
  char *end = NULL;
  unsigned long long parsed;
  if (text[0] < '0' || text[0] > '9') return 0;
  errno = 0;
  parsed = strtoull(text, &end, 10);
  if (errno != 0 || *end != '\0') return 0;
  *value = (uint64_t)parsed;
  return 1;
}

static int timeline_header(FILE *file) {
  char line[128];
  return fgets(line, sizeof(line), file) != NULL &&
         strcmp(line, "kws-afe-timeline-v1\n") == 0;
}

/* 1 = span, 0 = EOF, -1 = malformed. Fixed columns, no implicit defaults. */
static int timeline_read(FILE *file, kws_timeline_span_t *span) {
  char line[512];
  char *columns[10];
  size_t columns_count = 0u;
  char *token;
  uint64_t numbers[8];
  static const size_t integer_columns[] = {0u, 1u, 2u, 3u, 4u, 5u, 7u};
  char *end = NULL;
  double probability;
  if (fgets(line, sizeof(line), file) == NULL) return ferror(file) ? -1 : 0;
  if (strchr(line, '\n') == NULL) return -1;
  token = strtok(line, "\t\r\n ");
  while (token != NULL && columns_count < 10u) {
    columns[columns_count++] = token;
    token = strtok(NULL, "\t\r\n ");
  }
  if (columns_count != 9u || token != NULL) return -1;
  for (size_t i = 0u; i < 7u; ++i) {
    size_t column = integer_columns[i];
    if (!timeline_uint(columns[column], &numbers[column])) return -1;
  }
  if (numbers[1] == 0u || numbers[1] > UINT32_MAX || numbers[4] > 31u ||
      numbers[5] > UINT32_MAX || numbers[7] > UINT32_MAX ||
      strlen(columns[8]) != 64u) return -1;
  memset(span, 0, sizeof(*span));
  span->start = numbers[0];
  span->count = numbers[1];
  span->metadata.struct_size = (uint32_t)sizeof(span->metadata);
  span->metadata.api_version = KWS_FRAME_METADATA_ALIGNED_API_VERSION;
  span->metadata.stream_sequence = numbers[2];
  span->metadata.capture_timestamp_ns = numbers[3];
  span->metadata.flags = (uint32_t)numbers[4];
  span->metadata.lost_samples = (uint32_t)numbers[5];
  span->metadata.afe_latency_samples = (uint32_t)numbers[7];
  probability = strtod(columns[6], &end);
  if (*end != '\0' || !isfinite(probability) || probability < 0.0 ||
      probability > 1.0) return -1;
  /* Validate before float rounding; tiny nonnegative inputs may round to zero. */
  span->metadata.external_vad_probability = (float)probability;
  for (size_t i = 0u; i < 32u; ++i) {
    char hex[3] = {columns[8][2u * i], columns[8][2u * i + 1u], '\0'};
    if (strspn(hex, "0123456789abcdef") != 2u) return -1;
    span->metadata.afe_config_sha256[i] = (uint8_t)strtoul(hex, NULL, 16);
  }
  return 1;
}

static int timeline_validate(FILE *file, uint64_t samples) {
  kws_timeline_span_t previous = {0};
  kws_timeline_span_t span;
  uint64_t end = 0u;
  int have_previous = 0;
  int status;
  if (!timeline_header(file)) return 0;
  while ((status = timeline_read(file, &span)) == 1) {
    uint32_t reset = span.metadata.flags & 15u;
    uint64_t capture = span.metadata.capture_timestamp_ns;
    uint64_t duration = span.count * UINT64_C(62500);
    if (span.start != end || span.count > samples - end ||
        capture > (uint64_t)INT64_MAX - duration ||
        (span.metadata.lost_samples != 0u && reset == 0u)) return 0;
    if (have_previous != 0) {
      uint64_t expected_capture = previous.metadata.capture_timestamp_ns +
          (previous.count + span.metadata.lost_samples) * UINT64_C(62500);
      if ((span.metadata.flags & KWS_FRAME_CLOCK_RESET) == 0u &&
          capture != expected_capture) return 0;
      if (reset == 0u && (previous.metadata.stream_sequence == UINT64_MAX ||
          span.metadata.stream_sequence != previous.metadata.stream_sequence + 1u)) return 0;
      if (reset == 0u &&
          (span.metadata.afe_latency_samples != previous.metadata.afe_latency_samples ||
           memcmp(span.metadata.afe_config_sha256, previous.metadata.afe_config_sha256, 32u) != 0)) return 0;
    }
    previous = span;
    have_previous = 1;
    end += span.count;
  }
  if (status != 0 || end != samples || have_previous == 0) return 0;
  rewind(file);
  return timeline_header(file);
}
#endif
