#include "tool_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
  fprintf(stderr, "CHECK failed at line %d: %s\n", __LINE__, #x); \
  exit(1); } } while (0)

static unsigned cases;
static unsigned failures;

static void wr32(unsigned char *p, uint32_t value) {
  for (unsigned i = 0u; i < 4u; ++i) p[i] = (unsigned char)(value >> (8u * i));
}

static void check_wav(const char *name, const unsigned char *data, size_t size,
                      int accepted, uint32_t expected_bytes,
                      long expected_offset, size_t available_bytes) {
  FILE *file = tmpfile();
  uint32_t bytes = 0u;
  long offset = 0L;
  unsigned char pcm[4];
  int actual;
  CHECK(file != NULL);
  CHECK(fwrite(data, 1u, size, file) == size);
  CHECK(fseek(file, 0L, SEEK_SET) == 0);
  actual = kws_tool_open_wav(file, &bytes, &offset);
  ++cases;
  if (actual != accepted) {
    fprintf(stderr, "%s: accepted=%d expected=%d\n", name, actual, accepted);
    ++failures;
  } else if (accepted != 0) {
    CHECK(bytes == expected_bytes && offset == expected_offset);
    CHECK(fseek(file, offset, SEEK_SET) == 0);
    CHECK(bytes <= sizeof(pcm));
    CHECK(fread(pcm, 1u, bytes, file) == available_bytes);
    CHECK(memcmp(pcm, data + (size_t)offset, available_bytes) == 0);
  }
  CHECK(fclose(file) == 0);
}

static void check_header(const unsigned char header[44], int accepted) {
  check_wav("PCM layout", header, 44u, accepted, 0u, 44L, 0u);
}

static void check_riff_bounds(const unsigned char empty[44]) {
  unsigned char pcm[48];
  unsigned char changed[80];
  static const uint32_t short_sizes[] = {0u, 3u, 4u, 6u, 20u, 32u, 36u, 39u};
  memcpy(pcm, empty, 44u);
  wr32(pcm + 4u, 40u);
  wr32(pcm + 40u, 4u);
  pcm[44] = 0x34u; pcm[45] = 0x12u; pcm[46] = 0xccu; pcm[47] = 0xedu;
  check_wav("canonical PCM", pcm, sizeof(pcm), 1, 4u, 44L, 4u);
  for (size_t i = 0u; i < sizeof(short_sizes) / sizeof(short_sizes[0]); ++i) {
    memcpy(changed, pcm, sizeof(pcm));
    wr32(changed + 4u, short_sizes[i]);
    check_wav("RIFF boundary before chunk end", changed, sizeof(pcm), 0,
              0u, 0L, 0u);
  }

  memcpy(changed, pcm, sizeof(pcm));
  wr32(changed + 16u, UINT32_MAX);
  check_wav("fmt padded size overflow", changed, sizeof(pcm), 0, 0u, 0L, 0u);
  memcpy(changed + 12u, "JUNK", 4u);
  check_wav("unknown padded size overflow", changed, sizeof(pcm), 0, 0u, 0L, 0u);
  memcpy(changed, pcm, sizeof(pcm));
  wr32(changed + 40u, UINT32_MAX - 1u);
  check_wav("data exceeds RIFF", changed, sizeof(pcm), 0, 0u, 0L, 0u);

  /* Unknown odd-sized chunks include their pad byte in the RIFF bound. */
  memcpy(changed, pcm, 12u);
  memcpy(changed + 12u, "JUNK", 4u); wr32(changed + 16u, 3u);
  memcpy(changed + 20u, "abc", 3u); changed[23] = 0u;
  memcpy(changed + 24u, pcm + 12u, 36u); wr32(changed + 4u, 52u);
  check_wav("odd unknown with padding", changed, 60u, 1, 4u, 56L, 4u);
  wr32(changed + 4u, 15u);
  check_wav("unknown padding crosses RIFF", changed, 60u, 0, 0u, 0L, 0u);

  memcpy(changed, pcm, 36u); wr32(changed + 4u, 42u);
  wr32(changed + 16u, 17u); changed[36] = 0u; changed[37] = 0u;
  memcpy(changed + 38u, pcm + 36u, 12u);
  check_wav("extended fmt with padding", changed, 50u, 1, 4u, 46L, 4u);
  wr32(changed + 4u, 29u);
  check_wav("fmt padding crosses RIFF", changed, 50u, 0, 0u, 0L, 0u);

  memcpy(changed, pcm, sizeof(pcm));
  memcpy(changed + 48u, "tail", 4u);
  check_wav("legal bytes after RIFF", changed, 52u, 1, 4u, 44L, 4u);
  /* Preserve first-data-chunk behavior: no new full-file/trailer scan. */
  memcpy(changed + 48u, "JUNK", 4u); wr32(changed + 52u, UINT32_MAX);
  wr32(changed + 4u, 48u);
  check_wav("later chunks not scanned", changed, 56u, 1, 4u, 44L, 4u);
  memcpy(changed, pcm, sizeof(pcm)); wr32(changed + 4u, UINT32_MAX);
  check_wav("overdeclared RIFF keeps first-data policy", changed, 48u,
            1, 4u, 44L, 4u);
  /* Data truncation is still detected by callers while reading PCM. */
  check_wav("physical short PCM remains caller check", pcm, 46u,
            1, 4u, 44L, 2u);
}

int main(void) {
  static const unsigned char valid[44] = {
    'R','I','F','F',36,0,0,0,'W','A','V','E',
    'f','m','t',' ',16,0,0,0,1,0,1,0,
    0x80,0x3e,0,0,0x00,0x7d,0,0,2,0,16,0,
    'd','a','t','a',0,0,0,0
  };
  unsigned char mutated[44];
  check_header(valid, 1);
  /* Every byte matters: reject corruption in either PCM layout field. */
  for (size_t i = 28u; i < 34u; ++i) {
    memcpy(mutated, valid, sizeof(mutated));
    mutated[i] ^= 1u;
    check_header(mutated, 0);
  }
  check_riff_bounds(valid);
  CHECK(failures == 0u);
  printf("tool_io WAV layout/RIFF bounds: %u cases passed\n", cases);
  return 0;
}
