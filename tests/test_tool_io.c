#include "tool_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
  fprintf(stderr, "CHECK failed at line %d: %s\n", __LINE__, #x); \
  exit(1); } } while (0)

static void check_header(const unsigned char header[44], int accepted) {
  FILE *file = tmpfile();
  uint32_t bytes = 0u;
  long offset = 0L;
  CHECK(file != NULL);
  CHECK(fwrite(header, 1u, 44u, file) == 44u);
  CHECK(fseek(file, 0L, SEEK_SET) == 0);
  CHECK(kws_tool_open_wav(file, &bytes, &offset) == accepted);
  if (accepted != 0) CHECK(bytes == 0u && offset == 44L);
  CHECK(fclose(file) == 0);
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
  puts("tool_io WAV PCM layout: 7 cases passed");
  return 0;
}
