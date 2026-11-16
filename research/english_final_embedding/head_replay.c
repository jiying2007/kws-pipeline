#include "english_causal_head.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Replay-only executable; the runtime itself has no allocation or file I/O. */
static uint32_t u32(const unsigned char *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint64_t u64(const unsigned char *p) { return (uint64_t)u32(p) | ((uint64_t)u32(p + 4) << 32); }

int main(int argc, char **argv) {
  ehead_model *m = NULL;
  ehead_state *s = NULL;
  unsigned char *blob = NULL;
  FILE *weights = NULL, *stream = NULL;
  unsigned char header[12], row[12 + 4 * EHEAD_DIM];
  uint32_t i, count;
  int rc = 2;
  if (argc != 3) return 2;
  m = malloc(sizeof *m); s = malloc(sizeof *s); blob = malloc(EHEAD_BLOB_BYTES);
  if (!m || !s || !blob) goto done;
  weights = fopen(argv[1], "rb");
  if (!weights || fread(blob, 1, EHEAD_BLOB_BYTES, weights) != EHEAD_BLOB_BYTES ||
      fgetc(weights) != EOF || ferror(weights) || ehead_load(m, blob, EHEAD_BLOB_BYTES)) goto done;
  stream = fopen(argv[2], "rb");
  if (!stream || fread(header, 1, sizeof header, stream) != sizeof header ||
      memcmp(header, "EHSTR01\0", 8)) goto done;
  count = u32(header + 8);
  if (!count || count > 1000000) goto done;
  ehead_reset(s);
  for (i = 0; i < count; ++i) {
    unsigned d;
    float embedding[EHEAD_DIM];
    ehead_output output;
    if (fread(row, 1, sizeof row, stream) != sizeof row || u32(row + 8) > 1) goto done;
    for (d = 0; d < EHEAD_DIM; ++d) {
      uint32_t bits = u32(row + 12 + 4 * d);
      memcpy(&embedding[d], &bits, sizeof bits);
    }
    if (u32(row + 8)) ehead_discontinuity(s);
    if (ehead_step(m, s, embedding, u64(row), &output)) goto done;
    printf("%llu %d %u %.9g %.9g %.9g %.9g %.9g %.9g\n",
           (unsigned long long)output.available_after_sample, output.ready, output.keyword,
           (double)output.logits[0], (double)output.logits[1], (double)output.logits[2],
           (double)output.probabilities[0], (double)output.probabilities[1], (double)output.probabilities[2]);
  }
  if (fgetc(stream) != EOF || ferror(stream) || fflush(stdout)) goto done;
  rc = 0;
done:
  if (weights) fclose(weights);
  if (stream) fclose(stream);
  free(blob); free(s); free(m);
  return rc;
}
