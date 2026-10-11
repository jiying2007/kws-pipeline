#include "decoder.h"
#include <stdint.h>
#include <stdio.h>

/* Deterministic synthetic logits only. Compare the complete original decoder
 * state byte-for-byte after every frame across original, guard-off and guard-on
 * builds in default mode. Nothing calls a frontend or acoustic model. */
int main(void) {
  kws_decoder_t d;
  const uint16_t a[] = {1u, 2u, 3u, 4u}, b[] = {3u, 4u, 3u, 4u};
  kws_keyword_t kw[2] = {{0}};
  uint32_t random = 0x72589u;
  size_t bytes = sizeof(d);
#ifdef KWS_RESEARCH_CHILD_ADMISSION
  bytes = offsetof(kws_decoder_t, research_used_paths);
#endif
  kw[0].id = 1u; kw[0].tokens = a; kw[0].num_tokens = 4u;
  kw[1].id = 2u; kw[1].tokens = b; kw[1].num_tokens = 4u;
  kw[0].threshold = kw[1].threshold = 0.55f;
  kws_decoder_init(&d, 0.0f, 0.94f);
  if (kws_decoder_set_keywords(&d, kw, 2u, 5u) != KWS_OK) { return 2; }
  for (unsigned i = 0u; i < 20000u; ++i) {
    float logits[5];
    uint32_t id = 0u;
    float conf = 0.0f;
    for (unsigned t = 0u; t < 5u; ++t) {
      random = random * UINT32_C(1664525) + UINT32_C(1013904223);
      logits[t] = (float)(random % 1024u) / 64.0f - 8.0f;
    }
    int detected = kws_decoder_step(&d, logits, 5u, (i % 37u) < 20u, &id, &conf);
    if (fwrite(&d, bytes, 1u, stdout) != 1u ||
        fwrite(&detected, sizeof(detected), 1u, stdout) != 1u ||
        fwrite(&id, sizeof(id), 1u, stdout) != 1u ||
        fwrite(&conf, sizeof(conf), 1u, stdout) != 1u) { return 3; }
    if (i % 79u == 0u) { kws_decoder_reset(&d); }
    if (i % 211u == 0u) {
      kw[0].prefix_policy = (uint8_t)((i / 211u) % 3u);
      kw[1].prefix_policy = (uint8_t)((i / 211u) % 3u);
      kw[0].min_trailing_blanks = kw[1].min_trailing_blanks = 2u;
      kw[0].grace_frames = kw[1].grace_frames = 3u;
      if (kws_decoder_set_keywords(&d, kw, 2u, 5u) != KWS_OK) { return 4; }
    }
  }
  return ferror(stdout) != 0;
}
