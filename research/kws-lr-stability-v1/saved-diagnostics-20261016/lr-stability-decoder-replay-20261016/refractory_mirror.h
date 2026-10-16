#ifndef CONTROLLED_LP_REFRACTORY_MIRROR_H
#define CONTROLLED_LP_REFRACTORY_MIRROR_H
#include <stdint.h>

/* A diagnostic mirror of src/kws.c:323-344, not the product engine. */
typedef struct {
  uint64_t suppress_until_sample;
} suppression_mirror_t;

enum mirror_outcome { MIRROR_NO_HIT = 0, MIRROR_ACCEPTED = 1, MIRROR_SUPPRESSED = 2 };

static enum mirror_outcome suppression_apply(suppression_mirror_t *state,
                                             int decoder_hit,
                                             uint64_t end_sample) {
  if (decoder_hit == 0) {
    return MIRROR_NO_HIT;
  }
  if (end_sample >= state->suppress_until_sample) {
    state->suppress_until_sample = end_sample + UINT64_C(19200);
    return MIRROR_ACCEPTED;
  }
  return MIRROR_SUPPRESSED;
}
#endif
