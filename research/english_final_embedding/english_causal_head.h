#ifndef ENGLISH_CAUSAL_HEAD_H
#define ENGLISH_CAUSAL_HEAD_H
#include <stddef.h>
#include <stdint.h>

/* Research head only. The PCM frontend and its frozen assets are separate. */
#define EHEAD_DIM 96
#define EHEAD_HISTORY 16
#define EHEAD_HIDDEN 32
#define EHEAD_CLASSES 3
#define EHEAD_INPUT (EHEAD_DIM * EHEAD_HISTORY)
#define EHEAD_FLOATS (2 * EHEAD_DIM + EHEAD_HIDDEN * EHEAD_INPUT + EHEAD_HIDDEN + EHEAD_CLASSES * EHEAD_HIDDEN + EHEAD_CLASSES)
#define EHEAD_BLOB_BYTES (8 + 4 * EHEAD_FLOATS)
#define EHEAD_HOP_SAMPLES UINT64_C(1280)
#define EHEAD_FIRST_SAMPLES UINT64_C(16000)

typedef struct {
  float mean96[EHEAD_DIM], scale96[EHEAD_DIM];
  float w1[EHEAD_HIDDEN][EHEAD_INPUT], b1[EHEAD_HIDDEN];
  float w2[EHEAD_CLASSES][EHEAD_HIDDEN], b2[EHEAD_CLASSES];
} ehead_model;

typedef struct {
  float history[EHEAD_HISTORY][EHEAD_DIM];
  unsigned next, count, candidate, confirmations;
  uint64_t last_sample, blocked_until;
} ehead_state;

typedef struct {
  int ready;
  unsigned keyword; /* 0 = no event; 1 = Vosen; 2 = Mero. */
  uint64_t available_after_sample;
  float logits[EHEAD_CLASSES], probabilities[EHEAD_CLASSES];
} ehead_output;

/* EHKWS01\0 plus exactly EHEAD_FLOATS little-endian IEEE754 float32 values.
 * Order: mean96, scale96, w1 row-major, b1, w2 row-major, b2. */
int ehead_load(ehead_model *model, const unsigned char *bytes, size_t length);
void ehead_reset(ehead_state *state);
void ehead_discontinuity(ehead_state *state); /* Preserve absolute PCM clock. */
/* Call once per actual embedding, never at EOF without an embedding. Returns
 * -1 on invalid values/clock, otherwise 0. A successful 16th embedding makes
 * the first head output. Clock gaps require explicit discontinuity first. */
int ehead_step(const ehead_model *model, ehead_state *state,
               const float embedding[EHEAD_DIM], uint64_t available_after_sample,
               ehead_output *output);
#endif
