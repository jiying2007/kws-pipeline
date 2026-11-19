#include "english_causal_head.h"
#include <float.h>
#include <math.h>
#include <string.h>

static int load_array(float *values, size_t count, const unsigned char **bytes) {
  size_t i;
  for (i = 0; i < count; ++i) {
    const unsigned char *p = *bytes;
    uint32_t u = (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                 ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    memcpy(&values[i], &u, sizeof u);
    if (!isfinite(values[i])) return -1;
    *bytes += 4;
  }
  return 0;
}

int ehead_load(ehead_model *m, const unsigned char *bytes, size_t length) {
  unsigned i;
  const unsigned char *p;
  if (!m || !bytes || length != EHEAD_BLOB_BYTES || sizeof(float) != 4 ||
      FLT_RADIX != 2 || FLT_MANT_DIG != 24 || memcmp(bytes, "EHKWS01\0", 8)) return -1;
  p = bytes + 8;
  if (load_array(m->mean96, EHEAD_DIM, &p) ||
      load_array(m->scale96, EHEAD_DIM, &p)) return -1;
  for (i = 0; i < EHEAD_DIM; ++i) if (!(m->scale96[i] > 0.0f)) return -1;
  for (i = 0; i < EHEAD_HIDDEN; ++i)
    if (load_array(m->w1[i], EHEAD_INPUT, &p)) return -1;
  if (load_array(m->b1, EHEAD_HIDDEN, &p)) return -1;
  for (i = 0; i < EHEAD_CLASSES; ++i)
    if (load_array(m->w2[i], EHEAD_HIDDEN, &p)) return -1;
  return load_array(m->b2, EHEAD_CLASSES, &p);
}

void ehead_reset(ehead_state *s) {
  if (s) memset(s, 0, sizeof *s);
}

void ehead_discontinuity(ehead_state *s) {
  uint64_t clock;
  if (!s) return;
  clock = s->last_sample;
  ehead_reset(s);
  s->last_sample = clock;
}

static int score(const ehead_model *m, const ehead_state *s, ehead_output *o) {
  unsigned h, t, d, k;
  float hidden[EHEAD_HIDDEN], maximum, total = 0.0f;
  for (h = 0; h < EHEAD_HIDDEN; ++h) {
    float value = m->b1[h];
    for (t = 0; t < EHEAD_HISTORY; ++t) {
      const float *row = s->history[(s->next + t) % EHEAD_HISTORY];
      for (d = 0; d < EHEAD_DIM; ++d)
        value += m->w1[h][t * EHEAD_DIM + d] * row[d];
    }
    if (!isfinite(value)) return -1;
    hidden[h] = value > 0.0f ? value : 0.0f;
  }
  for (k = 0; k < EHEAD_CLASSES; ++k) {
    float value = m->b2[k];
    for (h = 0; h < EHEAD_HIDDEN; ++h) value += m->w2[k][h] * hidden[h];
    if (!isfinite(value)) return -1;
    o->logits[k] = value;
  }
  maximum = o->logits[0];
  for (k = 1; k < EHEAD_CLASSES; ++k)
    if (o->logits[k] > maximum) maximum = o->logits[k];
  for (k = 0; k < EHEAD_CLASSES; ++k) {
    o->probabilities[k] = expf(o->logits[k] - maximum);
    total += o->probabilities[k];
  }
  for (k = 0; k < EHEAD_CLASSES; ++k) o->probabilities[k] /= total;
  return 0;
}

int ehead_step(const ehead_model *m, ehead_state *s, const float embedding[EHEAD_DIM],
               uint64_t sample, ehead_output *o) {
  unsigned d, selected = 0, old_next, old_count;
  float normalized[EHEAD_DIM], old_row[EHEAD_DIM];
  if (!m || !s || !embedding || !o || s->next >= EHEAD_HISTORY ||
      s->count > EHEAD_HISTORY || sample < EHEAD_FIRST_SAMPLES ||
      sample <= s->last_sample || sample > UINT64_MAX - 13 * EHEAD_HOP_SAMPLES ||
      (s->count && sample - s->last_sample != EHEAD_HOP_SAMPLES)) return -1;
  for (d = 0; d < EHEAD_DIM; ++d) {
    if (!isfinite(embedding[d])) return -1;
    normalized[d] = (embedding[d] - m->mean96[d]) / m->scale96[d];
    if (!isfinite(normalized[d])) return -1;
  }
  memset(o, 0, sizeof *o);
  o->available_after_sample = sample;
  old_next = s->next; old_count = s->count;
  memcpy(old_row, s->history[old_next], sizeof old_row);
  memcpy(s->history[s->next], normalized, sizeof normalized);
  s->next = (s->next + 1) % EHEAD_HISTORY;
  if (s->count < EHEAD_HISTORY) ++s->count;
  if (s->count == EHEAD_HISTORY && score(m, s, o)) {
    memcpy(s->history[old_next], old_row, sizeof old_row);
    s->next = old_next; s->count = old_count;
    memset(o, 0, sizeof *o);
    return -1;
  }
  s->last_sample = sample;
  if (s->count < EHEAD_HISTORY) return 0;
  o->ready = 1;
  for (d = 1; d < EHEAD_CLASSES; ++d)
    if (o->probabilities[d] > o->probabilities[selected]) selected = d;
  if (!selected || o->probabilities[selected] < 0.90f || sample < s->blocked_until) {
    s->candidate = 0; s->confirmations = 0;
    return 0;
  }
  s->confirmations = s->candidate == selected ? s->confirmations + 1 : 1;
  s->candidate = selected;
  if (s->confirmations >= 3) {
    o->keyword = selected;
    s->blocked_until = sample + 13 * EHEAD_HOP_SAMPLES;
    s->candidate = 0; s->confirmations = 0;
  }
  return 0;
}
