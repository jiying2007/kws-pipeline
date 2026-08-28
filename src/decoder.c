#include "decoder.h"

#include <math.h>
#include <string.h>

#define NEG_INF (-1.0e30f)
#define SILENCE_RETENTION_LOG (-0.3566749439f)

/*
 * Approximate exp(x) for x in [-8, 0] without calling libm expf().
 * (1 + x / 256)^256 converges closely enough for decoder normalization,
 * while reducing the always-on cost to multiplies. Values below -8 have
 * negligible contribution to the softmax denominator and are dropped.
 */
static float fast_exp_nonpos(float x) {
  float y;

  if (x <= -8.0f) {
    return 0.0f;
  }
  if (x >= 0.0f) {
    return 1.0f;
  }

  y = 1.0f + x * (1.0f / 256.0f);
  y *= y;
  y *= y;
  y *= y;
  y *= y;
  y *= y;
  y *= y;
  y *= y;
  y *= y;
  return y;
}

static float approx_logsumexp(const float *x, uint16_t n) {
  float max_value = x[0];
  float sum = 0.0f;

  for (uint16_t i = 1u; i < n; ++i) {
    if (x[i] > max_value) {
      max_value = x[i];
    }
  }
  for (uint16_t i = 0u; i < n; ++i) {
    sum += fast_exp_nonpos(x[i] - max_value);
  }

  return max_value + logf(sum);
}

void kws_decoder_init(kws_decoder_t *d,
                      float token_boost,
                      float state_retention) {
  memset(d, 0, sizeof(*d));
  d->token_boost = token_boost;
  d->retention_log = logf(state_retention);
  d->node_count = 1u;
  d->nodes[0].first_child = UINT16_MAX;
  d->nodes[0].next_sibling = UINT16_MAX;
  d->nodes[0].terminal_keyword = -1;
  d->nodes[0].score = 0.0f;
}

static uint16_t find_or_add_child(kws_decoder_t *d,
                                  uint16_t parent,
                                  uint16_t token,
                                  int *ok) {
  uint16_t child = d->nodes[parent].first_child;

  while (child != UINT16_MAX) {
    if (d->nodes[child].token == token) {
      return child;
    }
    child = d->nodes[child].next_sibling;
  }

  if (d->node_count >= KWS_MAX_TRIE_NODES) {
    *ok = 0;
    return 0u;
  }

  child = d->node_count++;
  d->nodes[child].token = token;
  d->nodes[child].parent = parent;
  d->nodes[child].first_child = UINT16_MAX;
  d->nodes[child].next_sibling = d->nodes[parent].first_child;
  d->nodes[child].terminal_keyword = -1;
  d->nodes[child].depth = (uint16_t)(d->nodes[parent].depth + 1u);
  d->nodes[child].score = NEG_INF;
  d->nodes[parent].first_child = child;
  return child;
}

static int same_sequence(const kws_keyword_t *a, const kws_keyword_t *b) {
  if (a->num_tokens != b->num_tokens) {
    return 0;
  }
  return memcmp(a->tokens, b->tokens,
                (size_t)a->num_tokens * sizeof(a->tokens[0])) == 0;
}

static kws_status_t validate_keywords(const kws_keyword_t *keywords,
                                      size_t count,
                                      uint16_t vocab_size) {
  if ((keywords == NULL && count != 0u) || count > KWS_MAX_KEYWORDS) {
    return KWS_EINVAL;
  }

  for (size_t k = 0u; k < count; ++k) {
    if (keywords[k].tokens == NULL || keywords[k].num_tokens == 0u ||
        keywords[k].num_tokens > KWS_MAX_TOKENS_PER_KEYWORD ||
        !isfinite(keywords[k].threshold) || keywords[k].threshold <= 0.0f ||
        keywords[k].threshold >= 1.0f) {
      return KWS_EINVAL;
    }
    for (uint16_t i = 0u; i < keywords[k].num_tokens; ++i) {
      uint16_t token = keywords[k].tokens[i];
      if (token == 0u || token >= vocab_size) {
        return KWS_EBOUNDS;
      }
    }
    for (size_t prior = 0u; prior < k; ++prior) {
      if (keywords[prior].id == keywords[k].id ||
          same_sequence(&keywords[prior], &keywords[k]) != 0) {
        return KWS_EINVAL;
      }
    }
  }
  return KWS_OK;
}

kws_status_t kws_decoder_set_keywords(kws_decoder_t *d,
                                      const kws_keyword_t *keywords,
                                      size_t count,
                                      uint16_t vocab_size) {
  float boost;
  float retention;
  kws_status_t validation;

  validation = validate_keywords(keywords, count, vocab_size);
  if (validation != KWS_OK) {
    return validation;
  }

  boost = d->token_boost;
  retention = expf(d->retention_log);
  kws_decoder_init(d, boost, retention);
  d->keyword_count = (uint16_t)count;

  for (size_t k = 0u; k < count; ++k) {
    uint16_t node = 0u;
    int ok = 1;

    d->keyword_ids[k] = keywords[k].id;
    d->thresholds[k] = keywords[k].threshold;

    for (uint16_t i = 0u; i < keywords[k].num_tokens; ++i) {
      node = find_or_add_child(d, node, keywords[k].tokens[i], &ok);
      if (!ok) {
        return KWS_ENOMEM;
      }
    }
    d->nodes[node].terminal_keyword = (int16_t)k;
  }

  return KWS_OK;
}

void kws_decoder_reset(kws_decoder_t *d) {
  for (uint16_t i = 0u; i < d->node_count; ++i) {
    d->nodes[i].score = (i == 0u) ? 0.0f : NEG_INF;
    d->nodes[i].next_score = NEG_INF;
  }
}

static void max_assign(float *dst, float value) {
  if (value > *dst) {
    *dst = value;
  }
}

int kws_decoder_step(kws_decoder_t *d,
                     const float *logits,
                     uint16_t vocab_size,
                     int speech_active,
                     uint32_t *keyword_id,
                     float *confidence) {
  float norm = approx_logsumexp(logits, vocab_size);
  float best_conf = 0.0f;
  float decay = speech_active ? d->retention_log : SILENCE_RETENTION_LOG;
  int best_kw = -1;

  for (uint16_t i = 0u; i < d->node_count; ++i) {
    d->nodes[i].next_score = NEG_INF;
  }
  d->nodes[0].next_score = 0.0f;

  for (uint16_t i = 0u; i < d->node_count; ++i) {
    float base = (i == 0u) ? 0.0f : d->nodes[i].score;
    uint16_t child;

    if (i != 0u && base > NEG_INF / 2.0f) {
      max_assign(&d->nodes[i].next_score, base + decay);
    }
    if (base <= NEG_INF / 2.0f) {
      continue;
    }

    child = d->nodes[i].first_child;
    while (child != UINT16_MAX) {
      uint16_t token = d->nodes[child].token;
      float log_probability = logits[token] - norm + d->token_boost;
      max_assign(&d->nodes[child].next_score, base + log_probability);
      child = d->nodes[child].next_sibling;
    }
  }

  for (uint16_t i = 0u; i < d->node_count; ++i) {
    d->nodes[i].score = d->nodes[i].next_score;
    if (speech_active && d->nodes[i].terminal_keyword >= 0 &&
        d->nodes[i].score > NEG_INF / 2.0f) {
      int kw = d->nodes[i].terminal_keyword;
      float conf = expf(d->nodes[i].score / (float)d->nodes[i].depth);
      if (conf > 1.0f) {
        conf = 1.0f;
      }
      if (conf >= d->thresholds[kw] && conf > best_conf) {
        best_conf = conf;
        best_kw = kw;
      }
    }
  }

  if (best_kw >= 0) {
    *keyword_id = d->keyword_ids[best_kw];
    *confidence = best_conf;
    kws_decoder_reset(d);
    return 1;
  }

  return 0;
}
