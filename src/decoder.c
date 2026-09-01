#include "decoder.h"

#include <math.h>
#include <string.h>

#define NEG_INF (-1.0e30f)
#define SILENCE_RETENTION_LOG (-0.3566749439f)

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

static int blank_is_dominant(const float *logits, uint16_t vocab_size) {
  for (uint16_t i = 1u; i < vocab_size; ++i) {
    if (logits[i] > logits[0]) {
      return 0;
    }
  }
  return 1;
}

static void max_assign_pair(float *dst_score,
                            float *dst_acoustic,
                            float score,
                            float acoustic) {
  if (score > *dst_score) {
    *dst_score = score;
    *dst_acoustic = acoustic;
  }
}

static void clear_pending(kws_decoder_t *d) {
  d->pending_keyword = -1;
  d->pending_confidence = 0.0f;
  d->pending_depth = 0u;
  d->pending_age_frames = 0u;
  d->pending_blank_frames = 0u;
}

static void init_node_scores(kws_trie_node_t *node, int root) {
  node->score = root != 0 ? 0.0f : NEG_INF;
  node->blank_score = NEG_INF;
  node->next_score = NEG_INF;
  node->next_blank_score = NEG_INF;
  node->acoustic_score = root != 0 ? 0.0f : NEG_INF;
  node->blank_acoustic_score = NEG_INF;
  node->next_acoustic_score = NEG_INF;
  node->next_blank_acoustic_score = NEG_INF;
}

static void init_structure(kws_decoder_t *d,
                           float token_boost,
                           float retention_log) {
  memset(d, 0, sizeof(*d));
  d->token_boost = token_boost;
  d->retention_log = retention_log;
  d->node_count = 1u;
  d->nodes[0].first_child = UINT16_MAX;
  d->nodes[0].next_sibling = UINT16_MAX;
  d->nodes[0].terminal_keyword = -1;
  init_node_scores(&d->nodes[0], 1);
  clear_pending(d);
}

void kws_decoder_init(kws_decoder_t *d,
                      float token_boost,
                      float state_retention) {
  init_structure(d, token_boost, logf(state_retention));
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
  init_node_scores(&d->nodes[child], 0);
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
        keywords[k].threshold >= 1.0f ||
        keywords[k].prefix_policy > (uint8_t)KWS_PREFIX_GRACE ||
        (keywords[k].prefix_policy == (uint8_t)KWS_PREFIX_LONGEST &&
         keywords[k].min_trailing_blanks == 0u) ||
        (keywords[k].prefix_policy == (uint8_t)KWS_PREFIX_GRACE &&
         keywords[k].grace_frames == 0u)) {
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
  float retention_log;
  kws_status_t validation;

  validation = validate_keywords(keywords, count, vocab_size);
  if (validation != KWS_OK) {
    return validation;
  }

  boost = d->token_boost;
  retention_log = d->retention_log;
  init_structure(d, boost, retention_log);
  d->keyword_count = (uint16_t)count;

  for (size_t k = 0u; k < count; ++k) {
    uint16_t node = 0u;
    int ok = 1;

    d->keyword_ids[k] = keywords[k].id;
    d->thresholds[k] = keywords[k].threshold;
    d->min_trailing_blanks[k] = keywords[k].min_trailing_blanks;
    d->priorities[k] = keywords[k].priority;
    d->prefix_policies[k] = keywords[k].prefix_policy;
    d->grace_frames[k] = keywords[k].grace_frames;

    for (uint16_t i = 0u; i < keywords[k].num_tokens; ++i) {
      node = find_or_add_child(d, node, keywords[k].tokens[i], &ok);
      if (!ok) {
        return KWS_ENOMEM;
      }
    }
    d->nodes[node].terminal_keyword = (int16_t)k;
    d->terminal_nodes[k] = node;
  }

  return KWS_OK;
}

void kws_decoder_reset(kws_decoder_t *d) {
  for (uint16_t i = 0u; i < d->node_count; ++i) {
    init_node_scores(&d->nodes[i], i == 0u);
  }
  clear_pending(d);
}

static int immediate_better(const kws_decoder_t *d,
                            int candidate,
                            float candidate_conf,
                            uint16_t candidate_depth,
                            int current,
                            float current_conf,
                            uint16_t current_depth) {
  if (current < 0) {
    return 1;
  }
  if (d->priorities[candidate] != d->priorities[current]) {
    return d->priorities[candidate] > d->priorities[current];
  }
  if (candidate_depth != current_depth) {
    return candidate_depth > current_depth;
  }
  return candidate_conf > current_conf;
}

static int pending_better(const kws_decoder_t *d,
                          int candidate,
                          float candidate_conf,
                          uint16_t candidate_depth) {
  int current = d->pending_keyword;
  if (current < 0) {
    return 1;
  }
  if (candidate_depth != d->pending_depth) {
    return candidate_depth > d->pending_depth;
  }
  if (d->priorities[candidate] != d->priorities[current]) {
    return d->priorities[candidate] > d->priorities[current];
  }
  return candidate_conf > d->pending_confidence;
}

static void offer_pending(kws_decoder_t *d,
                          int kw,
                          float conf,
                          uint16_t depth) {
  if (d->pending_keyword == kw) {
    if (conf > d->pending_confidence) {
      d->pending_confidence = conf;
    }
    return;
  }
  if (pending_better(d, kw, conf, depth) != 0) {
    d->pending_keyword = (int16_t)kw;
    d->pending_confidence = conf;
    d->pending_depth = depth;
    d->pending_age_frames = 0u;
    d->pending_blank_frames = 0u;
  }
}

static int pending_ready(const kws_decoder_t *d) {
  int kw = d->pending_keyword;
  uint8_t policy;
  if (kw < 0) {
    return 0;
  }
  policy = d->prefix_policies[kw];
  if (d->pending_blank_frames < d->min_trailing_blanks[kw]) {
    return 0;
  }
  if (policy == (uint8_t)KWS_PREFIX_LONGEST) {
    return 1;
  }
  if (policy == (uint8_t)KWS_PREFIX_GRACE) {
    return d->pending_age_frames >= d->grace_frames[kw];
  }
  return 1;
}

int kws_decoder_step(kws_decoder_t *d,
                     const float *logits,
                     uint16_t vocab_size,
                     int speech_active,
                     uint32_t *keyword_id,
                     float *confidence) {
  float norm = approx_logsumexp(logits, vocab_size);
  float immediate_conf = 0.0f;
  uint16_t immediate_depth = 0u;
  float decay = speech_active ? d->retention_log : SILENCE_RETENTION_LOG;
  int blank_dominant = blank_is_dominant(logits, vocab_size);
  int immediate_kw = -1;

  if (d->pending_keyword >= 0) {
    if (d->pending_age_frames != UINT16_MAX) {
      d->pending_age_frames++;
    }
    if (blank_dominant != 0) {
      if (d->pending_blank_frames != UINT16_MAX) {
        d->pending_blank_frames++;
      }
    } else {
      d->pending_blank_frames = 0u;
    }
  }

  for (uint16_t i = 0u; i < d->node_count; ++i) {
    d->nodes[i].next_score = NEG_INF;
    d->nodes[i].next_blank_score = NEG_INF;
    d->nodes[i].next_acoustic_score = NEG_INF;
    d->nodes[i].next_blank_acoustic_score = NEG_INF;
  }
  d->nodes[0].next_score = 0.0f;
  d->nodes[0].next_acoustic_score = 0.0f;

  for (uint16_t i = 0u; i < d->node_count; ++i) {
    float nonblank = (i == 0u) ? 0.0f : d->nodes[i].score;
    float separated = (i == 0u) ? NEG_INF : d->nodes[i].blank_score;
    float nonblank_acoustic =
        (i == 0u) ? 0.0f : d->nodes[i].acoustic_score;
    float separated_acoustic =
        (i == 0u) ? NEG_INF : d->nodes[i].blank_acoustic_score;
    uint16_t child;

    if (i != 0u) {
      if (nonblank > NEG_INF / 2.0f) {
        if (blank_dominant != 0) {
          max_assign_pair(&d->nodes[i].next_blank_score,
                          &d->nodes[i].next_blank_acoustic_score,
                          nonblank + decay, nonblank_acoustic);
        } else {
          max_assign_pair(&d->nodes[i].next_score,
                          &d->nodes[i].next_acoustic_score,
                          nonblank + decay, nonblank_acoustic);
        }
      }
      if (separated > NEG_INF / 2.0f) {
        max_assign_pair(&d->nodes[i].next_blank_score,
                        &d->nodes[i].next_blank_acoustic_score,
                        separated + decay, separated_acoustic);
      }
    }

    child = d->nodes[i].first_child;
    while (child != UINT16_MAX) {
      uint16_t token = d->nodes[child].token;
      int repeated_token = i != 0u && token == d->nodes[i].token;
      float base;
      float base_acoustic;

      if (repeated_token != 0) {
        base = separated;
        base_acoustic = separated_acoustic;
      } else if (nonblank >= separated) {
        base = nonblank;
        base_acoustic = nonblank_acoustic;
      } else {
        base = separated;
        base_acoustic = separated_acoustic;
      }

      if (base > NEG_INF / 2.0f) {
        float acoustic_log_probability = logits[token] - norm;
        float search_log_probability =
            acoustic_log_probability + d->token_boost;
        max_assign_pair(&d->nodes[child].next_score,
                        &d->nodes[child].next_acoustic_score,
                        base + search_log_probability,
                        base_acoustic + acoustic_log_probability);
      }
      child = d->nodes[child].next_sibling;
    }
  }

  for (uint16_t i = 0u; i < d->node_count; ++i) {
    float terminal_score;
    float terminal_acoustic;
    d->nodes[i].score = d->nodes[i].next_score;
    d->nodes[i].blank_score = d->nodes[i].next_blank_score;
    d->nodes[i].acoustic_score = d->nodes[i].next_acoustic_score;
    d->nodes[i].blank_acoustic_score =
        d->nodes[i].next_blank_acoustic_score;
    if (d->nodes[i].score >= d->nodes[i].blank_score) {
      terminal_score = d->nodes[i].score;
      terminal_acoustic = d->nodes[i].acoustic_score;
    } else {
      terminal_score = d->nodes[i].blank_score;
      terminal_acoustic = d->nodes[i].blank_acoustic_score;
    }
    if (speech_active && d->nodes[i].terminal_keyword >= 0 &&
        terminal_score > NEG_INF / 2.0f &&
        terminal_acoustic > NEG_INF / 2.0f) {
      int kw = d->nodes[i].terminal_keyword;
      float conf = expf(terminal_acoustic / (float)d->nodes[i].depth);
      if (conf > 1.0f) {
        conf = 1.0f;
      }
      if (conf >= d->thresholds[kw]) {
        if (d->prefix_policies[kw] == (uint8_t)KWS_PREFIX_IMMEDIATE) {
          if (immediate_better(d, kw, conf, d->nodes[i].depth,
                               immediate_kw, immediate_conf,
                               immediate_depth) != 0) {
            immediate_kw = kw;
            immediate_conf = conf;
            immediate_depth = d->nodes[i].depth;
          }
        } else {
          offer_pending(d, kw, conf, d->nodes[i].depth);
        }
      }
    }
  }

  if (immediate_kw >= 0) {
    *keyword_id = d->keyword_ids[immediate_kw];
    *confidence = immediate_conf;
    kws_decoder_reset(d);
    return 1;
  }

  if (pending_ready(d) != 0) {
    int kw = d->pending_keyword;
    *keyword_id = d->keyword_ids[kw];
    *confidence = d->pending_confidence;
    kws_decoder_reset(d);
    return 1;
  }

  return 0;
}
