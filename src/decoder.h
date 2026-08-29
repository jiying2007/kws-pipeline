#ifndef KWS_PIPELINE_DECODER_H
#define KWS_PIPELINE_DECODER_H

#include <stddef.h>
#include <stdint.h>

#include "kws_pipeline/kws.h"

typedef struct kws_trie_node {
  uint16_t token;
  uint16_t parent;
  uint16_t first_child;
  uint16_t next_sibling;
  int16_t terminal_keyword;
  uint16_t depth;
  float score;
  float blank_score;
  float next_score;
  float next_blank_score;
} kws_trie_node_t;

typedef struct kws_decoder {
  kws_trie_node_t nodes[KWS_MAX_TRIE_NODES];
  uint16_t node_count;
  uint16_t keyword_count;
  uint32_t keyword_ids[KWS_MAX_KEYWORDS];
  uint16_t terminal_nodes[KWS_MAX_KEYWORDS];
  float thresholds[KWS_MAX_KEYWORDS];
  uint8_t min_trailing_blanks[KWS_MAX_KEYWORDS];
  uint8_t priorities[KWS_MAX_KEYWORDS];
  uint8_t prefix_policies[KWS_MAX_KEYWORDS];
  uint8_t grace_frames[KWS_MAX_KEYWORDS];
  float token_boost;
  float retention_log;
  int16_t pending_keyword;
  float pending_confidence;
  uint16_t pending_depth;
  uint16_t pending_age_frames;
  uint16_t pending_blank_frames;
} kws_decoder_t;

void kws_decoder_init(kws_decoder_t *d,
                      float token_boost,
                      float state_retention);
kws_status_t kws_decoder_set_keywords(kws_decoder_t *d,
                                      const kws_keyword_t *keywords,
                                      size_t count,
                                      uint16_t vocab_size);
void kws_decoder_reset(kws_decoder_t *d);
int kws_decoder_step(kws_decoder_t *d,
                     const float *logits,
                     uint16_t vocab_size,
                     int speech_active,
                     uint32_t *keyword_id,
                     float *confidence);

#endif
