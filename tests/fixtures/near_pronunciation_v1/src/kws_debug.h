#ifndef KWS_PIPELINE_KWS_DEBUG_H
#define KWS_PIPELINE_KWS_DEBUG_H

#include <stddef.h>
#include <stdint.h>

#include "kws_pipeline/kws.h"

/*
 * Development-only, repo-internal acoustic/decoder bridge.
 *
 * This header is intentionally not installed.  It exists so hosted diagnostic
 * tools can capture the exact int8 acoustic logits produced by the shipping
 * engine and replay those frames through the exact shipping decoder/event path
 * without re-running frontend or model inference.
 */

typedef struct kws_decoder_debug_node_state {
  uint16_t token;
  uint16_t parent;
  uint16_t depth;
  int16_t terminal_keyword_index;
  float score;
  float blank_score;
  float acoustic_score;
  float blank_acoustic_score;
} kws_decoder_debug_node_state_t;

typedef struct kws_decoder_debug_state {
  uint16_t node_count;
  uint16_t keyword_count;
  uint16_t inactive_frames;
  int16_t pending_keyword_index;
  float pending_confidence;
  uint16_t pending_depth;
  uint16_t pending_age_frames;
  uint16_t pending_blank_frames;
  float token_boost;
  float retention_log;
  float silence_retention_log;
  float fuzzy_child_retention_cost_log;
} kws_decoder_debug_state_t;

int kws_engine_debug_copy_last_frame(const kws_engine_t *engine,
                                     uint64_t *out_frame_number,
                                     uint64_t *out_end_sample,
                                     int *out_speech_active,
                                     float *out_logits,
                                     size_t logits_capacity,
                                     uint16_t *out_vocab_size);


kws_status_t kws_engine_debug_copy_decoder_state(
    const kws_engine_t *engine,
    kws_decoder_debug_state_t *out_state,
    kws_decoder_debug_node_state_t *out_nodes,
    size_t node_capacity);

kws_status_t kws_engine_debug_set_decoder_search_policy(
    kws_engine_t *engine,
    float blank_retention,
    float fuzzy_child_cost_log);

/* Detection outputs follow the optional-output contract of
 * kws_engine_accept_pcm16*(), including clearing out_detected on failure. */
kws_status_t kws_engine_debug_replay_frame(kws_engine_t *engine,
                                           const float *logits,
                                           uint16_t vocab_size,
                                           int speech_active,
                                           uint64_t end_sample,
                                           kws_detection_t *out_detection,
                                           int *out_detected);

#endif
