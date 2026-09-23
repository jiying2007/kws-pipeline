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

int kws_engine_debug_copy_last_frame(const kws_engine_t *engine,
                                     uint64_t *out_frame_number,
                                     uint64_t *out_end_sample,
                                     int *out_speech_active,
                                     float *out_logits,
                                     size_t logits_capacity,
                                     uint16_t *out_vocab_size);

kws_status_t kws_engine_debug_replay_frame(kws_engine_t *engine,
                                           const float *logits,
                                           uint16_t vocab_size,
                                           int speech_active,
                                           uint64_t end_sample,
                                           kws_detection_t *out_detection,
                                           int *out_detected);

#endif
