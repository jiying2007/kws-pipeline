#ifndef KWS_PIPELINE_KWS_H
#define KWS_PIPELINE_KWS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KWS_MODEL_VERSION 2u
#define KWS_KEYWORD_PACK_VERSION 3u
#define KWS_SAMPLE_RATE_HZ 16000u
#define KWS_FRAME_LENGTH_SAMPLES 400u
#define KWS_FRAME_HOP_SAMPLES 320u
#define KWS_MAX_PCM_BLOCK_SAMPLES KWS_FRAME_HOP_SAMPLES
#define KWS_MAX_KEYWORDS 16u
#define KWS_MAX_TOKENS_PER_KEYWORD 16u
#define KWS_MAX_TRIE_NODES (1u + KWS_MAX_KEYWORDS * KWS_MAX_TOKENS_PER_KEYWORD)
#define KWS_MAX_FEATURE_DIM 40u
#define KWS_MAX_HIDDEN_DIM 64u
#define KWS_MAX_VOCAB_SIZE 512u

#define KWS_FRONTEND_LOGMEL 0u
#define KWS_FRONTEND_PCEN_LITE 1u
#define KWS_FRAME_METADATA_API_VERSION 1u
#define KWS_ENGINE_STATS_V2_API_VERSION 1u
#define KWS_BUILD_INFO_API_VERSION 1u

typedef enum kws_prefix_policy {
  KWS_PREFIX_IMMEDIATE = 0,
  KWS_PREFIX_LONGEST = 1,
  KWS_PREFIX_GRACE = 2
} kws_prefix_policy_t;

typedef enum kws_status {
  KWS_OK = 0,
  KWS_EINVAL = -1,
  KWS_EFORMAT = -2,
  KWS_ENOMEM = -3,
  KWS_EBOUNDS = -4
} kws_status_t;

typedef struct kws_model {
  uint16_t feature_dim;
  uint16_t hidden_dim;
  uint16_t vocab_size;
  uint16_t frontend_kind;
  uint32_t sample_rate_hz;
  uint32_t frame_length_samples;
  uint32_t frame_hop_samples;
  uint64_t vocab_fingerprint;
  float wx_scale;
  float wh_scale;
  float wo_scale;
  const int8_t *wx;
  const int8_t *wh;
  const float *bh;
  const int8_t *wo;
  const float *bo;
} kws_model_t;

typedef struct kws_keyword {
  uint32_t id;
  const uint16_t *tokens;
  uint16_t num_tokens;
  float threshold;
  uint8_t min_trailing_blanks;
  uint8_t priority;
  uint8_t prefix_policy;
  uint8_t grace_frames;
} kws_keyword_t;

typedef struct kws_keyword_pack {
  kws_keyword_t keywords[KWS_MAX_KEYWORDS];
  uint16_t token_storage[KWS_MAX_KEYWORDS][KWS_MAX_TOKENS_PER_KEYWORD];
  size_t keyword_count;
  uint64_t vocab_fingerprint;
} kws_keyword_pack_t;

typedef struct kws_config {
  float min_speech_dbfs;
  float token_boost;
  float state_retention;
  uint32_t refractory_ms;
} kws_config_t;

typedef struct kws_detection {
  uint32_t keyword_id;
  float confidence;
  uint64_t end_sample;
} kws_detection_t;

typedef struct kws_engine_stats {
  uint64_t processed_samples;
  uint64_t processed_frames;
  uint64_t speech_frames;
  uint64_t blank_top1_frames;
  uint64_t decoder_hits;
  uint64_t refractory_suppressed;
  uint64_t detections;
  uint16_t keyword_count;
  uint16_t trie_nodes;
  int16_t pending_keyword_index;
  uint16_t pending_age_frames;
  float max_detection_confidence;
} kws_engine_stats_t;

typedef uint32_t kws_frame_flags_t;
enum {
  KWS_FRAME_DISCONTINUITY = 1u << 0,
  KWS_FRAME_XRUN = 1u << 1,
  KWS_FRAME_CODEC_REOPEN = 1u << 2,
  KWS_FRAME_CLOCK_RESET = 1u << 3,
  KWS_FRAME_EXTERNAL_VAD_VALID = 1u << 4
};

typedef struct kws_frame_metadata {
  uint32_t struct_size;
  uint32_t api_version;
  kws_frame_flags_t flags;
  uint32_t lost_samples;
  uint64_t stream_sequence;
  uint64_t capture_timestamp_ns;
  float external_vad_probability;
  uint32_t afe_latency_samples;
  uint8_t afe_config_sha256[32];
  uint32_t reserved[8];
} kws_frame_metadata_t;

typedef struct kws_engine_stats_v2 {
  uint32_t struct_size;
  uint32_t api_version;
  uint64_t processed_samples;
  uint64_t processed_frames;
  uint64_t speech_frames;
  uint64_t blank_top1_frames;
  uint64_t decoder_hits;
  uint64_t refractory_suppressed;
  uint64_t detections;
  uint64_t discontinuities;
  uint64_t lost_samples;
  uint64_t external_vad_frames;
  uint64_t last_stream_sequence;
  uint64_t last_capture_timestamp_ns;
  uint32_t afe_latency_samples;
  uint16_t keyword_count;
  uint16_t trie_nodes;
  int16_t pending_keyword_index;
  uint16_t pending_age_frames;
  float max_detection_confidence;
  uint8_t afe_config_sha256[32];
  uint32_t reserved[8];
} kws_engine_stats_v2_t;

typedef struct kws_build_info {
  uint32_t struct_size;
  uint32_t api_version;
  const char *version;
  const char *source_revision;
  const char *compiler_id;
  const char *compiler_version;
  const char *target_triple;
  const char *build_type;
  const char *config_digest;
  uint32_t reserved[8];
} kws_build_info_t;

typedef struct kws_engine kws_engine_t;

kws_status_t kws_model_open(const void *blob,
                            size_t blob_bytes,
                            kws_model_t *out_model);

kws_status_t kws_keyword_pack_open(const void *blob,
                                   size_t blob_bytes,
                                   const kws_model_t *model,
                                   kws_keyword_pack_t *out_pack);

kws_config_t kws_default_config(void);
size_t kws_engine_required_bytes(const kws_model_t *model);
size_t kws_engine_required_alignment(void);

kws_status_t kws_engine_init(void *arena,
                             size_t arena_bytes,
                             const kws_model_t *model,
                             const kws_config_t *config,
                             kws_engine_t **out_engine);

kws_status_t kws_engine_set_keywords(kws_engine_t *engine,
                                     const kws_keyword_t *keywords,
                                     size_t keyword_count,
                                     uint64_t vocab_fingerprint);

kws_status_t kws_engine_set_keyword_pack(kws_engine_t *engine,
                                         const kws_keyword_pack_t *pack);

void kws_engine_reset(kws_engine_t *engine);

kws_status_t kws_engine_accept_pcm16(kws_engine_t *engine,
                                     const int16_t *samples,
                                     size_t sample_count,
                                     kws_detection_t *out_detection,
                                     int *out_detected);

kws_status_t kws_engine_accept_pcm16_ex(kws_engine_t *engine,
                                        const int16_t *samples,
                                        size_t sample_count,
                                        const kws_frame_metadata_t *metadata,
                                        kws_detection_t *out_detection,
                                        int *out_detected);

uint64_t kws_engine_processed_samples(const kws_engine_t *engine);

kws_status_t kws_engine_get_stats(const kws_engine_t *engine,
                                  kws_engine_stats_t *out_stats);

kws_status_t kws_engine_get_stats_v2(const kws_engine_t *engine,
                                     kws_engine_stats_v2_t *out_stats);

const kws_build_info_t *kws_build_info(void);

#ifdef __cplusplus
}
#endif

#endif
