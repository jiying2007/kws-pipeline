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

typedef enum kws_discontinuity_reason {
  KWS_DISCONTINUITY_XRUN = 1,
  KWS_DISCONTINUITY_ROUTE_CHANGE = 2,
  KWS_DISCONTINUITY_CLOCK_RESET = 3,
  KWS_DISCONTINUITY_SUSPEND_RESUME = 4
} kws_discontinuity_reason_t;

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

/* One keyword.  Field defaults and validation ranges live in
 * configs/parameter-contract.json; tools/compile_keywords.py emits these
 * records into a KWKP v3 pack and src/keyword_pack.c revalidates them on load,
 * so a pack that bypasses the compiler is still rejected. */
typedef struct kws_keyword {
  uint32_t id;
  const uint16_t *tokens;
  uint16_t num_tokens;
  /* Acceptance threshold on the emitted confidence, in (0,1). */
  float threshold;
  /* Blank-separated frames required before a held terminal may fire, 0..8. */
  uint8_t min_trailing_blanks;
  /* Arbitration rank, 0..15.  Higher wins; ties break on depth, then
   * confidence, so duplicate ranks stay deterministic. */
  uint8_t priority;
  /* kws_prefix_policy_t. */
  uint8_t prefix_policy;
  /* Hold window for KWS_PREFIX_GRACE, 0..32 frames. */
  uint8_t grace_frames;
} kws_keyword_t;

typedef struct kws_keyword_pack {
  kws_keyword_t keywords[KWS_MAX_KEYWORDS];
  uint16_t token_storage[KWS_MAX_KEYWORDS][KWS_MAX_TOKENS_PER_KEYWORD];
  size_t keyword_count;
  uint64_t vocab_fingerprint;
} kws_keyword_pack_t;

/* L2 product configuration.  Every field is validated against the ranges in
 * configs/parameter-contract.json when the engine is initialised; an
 * out-of-range value is rejected with KWS_EINVAL.  Changing any field whose
 * contract entry sets invalidates_thresholds invalidates the calibrated
 * per-keyword thresholds.  See docs/RUNTIME_CONFIG.md. */
typedef struct kws_config {
  /* Speech gate applied to the post-AFE frame energy, in dBFS. */
  float min_speech_dbfs;
  /* DEPRECATED and INEFFECTIVE.  Added once per trie depth, so it is a constant
   * offset on the search score; the emitted confidence uses exp(acoustic/depth)
   * and the retention gate subtracts token_boost*depth, which cancels it
   * exactly.  Retained for source and ABI compatibility only. */
  float token_boost;
  /* Per-frame retention factor for a live prefix on a speech frame, in (0,1). */
  float state_retention;
  /* Suppression window after an emitted detection, in milliseconds. */
  uint32_t refractory_ms;
  /* Decision threshold on the external VAD probability carried in frame
   * metadata, in (0,1). */
  float external_vad_threshold;
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
  uint64_t discontinuities;
  uint16_t keyword_count;
  uint16_t trie_nodes;
  int16_t pending_keyword_index;
  uint16_t pending_age_frames;
  uint32_t last_discontinuity_reason;
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
  /* Must be zero. A non-zero reserved word means the caller was built against a
   * newer ABI than this engine understands, and kws_engine_accept_pcm16_ex()
   * rejects the frame rather than silently dropping that field. */
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

/* ---------------------------------------------------------------------------
 * API contract
 *
 * Arena ownership
 *   The caller owns the memory passed to kws_engine_init(). The engine never
 *   allocates, never frees and never retains a pointer past the arena's
 *   lifetime. Release the arena only after the last call on that engine.
 *
 * Read-only blob views
 *   kws_model_open() and kws_keyword_pack_open() do not copy. The structures
 *   they return point into the caller's blob, so that blob must outlive every
 *   engine built from it.
 *
 * Concurrency
 *   Engines are independent: several may run in parallel on separate arenas.
 *   A single engine is not thread-safe, so serialise calls per engine. There
 *   is no locking, no thread and no file I/O anywhere on the audio path.
 *
 * Block size
 *   kws_engine_accept_pcm16*() accepts at most KWS_MAX_PCM_BLOCK_SAMPLES
 *   samples per call and returns KWS_EBOUNDS beyond that. The cap is one frame
 *   hop, which is what bounds the worst-case work per call.
 *
 * Return codes
 *   KWS_OK on success. KWS_EINVAL for a malformed argument, including a
 *   rejected config or a frame whose metadata breaks the ABI contract;
 *   KWS_EBOUNDS for an oversized block; KWS_EFORMAT for a blob or vocabulary
 *   fingerprint mismatch; KWS_ENOMEM when the arena or the trie cannot hold
 *   the request. A failing call never reports a detection, and a failing
 *   kws_engine_init() sets *out_engine to NULL.
 *
 * Optional outputs
 *   out_detection may be NULL when only the detected flag is wanted, and
 *   out_detected may be NULL when only the detection is wanted. Outputs are
 *   written only on KWS_OK.
 * ------------------------------------------------------------------------ */

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

kws_status_t kws_engine_notify_discontinuity(
    kws_engine_t *engine,
    kws_discontinuity_reason_t reason);

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

/* Legacy fixed-layout stats. Kept for ABI compatibility; the structure cannot
 * grow, so every new counter goes to the v2 structure only. Prefer
 * kws_engine_get_stats_v2() in new code. */
kws_status_t kws_engine_get_stats(const kws_engine_t *engine,
                                  kws_engine_stats_t *out_stats);

/* Standard stats interface: self-describing through struct_size and
 * api_version, and the one to extend. */
kws_status_t kws_engine_get_stats_v2(const kws_engine_t *engine,
                                     kws_engine_stats_v2_t *out_stats);

const kws_build_info_t *kws_build_info(void);

#ifdef __cplusplus
}
#endif

#endif
