/* Test-only CLI compilation: actual stdio and cleanup, synthetic compute/hash
 * providers. Never ship or treat these outputs as acoustic/board evidence. */
#define _POSIX_C_SOURCE 200809L
#define kws_engine_accept_pcm16 kws_test_accept_pcm16
#define kws_engine_accept_pcm16_ex kws_test_accept_pcm16_ex
#define kws_frontend_push kws_test_frontend_push
#define kws_sha256_file_hex kws_test_sha256_file_hex
#define clock_gettime kws_test_clock_gettime
#ifdef KWS_STDOUT_TEST_BENCH
#define kws_engine_get_stats kws_test_get_stats
#define clock kws_test_clock
#endif
#define main kws_tested_main
#include KWS_STDOUT_TEST_TOOL
#undef main

#include "frontend.h"
#include "kws_debug.h"
#include "sha256.h"
#include <string.h>
#include <time.h>

#ifndef KWS_STDOUT_TEST_BENCH
static uint64_t synthetic_samples;
#endif

kws_status_t kws_test_accept_pcm16_ex(kws_engine_t *engine,
    const int16_t *samples, size_t count, const kws_frame_metadata_t *metadata,
    kws_detection_t *hit, int *detected) {
#ifndef KWS_STDOUT_TEST_BENCH
  const float logits[5] = {-8.0f, 8.0f, -8.0f, -8.0f, -8.0f};
#endif
  (void)samples;
  (void)metadata;
  if (detected != NULL) *detected = 0;
#ifdef KWS_STDOUT_TEST_BENCH
  (void)engine;
  (void)count;
  (void)hit;
#else
  synthetic_samples += count;
  if (synthetic_samples == 400u) {
    return kws_engine_debug_replay_frame(engine, logits, 5u, 1,
                                         synthetic_samples, hit, detected);
  }
#endif
  return KWS_OK;
}

kws_status_t kws_test_accept_pcm16(kws_engine_t *engine,
    const int16_t *samples, size_t count, kws_detection_t *hit, int *detected) {
  return kws_test_accept_pcm16_ex(engine, samples, count, NULL, hit, detected);
}

int kws_test_frontend_push(kws_frontend_t *frontend, int16_t sample,
                          float *features) {
  (void)sample;
  frontend->fill++;
  if (frontend->fill != 400u) return 0;
  for (uint16_t i = 0u; i < frontend->feature_dim; ++i) features[i] = 0.0f;
  frontend->last_dbfs = -120.0f;
  return 1;
}

int kws_test_sha256_file_hex(const char *path, char out[65]) {
  (void)path;
  memcpy(out, "ece44b47bd378c20dd254220b368e41143ec678cbab9dc56901513026ed8d402", 65u);
  return 1;
}

int kws_test_clock_gettime(clockid_t clock_id, struct timespec *value) {
  static long tick;
  (void)clock_id;
  tick += 100000L;
  value->tv_sec = 0;
  value->tv_nsec = tick;
  return 0;
}

const kws_build_info_t *kws_build_info(void) {
  static const kws_build_info_t value = {
      .struct_size = sizeof(kws_build_info_t), .api_version = 1u,
      .version = "synthetic-stdout-test", .source_revision = "synthetic",
      .compiler_id = "synthetic", .compiler_version = "synthetic",
      .target_triple = "synthetic", .build_type = "synthetic",
      .config_digest = "synthetic",
  };
  return &value;
}

#ifdef KWS_STDOUT_TEST_BENCH
kws_status_t kws_test_get_stats(const kws_engine_t *engine,
                                kws_engine_stats_t *stats) {
  kws_decoder_debug_state_t state;
  kws_decoder_debug_node_state_t nodes[KWS_MAX_TRIE_NODES];
  kws_status_t status = kws_engine_debug_copy_decoder_state(
      engine, &state, nodes, KWS_MAX_TRIE_NODES);
  if (status != KWS_OK) return status;
  memset(stats, 0, sizeof(*stats));
  stats->processed_frames = 1u;
  stats->keyword_count = state.keyword_count;
  stats->trie_nodes = state.node_count;
  return KWS_OK;
}

clock_t kws_test_clock(void) {
  static clock_t tick;
  tick += (clock_t)(CLOCKS_PER_SEC / 1000);
  return tick;
}
#endif

int main(int argc, char **argv) {
  /* Small output must remain buffered until the real CLI's final flush. */
  static char buffer[16384];
  if (setvbuf(stdout, buffer, _IOFBF, sizeof(buffer)) != 0) return 3;
#ifdef KWS_STDOUT_TEST_BENCH
  (void)argc;
  (void)argv;
  return kws_tested_main();
#else
  return kws_tested_main(argc, argv);
#endif
}
