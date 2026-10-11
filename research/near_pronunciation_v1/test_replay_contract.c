/* Include only the derived replay CLI to exercise its input allowlist. */
#define main research_replay_main
#include "kws_decoder_replay.c"
#undef main

#define CHECK(x) do { if (!(x)) { \
  fprintf(stderr, "CHECK failed: %s:%d: %s\n", __FILE__, __LINE__, #x); \
  exit(1); } } while (0)

int main(void) {
  const char *sha = "ece44b47bd378c20dd254220b368e41143ec678cbab9dc56901513026ed8d402";
  static const uint16_t a[] = {1u, 2u, 3u, 4u}, b[] = {3u, 4u, 3u, 4u};
  static const int8_t wx[32u * 64u] = {0}, wh[64u * 64u] = {0}, wo[5u * 64u] = {0};
  static const float bh[64u] = {0}, bo[5u] = {0};
  kws_model_t model = {0};
  kws_keyword_pack_t pack = {0};
  kws_engine_t *engine = NULL;
  kws_config_t config = kws_default_config();
  model.feature_dim = 32u; model.hidden_dim = 64u; model.vocab_size = 5u;
  model.vocab_fingerprint = UINT64_C(0x316aea1a94b303bf);
  model.sample_rate_hz = KWS_SAMPLE_RATE_HZ;
  model.frame_length_samples = KWS_FRAME_LENGTH_SAMPLES;
  model.frame_hop_samples = KWS_FRAME_HOP_SAMPLES;
  model.wx_scale = model.wh_scale = model.wo_scale = 1.0f;
  model.wx = wx; model.wh = wh; model.wo = wo; model.bh = bh; model.bo = bo;
  pack.keyword_count = 2u; pack.vocab_fingerprint = model.vocab_fingerprint;
  pack.keywords[0].id = 1u; pack.keywords[0].tokens = a;
  pack.keywords[1].id = 2u; pack.keywords[1].tokens = b;
  pack.keywords[0].num_tokens = pack.keywords[1].num_tokens = 4u;
  pack.keywords[0].threshold = pack.keywords[1].threshold = 0.55f;
  CHECK(research_inputs_valid(&model, &pack, sha) == 1);
  CHECK(research_inputs_valid(&model, &pack, "wrong") == 0);
  model.vocab_size = 6u; CHECK(research_inputs_valid(&model, &pack, sha) == 0);
  model.vocab_size = 5u;
  pack.keywords[0].threshold = 0.56f;
  CHECK(research_inputs_valid(&model, &pack, sha) == 0);
  pack.keywords[0].threshold = 0.55f;
  pack.keywords[0].num_tokens = 2u;
  CHECK(research_inputs_valid(&model, &pack, sha) == 0);
  pack.keywords[0].num_tokens = 4u;
  pack.keywords[0].tokens = b; CHECK(research_inputs_valid(&model, &pack, sha) == 0);
  pack.keywords[0].tokens = a;
  pack.keywords[0].priority = 1u; CHECK(research_inputs_valid(&model, &pack, sha) == 0);
  pack.keywords[0].priority = 0u;

  /* Model tensors are never evaluated. The engine is used ONLY for its exact
   * decoder/event/refractory bridge with caller-provided synthetic logits. */
  size_t bytes = kws_engine_required_bytes(&model);
  void *arena = malloc(bytes);
  CHECK(arena != NULL && bytes != 0u);
  CHECK(kws_engine_init(arena, bytes, &model, &config, &engine) == KWS_OK);
  CHECK(kws_engine_set_keyword_pack(engine, &pack) == KWS_OK);
  CHECK(kws_engine_research_set_child_admission(engine,
          KWS_RESEARCH_SINGLE_FUZZY_WO) == KWS_OK);
  uint64_t sample = 0u;
  for (unsigned phrase = 0u; phrase < 2u; ++phrase) {
    for (unsigned t = 0u; t < 4u; ++t) {
      float logits[5] = {-8.0f, -8.0f, -8.0f, -8.0f, -8.0f};
      int detected = 0;
      kws_detection_t hit = {0};
      logits[a[t]] = 8.0f;
      if (t == 3u) { logits[0] = 8.25f; }
      sample += KWS_FRAME_HOP_SAMPLES;
      CHECK(kws_engine_debug_replay_frame(engine, logits, 5u, 1, sample,
                                          &hit, &detected) == KWS_OK);
      CHECK(detected == (phrase == 0u && t == 3u));
      if (detected != 0) { CHECK(hit.keyword_id == 1u); }
    }
  }
  kws_engine_stats_t stats;
  CHECK(kws_engine_get_stats(engine, &stats) == KWS_OK);
  CHECK(stats.decoder_hits == 2u && stats.detections == 1u);
  CHECK(stats.refractory_suppressed == 1u);
  kws_decoder_debug_state_t state;
  kws_decoder_debug_node_state_t nodes[KWS_MAX_TRIE_NODES];
  CHECK(kws_engine_debug_copy_decoder_state(engine, &state, nodes,
                                           KWS_MAX_TRIE_NODES) == KWS_OK);
  CHECK(state.research_child_admission == KWS_RESEARCH_SINGLE_FUZZY_WO);
  for (unsigned n = 0u; n < state.node_count; ++n) {
    CHECK(nodes[n].research_used_score == -1.0e30f);
    CHECK(nodes[n].research_used_blank_score == -1.0e30f);
  }
  printf("research replay contract tests: ok; engine_bytes=%zu\n", bytes);
  free(arena);
  return 0;
}
