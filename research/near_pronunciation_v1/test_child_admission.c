#include "decoder.h"
#include "kws_parameter_limits.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef KWS_RESEARCH_CHILD_ADMISSION
#error This test requires the explicit offline research compile guard.
#endif
#define CHECK(x) do { if (!(x)) { \
  fprintf(stderr, "CHECK failed: %s:%d: %s\n", __FILE__, __LINE__, #x); \
  exit(1); } } while (0)
#define DEAD (-1.0e30f)

static const uint16_t k1[] = {1u, 2u, 3u, 4u};
static const uint16_t k2[] = {3u, 4u, 3u, 4u};

static void configure(kws_decoder_t *d, kws_research_child_admission_t mode) {
  kws_keyword_t kw[2] = {{0}};
  kw[0].id = 1u; kw[0].tokens = k1; kw[0].num_tokens = 4u;
  kw[1].id = 2u; kw[1].tokens = k2; kw[1].num_tokens = 4u;
  kw[0].threshold = kw[1].threshold = 0.55f;
  kws_decoder_init(d, 0.0f, 0.94f);
  CHECK(kws_decoder_set_keywords(d, kw, 2u, 5u) == KWS_OK);
  CHECK(kws_decoder_research_set_child_admission(d, mode) == KWS_OK);
}

static int frame(kws_decoder_t *d, uint16_t top, int weak, int active,
                 uint32_t *id) {
  float logits[5] = {-8.0f, -8.0f, -8.0f, -8.0f, -8.0f};
  float conf = 0.0f;
  logits[top] = 8.0f;
  if (weak >= 0) { logits[(unsigned)weak] = 7.75f; }
  return kws_decoder_step(d, logits, 5u, active, id, &conf);
}

static void reset_clean(const kws_decoder_t *d) {
  CHECK(d->nodes[0].score == 0.0f);
  for (uint16_t i = 0u; i < d->node_count; ++i) {
    const kws_research_path_state_t *s = &d->research_used_paths[i];
    CHECK(s->score == DEAD && s->blank_score == DEAD);
    CHECK(s->acoustic_score == DEAD && s->blank_acoustic_score == DEAD);
    CHECK(s->next_score == DEAD && s->next_blank_score == DEAD);
    CHECK(s->next_acoustic_score == DEAD && s->next_blank_acoustic_score == DEAD);
    if (i != 0u) { CHECK(d->nodes[i].score == DEAD); }
  }
}

static void canonical_and_one_weak(kws_research_child_admission_t mode) {
  kws_decoder_t d;
  uint32_t id = 0u;
  const uint16_t *seqs[] = {k1, k2};
  for (unsigned seq = 0u; seq < 2u; ++seq) {
    configure(&d, mode);
    for (unsigned t = 0u; t < 4u; ++t) {
      CHECK(frame(&d, seqs[seq][t], -1, 1, &id) == (t == 3u));
    }
    CHECK(id == seq + 1u);
    reset_clean(&d);
    CHECK(d.research_child_admission == (uint8_t)mode);

    configure(&d, mode);
    for (unsigned t = 0u; t < 3u; ++t) {
      CHECK(frame(&d, seqs[seq][t], -1, 1, &id) == 0);
    }
    CHECK(frame(&d, 0u, 4, 1, &id) ==
          (mode == KWS_RESEARCH_SINGLE_FUZZY_WO));
    if (mode == KWS_RESEARCH_SINGLE_FUZZY_WO) { reset_clean(&d); }
  }
}

static void denials_and_root_rules(kws_research_child_admission_t mode) {
  kws_decoder_t d;
  uint32_t id = 0u;
  configure(&d, mode);
  CHECK(frame(&d, 3u, -1, 1, &id) == 0);
  CHECK(frame(&d, 0u, 4, 1, &id) == 0);
  CHECK(frame(&d, 3u, -1, 1, &id) == 0);
  CHECK(frame(&d, 0u, 4, 1, &id) == 0); /* Never two weak wo. */
  CHECK(d.nodes[d.terminal_nodes[1]].score == DEAD);
  CHECK(d.research_used_paths[d.terminal_nodes[1]].score == DEAD);

  configure(&d, mode);
  CHECK(frame(&d, 1u, -1, 1, &id) == 0);
  CHECK(frame(&d, 0u, 2, 1, &id) == 0); /* Non-wo cannot be fuzzy. */
  CHECK(frame(&d, 3u, -1, 1, &id) == 0);
  CHECK(frame(&d, 4u, -1, 1, &id) == 0);

  /* Existing bounded ambiguous roots remain valid in both arms. */
  for (unsigned root_top = 0u; root_top <= 3u; root_top += 3u) {
    configure(&d, mode);
    CHECK(frame(&d, (uint16_t)root_top, 1, 1, &id) == 0);
    CHECK(frame(&d, 2u, -1, 1, &id) == 0);
    CHECK(frame(&d, 3u, -1, 1, &id) == 0);
    CHECK(frame(&d, 4u, -1, 1, &id) == 1);
    CHECK(id == 1u);
  }
  configure(&d, mode);
  CHECK(frame(&d, 2u, 1, 1, &id) == 0); /* Unrelated top blocks root. */
  CHECK(frame(&d, 2u, -1, 1, &id) == 0);
  CHECK(frame(&d, 3u, -1, 1, &id) == 0);
  CHECK(frame(&d, 4u, -1, 1, &id) == 0);

  /* Isolated 小窝, partial 你好 and reordered tokens are not whole phrases. */
  configure(&d, mode);
  CHECK(frame(&d, 3u, -1, 1, &id) == 0);
  CHECK(frame(&d, 4u, -1, 1, &id) == 0);
  kws_decoder_reset(&d);
  CHECK(frame(&d, 1u, -1, 1, &id) == 0);
  CHECK(frame(&d, 2u, -1, 1, &id) == 0);
  kws_decoder_reset(&d);
  CHECK(frame(&d, 4u, -1, 1, &id) == 0);
  CHECK(frame(&d, 3u, -1, 1, &id) == 0);
  CHECK(frame(&d, 2u, -1, 1, &id) == 0);
  CHECK(frame(&d, 1u, -1, 1, &id) == 0);
}

static void reset_and_budget(void) {
  kws_decoder_t d, before;
  uint32_t id = 0u;
  configure(&d, KWS_RESEARCH_SINGLE_FUZZY_WO);
  CHECK(frame(&d, 3u, -1, 1, &id) == 0);
  CHECK(frame(&d, 0u, 4, 1, &id) == 0);
  uint16_t prefix = d.nodes[d.terminal_nodes[1]].parent;
  prefix = d.nodes[prefix].parent;
  CHECK(d.research_used_paths[prefix].score > DEAD / 2.0f);
  CHECK(fabsf(d.research_used_paths[prefix].score -
              d.research_used_paths[prefix].acoustic_score + 8.25f) < 0.00001f);
  before = d;
  CHECK(kws_decoder_research_set_child_admission(&d,
          (kws_research_child_admission_t)9) == KWS_EINVAL);
  CHECK(memcmp(&d, &before, sizeof(d)) == 0);
  CHECK(kws_decoder_research_set_child_admission(NULL,
          KWS_RESEARCH_STRICT_CHILDREN) == KWS_EINVAL);
  CHECK(kws_decoder_debug_set_search_policy(&d, 0.7f, -16.0f) == KWS_EINVAL);
  CHECK(memcmp(&d, &before, sizeof(d)) == 0);

  /* A used lane must expire at the same total search-only budget. */
  for (unsigned i = 0u; i < 23u; ++i) {
    CHECK(frame(&d, 0u, -1, 1, &id) == 0);
  }
  CHECK(frame(&d, 3u, -1, 1, &id) == 0);
  CHECK(frame(&d, 4u, -1, 1, &id) == 0);
  kws_decoder_reset(&d); reset_clean(&d);
  CHECK(d.research_child_admission == KWS_RESEARCH_SINGLE_FUZZY_WO);
  CHECK(frame(&d, 3u, -1, 1, &id) == 0);
  CHECK(frame(&d, 0u, 4, 1, &id) == 0);
  for (unsigned i = 0u; i < KWS_DECODER_BOUNDARY_RESET_INACTIVE_FRAMES; ++i) {
    CHECK(frame(&d, 0u, -1, 0, &id) == 0);
  }
  reset_clean(&d);
  CHECK(frame(&d, 3u, -1, 1, &id) == 0);
  CHECK(frame(&d, 4u, -1, 1, &id) == 0);

  /* Legal path-state fixture: a stronger used-budget path must not overwrite
   * the weaker unspent route before that route spends its one exception. */
  configure(&d, KWS_RESEARCH_SINGLE_FUZZY_WO);
  uint16_t parent = d.nodes[d.terminal_nodes[1]].parent;
  d.nodes[parent].score = d.nodes[parent].acoustic_score = -9.0f;
  d.research_used_paths[parent].score = -8.30f;
  d.research_used_paths[parent].acoustic_score = -0.05f;
  CHECK(frame(&d, 0u, 4, 1, &id) == 0);
  const kws_research_path_state_t *terminal = &d.research_used_paths[d.terminal_nodes[1]];
  CHECK(terminal->score > DEAD / 2.0f);
  CHECK(terminal->acoustic_score < -9.0f);
  CHECK(fabsf(terminal->score - terminal->acoustic_score + 8.25f) < 0.00001f);
  CHECK(d.research_used_paths[0].score == DEAD); /* No used-budget root seed. */
}

static void repeated_tokens(void) {
  const uint16_t tokens[] = {3u, 4u, 4u};
  kws_keyword_t kw = {0};
  kws_decoder_t d;
  uint32_t id = 0u;
  kw.id = 7u; kw.tokens = tokens; kw.num_tokens = 3u; kw.threshold = 0.55f;
  for (unsigned separator = 0u; separator < 2u; ++separator) {
    kws_decoder_init(&d, 0.0f, 0.94f);
    CHECK(kws_decoder_set_keywords(&d, &kw, 1u, 5u) == KWS_OK);
    CHECK(kws_decoder_research_set_child_admission(&d,
            KWS_RESEARCH_SINGLE_FUZZY_WO) == KWS_OK);
    CHECK(frame(&d, 3u, -1, 1, &id) == 0);
    CHECK(frame(&d, 0u, 4, 1, &id) == 0);
    if (separator != 0u) { CHECK(frame(&d, 0u, -1, 1, &id) == 0); }
    CHECK(frame(&d, 4u, -1, 1, &id) == (int)separator);
  }
}

static void immediate_blank_gate(void) {
  kws_decoder_t d;
  uint32_t id = 0u;
  configure(&d, KWS_RESEARCH_SINGLE_FUZZY_WO);
  d.min_trailing_blanks[0] = 1u;
  CHECK(frame(&d, 1u, -1, 1, &id) == 0);
  CHECK(frame(&d, 2u, -1, 1, &id) == 0);
  CHECK(frame(&d, 3u, -1, 1, &id) == 0);
  CHECK(frame(&d, 0u, 4, 1, &id) == 0);
  CHECK(frame(&d, 0u, -1, 1, &id) == 1);
  CHECK(id == 1u); reset_clean(&d);
}

static void exact_boundaries(void) {
  kws_decoder_t d;
  uint32_t id = 0u;
  float confidence = 0.0f;
  /* Inclusive root margin remains 0.5, not a strict-only root filter. */
  for (unsigned outside = 0u; outside < 2u; ++outside) {
    configure(&d, KWS_RESEARCH_STRICT_CHILDREN);
    float logits[5] = {8.0f, outside != 0u ? 7.49f : 7.50f,
                       -8.0f, -8.0f, -8.0f};
    CHECK(kws_decoder_step(&d, logits, 5u, 1, &id, &confidence) == 0);
    CHECK(frame(&d, 2u, -1, 1, &id) == 0);
    CHECK(frame(&d, 3u, -1, 1, &id) == 0);
    CHECK(frame(&d, 4u, -1, 1, &id) == (outside == 0u));
  }
  /* Terminal search-only boundary remains inclusive at -16. A legal
   * used-budget prefix can spend retention up to, but not beyond, that bound. */
  for (unsigned outside = 0u; outside < 2u; ++outside) {
    configure(&d, KWS_RESEARCH_SINGLE_FUZZY_WO);
    uint16_t parent = d.nodes[d.terminal_nodes[0]].parent;
    d.research_used_paths[parent].score = outside != 0u ? -16.01f : -16.0f;
    d.research_used_paths[parent].acoustic_score = 0.0f;
    CHECK(frame(&d, 4u, -1, 1, &id) == (outside == 0u));
  }
  /* The old cost=-16 workaround really is not strict. */
  configure(&d, KWS_RESEARCH_STRICT_CHILDREN);
  d.research_child_admission = 0u;
  CHECK(kws_decoder_debug_set_search_policy(&d, 0.7f, -16.0f) == KWS_OK);
  CHECK(frame(&d, 1u, -1, 1, &id) == 0);
  CHECK(frame(&d, 2u, -1, 1, &id) == 0);
  CHECK(frame(&d, 3u, -1, 1, &id) == 0);
  CHECK(frame(&d, 0u, 4, 1, &id) == 1);
}

int main(void) {
  canonical_and_one_weak(KWS_RESEARCH_STRICT_CHILDREN);
  canonical_and_one_weak(KWS_RESEARCH_SINGLE_FUZZY_WO);
  denials_and_root_rules(KWS_RESEARCH_STRICT_CHILDREN);
  denials_and_root_rules(KWS_RESEARCH_SINGLE_FUZZY_WO);
  reset_and_budget(); repeated_tokens(); immediate_blank_gate(); exact_boundaries();
  puts("research child-admission tests: ok");
  return 0;
}
