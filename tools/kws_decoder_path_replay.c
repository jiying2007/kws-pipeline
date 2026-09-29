#include "kws_pipeline/kws.h"
#include "kws_debug.h"
#include "kws_trace_io.h"
#include "kws_parameter_limits.h"
#include "sha256.h"
#include "tool_io.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEAD_SCORE (-5.0e29f)

typedef struct path_provenance {
  uint16_t token_advances;
  uint16_t exact_top_advances;
  uint16_t fuzzy_advances;
  float fuzzy_logit_gap_sum;
  float fuzzy_logit_gap_max;
  uint32_t fuzzy_target_rank_sum;
  uint16_t fuzzy_target_rank_max;
  uint64_t fuzzy_event_frame_index[KWS_MAX_TOKENS_PER_KEYWORD];
  uint16_t fuzzy_event_depth[KWS_MAX_TOKENS_PER_KEYWORD];
  uint16_t fuzzy_event_target_token[KWS_MAX_TOKENS_PER_KEYWORD];
  uint16_t fuzzy_event_top_token[KWS_MAX_TOKENS_PER_KEYWORD];
  float fuzzy_event_logit_gap[KWS_MAX_TOKENS_PER_KEYWORD];
  uint16_t fuzzy_event_target_rank[KWS_MAX_TOKENS_PER_KEYWORD];
  uint16_t root_exact_starts;
  uint16_t root_ambiguous_starts;
  uint16_t same_token_retentions;
  uint16_t blank_retentions;
} path_provenance_t;

typedef struct shadow_lane {
  float score;
  float acoustic;
  path_provenance_t provenance;
} shadow_lane_t;

typedef struct shadow_node {
  shadow_lane_t nonblank;
  shadow_lane_t blank;
  shadow_lane_t next_nonblank;
  shadow_lane_t next_blank;
} shadow_node_t;

typedef struct path_snapshot {
  int valid;
  float retention_log;
  float confidence;
  path_provenance_t provenance;
} path_snapshot_t;

typedef struct keyword_summary {
  uint16_t target_depth;
  uint16_t max_depth_reached;
  uint64_t root_alive_frames;
  uint64_t terminal_alive_frames;
  uint64_t terminal_speech_alive_frames;
  uint64_t terminal_retention_pass_frames;
  uint64_t terminal_threshold_pass_frames;
  uint64_t pending_frames;
  uint64_t detections;
  float max_terminal_confidence;
  float max_terminal_retention_log;
  path_snapshot_t best_retention_path;
  path_snapshot_t best_confidence_retention_pass_path;
  path_snapshot_t best_eligible_path;
} keyword_summary_t;

static uint16_t saturating_inc(uint16_t value) {
  return value == UINT16_MAX ? value : (uint16_t)(value + 1u);
}

static float shadow_fast_exp_nonpos(float x) {
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

static float shadow_logsumexp(const float *x, uint16_t n) {
  float max_value = x[0];
  float sum = 0.0f;
  for (uint16_t i = 1u; i < n; ++i) {
    if (x[i] > max_value) {
      max_value = x[i];
    }
  }
  for (uint16_t i = 0u; i < n; ++i) {
    sum += shadow_fast_exp_nonpos(x[i] - max_value);
  }
  return max_value + logf(sum);
}

static uint16_t shadow_dominant_token(const float *logits,
                                      uint16_t vocab_size) {
  uint16_t top = 0u;
  for (uint16_t i = 1u; i < vocab_size; ++i) {
    if (logits[i] > logits[top]) {
      top = i;
    }
  }
  return top;
}

static uint16_t shadow_target_rank(const float *logits,
                                   uint16_t vocab_size,
                                   uint16_t token) {
  uint16_t rank = 1u;
  float target = logits[token];
  for (uint16_t i = 0u; i < vocab_size; ++i) {
    if (i != token && logits[i] > target) {
      rank = saturating_inc(rank);
    }
  }
  return rank;
}

static int shadow_is_root_token(const kws_decoder_debug_node_state_t *nodes,
                                uint16_t node_count,
                                uint16_t token) {
  for (uint16_t i = 1u; i < node_count; ++i) {
    if (nodes[i].depth == 1u && nodes[i].token == token) {
      return 1;
    }
  }
  return 0;
}

static void shadow_dead_lane(shadow_lane_t *lane) {
  memset(lane, 0, sizeof(*lane));
  lane->score = -1.0e30f;
  lane->acoustic = -1.0e30f;
}

static void shadow_reset(shadow_node_t *shadow, uint16_t node_count) {
  for (uint16_t i = 0u; i < node_count; ++i) {
    shadow_dead_lane(&shadow[i].nonblank);
    shadow_dead_lane(&shadow[i].blank);
    shadow_dead_lane(&shadow[i].next_nonblank);
    shadow_dead_lane(&shadow[i].next_blank);
  }
  shadow[0].nonblank.score = 0.0f;
  shadow[0].nonblank.acoustic = 0.0f;
}

static void shadow_assign(shadow_lane_t *dst,
                          float score,
                          float acoustic,
                          path_provenance_t provenance) {
  if (score > dst->score) {
    dst->score = score;
    dst->acoustic = acoustic;
    dst->provenance = provenance;
  }
}

static int shadow_lane_alive(const shadow_lane_t *lane) {
  return lane->score > DEAD_SCORE && lane->acoustic > DEAD_SCORE;
}

static void update_best_retention(path_snapshot_t *dst,
                                  float retention_log,
                                  float confidence,
                                  path_provenance_t provenance) {
  if (dst->valid == 0 || retention_log > dst->retention_log ||
      (retention_log == dst->retention_log && confidence > dst->confidence)) {
    dst->valid = 1;
    dst->retention_log = retention_log;
    dst->confidence = confidence;
    dst->provenance = provenance;
  }
}

static void update_best_confidence(path_snapshot_t *dst,
                                   float retention_log,
                                   float confidence,
                                   path_provenance_t provenance) {
  if (dst->valid == 0 || confidence > dst->confidence ||
      (confidence == dst->confidence && retention_log > dst->retention_log)) {
    dst->valid = 1;
    dst->retention_log = retention_log;
    dst->confidence = confidence;
    dst->provenance = provenance;
  }
}

static int close_shadow_value(float actual, float expected) {
  if (actual < DEAD_SCORE && expected < DEAD_SCORE) {
    return 1;
  }
  return fabsf(actual - expected) <= 2.0e-4f;
}

static int verify_shadow_nodes(const shadow_node_t *shadow,
                               const kws_decoder_debug_node_state_t *nodes,
                               uint16_t node_count) {
  for (uint16_t i = 0u; i < node_count; ++i) {
    if (!close_shadow_value(nodes[i].score, shadow[i].nonblank.score) ||
        !close_shadow_value(nodes[i].blank_score, shadow[i].blank.score) ||
        !close_shadow_value(nodes[i].acoustic_score, shadow[i].nonblank.acoustic) ||
        !close_shadow_value(nodes[i].blank_acoustic_score,
                            shadow[i].blank.acoustic)) {
      return 0;
    }
  }
  return 1;
}

static void shadow_step(
    shadow_node_t *shadow,
    const kws_decoder_debug_node_state_t *node_meta,
    const kws_decoder_debug_state_t *decoder,
    const kws_keyword_pack_t *pack,
    const uint16_t *terminal_nodes,
    keyword_summary_t *summary,
    const float *logits,
    uint16_t vocab_size,
    int speech_active,
    uint64_t frame_index) {
  float norm = shadow_logsumexp(logits, vocab_size);
  float decay = speech_active != 0 ? decoder->retention_log
                                   : decoder->silence_retention_log;
  uint16_t top_token = shadow_dominant_token(logits, vocab_size);
  int blank_dominant = top_token == 0u;
  int top_is_keyword_root =
      top_token != 0u &&
      shadow_is_root_token(node_meta, decoder->node_count, top_token) != 0;

  for (uint16_t i = 0u; i < decoder->node_count; ++i) {
    shadow_dead_lane(&shadow[i].next_nonblank);
    shadow_dead_lane(&shadow[i].next_blank);
  }
  shadow[0].next_nonblank.score = 0.0f;
  shadow[0].next_nonblank.acoustic = 0.0f;

  for (uint16_t i = 0u; i < decoder->node_count; ++i) {
    const shadow_lane_t *nonblank = &shadow[i].nonblank;
    const shadow_lane_t *separated = &shadow[i].blank;

    if (i != 0u) {
      if (shadow_lane_alive(nonblank) != 0) {
        if (blank_dominant != 0) {
          path_provenance_t provenance = nonblank->provenance;
          provenance.blank_retentions =
              saturating_inc(provenance.blank_retentions);
          shadow_assign(&shadow[i].next_blank,
                        nonblank->score + decoder->silence_retention_log,
                        nonblank->acoustic, provenance);
        } else if (top_token == node_meta[i].token) {
          path_provenance_t provenance = nonblank->provenance;
          provenance.same_token_retentions =
              saturating_inc(provenance.same_token_retentions);
          shadow_assign(&shadow[i].next_nonblank,
                        nonblank->score + decay,
                        nonblank->acoustic, provenance);
        }
      }
      if (shadow_lane_alive(separated) != 0 && blank_dominant != 0) {
        path_provenance_t provenance = separated->provenance;
        provenance.blank_retentions =
            saturating_inc(provenance.blank_retentions);
        shadow_assign(&shadow[i].next_blank,
                      separated->score + decoder->silence_retention_log,
                      separated->acoustic, provenance);
      }
    }

    for (uint16_t child = 1u; child < decoder->node_count; ++child) {
      if (node_meta[child].parent != i) {
        continue;
      }
      uint16_t token = node_meta[child].token;
      int repeated_token = i != 0u && token == node_meta[i].token;
      const shadow_lane_t *base_lane;
      if (repeated_token != 0) {
        base_lane = separated;
      } else if (nonblank->score >= separated->score) {
        base_lane = nonblank;
      } else {
        base_lane = separated;
      }

      if (shadow_lane_alive(base_lane) == 0) {
        continue;
      }
      if (i == 0u && top_token != token &&
          !((blank_dominant != 0 || top_is_keyword_root != 0) &&
            logits[top_token] - logits[token] <=
                KWS_ROOT_START_LOGIT_MARGIN)) {
        continue;
      }

      float acoustic_log_probability = logits[token] - norm;
      float search_log_probability =
          acoustic_log_probability + decoder->token_boost;
      path_provenance_t provenance = base_lane->provenance;
      provenance.token_advances = saturating_inc(provenance.token_advances);
      if (i == 0u) {
        if (top_token == token) {
          provenance.root_exact_starts =
              saturating_inc(provenance.root_exact_starts);
        } else {
          provenance.root_ambiguous_starts =
              saturating_inc(provenance.root_ambiguous_starts);
        }
      } else if (top_token == token) {
        provenance.exact_top_advances =
            saturating_inc(provenance.exact_top_advances);
      } else {
        float fuzzy_gap = logits[top_token] - logits[token];
        uint16_t target_rank =
            shadow_target_rank(logits, vocab_size, token);
        uint16_t event_index = provenance.fuzzy_advances;
        if (event_index < KWS_MAX_TOKENS_PER_KEYWORD) {
          provenance.fuzzy_event_frame_index[event_index] = frame_index;
          provenance.fuzzy_event_depth[event_index] = node_meta[child].depth;
          provenance.fuzzy_event_target_token[event_index] = token;
          provenance.fuzzy_event_top_token[event_index] = top_token;
          provenance.fuzzy_event_logit_gap[event_index] = fuzzy_gap;
          provenance.fuzzy_event_target_rank[event_index] = target_rank;
        }
        provenance.fuzzy_advances = saturating_inc(provenance.fuzzy_advances);
        provenance.fuzzy_logit_gap_sum += fuzzy_gap;
        if (fuzzy_gap > provenance.fuzzy_logit_gap_max) {
          provenance.fuzzy_logit_gap_max = fuzzy_gap;
        }
        provenance.fuzzy_target_rank_sum += (uint32_t)target_rank;
        if (target_rank > provenance.fuzzy_target_rank_max) {
          provenance.fuzzy_target_rank_max = target_rank;
        }
        search_log_probability += decoder->fuzzy_child_retention_cost_log;
      }
      shadow_assign(&shadow[child].next_nonblank,
                    base_lane->score + search_log_probability,
                    base_lane->acoustic + acoustic_log_probability,
                    provenance);
    }
  }

  for (uint16_t i = 0u; i < decoder->node_count; ++i) {
    shadow[i].nonblank = shadow[i].next_nonblank;
    shadow[i].blank = shadow[i].next_blank;
  }

  if (speech_active == 0) {
    return;
  }
  for (size_t k = 0u; k < pack->keyword_count; ++k) {
    uint16_t terminal_index = terminal_nodes[k];
    const shadow_lane_t *lane =
        shadow[terminal_index].nonblank.score >=
                shadow[terminal_index].blank.score
            ? &shadow[terminal_index].nonblank
            : &shadow[terminal_index].blank;
    if (shadow_lane_alive(lane) == 0) {
      continue;
    }
    float retention_log =
        lane->score - lane->acoustic -
        decoder->token_boost * (float)node_meta[terminal_index].depth;
    float confidence =
        expf(lane->acoustic / (float)node_meta[terminal_index].depth);
    if (confidence > 1.0f) {
      confidence = 1.0f;
    }
    update_best_retention(&summary[k].best_retention_path,
                          retention_log, confidence, lane->provenance);
    if (retention_log >= KWS_MIN_PATH_RETENTION_LOG) {
      update_best_confidence(
          &summary[k].best_confidence_retention_pass_path,
          retention_log, confidence, lane->provenance);
      if (confidence >= pack->keywords[k].threshold) {
        update_best_confidence(&summary[k].best_eligible_path,
                               retention_log, confidence, lane->provenance);
      }
    }
  }
}

static void print_provenance(const path_provenance_t *value) {
  fprintf(stdout,
          "{\"token_advances\":%u,\"exact_top_advances\":%u,"
          "\"fuzzy_advances\":%u,"
          "\"fuzzy_logit_gap_sum\":%.9g,\"fuzzy_logit_gap_max\":%.9g,"
          "\"fuzzy_target_rank_sum\":%u,\"fuzzy_target_rank_max\":%u,"
          "\"fuzzy_events\":[",
          value->token_advances, value->exact_top_advances,
          value->fuzzy_advances, (double)value->fuzzy_logit_gap_sum,
          (double)value->fuzzy_logit_gap_max,
          (unsigned int)value->fuzzy_target_rank_sum,
          value->fuzzy_target_rank_max);
  for (uint16_t i = 0u;
       i < value->fuzzy_advances && i < KWS_MAX_TOKENS_PER_KEYWORD; ++i) {
    if (i != 0u) {
      fputc(',', stdout);
    }
    fprintf(stdout,
            "{\"frame_index\":%llu,\"depth\":%u,"
            "\"target_token\":%u,\"top_token\":%u,"
            "\"logit_gap\":%.9g,\"target_rank\":%u}",
            (unsigned long long)value->fuzzy_event_frame_index[i],
            value->fuzzy_event_depth[i], value->fuzzy_event_target_token[i],
            value->fuzzy_event_top_token[i],
            (double)value->fuzzy_event_logit_gap[i],
            value->fuzzy_event_target_rank[i]);
  }
  fprintf(stdout,
          "],\"root_exact_starts\":%u,\"root_ambiguous_starts\":%u,"
          "\"same_token_retentions\":%u,\"blank_retentions\":%u}",
          value->root_exact_starts, value->root_ambiguous_starts,
          value->same_token_retentions, value->blank_retentions);
}

static void print_snapshot(const path_snapshot_t *value) {
  if (value->valid == 0) {
    fputs("null", stdout);
    return;
  }
  fprintf(stdout, "{\"retention_log\":%.9g,\"confidence\":%.9g,"
                  "\"provenance\":",
          (double)value->retention_log, (double)value->confidence);
  print_provenance(&value->provenance);
  fputc('}', stdout);
}

static int node_alive(const kws_decoder_debug_node_state_t *node) {
  return node->score > DEAD_SCORE || node->blank_score > DEAD_SCORE;
}

static int keyword_index_for_id(const kws_keyword_pack_t *pack,
                                uint32_t keyword_id) {
  for (size_t i = 0u; i < pack->keyword_count; ++i) {
    if (pack->keywords[i].id == keyword_id) {
      return (int)i;
    }
  }
  return -1;
}

int main(int argc, char **argv) {
  uint8_t *model_blob = NULL;
  uint8_t *pack_blob = NULL;
  size_t model_bytes = 0u;
  size_t pack_bytes = 0u;
  kws_model_t model;
  kws_keyword_pack_t pack;
  kws_config_t config = kws_default_config();
  kws_engine_t *engine = NULL;
  void *arena = NULL;
  kws_trace_reader_t reader = {0};
  kws_trace_header_t trace = {0};
  float logits[KWS_MAX_VOCAB_SIZE];
  char model_sha256[65];
  kws_decoder_debug_state_t decoder_state;
  kws_decoder_debug_node_state_t nodes[KWS_MAX_TRIE_NODES];
  uint16_t terminal_nodes[KWS_MAX_KEYWORDS];
  uint16_t root_nodes[KWS_MAX_KEYWORDS];
  uint8_t path_nodes[KWS_MAX_KEYWORDS][KWS_MAX_TRIE_NODES];
  keyword_summary_t summary[KWS_MAX_KEYWORDS];
  shadow_node_t shadow[KWS_MAX_TRIE_NODES];
  uint16_t shadow_inactive_frames = 0u;
  uint64_t frame_count = 0u;
  int reader_open = 0;
  int exit_code = 1;

  if (argc != 5) {
    fprintf(stderr,
            "usage: %s model.kwm keywords.kwk trace.kwtr recording-id\n",
            argv[0]);
    return 2;
  }

  memset(&decoder_state, 0, sizeof(decoder_state));
  memset(nodes, 0, sizeof(nodes));
  memset(terminal_nodes, 0, sizeof(terminal_nodes));
  memset(root_nodes, 0, sizeof(root_nodes));
  memset(path_nodes, 0, sizeof(path_nodes));
  memset(summary, 0, sizeof(summary));
  memset(shadow, 0, sizeof(shadow));
  for (size_t i = 0u; i < KWS_MAX_KEYWORDS; ++i) {
    summary[i].max_terminal_confidence = -INFINITY;
    summary[i].max_terminal_retention_log = -INFINITY;
  }

  if (kws_tool_read_file(argv[1], &model_blob, &model_bytes) == 0 ||
      kws_tool_read_file(argv[2], &pack_blob, &pack_bytes) == 0 ||
      kws_model_open(model_blob, model_bytes, &model) != KWS_OK ||
      kws_keyword_pack_open(pack_blob, pack_bytes, &model, &pack) != KWS_OK ||
      kws_sha256_file_hex(argv[1], model_sha256) == 0) {
    fprintf(stderr, "cannot open model/keyword pack\n");
    goto cleanup;
  }
  if (pack.keyword_count == 0u || pack.keyword_count > KWS_MAX_KEYWORDS) {
    fprintf(stderr, "keyword pack count is invalid\n");
    goto cleanup;
  }
  if (!kws_trace_reader_open(&reader, argv[3], &trace)) {
    fprintf(stderr, "cannot open acoustic trace: %s\n", argv[3]);
    goto cleanup;
  }
  reader_open = 1;
  if (trace.vocab_size != model.vocab_size ||
      trace.frontend_kind != model.frontend_kind ||
      trace.sample_rate_hz != model.sample_rate_hz ||
      trace.frame_length_samples != model.frame_length_samples ||
      trace.frame_hop_samples != model.frame_hop_samples ||
      trace.vocab_fingerprint != model.vocab_fingerprint ||
      strcmp(trace.model_sha256, model_sha256) != 0) {
    fprintf(stderr, "acoustic trace does not match model\n");
    goto cleanup;
  }

  arena = malloc(kws_engine_required_bytes(&model));
  if (arena == NULL ||
      kws_engine_init(arena, kws_engine_required_bytes(&model), &model, &config,
                      &engine) != KWS_OK ||
      kws_engine_set_keyword_pack(engine, &pack) != KWS_OK ||
      kws_engine_debug_copy_decoder_state(
          engine, &decoder_state, nodes, KWS_MAX_TRIE_NODES) != KWS_OK) {
    fprintf(stderr, "cannot initialize replay engine/debug state\n");
    goto cleanup;
  }

  for (size_t k = 0u; k < pack.keyword_count; ++k) {
    int terminal = -1;
    for (uint16_t i = 0u; i < decoder_state.node_count; ++i) {
      if (nodes[i].terminal_keyword_index == (int16_t)k) {
        if (terminal >= 0) {
          fprintf(stderr, "keyword has multiple terminal nodes\n");
          goto cleanup;
        }
        terminal = (int)i;
      }
    }
    if (terminal <= 0) {
      fprintf(stderr, "keyword terminal node is missing\n");
      goto cleanup;
    }
    terminal_nodes[k] = (uint16_t)terminal;
    summary[k].target_depth = nodes[terminal].depth;
    uint16_t current = (uint16_t)terminal;
    while (current != 0u) {
      if (current >= decoder_state.node_count) {
        fprintf(stderr, "decoder parent chain is invalid\n");
        goto cleanup;
      }
      path_nodes[k][current] = 1u;
      if (nodes[current].depth == 1u) {
        root_nodes[k] = current;
      }
      current = nodes[current].parent;
    }
    if (root_nodes[k] == 0u) {
      fprintf(stderr, "keyword root node is missing\n");
      goto cleanup;
    }
  }
  shadow_reset(shadow, decoder_state.node_count);

  for (;;) {
    uint64_t end_sample = 0u;
    int speech_active = 0;
    int detected = 0;
    kws_detection_t hit = {0u, 0.0f, 0u};
    int status = kws_trace_reader_next(&reader, &end_sample, &speech_active,
                                       logits, KWS_MAX_VOCAB_SIZE);
    if (status == 0) {
      break;
    }
    if (status < 0) {
      fprintf(stderr, "invalid acoustic trace frame\n");
      goto cleanup;
    }
    if (speech_active != 0) {
      shadow_inactive_frames = 0u;
    } else if (shadow_inactive_frames != UINT16_MAX) {
      shadow_inactive_frames++;
    }
    shadow_step(shadow, nodes, &decoder_state, &pack, terminal_nodes, summary,
                logits, trace.vocab_size, speech_active, frame_count);

    if (kws_engine_debug_replay_frame(engine, logits, trace.vocab_size,
                                      speech_active, end_sample, &hit,
                                      &detected) != KWS_OK ||
        kws_engine_debug_copy_decoder_state(
            engine, &decoder_state, nodes, KWS_MAX_TRIE_NODES) != KWS_OK) {
      fprintf(stderr, "invalid acoustic trace/debug frame\n");
      goto cleanup;
    }
    frame_count++;

    if (detected != 0) {
      shadow_reset(shadow, decoder_state.node_count);
      shadow_inactive_frames = 0u;
    } else if (speech_active == 0 &&
               shadow_inactive_frames >=
                   KWS_DECODER_BOUNDARY_RESET_INACTIVE_FRAMES) {
      shadow_reset(shadow, decoder_state.node_count);
      shadow_inactive_frames =
          (uint16_t)KWS_DECODER_BOUNDARY_RESET_INACTIVE_FRAMES;
    }
    if (decoder_state.inactive_frames != shadow_inactive_frames ||
        verify_shadow_nodes(shadow, nodes, decoder_state.node_count) == 0) {
      fprintf(stderr,
              "shadow decoder diverged from exact runtime at frame %llu\n",
              (unsigned long long)(frame_count - 1u));
      goto cleanup;
    }

    if (detected != 0) {
      int k = keyword_index_for_id(&pack, hit.keyword_id);
      if (k < 0) {
        fprintf(stderr, "decoder emitted unknown keyword id\n");
        goto cleanup;
      }
      summary[k].detections++;
    }

    for (size_t k = 0u; k < pack.keyword_count; ++k) {
      keyword_summary_t *out = &summary[k];
      if (node_alive(&nodes[root_nodes[k]]) != 0) {
        out->root_alive_frames++;
      }
      for (uint16_t i = 1u; i < decoder_state.node_count; ++i) {
        if (path_nodes[k][i] != 0u && node_alive(&nodes[i]) != 0 &&
            nodes[i].depth > out->max_depth_reached) {
          out->max_depth_reached = nodes[i].depth;
        }
      }
      if (decoder_state.pending_keyword_index == (int16_t)k) {
        out->pending_frames++;
      }

      const kws_decoder_debug_node_state_t *terminal =
          &nodes[terminal_nodes[k]];
      if (node_alive(terminal) != 0) {
        float terminal_score;
        float terminal_acoustic;
        float retention_log;
        float confidence;
        out->terminal_alive_frames++;
        if (terminal->score >= terminal->blank_score) {
          terminal_score = terminal->score;
          terminal_acoustic = terminal->acoustic_score;
        } else {
          terminal_score = terminal->blank_score;
          terminal_acoustic = terminal->blank_acoustic_score;
        }
        if (terminal_score > DEAD_SCORE && terminal_acoustic > DEAD_SCORE) {
          retention_log =
              terminal_score - terminal_acoustic -
              decoder_state.token_boost * (float)terminal->depth;
          confidence = expf(terminal_acoustic / (float)terminal->depth);
          if (confidence > 1.0f) {
            confidence = 1.0f;
          }
          if (retention_log > out->max_terminal_retention_log) {
            out->max_terminal_retention_log = retention_log;
          }
          if (confidence > out->max_terminal_confidence) {
            out->max_terminal_confidence = confidence;
          }
          if (speech_active != 0) {
            out->terminal_speech_alive_frames++;
            if (retention_log >= KWS_MIN_PATH_RETENTION_LOG) {
              out->terminal_retention_pass_frames++;
              if (confidence >= pack.keywords[k].threshold) {
                out->terminal_threshold_pass_frames++;
              }
            }
          }
        }
      }
    }
  }

  if (!kws_trace_reader_close(&reader)) {
    reader_open = 0;
    fprintf(stderr, "acoustic trace length/trailer is invalid\n");
    goto cleanup;
  }
  reader_open = 0;

  fputs("{\"recording\":", stdout);
  kws_tool_print_json_string(stdout, argv[4]);
  fprintf(stdout,
          ",\"frames\":%llu,\"decoder\":{\"token_boost\":%.9g,"
          "\"retention_log\":%.9g,\"silence_retention_log\":%.9g,"
          "\"fuzzy_child_retention_cost_log\":%.9g},\"keywords\":[",
          (unsigned long long)frame_count,
          (double)decoder_state.token_boost,
          (double)decoder_state.retention_log,
          (double)decoder_state.silence_retention_log,
          (double)decoder_state.fuzzy_child_retention_cost_log);
  for (size_t k = 0u; k < pack.keyword_count; ++k) {
    const keyword_summary_t *row = &summary[k];
    if (k != 0u) {
      fputc(',', stdout);
    }
    fprintf(stdout,
            "{\"keyword_id\":%u,\"target_depth\":%u,"
            "\"max_depth_reached\":%u,\"root_alive_frames\":%llu,"
            "\"terminal_alive_frames\":%llu,"
            "\"terminal_speech_alive_frames\":%llu,"
            "\"terminal_retention_pass_frames\":%llu,"
            "\"terminal_threshold_pass_frames\":%llu,"
            "\"pending_frames\":%llu,\"detections\":%llu,"
            "\"threshold\":%.9g,\"max_terminal_confidence\":",
            pack.keywords[k].id, row->target_depth, row->max_depth_reached,
            (unsigned long long)row->root_alive_frames,
            (unsigned long long)row->terminal_alive_frames,
            (unsigned long long)row->terminal_speech_alive_frames,
            (unsigned long long)row->terminal_retention_pass_frames,
            (unsigned long long)row->terminal_threshold_pass_frames,
            (unsigned long long)row->pending_frames,
            (unsigned long long)row->detections,
            (double)pack.keywords[k].threshold);
    if (isfinite(row->max_terminal_confidence)) {
      fprintf(stdout, "%.9g", (double)row->max_terminal_confidence);
    } else {
      fputs("null", stdout);
    }
    fputs(",\"max_terminal_retention_log\":", stdout);
    if (isfinite(row->max_terminal_retention_log)) {
      fprintf(stdout, "%.9g", (double)row->max_terminal_retention_log);
    } else {
      fputs("null", stdout);
    }
    fputs(",\"selected_path\":{\"best_retention\":", stdout);
    print_snapshot(&row->best_retention_path);
    fputs(",\"best_confidence_retention_pass\":", stdout);
    print_snapshot(&row->best_confidence_retention_pass_path);
    fputs(",\"best_eligible\":", stdout);
    print_snapshot(&row->best_eligible_path);
    fputs("}}", stdout);
  }
  fputs("]}\n", stdout);
  exit_code = ferror(stdout) != 0 ? 1 : 0;

cleanup:
  if (reader_open != 0) {
    (void)kws_trace_reader_close(&reader);
  }
  free(arena);
  free(pack_blob);
  free(model_blob);
  return exit_code;
}
