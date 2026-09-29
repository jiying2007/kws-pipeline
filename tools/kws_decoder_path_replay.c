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
} keyword_summary_t;

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
    if (status < 0 ||
        kws_engine_debug_replay_frame(engine, logits, trace.vocab_size,
                                      speech_active, end_sample, &hit,
                                      &detected) != KWS_OK ||
        kws_engine_debug_copy_decoder_state(
            engine, &decoder_state, nodes, KWS_MAX_TRIE_NODES) != KWS_OK) {
      fprintf(stderr, "invalid acoustic trace/debug frame\n");
      goto cleanup;
    }
    frame_count++;

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
    fputc('}', stdout);
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
