#include "kws_pipeline/kws.h"
#include "tool_io.h"
#include "kws_debug.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#endif

#define BLOCK_SAMPLES 320u


static void print_state(const kws_decoder_debug_state_t *state, const kws_decoder_debug_node_state_t *nodes) {
  fprintf(stderr,"{\"inactive\":%u,\"pending\":%d,\"pending_age\":%u,\"pending_blanks\":%u,\"nodes\":[",(unsigned)state->inactive_frames,(int)state->pending_keyword_index,(unsigned)state->pending_age_frames,(unsigned)state->pending_blank_frames);
  for(uint16_t i=0u;i<state->node_count;++i){
    const kws_decoder_debug_node_state_t *n=&nodes[i];
    fprintf(stderr,"%s[%u,%u,%u,%d,%.9g,%.9g,%.9g,%.9g]",i?",":"",(unsigned)n->token,(unsigned)n->parent,(unsigned)n->depth,(int)n->terminal_keyword_index,(double)n->score,(double)n->blank_score,(double)n->acoustic_score,(double)n->blank_acoustic_score);
  }
  fputs("]}",stderr);
}

int main(int argc, char **argv) {
  uint8_t *model_blob = NULL;
  uint8_t *pack_blob = NULL;
  size_t model_bytes = 0u;
  size_t pack_bytes = 0u;
  kws_model_t model;
  kws_keyword_pack_t pack;
  kws_engine_t *engine = NULL;
  void *arena = NULL;
  int16_t pcm[BLOCK_SAMPLES];
  int exit_code = 1;
  uint64_t fed_samples = 0u;
  uint64_t trace_start = strtoull(getenv("KWS_TRACE_START_SAMPLE"), NULL, 10);

  if (argc != 4) {
    fprintf(stderr, "usage: %s model.kwm keywords.kwk recording-id < pcm16le.raw\n", argv[0]);
    return 2;
  }
#if defined(_WIN32)
  /* Text mode would translate CRLF and treat 0x1A as end of file, either of
   * which corrupts a raw PCM16 stream arriving from a pipe or a redirect. */
  if (_setmode(_fileno(stdin), _O_BINARY) == -1) {
    fprintf(stderr, "cannot switch stdin to binary mode\n");
    return 2;
  }
#endif
  if (kws_tool_read_file(argv[1], &model_blob, &model_bytes) == 0 ||
      kws_tool_read_file(argv[2], &pack_blob, &pack_bytes) == 0) {
    fprintf(stderr, "cannot read model or keyword pack\n");
    goto cleanup;
  }
  if (kws_model_open(model_blob, model_bytes, &model) != KWS_OK ||
      kws_keyword_pack_open(pack_blob, pack_bytes, &model, &pack) != KWS_OK) {
    fprintf(stderr, "invalid model or keyword pack\n");
    goto cleanup;
  }
  arena = malloc(kws_engine_required_bytes(&model));
  if (arena == NULL ||
      kws_engine_init(arena, kws_engine_required_bytes(&model), &model, NULL,
                      &engine) != KWS_OK ||
      kws_engine_set_keyword_pack(engine, &pack) != KWS_OK) {
    fprintf(stderr, "cannot initialize KWS engine\n");
    goto cleanup;
  }

  for (;;) {
    size_t got = fread(pcm, sizeof(pcm[0]), BLOCK_SAMPLES, stdin);
    if (got == 0u) {
      if (ferror(stdin) != 0) {
        fprintf(stderr, "raw PCM input error\n");
        goto cleanup;
      }
      break;
    }
    {
      kws_detection_t hit;
      int detected = 0;
      kws_decoder_debug_state_t before, after;
      kws_decoder_debug_node_state_t before_nodes[KWS_MAX_TRIE_NODES], after_nodes[KWS_MAX_TRIE_NODES];
      int trace = fed_samples + got >= trace_start;
      if (trace && kws_engine_debug_copy_decoder_state(engine, &before, before_nodes, KWS_MAX_TRIE_NODES) != KWS_OK) goto cleanup;
      fed_samples += got;
      if (kws_engine_accept_pcm16(engine, pcm, got, &hit, &detected) != KWS_OK) {
        fprintf(stderr, "KWS runtime error\n");
        goto cleanup;
      }
      if (trace) {
        float logits[KWS_MAX_VOCAB_SIZE]; uint64_t frame=0u,end=0u; uint16_t vocab=0u; int speech=0;
        if (kws_engine_debug_copy_last_frame(engine,&frame,&end,&speech,logits,KWS_MAX_VOCAB_SIZE,&vocab) != 1 || kws_engine_debug_copy_decoder_state(engine,&after,after_nodes,KWS_MAX_TRIE_NODES) != KWS_OK) goto cleanup;
        fprintf(stderr,"{\"frame\":%llu,\"end_sample\":%llu,\"speech_active\":%d,\"detected\":%d,\"logits\":[",(unsigned long long)frame,(unsigned long long)end,speech,detected);
        for(uint16_t i=0u;i<vocab;++i) fprintf(stderr,"%s%.9g",i?",":"",(double)logits[i]);
        fputs("],\"before\":",stderr);print_state(&before,before_nodes);fputs(",\"after\":",stderr);print_state(&after,after_nodes);fputs("}\n",stderr);
      }
      if (detected != 0) {
        fputs("{\"recording\":", stdout);
        kws_tool_print_json_string(stdout, argv[3]);
        fprintf(stdout,
                ",\"keyword_id\":%u,\"time_s\":%.6f,\"confidence\":%.6f}\n",
                hit.keyword_id,
                (double)hit.end_sample / (double)KWS_SAMPLE_RATE_HZ,
                (double)hit.confidence);
        if (fflush(stdout) != 0) {
          goto cleanup;
        }
      }
    }
    if (got != BLOCK_SAMPLES) {
      if (ferror(stdin) != 0) {
        fprintf(stderr, "raw PCM input error\n");
        goto cleanup;
      }
      break;
    }
  }
  exit_code = 0;

cleanup:
  free(arena);
  free(pack_blob);
  free(model_blob);
  return exit_code;
}
