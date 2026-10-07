/* Research-only bounded C11 port of pinned WeKws stream decoder.
 * See README.md for upstream identity, semantics, bounds and tie policy. */
#ifndef A20_DECODER_H
#define A20_DECODER_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define A20D_CLASSES 6
#define A20D_SCORE_BEAM 3
#define A20D_PATH_BEAM 20
#define A20D_PREFIX_CAP 128
#define A20D_NEXT_CAP (A20D_SCORE_BEAM * A20D_PATH_BEAM * 2)
#define A20D_LIVE_NODES (A20D_PATH_BEAM * A20D_PREFIX_CAP)
#define A20D_NODE_CAP (A20D_LIVE_NODES + A20D_NEXT_CAP)
#define A20D_DOWNSAMPLE 3

typedef enum {
    A20D_OK = 0, A20D_INVALID = 1, A20D_CAPACITY = 2, A20D_OVERFLOW = 3
} a20d_status;
typedef struct { int64_t frame; double prob; int32_t token; } a20d_node;
typedef struct {
    double pb, pnb;
    uint16_t len;
    uint16_t node[A20D_PREFIX_CAP];
} a20d_hyp;
typedef struct {
    int32_t valid;             /* 0 for an empty input chunk (upstream {}). */
    int32_t state;             /* 1 for activation, otherwise 0. */
    int32_t keyword;           /* 1=你好小窝, 2=小窝小窝; 0 when inactive. */
    int64_t start_frame, end_frame; /* Original 10 ms grid; -1 if inactive. */
    double score;             /* sqrt(product), not a length-normalized score. */
    size_t rows_decoded;       /* Stops at first activation. */
} a20d_result;
typedef struct {
    uint32_t magic;
    uint16_t hyp_count, node_count;
    int64_t total_frames, last_active_pos;
    double hit_score;
    a20d_result result;
    a20d_hyp hyps[A20D_PATH_BEAM];
    a20d_node nodes[A20D_NODE_CAP];
} a20d_decoder;
typedef struct {
    a20d_decoder work;         /* Transactional copy: errors do not change state. */
    a20d_hyp next[A20D_NEXT_CAP];
    a20d_node compact[A20D_LIVE_NODES];
    uint16_t remap[A20D_NODE_CAP];
} a20d_workspace;
typedef struct {
    size_t len;
    double pb, pnb;
    int32_t token[A20D_PREFIX_CAP];
    int64_t frame[A20D_PREFIX_CAP];
    double prob[A20D_PREFIX_CAP];
} a20d_hyp_view;

size_t a20d_decoder_bytes(void);
size_t a20d_workspace_bytes(void);
a20d_status a20d_init(a20d_decoder *decoder);
a20d_status a20d_reset(a20d_decoder *decoder); /* Keeps clock/interval/result. */
a20d_status a20d_reset_all(a20d_decoder *decoder);
a20d_status a20d_softmax6(const float logits[A20D_CLASSES], float probs[A20D_CLASSES]);
/* CPU six-element topk order, including libstdc++-style tie behavior. */
a20d_status a20d_top3(const float probs[A20D_CLASSES], int32_t indices[A20D_SCORE_BEAM]);
/* Contiguous rows*6 float32. Whole input checked, even activation-skipped tail.
 * Caller provides disjoint initialized state, scratch, input and output.
 * No allocation or I/O. On any error, state and result are unchanged. */
a20d_status a20d_process_probs(a20d_decoder *, a20d_workspace *, const float *, size_t rows, a20d_result *);
a20d_status a20d_process_logits(a20d_decoder *, a20d_workspace *, const float *, size_t rows, a20d_result *);
a20d_status a20d_get_hyp(const a20d_decoder *, size_t index, a20d_hyp_view *);
const char *a20d_status_string(a20d_status);
#ifdef __cplusplus
}
#endif
#endif
