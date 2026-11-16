#ifndef ENGLISH_FIXED_FRONTEND_H
#define ENGLISH_FIXED_FRONTEND_H
#include <stddef.h>
#include <stdint.h>

/* A static, float32 subset for two pinned feature graphs, not an ONNX loader.
 * Compile with -fno-fast-math -ffp-contract=off. No core allocation or threads.
 * Descriptors/constants are immutable; input, workspace and state are owned by
 * the caller. One state/workspace per concurrent stream. */
#define EF_MAX_RANK 6
#define EF_MAX_INPUTS 8
#define EF_NONE UINT32_MAX
#define EF_PCM_WINDOW 16000
#define EF_HOP 1280
#define EF_MEL_FRAMES 97
#define EF_MEL_BINS 32
#define EF_EMBED_FRAMES 76
#define EF_EMBED_DIM 96

enum ef_status { EF_OK = 0, EF_INVALID = -1, EF_WORKSPACE = -2,
                 EF_NONFINITE = -3, EF_CLOCK = -4, EF_CALLBACK = -5 };
enum ef_storage { EF_INPUT = 0, EF_CONSTANT = 1, EF_WORK = 2 };
enum ef_op { EF_ADD = 1, EF_SUB, EF_MUL, EF_DIV, EF_POW, EF_MAX,
             EF_LOG, EF_CLIP, EF_LEAKY_RELU, EF_TRANSPOSE, EF_REDUCE_MAX,
             EF_MATMUL, EF_CONV, EF_MAX_POOL };

typedef struct {
    uint32_t rank, dims[EF_MAX_RANK], count, storage, offset;
} ef_tensor;

typedef struct {
    uint32_t op, input_count, inputs[EF_MAX_INPUTS], output;
    /* CONV/POOL: spatial rank, groups, kH,kW,sH,sW,dH,dW,
     *             padH_begin,padW_begin,padH_end,padW_end,ceil_mode.
     * 1D maps to H, W=1. auto_pad is explicitly resolved by the exporter.
     * TRANSPOSE: complete permutation. REDUCE_MAX: axes bitmask, keepdims.
     * LEAKY_RELU: alpha. All unused entries are zero. */
    int32_t p[16];
    float alpha;
} ef_node;

typedef struct {
    const ef_tensor *tensors;
    uint32_t tensor_count;
    const ef_node *nodes;
    uint32_t node_count, input, output;
    const float *constants;
    uint32_t constant_count, workspace_floats;
} ef_graph;

/* Trace is optional, synchronous, and receives an output before reuse.
 * Copy values in the callback if they must outlive this invocation. */
typedef int (*ef_trace_fn)(void *context, uint32_t node_index,
                           const ef_tensor *tensor, const float *values);
/* stage 1: raw 97x32 mel; stage 2: normalized 76x32; stage 3: embedding96. */
typedef int (*ef_feature_trace_fn)(void *context, uint32_t stage,
                                  uint64_t available_after_sample,
                                  const float *values, size_t count);
size_t ef_graph_workspace_bytes(const ef_graph *graph);
int ef_graph_validate(const ef_graph *graph);
int ef_graph_run(const ef_graph *graph, const float *input,
                 float *workspace, size_t workspace_floats,
                 const float **output, ef_trace_fn trace, void *trace_context);

typedef struct {
    const ef_graph *mel, *embedding;
    float *workspace;
    size_t workspace_floats;
    int16_t ring[EF_PCM_WINDOW];
    uint32_t write, filled;
    uint64_t observed, next_end;
    int failed, initialized;
    ef_feature_trace_fn trace;
    void *trace_context;
} ef_frontend;

/* available_after_sample is the absolute count of observed PCM samples.
 * The pointer is valid only during the callback. Feed it directly to the
 * separately exported head; no head or event thresholds are changed here. */
typedef int (*ef_embedding_fn)(void *context, uint64_t available_after_sample,
                               const float embedding[EF_EMBED_DIM]);
size_t ef_frontend_workspace_bytes(const ef_graph *mel, const ef_graph *embedding);
int ef_frontend_init(ef_frontend *state, const ef_graph *mel,
                     const ef_graph *embedding, float *workspace,
                     size_t workspace_floats);
void ef_frontend_reset(ef_frontend *state);
/* Clears acoustic history and moves to an explicitly declared absolute input
 * offset; no samples or embeddings are invented. Also clear head history. */
int ef_frontend_discontinuity(ef_frontend *state, uint64_t next_input_sample);
/* first_sample is zero-based and must equal observed. On failure consumed
 * reports actual accepted input. State is poisoned after an execution or
 * callback failure; reset/discontinuity is required before another feed.
 * count=0 emits nothing: there is deliberately no EOF-padding API. */
int ef_frontend_feed(ef_frontend *state, const int16_t *pcm, size_t count,
                     uint64_t first_sample, ef_embedding_fn callback,
                     void *context, size_t *consumed);
#endif
