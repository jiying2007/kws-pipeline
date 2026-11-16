#ifndef EKWS_RESEARCH_STREAM_H
#define EKWS_RESEARCH_STREAM_H
#include <stddef.h>
#include <stdint.h>

/* Research ABI 1. This is not the KWSP/KWKP model or kws_engine ABI.
 * No allocation, I/O, resampling, threads, threshold selection, or EOF padding.
 * One caller-owned arena and workspace per stream; one owner at a time. */
#define EKWS_RESEARCH_ABI 1u
#define EKWS_SAMPLE_RATE 16000u
#define EKWS_HEAD_BLOB_BYTES 197908u
#define EKWS_CLASSES 3u

typedef struct ekws_stream ekws_stream;
enum ekws_status {
    EKWS_OK = 0, EKWS_INVALID = -1, EKWS_MEMORY = -2, EKWS_CLOCK = -3,
    EKWS_REENTRANT = -4, EKWS_POISONED = -5, EKWS_FRONTEND = -6,
    EKWS_HEAD = -7, EKWS_CALLBACK = -8
};
enum ekws_gap_reason {
    EKWS_XRUN = 1, EKWS_ROUTE_CHANGE = 2, EKWS_CLOCK_RESET = 3, EKWS_SUSPEND_RESUME = 4
};

typedef struct {
    uint32_t keyword_id; /* 1 Hey Vosen, 2 Hey Mero; no callback for class 0. */
    uint64_t available_after_sample; /* Observed sample count, never a word-end label. */
    float confidence; /* Uncalibrated original head probability. */
} ekws_event;

typedef struct {
    uint32_t ready, keyword_id;
    uint64_t available_after_sample;
    float logits[EKWS_CLASSES], probabilities[EKWS_CLASSES];
} ekws_score;

/* Synchronous borrowed values, valid only during this call. Return zero on
 * success. A nonzero callback result poisons the stream: consumed audio and
 * already delivered events must not be replayed. No API is reentrant. */
typedef int (*ekws_event_fn)(void *context, const ekws_event *event);
typedef int (*ekws_score_fn)(void *context, const ekws_score *score);
typedef struct {
    ekws_event_fn event;
    void *event_context;
    ekws_score_fn score; /* Optional diagnostic observer; no additional inference. */
    void *score_context;
} ekws_callbacks;

typedef struct {
    uint64_t accepted_samples, embeddings, ready_outputs;
    uint64_t detected_events, delivered_events, callback_failures, feed_failures;
    uint64_t resets, discontinuities;
    uint64_t next_input_sample, last_embedding_sample, last_event_sample;
    uint32_t frontend_buffered_samples, head_history_count, last_gap_reason;
    uint32_t poisoned, counters_saturated;
} ekws_stats;

size_t ekws_stream_bytes(void);
size_t ekws_stream_alignment(void);
size_t ekws_workspace_bytes(void);
size_t ekws_workspace_alignment(void);

/* Initialize only an inactive candidate arena. All three input ranges must be
 * disjoint and aligned as queried. Failure leaves *out unchanged, but the
 * candidate arena is not usable. Head values are copied; the blob can then be
 * released. The caller must authenticate actual artifact hashes before init.
 * The workspace/arena and callback contexts must outlive the stream. */
int ekws_stream_init(void *arena, size_t arena_bytes, float *workspace,
                     size_t workspace_bytes, const unsigned char *head_blob,
                     size_t head_blob_bytes, const ekws_callbacks *callbacks,
                     ekws_stream **out);
/* first_sample is zero-based and must equal stats.next_input_sample. Arbitrary
 * chunk sizes are accepted; one call may emit several events. *consumed is
 * exact even on failure. A zero-length call emits nothing. PCM is borrowed and
 * immutable for this entire call (also inside callbacks); it must not overlap
 * the stream arena or workspace. Output/config pointers must not alias those
 * ranges or the PCM input. Reusing workspace as an input buffer is invalid. */
int ekws_stream_accept_pcm16(ekws_stream *stream, const int16_t *pcm,
                            size_t count, uint64_t first_sample, size_t *consumed);
/* Reset starts a NEW logical recording at sample 0. Both acoustic/head history
 * are cleared; cumulative counters remain and resets increments. */
int ekws_stream_reset(ekws_stream *stream);
/* Declare the next absolute observed input position after a gap, never less
 * than next_input_sample. Clear both histories and cooldown without creating
 * samples/embeddings. A backwards device clock requires a new logical reset. */
int ekws_stream_discontinuity(ekws_stream *stream, uint64_t next_input_sample,
                             enum ekws_gap_reason reason);
int ekws_stream_get_stats(const ekws_stream *stream, ekws_stats *out);
#endif
