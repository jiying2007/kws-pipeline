#include "ekws_research_stream.h"
#include "fixed_frontend.h"
#include "fixed_models.h"
#include "english_causal_head.h"
#include <string.h>

#define EKWS_MAGIC UINT32_C(0x45525331)
struct ekws_stream {
    uint32_t magic;
    int busy, poisoned, callback_status;
    ef_frontend frontend;
    ehead_model model;
    ehead_state head;
    ekws_callbacks callbacks;
    ekws_stats stats;
};

static int valid(const ekws_stream *s) { return s && s->magic == EKWS_MAGIC; }
static void add(ekws_stream *s, uint64_t *value, uint64_t delta) {
    if (delta > UINT64_MAX - *value) {
        *value = UINT64_MAX; s->stats.counters_saturated = 1;
    } else *value += delta;
}
static int disjoint(const void *a, size_t an, const void *b, size_t bn) {
    uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    if (an > UINTPTR_MAX - x || bn > UINTPTR_MAX - y) return 0;
    return x + an <= y || y + bn <= x;
}

size_t ekws_stream_bytes(void) { return sizeof(ekws_stream); }
size_t ekws_stream_alignment(void) { return _Alignof(ekws_stream); }
size_t ekws_workspace_bytes(void) {
    return ef_frontend_workspace_bytes(&ef_fixed_mel, &ef_fixed_embedding);
}
size_t ekws_workspace_alignment(void) { return _Alignof(float); }

int ekws_stream_init(void *arena, size_t arena_bytes, float *workspace,
                     size_t workspace_bytes, const unsigned char *blob,
                     size_t blob_bytes, const ekws_callbacks *callbacks,
                     ekws_stream **out) {
    ekws_stream *s;
    ekws_callbacks chosen;
    size_t needed = ekws_workspace_bytes();
    if (!arena || !workspace || !blob || !callbacks || !callbacks->event || !out ||
        blob_bytes != EKWS_HEAD_BLOB_BYTES || EHEAD_BLOB_BYTES != EKWS_HEAD_BLOB_BYTES)
        return EKWS_INVALID;
    if (!needed || arena_bytes < sizeof(ekws_stream) || workspace_bytes < needed ||
        (uintptr_t)arena % _Alignof(ekws_stream) || (uintptr_t)workspace % _Alignof(float) ||
        !disjoint(arena, sizeof(ekws_stream), workspace, workspace_bytes) ||
        !disjoint(arena, sizeof(ekws_stream), blob, blob_bytes) ||
        !disjoint(workspace, workspace_bytes, blob, blob_bytes)) return EKWS_MEMORY;
    chosen = *callbacks;
    s = (ekws_stream *)arena;
    memset(s, 0, sizeof(*s));
    if (ehead_load(&s->model, blob, blob_bytes)) return EKWS_HEAD;
    if (ef_frontend_init(&s->frontend, &ef_fixed_mel, &ef_fixed_embedding,
                         workspace, workspace_bytes / sizeof(float)) != EF_OK) return EKWS_FRONTEND;
    ehead_reset(&s->head);
    s->callbacks = chosen; s->magic = EKWS_MAGIC;
    *out = s;
    return EKWS_OK;
}

static int embedding_ready(void *context, uint64_t sample, const float embedding[EF_EMBED_DIM]) {
    ekws_stream *s = (ekws_stream *)context;
    ehead_output head;
    ekws_score score;
    ekws_event event;
    unsigned k;
    add(s, &s->stats.embeddings, 1);
    s->stats.last_embedding_sample = sample;
    if (ehead_step(&s->model, &s->head, embedding, sample, &head)) {
        s->callback_status = EKWS_HEAD; return -1;
    }
    if (head.ready) add(s, &s->stats.ready_outputs, 1);
    if (head.keyword) {
        add(s, &s->stats.detected_events, 1);
        s->stats.last_event_sample = sample;
    }
    if (s->callbacks.score) {
        score.ready = (uint32_t)head.ready; score.keyword_id = head.keyword;
        score.available_after_sample = head.available_after_sample;
        for (k = 0; k < EKWS_CLASSES; ++k) {
            score.logits[k] = head.logits[k]; score.probabilities[k] = head.probabilities[k];
        }
        if (s->callbacks.score(s->callbacks.score_context, &score)) {
            add(s, &s->stats.callback_failures, 1); s->callback_status = EKWS_CALLBACK; return -1;
        }
    }
    if (head.keyword) {
        event.keyword_id = head.keyword; event.available_after_sample = sample;
        event.confidence = head.probabilities[head.keyword];
        if (s->callbacks.event(s->callbacks.event_context, &event)) {
            add(s, &s->stats.callback_failures, 1); s->callback_status = EKWS_CALLBACK; return -1;
        }
        add(s, &s->stats.delivered_events, 1);
    }
    return 0;
}

int ekws_stream_accept_pcm16(ekws_stream *s, const int16_t *pcm, size_t count,
                            uint64_t first, size_t *consumed) {
    int result;
    if (consumed) *consumed = 0;
    if (!valid(s) || !consumed || (count && !pcm)) return EKWS_INVALID;
    if (s->busy) return EKWS_REENTRANT;
    if (s->poisoned) return EKWS_POISONED;
    if (first != s->frontend.observed || count > UINT64_MAX - first ||
        (count && first + count > UINT64_MAX - 13 * EHEAD_HOP_SAMPLES)) {
        add(s, &s->stats.feed_failures, 1); return EKWS_CLOCK;
    }
    s->busy = 1; s->callback_status = EKWS_OK;
    result = ef_frontend_feed(&s->frontend, pcm, count, first, embedding_ready, s, consumed);
    add(s, &s->stats.accepted_samples, (uint64_t)*consumed);
    s->busy = 0;
    if (result != EF_OK) {
        s->poisoned = 1; add(s, &s->stats.feed_failures, 1);
        return s->callback_status != EKWS_OK ? s->callback_status : EKWS_FRONTEND;
    }
    return EKWS_OK;
}

int ekws_stream_reset(ekws_stream *s) {
    if (!valid(s)) return EKWS_INVALID;
    if (s->busy) return EKWS_REENTRANT;
    ef_frontend_reset(&s->frontend); ehead_reset(&s->head);
    s->poisoned = 0; s->callback_status = EKWS_OK;
    s->stats.last_embedding_sample = s->stats.last_event_sample = 0;
    s->stats.last_gap_reason = 0;
    add(s, &s->stats.resets, 1);
    return EKWS_OK;
}

int ekws_stream_discontinuity(ekws_stream *s, uint64_t next, enum ekws_gap_reason reason) {
    if (!valid(s) || reason < EKWS_XRUN || reason > EKWS_SUSPEND_RESUME) return EKWS_INVALID;
    if (s->busy) return EKWS_REENTRANT;
    if (next > UINT64_MAX - EF_PCM_WINDOW - 13 * EHEAD_HOP_SAMPLES ||
        ef_frontend_discontinuity(&s->frontend, next) != EF_OK) return EKWS_CLOCK;
    ehead_discontinuity(&s->head);
    s->poisoned = 0; s->callback_status = EKWS_OK;
    s->stats.last_gap_reason = (uint32_t)reason;
    add(s, &s->stats.discontinuities, 1);
    return EKWS_OK;
}

int ekws_stream_get_stats(const ekws_stream *s, ekws_stats *out) {
    if (!valid(s) || !out) return EKWS_INVALID;
    if (s->busy) return EKWS_REENTRANT;
    *out = s->stats;
    out->next_input_sample = s->frontend.observed;
    out->frontend_buffered_samples = s->frontend.filled;
    out->head_history_count = s->head.count;
    out->poisoned = (uint32_t)(s->poisoned != 0);
    return EKWS_OK;
}
