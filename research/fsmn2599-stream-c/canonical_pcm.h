#ifndef KWS_CANONICAL_PCM_H
#define KWS_CANONICAL_PCM_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Research-only canonical frame-step transport. No CMVN, network or decoder.
 * Caller owns state, callbacks and user storage. Single-threaded, no API calls
 * on the same state from callbacks (including init). No allocation. */
enum { KCF_OK=0, KCF_ARGUMENT=1, KCF_STATE=2, KCF_ODD_EOF=3,
       KCF_BANK=4, KCF_NONFINITE=5, KCF_SINK=6, KCF_OVERFLOW=7 };
enum { KCF_EVENT_FBANK=1, KCF_EVENT_CONTEXT=2 };
enum { KCF_PHASE_OPEN=0, KCF_PHASE_EOF=1, KCF_PHASE_CLOSED=2,
       KCF_PHASE_FAILED=3 };
typedef int (*kcf_bank_fn)(void *user, const int16_t pcm[400], float row[80]);
/* Row is 80 floats for FBANK or 400 floats for CONTEXT, valid only during call.
 * index: global fbank j or context m. center: j for FBANK or 3*m for CONTEXT.
 * end_sample: exclusive required padded-stream sample (400 + 160*j).
 * eof: this event was made available by the one true EOF tail.
 * Return 0 iff accepted; nonzero is sticky fatal failure (no retry/replay).
 * An acknowledged FBANK event always precedes its immediately-ready CONTEXT. */
typedef int (*kcf_emit_fn)(void *user, int event, uint64_t index,
                         uint64_t center, uint64_t end_sample, int eof,
                         const float *row);
typedef struct {
    uint32_t magic;
    int phase, busy, error, callback_error, failed_event;
    uint64_t failed_index;
    size_t used;
    unsigned have_low;
    uint8_t low;
    uint64_t input_bytes, real_samples, padded_samples;
    uint64_t fbank_attempted, fbank_frames, fbank_accepted;
    uint64_t context_attempted, context_accepted;
    unsigned eof_attempted, eof_completed, zero_samples;
    int16_t pcm[400];
    float history[5][80];
    float context[400];
    kcf_bank_fn bank;
    kcf_emit_fn emit;
    void *bank_user, *emit_user;
} kcf_pcm;
int kcf_pcm_init(kcf_pcm *s, kcf_bank_fn bank, void *bank_user,
                 kcf_emit_fn emit, void *emit_user);
/* Arbitrary little-endian byte chunks, including odd boundaries and zero size.
 * consumed is mandatory. On failure it identifies the consumed prefix; the
 * stream is poisoned after processing/callback failure, never silently skips.
 * Caller must not alias data/consumed with state or mutate it during a call. */
int kcf_pcm_feed(kcf_pcm *s, const uint8_t *data, size_t bytes, size_t *consumed);
/* Only at real recording EOF, once. Reject a pending odd byte BEFORE adding
 * zeros. Append exactly 4800 samples, retain residual PCM for audit, never pad
 * right context, never flush at I/O or decoder-window boundaries. */
int kcf_pcm_finish(kcf_pcm *s);
#ifdef __cplusplus
}
#endif
#endif
