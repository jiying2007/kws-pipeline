#include "mero_ctc.h"
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

int en_v5_mero_init(en_v5_mero_state *s, uint64_t max_gap_samples,
                    uint64_t cooldown_samples) {
    if (!s || max_gap_samples==0u || max_gap_samples>UINT64_MAX/2u ||
        cooldown_samples>UINT64_MAX/2u) return EN_V5_INVALID;
    memset(s,0,sizeof(*s));
    s->max_token_gap_samples=max_gap_samples;
    s->cooldown_samples=cooldown_samples;
    return EN_V5_OK;
}

int en_v5_mero_reset(en_v5_mero_state *s) {
    uint64_t gap,cool,count;
    if (!s || !s->max_token_gap_samples) return EN_V5_INVALID;
    gap=s->max_token_gap_samples;
    cool=s->cooldown_samples;
    count=s->event_count;
    memset(s,0,sizeof(*s));
    s->max_token_gap_samples=gap;
    s->cooldown_samples=cool;
    s->event_count=count;
    return EN_V5_OK;
}

int en_v5_mero_discontinuity(en_v5_mero_state *s,
                             uint64_t next_sample) {
    if (!s || !s->max_token_gap_samples) return EN_V5_INVALID;
    if (s->clock_initialized && next_sample<=s->last_sample)
        return EN_V5_CLOCK;
    s->progress=0u;
    s->previous_argmax=0u;
    s->clock_initialized=1u;
    s->last_sample=next_sample;
    s->last_progress_sample=0u;
    s->cooldown_until=0u;
    return EN_V5_OK;
}

static uint64_t bounded_add(uint64_t a,uint64_t b) {
    return b>UINT64_MAX-a ? UINT64_MAX : a+b;
}

int en_v5_mero_accept(en_v5_mero_state *s, const float logits[4],
                       uint64_t available_after_sample,
                       en_v5_mero_event *out) {
    unsigned tok=0u, i;
    float maximum;
    if (!s || !logits || !out || !s->max_token_gap_samples)
        return EN_V5_INVALID;
    if (s->clock_initialized && available_after_sample<=s->last_sample)
        return EN_V5_CLOCK;
    for (i=0u;i<EN_V5_MERO_TOKEN_COUNT;++i) {
        if (!isfinite(logits[i])) return EN_V5_NONFINITE;
    }
    maximum=logits[0];
    for (i=1u;i<EN_V5_MERO_TOKEN_COUNT;++i) {
        if (logits[i]>maximum) {maximum=logits[i]; tok=i;}
    }
    /* State and event are committed only after all validation succeeds. */
    out->detected=0u;
    out->keyword_id=0u;
    out->available_after_sample=available_after_sample;
    s->clock_initialized=1u;
    s->last_sample=available_after_sample;
    if (s->progress!=0u &&
        available_after_sample-s->last_progress_sample>
            s->max_token_gap_samples) {
        s->progress=0u;
    }
    if (available_after_sample<s->cooldown_until) {
        s->progress=0u;
        s->previous_argmax=tok;
        return EN_V5_OK;
    }
    /* Standard greedy CTC collapse: repetition without an intervening
     * blank is one token, blank is non-emitting but breaks repetition. */
    if (tok==s->previous_argmax) return EN_V5_OK;
    s->previous_argmax=tok;
    if (tok==0u) return EN_V5_OK;
    if (tok==1u) s->progress=1u;
    else if (tok==2u && s->progress==1u) ++s->progress;
    else if (tok==3u && s->progress==2u) {
        s->progress=0u;
        s->last_progress_sample=available_after_sample;
        s->cooldown_until=bounded_add(available_after_sample,
                                      s->cooldown_samples);
        if (s->event_count!=UINT64_MAX) ++s->event_count;
        out->detected=1u;
        out->keyword_id=EN_V5_MERO_KEYWORD_ID;
        return EN_V5_OK;
    } else s->progress=0u;
    if (s->progress>0u) s->last_progress_sample=available_after_sample;
    return EN_V5_OK;
}
