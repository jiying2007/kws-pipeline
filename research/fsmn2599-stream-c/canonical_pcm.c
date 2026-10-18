/* New independent implementation of the frozen Python frame-step contract.
 * Source and provenance: NOTICE.md, source-binding.json. Apache-2.0. */
#include "canonical_pcm.h"
#include <math.h>
#include <string.h>
#define KCF_MAGIC UINT32_C(0x4b434631)
_Static_assert(sizeof(kcf_pcm) <= 8192, "bounded stream state");

static int fail(kcf_pcm *s, int error, int event, uint64_t index, int cb) {
    s->error=error; s->failed_event=event; s->failed_index=index;
    s->callback_error=cb; s->phase=KCF_PHASE_FAILED;
    return error;
}
static int push_sample(kcf_pcm *s, int16_t sample) {
    float row[80];
    int rc;
    uint64_t j;
    s->pcm[s->used++]=sample;
    s->padded_samples++;
    if (s->used < 400) return KCF_OK;
    j=s->fbank_frames;
    s->fbank_attempted++;
    for (unsigned k=0;k<80;k++) row[k]=NAN;
    rc=s->bank(s->bank_user,s->pcm,row);
    if (rc) return fail(s,KCF_BANK,KCF_EVENT_FBANK,j,rc);
    for (unsigned k=0;k<80;k++)
        if (!isfinite(row[k])) return fail(s,KCF_NONFINITE,KCF_EVENT_FBANK,j,0);
    s->fbank_frames++;
    rc=s->emit(s->emit_user,KCF_EVENT_FBANK,j,j,s->padded_samples,
               s->phase==KCF_PHASE_EOF,row);
    if (rc) return fail(s,KCF_SINK,KCF_EVENT_FBANK,j,rc);
    s->fbank_accepted++;
    memcpy(s->history[j%5],row,sizeof(row));
    if (j>=2 && (j-2)%3==0) {
        uint64_t center=j-2, m=center/3;
        /* For center 0: frame indices [0,0,0,1,2]. Later: center-2..center+2.
         * Ring contains exactly the newest five rows, no growing buffer. */
        for (unsigned k=0;k<5;k++) {
            uint64_t index=(center+k>=2)?center+k-2:0;
            memcpy(s->context+k*80,s->history[index%5],80*sizeof(float));
        }
        s->context_attempted++;
        rc=s->emit(s->emit_user,KCF_EVENT_CONTEXT,m,center,s->padded_samples,
                   s->phase==KCF_PHASE_EOF,s->context);
        if (rc) return fail(s,KCF_SINK,KCF_EVENT_CONTEXT,m,rc);
        s->context_accepted++;
    }
    memmove(s->pcm,s->pcm+160,240*sizeof(int16_t));
    s->used=240;
    return KCF_OK;
}
int kcf_pcm_init(kcf_pcm *s, kcf_bank_fn bank, void *bank_user,
                 kcf_emit_fn emit, void *emit_user) {
    if (!s || !bank || !emit) return KCF_ARGUMENT;
    memset(s,0,sizeof(*s));
    s->magic=KCF_MAGIC; s->bank=bank; s->bank_user=bank_user;
    s->emit=emit; s->emit_user=emit_user;
    return KCF_OK;
}
int kcf_pcm_feed(kcf_pcm *s, const uint8_t *data, size_t bytes, size_t *consumed) {
    int rc=KCF_OK;
    if (!consumed) return KCF_ARGUMENT;
    *consumed=0;
    if (!s || (!data && bytes)) return KCF_ARGUMENT;
    if (s->magic!=KCF_MAGIC || s->busy) return KCF_STATE;
    if (s->error) return s->error;
    if (s->phase!=KCF_PHASE_OPEN) return KCF_STATE;
    /* Reserve tail headroom; this also bounds all frame/index arithmetic. */
    if (bytes > UINT64_MAX-UINT64_C(9600)-s->input_bytes)
        return fail(s,KCF_OVERFLOW,0,0,0);
    s->busy=1;
    for (size_t i=0;i<bytes;i++) {
        uint8_t byte=data[i];
        s->input_bytes++; *consumed=i+1;
        if (!s->have_low) {s->low=byte; s->have_low=1;}
        else {
            uint32_t word=(uint32_t)s->low|((uint32_t)byte<<8);
            /* Avoid implementation-defined uint16_t -> int16_t overflow. */
            int32_t signed_word=(word<32768u)?(int32_t)word:(int32_t)word-65536;
            s->have_low=0; s->real_samples++;
            rc=push_sample(s,(int16_t)signed_word);
            if (rc) break;
        }
    }
    s->busy=0;
    return rc;
}
int kcf_pcm_finish(kcf_pcm *s) {
    int rc=KCF_OK;
    if (!s) return KCF_ARGUMENT;
    if (s->magic!=KCF_MAGIC || s->busy) return KCF_STATE;
    if (s->error) return s->error;
    if (s->phase!=KCF_PHASE_OPEN || s->eof_attempted) return KCF_STATE;
    s->eof_attempted=1;
    if (s->have_low) return fail(s,KCF_ODD_EOF,0,0,0);
    s->busy=1; s->phase=KCF_PHASE_EOF;
    for (unsigned i=0;i<4800;i++) {
        s->zero_samples++;
        rc=push_sample(s,0);
        if (rc) break;
    }
    if (!rc) {s->eof_completed=1; s->phase=KCF_PHASE_CLOSED;}
    s->busy=0;
    return rc;
}
