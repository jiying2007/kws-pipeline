#ifndef KWS_CANONICAL_FBANK_H
#define KWS_CANONICAL_FBANK_H
#include <stddef.h>
#include <stdint.h>
/* Fixed researched contract, no generic configurable frontend. Tables must be
 * exported from the frozen runtime; this package deliberately has NO fake or
 * donor-Hamming fallback. Descriptor is a shape/version gate, not a hash proof:
 * runner must verify export manifest and array hashes before calling init.
 * Caller must preserve table lifetime/immutability throughout all calls. */
#define KCF_TABLE_SCHEMA UINT32_C(1)
typedef struct {
    uint32_t schema;
    size_t povey_count, mel_count;
    const float *povey; /* 400 float32; Torch hann(periodic=False).pow(.85) */
    const float *mel;   /* dense row-major 80*257, exact zero Nyquist column */
} kcf_tables;
typedef struct {
    uint32_t magic;
    const kcf_tables *tables;
    float re[512], im[512];
} kcf_fbank;
typedef struct {
    float dc[400], preemphasis[400], windowed[512];
    float magnitude[257], power[257], mel[80], logfbank[80];
} kcf_trace;
enum { KCF_FBANK_OK=0, KCF_FBANK_ARGUMENT=1, KCF_FBANK_TABLE=2,
       KCF_FBANK_STATE=3, KCF_FBANK_NONFINITE=4 };
int kcf_fbank_init(kcf_fbank *s, const kcf_tables *tables);
/* Exact 400 raw PCM16 amplitudes -> 80 raw logfbank float32. No scaling/CMVN.
 * Optional trace is for later diagnostic stage comparisons. Caller must not
 * overlap state, pcm, output, trace or immutable tables; not reentrant. */
int kcf_fbank_frame(kcf_fbank *s, const int16_t pcm[400], float row[80],
                    kcf_trace *trace);
int kcf_fbank_callback(void *state, const int16_t pcm[400], float row[80]);
#endif
