/* SPDX-License-Identifier: Apache-2.0
 * R1 research PCM stream. Fixed exported model identity, no hidden EOF audio.
 * Single threaded. Callback must not re-enter, destroy or mutate this handle.
 * Input buffers must remain valid throughout synchronous feed. */
#ifndef R1_STREAM_H
#define R1_STREAM_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define R1_OUTPUTS 2601u
/* feature/logit pointers are valid only within the callback. source_end is
 * the exact exclusive input dependency end, NOT a keyword end annotation.
 * delivery_end includes transport batching. No synthetic rows are emitted. */
typedef int (*r1_sink)(void *,uint64_t,uint64_t,uint64_t,const float *,const float *);
typedef struct r1_stream r1_stream;
r1_stream *r1_create(const unsigned char *model,size_t model_bytes,
    const unsigned char *tables,size_t table_bytes,r1_sink sink,void *user);
int r1_feed(r1_stream *s,const unsigned char *data,size_t bytes,size_t *consumed);
int r1_finish(r1_stream *s);
int r1_reset(r1_stream *s);
void r1_destroy(r1_stream *s);
size_t r1_state_bytes(void);
size_t r1_weight_bytes(void);
uint64_t r1_frames(const r1_stream *s);
/* Network-only validation path; fixed400D features. Not an audio model bypass.
 * Must not mix with audio feed on the same handle; reset between modes. */
int r1_feature(r1_stream *s,const float *feature,float *logits);
#ifdef __cplusplus
}
#endif
#endif
