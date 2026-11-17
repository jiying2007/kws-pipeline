#ifndef EKWS_STREAM_TRACE_FORMAT_H
#define EKWS_STREAM_TRACE_FORMAT_H
#include <stdint.h>

/* Numeric diagnostic only. Link the ORIGINAL installed archive with
 * -Wl,--wrap=ehead_step. Do not use this consumer for performance measurements.
 *
 * All integers are unsigned little-endian. Every float is its exact IEEE754
 * binary32 bit pattern encoded as little-endian uint32, including signed zero.
 * No C structs are written and there is no implicit padding or alignment.
 *
 * Header (40 bytes): "EKWSTRC1" (8 bytes, no NUL), u32 format_version,
 * u32 public_ABI, u32 sample_rate, u32 chunk_samples, u32 embedding_dimension,
 * u32 class_count, u64 input_samples.
 *
 * Each record: u8 type, u32 payload_bytes, then exactly that many bytes.
 * EMBEDDING (400-byte payload): u64 zero_based_embedding_ordinal,
 *   u64 available_after_sample, 96 x f32 embedding in original index order.
 * SCORE (48-byte payload): u64 matching_embedding_ordinal, u32 ready,
 *   u32 keyword_id, u64 available_after_sample, 3 x f32 logits then
 *   3 x f32 probabilities, each in original class order 0, 1, 2.
 * EVENT (32-byte payload): u64 zero_based_event_ordinal,
 *   u64 matching_embedding_ordinal, u32 keyword_id,
 *   u64 available_after_sample, f32 confidence.
 * END (48-byte payload): u64 accepted_samples, u64 embedding_count,
 *   u64 score_count, u64 ready_count, u64 event_count, u64 feed_call_count.
 *
 * Order: EMBEDDING, SCORE, optional EVENT, repeated; one END last. SCORE
 * includes the first 15 not-ready outputs, not only ready outputs. A successful
 * 16-second input has 188 embeddings, 188 scores and 173 ready scores.
 * No time, path, source-build identity or performance value enters the stream.
 * An incomplete file or missing END is never a valid numeric-parity result.
 */
#define EKWS_TRACE_FORMAT_VERSION 1u
#define EKWS_TRACE_HEADER_BYTES 40u
#define EKWS_TRACE_RECORD_PREFIX_BYTES 5u
#define EKWS_TRACE_EMBED_DIM 96u
#define EKWS_TRACE_CLASSES 3u
#define EKWS_TRACE_EMBED_PAYLOAD_BYTES 400u
#define EKWS_TRACE_SCORE_PAYLOAD_BYTES 48u
#define EKWS_TRACE_EVENT_PAYLOAD_BYTES 32u
#define EKWS_TRACE_END_PAYLOAD_BYTES 48u
enum ekws_trace_record_type {
    EKWS_TRACE_EMBEDDING = 1,
    EKWS_TRACE_SCORE = 2,
    EKWS_TRACE_EVENT = 3,
    EKWS_TRACE_END = 255
};
#endif
