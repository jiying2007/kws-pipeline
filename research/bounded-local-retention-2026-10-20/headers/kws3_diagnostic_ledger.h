/* Source-only diagnostic ledger. No decoder execution is authorized by this file. */
#ifndef KWS3_DIAGNOSTIC_LEDGER_H
#define KWS3_DIAGNOSTIC_LEDGER_H
#include <stddef.h>
#include <stdint.h>
#define KWS3_LEDGER_CAPACITY 32768
/* A capture session must contain exactly one decoder, on one thread. */
typedef struct {
    uint64_t sequence;
    const char *kind;
    int64_t row, window_start, last_active_or_accepted;
    uint32_t keyword_id;
    size_t path_index;
    int node_offset;
    int64_t node_frames[4];
    double node_probabilities[4], score, threshold;
    int has_nodes, has_probabilities, has_threshold;
    int duration_too_short, duration_too_long, before_last_active;
    size_t hypothesis_count;
    int decoder_status;
} kws3_ledger_record;
typedef int (*kws3_ledger_sink)(void *, const kws3_ledger_record *);
/* Call begin before a capture; never during an active decoder call. */
void kws3_ledger_begin(void);
/* sticky validity: zero on capacity loss, missing sink or callback failure. */
int kws3_ledger_valid(void);
size_t kws3_ledger_size(void);
size_t kws3_ledger_record_size(void);
uint64_t kws3_ledger_dropped(void);
const kws3_ledger_record *kws3_ledger_records(void);
/* Invoke only AFTER the decoder call returns. Appends are never drained in decoder code.
 * Success consumes pending records without erasing their session sequence.
 * Sink must not call any decoder/ledger function; return zero on successful append.
 * Caller owns serialization, session/recording identity and persistence. */
int kws3_ledger_drain(kws3_ledger_sink sink, void *user);
#endif
