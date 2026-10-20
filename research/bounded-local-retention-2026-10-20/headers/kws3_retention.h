/* SPDX-License-Identifier: Apache-2.0
 * Research-only fixed five-row local-retention policy, version 1.
 * Read-only diagnostics: querying does not mean EOF and does not finalize/reset.
 */
#ifndef KWS3_RETENTION_H
#define KWS3_RETENTION_H
#include "kws3.h"
#ifdef __cplusplus
extern "C" {
#endif
#define KWS3_RETENTION_VERSION 1
#define KWS3_RETENTION_ROWS INT64_C(5)
#define KWS3_RETENTION_INACTIVE INT64_C(-1)
typedef struct {
    int64_t start, stop, last_active_pos, anchor_row, deadline_row;
    uint64_t hypothesis_count;
} kws3_retention_window;
typedef struct {
    uint64_t episodes_started, low_repeated, strong_closed, expired;
    uint64_t window_censored, active_episodes, peak_active_episodes;
    uint64_t rows_committed;
    uint32_t live_windows, version;
    kws3_retention_window windows[KWS3_WINDOWS];
} kws3_retention_summary;
/* These are committed-state counters only. Failed-row attempts remain visible
 * in the original diagnostic ledger, but never increment this summary.
 * At natural EOF the caller labels active_episodes as right-censored externally.
 * window_censored episodes have already retired and must not be counted twice.
 * A decoder whose legacy finish was called is unsupported: returns KWS3_FINISHED.
 */
kws3_status kws3_get_retention_summary(const kws3_decoder *decoder,
                                       kws3_retention_summary *out);
size_t kws3_retention_summary_bytes(void);
size_t kws3_retention_window_bytes(void);
#ifdef __cplusplus
}
#endif
#endif
