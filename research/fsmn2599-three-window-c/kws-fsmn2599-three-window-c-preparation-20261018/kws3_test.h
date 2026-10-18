/* SPDX-License-Identifier: Apache-2.0. Test-only white-box ABI. */
#ifndef KWS3_TEST_H
#define KWS3_TEST_H
#ifndef KWS3_TESTING
#error "Test hooks must not be exposed by production builds"
#endif
#include "kws3.h"
typedef struct { int64_t frame; double prob; } kws3_test_node;
typedef struct {
    double pb,pnb;
    size_t length;
    uint16_t tokens[KWS3_WINDOW_ROWS];
    kws3_test_node nodes[KWS3_WINDOW_ROWS];
} kws3_test_hyp;
kws3_status kws3_test_top3(kws3_decoder *, const double *, size_t,
                         uint16_t ids[3], double probabilities[3]);
kws3_status kws3_test_beam(kws3_decoder *, const double *,
                         const kws3_test_hyp *, size_t, int64_t,
                         kws3_test_hyp *, size_t, size_t *);
kws3_status kws3_test_snapshot(const kws3_decoder *, size_t window_index,
                         int64_t *start, int64_t *last_active,
                         kws3_test_hyp *, size_t, size_t *);
kws3_status kws3_test_detect_seed(const kws3_test_hyp *, size_t,
                         int64_t last_active, int64_t decision_row,
                         kws3_event *, int *activated, int64_t *new_last_active);
/* detect_seed: when inactive, only event.score is a diagnostic value (zero
 * if no matching keyword). It is not an emitted candidate event. */
kws3_status kws3_test_merge(kws3_decoder *, const kws3_event *, size_t,
                         int64_t row, kws3_event *, size_t, size_t *);
#endif
