/* SPDX-License-Identifier: Apache-2.0
 * Explicit offline research facade. No production or product-clock certification. */
#include "kws3.h"
#include "kws3_retention.h"
#include <string.h>
/* No caller-provided threshold, retention rows, reset, finish, or padding API. */
kws3_status kr_create(const char *mode, const char *recording_id, kws3_decoder **out) {
    if (out) *out = NULL;
    if (!mode || strcmp(mode, "bounded-retention-research-v1") != 0 || !out)
        return KWS3_ARGUMENT;
    return kws3_create_with_threshold(recording_id, 0.7, out);
}
kws3_status kr_step_f32(kws3_decoder *d, const float *row, size_t classes,
                        kws3_event *events, size_t cap, size_t *count) {
    return kws3_step_f32(d, row, classes, events, cap, count);
}
kws3_status kr_summary(const kws3_decoder *d, kws3_summary *out) { return kws3_get_summary(d,out); }
kws3_status kr_retention(const kws3_decoder *d, kws3_retention_summary *out) { return kws3_get_retention_summary(d,out); }
kws3_status kr_threshold(const kws3_decoder *d, double *v, uint64_t *n) { return kws3_get_threshold_status(d,v,n); }
void kr_destroy(kws3_decoder *d) { kws3_destroy(d); }
size_t kr_storage_bytes(void) { return kws3_storage_bytes(); }
size_t kr_event_bytes(void) { return sizeof(kws3_event); }
size_t kr_summary_bytes(void) { return sizeof(kws3_summary); }
size_t kr_retention_bytes(void) { return sizeof(kws3_retention_summary); }
